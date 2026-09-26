// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Burn port of ARDY's two-stage denoiser and causal FSQ tokenizer.

use crate::{config::ArdyConfig, weights::Weights};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{
        Bool, Tensor, TensorData, activation, module::attention, ops::AttentionModuleOptions,
    },
};
use burn_human_motion::artifacts::{Manifest, PartReader};

pub struct Ardy<B: Backend> {
    pub config: ArdyConfig,
    pub weights: Weights<B>,
}

impl<B: Backend> Ardy<B> {
    pub async fn load(
        manifest: &Manifest,
        reader: &mut impl PartReader,
        device: &B::Device,
        progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        let (weights, config) = Weights::load(manifest, reader, device, progress).await?;
        Ok(Self { config, weights })
    }

    /// Post-norm transformer with Burn's fused attention primitive. No explicit
    /// sequence-by-sequence score tensor is constructed by this implementation.
    fn transformer(
        &self,
        name: &str,
        mut x: Tensor<B, 3>,
        padding: Option<Tensor<B, 4, Bool>>,
        causal: bool,
        heads: usize,
    ) -> Tensor<B, 3> {
        let [batch, time, dim] = x.dims();
        for i in 0..8 {
            let p = format!("{name}.seqTransEncoder.layers.{i}");
            let qkv = self.weights.linear_named(
                &format!("{p}.self_attn.in_proj_weight"),
                &format!("{p}.self_attn.in_proj_bias"),
                x.clone(),
            );
            let project = |start| {
                qkv.clone()
                    .slice([0..batch, 0..time, start..start + dim])
                    .reshape([batch, time, heads, dim / heads])
                    .swap_dims(1, 2)
            };
            let attended = attention(
                project(0),
                project(dim),
                project(2 * dim),
                padding.clone(),
                None,
                AttentionModuleOptions {
                    is_causal: causal,
                    ..Default::default()
                },
            )
            .swap_dims(1, 2)
            .reshape([batch, time, dim]);
            x = self.weights.norm(
                &format!("{p}.norm1"),
                x + self
                    .weights
                    .linear(&format!("{p}.self_attn.out_proj"), attended),
            );
            // PyTorch uses exact (erf) GELU for these checkpoints.
            let ff = self.weights.linear(&format!("{p}.linear1"), x.clone());
            let ff = activation::gelu(ff);
            x = self.weights.norm(
                &format!("{p}.norm2"),
                x + self.weights.linear(&format!("{p}.linear2"), ff),
            );
        }
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn backbone(
        &self,
        name: &str,
        x: Tensor<B, 3>,
        valid: Tensor<B, 2, Bool>,
        text: Tensor<B, 3>,
        step: usize,
        heading: Tensor<B, 2>,
        history_tokens: usize,
    ) -> Tensor<B, 3> {
        let [batch, time, dim] = x.dims();
        let device = &self.weights.device;
        let text = self.weights.linear(&format!("{name}.embed_text"), text);
        let step_pe = position_encoding::<B>(1, dim, step as isize, device).expand([batch, 1, dim]);
        let step = self
            .weights
            .linear(&format!("{name}.embed_timestep.time_embed.0"), step_pe);
        let step = self.weights.linear(
            &format!("{name}.embed_timestep.time_embed.2"),
            activation::silu(step),
        );
        let heading =
            Tensor::cat(vec![heading.clone().cos(), heading.sin()], 1).reshape([batch, 1, 2]);
        let heading = self
            .weights
            .linear(&format!("{name}.linear_first_heading_angle"), heading);
        let prefix = Tensor::cat(vec![text, step, heading], 1)
            + self
                .weights
                .tensor::<2>(&format!("{name}.learned_prefix_embedding.embedding.weight"))
                .unsqueeze_dim(0);
        let x = x + position_encoding(time, dim, -(history_tokens as isize), device);
        let input = Tensor::cat(vec![prefix, x], 1);
        let prefix_valid = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(vec![true; batch * 3], [batch, 3]),
            device,
        );
        let mask = Tensor::cat(vec![prefix_valid, valid], 1)
            .bool_not()
            .reshape([batch, 1, 1, time + 3])
            .expand([batch, 8, time + 3, time + 3]);
        let output = self.transformer(name, input, Some(mask), false, 8).slice([
            0..batch,
            3..time + 3,
            0..dim,
        ]);
        self.weights
            .linear(&format!("{name}.output_linear"), output)
    }

    /// A dense window supports a shared history length, 40 generation frames,
    /// and sparse future constraints. Batches may have different conditioning.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        x: Tensor<B, 3>,
        text: Tensor<B, 3>,
        heading: Tensor<B, 2>,
        observed: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        history_frames: usize,
        step: usize,
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens, dim] = x.dims();
        let frames = tokens * 4;
        ensure!(
            dim == 148
                && batch > 0
                && tokens <= 50
                && history_frames.is_multiple_of(4)
                && history_frames + 40 <= frames
                && step < 10,
            "invalid denoiser window"
        );
        ensure!(
            text.dims() == [batch, 1, 4096]
                && heading.dims() == [batch, 1]
                && observed.dims() == [batch, frames, 330]
                && mask.dims() == [batch, frames, 330],
            "conditioning shape mismatch"
        );
        let device = &self.weights.device;
        let h = history_frames / 4;
        let role = |start: usize, end: usize| {
            Tensor::<B, 3>::from_data(
                TensorData::new(
                    (0..batch * tokens)
                        .map(|i| {
                            if (start..end).contains(&(i % tokens)) {
                                1.0f32
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>(),
                    [batch, tokens, 1],
                ),
                device,
            )
        };
        let history = role(0, h);
        let generation = role(h, h + 10);
        let future = role(h + 10, tokens)
            * mask
                .clone()
                .reshape([batch, tokens, 4 * 330])
                .sum_dim(2)
                .greater_elem(0.0)
                .float();
        let valid = (history.clone() + generation.clone() + future.clone())
            .squeeze_dim::<2>(2)
            .greater_elem(0.0);
        let repeat_frames = |v: Tensor<B, 3>| {
            v.unsqueeze_dim::<4>(2)
                .expand([batch, tokens, 4, 1])
                .reshape([batch, frames, 1])
        };
        let root = x
            .clone()
            .slice([0..batch, 0..tokens, 0..20])
            .reshape([batch, frames, 5]);
        let latent = x.clone().slice([0..batch, 0..tokens, 20..148]);
        let root_mask = mask.clone().slice([0..batch, 0..frames, 0..5]);
        let root = root * (root_mask.clone().neg() + 1.0)
            + observed.clone().slice([0..batch, 0..frames, 0..5]) * root_mask;
        let body_obs = observed
            .slice([0..batch, 0..frames, 5..330])
            .reshape([batch, tokens, 1300]);
        let mask = mask.reshape([batch, tokens, 1320]);
        // This order is intentionally different from the history hybrid [root, latent].
        let extended = Tensor::cat(
            vec![
                latent.clone(),
                root.clone().reshape([batch, tokens, 20]),
                body_obs.clone(),
                mask.clone(),
            ],
            2,
        );
        let future_proj = self.weights.linear(
            "future_constraints_proj",
            extended.clone().slice([0..batch, 0..tokens, 128..2768]),
        ) * future;
        let root_input = self
            .weights
            .linear("global_root_hybrid_constraints_proj", extended)
            * generation.clone()
            + self.weights.linear("global_root_hybrid_proj", x) * history.clone()
            + future_proj.clone();
        let root_pred = self
            .backbone(
                "root_model",
                root_input,
                valid.clone(),
                text.clone(),
                step,
                heading.clone(),
                h,
            )
            .reshape([batch, frames, 5]);
        let root_pred =
            root_pred * repeat_frames(generation.clone()) + root * repeat_frames(history.clone());
        let local_root = self
            .local_root(root_pred.clone(), history_frames + 40)
            .reshape([batch, tokens, 16]);
        let body_input = Tensor::cat(vec![local_root, latent.clone()], 2);
        let body_extended = Tensor::cat(vec![body_input.clone(), body_obs, mask], 2);
        let body_input = self
            .weights
            .linear("local_root_hybrid_constraints_proj", body_extended)
            * generation.clone()
            + self.weights.linear("local_root_hybrid_proj", body_input) * history.clone()
            + future_proj;
        let body_pred = self.backbone("body_model", body_input, valid, text, step, heading, h)
            * generation
            + latent * history;
        Ok(Tensor::cat(
            vec![root_pred.reshape([batch, tokens, 20]), body_pred],
            2,
        ))
    }

    pub fn encode(&self, motion: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, frames, dim] = motion.dims();
        ensure!(
            dim == 330 && frames >= 4 && frames.is_multiple_of(4) && frames <= 200,
            "invalid encoder input"
        );
        let tokens = frames / 4;
        let body = motion
            .clone()
            .slice([0..batch, 0..frames, 5..330])
            .reshape([batch, tokens, 1300]);
        let x = self.weights.linear("encoder.input_proj", body)
            + position_encoding(tokens, 512, 0, &self.weights.device);
        let x = self.transformer("encoder", x, None, true, 4);
        let x = self.weights.linear("encoder.output_proj", x);
        let half = 63.0f32 * 1.001 / 2.0;
        let quantized = (((x + (0.5 / half).atanh()).tanh() * half - 0.5).round()) / 32.0;
        let latent = self.normalize_latent(quantized);
        Ok(Tensor::cat(
            vec![
                motion
                    .slice([0..batch, 0..frames, 0..5])
                    .reshape([batch, tokens, 20]),
                latent,
            ],
            2,
        ))
    }

    pub fn decode(&self, hybrid: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, tokens, dim] = hybrid.dims();
        ensure!(
            dim == 148 && tokens > 0 && tokens <= 50,
            "invalid decoder input"
        );
        let frames = tokens * 4;
        let root = hybrid
            .clone()
            .slice([0..batch, 0..tokens, 0..20])
            .reshape([batch, frames, 5]);
        let latent = hybrid.slice([0..batch, 0..tokens, 20..148]);
        let quantized = (self.unnormalize_latent(latent).clamp(-1.0, 1.0) * 32.0).round() / 32.0;
        let local = self
            .local_root(root.clone(), frames)
            .reshape([batch, tokens, 16]);
        let x = self.weights.linear("decoder.input_proj", quantized);
        let x = activation::relu(self.weights.linear(
            "decoder.external_cond_blocks.0",
            Tensor::cat(vec![x, local], 2),
        )) + position_encoding(tokens, 512, 0, &self.weights.device);
        let x = self.transformer("decoder", x, None, true, 4);
        let output = self
            .weights
            .linear("decoder.output_proj", x)
            .reshape([batch, frames, 329]);
        Ok(Tensor::cat(
            vec![root, output.slice([0..batch, 0..frames, 4..329])],
            2,
        ))
    }
}

pub fn position_encoding<B: Backend>(
    length: usize,
    dim: usize,
    origin: isize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut data = Vec::with_capacity(length * dim);
    for t in 0..length {
        for d in 0..dim {
            let angle = (t as isize + origin) as f32
                * 10000.0f32.powf(-(((d / 2) * 2) as f32) / dim as f32);
            data.push(if d % 2 == 0 { angle.sin() } else { angle.cos() });
        }
    }
    Tensor::from_data(TensorData::new(data, [1, length, dim]), device)
}

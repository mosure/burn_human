//! Deterministic DDIM and bounded autoregressive windows. Exact input noise is
//! injectable for parity; a seed alone is never treated as cross-runtime evidence.

use crate::{
    Ardy,
    representation::{decode_clip, trajectory_conditions},
};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_human_motion::{MotionClip, MotionRequest, TextEmbedding};

pub fn schedule(steps: usize) -> Result<Vec<(usize, f32, f32)>> {
    ensure!((1..=10).contains(&steps), "DDIM steps must be 1..10");
    let mut alpha = 1.0f32;
    let mut base = Vec::new();
    let abar = |t: f64| {
        ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2)
            .cos()
            .powi(2)
    };
    for i in 0..10 {
        let beta = (1.0 - abar((i + 1) as f64 / 10.0) / abar(i as f64 / 10.0)).min(0.999) as f32;
        alpha *= 1.0 - beta;
        base.push(alpha.max(1e-9));
    }
    let mut previous = 1.0;
    Ok((0..steps)
        .map(|i| {
            let mapped = (i as f32 * 9.0 / (steps - 1).max(1) as f32).round_ties_even() as usize;
            let value = (mapped, base[mapped], previous);
            previous = base[mapped];
            value
        })
        .collect())
}

impl<B: Backend> Ardy<B> {
    /// Inputs already use the centered window coordinate frame. Only the next
    /// forty frames are denoised; history is retained byte-for-byte on device.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_window(
        &self,
        mut x: Tensor<B, 3>,
        text: Tensor<B, 3>,
        heading: Tensor<B, 2>,
        observed: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        history_frames: usize,
        steps: usize,
        guidance: [f32; 2],
        mut progress: impl FnMut(usize),
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens, dim] = x.dims();
        ensure!(
            guidance
                .iter()
                .all(|v| v.is_finite() && (0.0..=10.0).contains(v)),
            "invalid guidance"
        );
        ensure!(
            dim == 148 && history_frames / 4 + 10 <= tokens,
            "invalid sample window"
        );
        let text3 = Tensor::cat(
            vec![
                text.clone(),
                Tensor::zeros_like(&text),
                Tensor::zeros_like(&text),
            ],
            0,
        );
        let observed3 = Tensor::cat(
            vec![
                Tensor::zeros_like(&observed),
                observed.clone(),
                Tensor::zeros_like(&observed),
            ],
            0,
        );
        let mask3 = Tensor::cat(
            vec![
                Tensor::zeros_like(&mask),
                mask.clone(),
                Tensor::zeros_like(&mask),
            ],
            0,
        );
        let heading3 = Tensor::cat(vec![heading.clone(), heading.clone(), heading], 0);
        let start = history_frames / 4;
        for (i, (mapped, alpha, previous)) in schedule(steps)?.into_iter().rev().enumerate() {
            let x3 = Tensor::cat(vec![x.clone(), x.clone(), x.clone()], 0);
            let pred = self.denoise(
                x3,
                text3.clone(),
                heading3.clone(),
                observed3.clone(),
                mask3.clone(),
                history_frames,
                mapped,
            )?;
            let uncond = pred
                .clone()
                .slice([2 * batch..3 * batch, start..start + 10, 0..dim]);
            let clean = uncond.clone()
                + (pred.clone().slice([0..batch, start..start + 10, 0..dim]) - uncond.clone())
                    * guidance[0]
                + (pred.slice([batch..2 * batch, start..start + 10, 0..dim]) - uncond)
                    * guidance[1];
            let current = x.clone().slice([0..batch, start..start + 10, 0..dim]);
            let epsilon = (current / alpha.sqrt() - clean.clone()) / (1.0 / alpha - 1.0).sqrt();
            let next = clean * previous.sqrt() + epsilon * (1.0 - previous).sqrt();
            x = x.slice_assign([0..batch, start..start + 10, 0..dim], next);
            progress(i + 1);
        }
        Ok(x.slice([0..batch, 0..start + 10, 0..dim]))
    }

    /// Generate a complete clip while retaining at most 160 history frames and
    /// 200 total conditioning frames. Only decoded motion leaves the GPU.
    /// Callback receives completed frames and can cancel between windows.
    pub async fn generate(
        &self,
        request: &MotionRequest,
        embedding: &TextEmbedding,
        mut progress: impl FnMut(usize) -> bool,
    ) -> Result<MotionClip> {
        request.validate()?;
        embedding.validate(&request.prompt, 4096)?;
        let (observed, mask) = trajectory_conditions(&self.config, request)?;
        let device = &self.weights.device;
        let text = Tensor::from_data(
            TensorData::new(embedding.values.clone(), [1, 1, 4096]),
            device,
        );
        let mut result: Vec<f32> = Vec::new();
        let mut random = NormalNoise::new(request.seed);
        let windows = request.frames.div_ceil(40);
        for window in 0..windows {
            ensure!(progress(window * 40), "generation cancelled");
            let generated = window * 40;
            let history_frames = generated.min(request.history_frames);
            let remaining = request.frames.saturating_sub(generated);
            let total = (history_frames + remaining.max(40)).min(200).div_ceil(4) * 4;
            let tokens = total / 4;
            let mut center = [0.0f32; 3];
            let mut heading = 0.0;
            let mut x = Tensor::zeros([1, tokens, 148], device);
            if history_frames > 0 {
                let mut history =
                    result[(generated - history_frames) * 330..generated * 330].to_vec();
                for d in [0, 2] {
                    center[d] = history[(history_frames - 1) * 330 + d]
                        * self.config.motion_stats.scale(d)
                        + self.config.motion_stats.mean[d];
                }
                let c = history[3] * self.config.motion_stats.scale(3)
                    + self.config.motion_stats.mean[3];
                let s = history[4] * self.config.motion_stats.scale(4)
                    + self.config.motion_stats.mean[4];
                heading = s.atan2(c);
                // Encode before centering, as upstream does (the encoder uses body only).
                let h = self.encode(Tensor::from_data(
                    TensorData::new(history.clone(), [1, history_frames, 330]),
                    device,
                ))?;
                for values in history.as_chunks_mut::<330>().0.iter_mut() {
                    for d in [0, 2] {
                        values[d] -= center[d] / self.config.motion_stats.scale(d);
                    }
                }
                let roots: Vec<f32> = history
                    .as_chunks::<330>()
                    .0
                    .iter()
                    .flat_map(|v| v[..5].iter().copied())
                    .collect();
                let root =
                    Tensor::from_data(TensorData::new(roots, [1, history_frames / 4, 20]), device);
                let h = Tensor::cat(
                    vec![root, h.slice([0..1, 0..history_frames / 4, 20..148])],
                    2,
                );
                x = x.slice_assign([0..1, 0..history_frames / 4, 0..148], h);
            }
            let noise: Vec<f32> = (0..10 * 148).map(|_| random.sample()).collect();
            x = x.slice_assign(
                [0..1, history_frames / 4..history_frames / 4 + 10, 0..148],
                Tensor::from_data(TensorData::new(noise, [1, 10, 148]), device),
            );
            let mut obs = vec![0.0; total * 330];
            let mut msk = vec![0.0; total * 330];
            for f in history_frames..total {
                let global = generated + f - history_frames;
                if global >= request.frames {
                    break;
                }
                obs[f * 330..(f + 1) * 330]
                    .copy_from_slice(&observed[global * 330..(global + 1) * 330]);
                msk[f * 330..(f + 1) * 330]
                    .copy_from_slice(&mask[global * 330..(global + 1) * 330]);
                for d in [0, 2] {
                    obs[f * 330 + d] -=
                        center[d] / self.config.motion_stats.scale(d) * msk[f * 330 + d];
                }
            }
            let hybrid = self.sample_window(
                x,
                text.clone(),
                Tensor::from_data([[heading]], device),
                Tensor::from_data(TensorData::new(obs, [1, total, 330]), device),
                Tensor::from_data(TensorData::new(msk, [1, total, 330]), device),
                history_frames,
                request.diffusion_steps,
                [request.text_guidance, request.trajectory_guidance],
                |_| {},
            )?;
            let frames = history_frames + 40;
            let mut translation = [0.0; 5];
            for d in [0, 2] {
                translation[d] = center[d] / self.config.motion_stats.scale(d);
            }
            let root = hybrid
                .clone()
                .slice([0..1, 0..frames / 4, 0..20])
                .reshape([1, frames, 5])
                + Tensor::from_data(TensorData::new(translation.to_vec(), [1, 1, 5]), device);
            let hybrid = Tensor::cat(
                vec![
                    root.reshape([1, frames / 4, 20]),
                    hybrid.slice([0..1, 0..frames / 4, 20..148]),
                ],
                2,
            );
            let output = self
                .decode(hybrid)?
                .slice([0..1, history_frames..frames, 0..330]);
            let data = output
                .into_data_async()
                .await
                .map_err(|e| anyhow::anyhow!("device read: {e}"))?;
            result.extend(data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?);
        }
        result.truncate(request.frames * 330);
        progress(request.frames);
        decode_clip(
            &self.config,
            &result,
            format!(
                "{}@{}; seed={}; steps={}; prompt={}",
                crate::config::MODEL_ID,
                crate::config::MODEL_REVISION,
                request.seed,
                request.diffusion_steps,
                request.prompt
            ),
        )
    }
}

/// Stable host RNG for request reproducibility; reference tests supply tensors.
pub struct NormalNoise {
    state: u64,
    spare: Option<f32>,
}
impl NormalNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }
    fn uniform(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (((z >> 40) as f32) + 0.5) / 16777216.0
    }
    pub fn sample(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = std::f32::consts::TAU * self.uniform();
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

//! Deterministic DDIM and bounded autoregressive windows. Exact input noise is
//! injectable for parity; a seed alone is never treated as cross-runtime evidence.

use crate::Ardy;
use anyhow::{Result, ensure};
use burn::{prelude::Backend, tensor::Tensor};
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

struct SampleState<B: Backend> {
    x: Tensor<B, 3>,
    text: Tensor<B, 3>,
    heading: Tensor<B, 2>,
    observed: Tensor<B, 3>,
    mask: Tensor<B, 3>,
    history_frames: usize,
    guidance: [f32; 2],
}
impl<B: Backend> SampleState<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Tensor<B, 3>,
        text: Tensor<B, 3>,
        heading: Tensor<B, 2>,
        observed: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        history_frames: usize,
        guidance: [f32; 2],
    ) -> Result<Self> {
        let [_, tokens, dim] = x.dims();
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
        Ok(Self {
            x,
            text: text3,
            heading: heading3,
            observed: observed3,
            mask: mask3,
            history_frames,
            guidance,
        })
    }
    fn advance(
        mut self,
        model: &Ardy<B>,
        (mapped, alpha, previous): (usize, f32, f32),
    ) -> Result<Self> {
        let [batch, _, dim] = self.x.dims();
        let start = self.history_frames / 4;
        let x3 = Tensor::cat(vec![self.x.clone(), self.x.clone(), self.x.clone()], 0);
        let pred = model.denoise(
            x3,
            self.text.clone(),
            self.heading.clone(),
            self.observed.clone(),
            self.mask.clone(),
            self.history_frames,
            mapped,
        )?;
        let uncond = pred
            .clone()
            .slice([2 * batch..3 * batch, start..start + 10, 0..dim]);
        let clean = uncond.clone()
            + (pred.clone().slice([0..batch, start..start + 10, 0..dim]) - uncond.clone())
                * self.guidance[0]
            + (pred.slice([batch..2 * batch, start..start + 10, 0..dim]) - uncond)
                * self.guidance[1];
        let current = self.x.clone().slice([0..batch, start..start + 10, 0..dim]);
        let epsilon = (current / alpha.sqrt() - clean.clone()) / (1.0 / alpha - 1.0).sqrt();
        let next = clean * previous.sqrt() + epsilon * (1.0 - previous).sqrt();
        self.x = self
            .x
            .slice_assign([0..batch, start..start + 10, 0..dim], next);
        Ok(self)
    }
    fn finish(self) -> Tensor<B, 3> {
        let [batch, _, dim] = self.x.dims();
        self.x
            .slice([0..batch, 0..self.history_frames / 4 + 10, 0..dim])
    }
}

impl<B: Backend> Ardy<B> {
    /// Inputs already use the centered window coordinate frame. Only the next
    /// forty frames are denoised; history is retained byte-for-byte on device.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_window(
        &self,
        x: Tensor<B, 3>,
        text: Tensor<B, 3>,
        heading: Tensor<B, 2>,
        observed: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        history_frames: usize,
        steps: usize,
        guidance: [f32; 2],
        mut progress: impl FnMut(usize),
    ) -> Result<Tensor<B, 3>> {
        let mut state =
            SampleState::new(x, text, heading, observed, mask, history_frames, guidance)?;
        for (i, step) in schedule(steps)?.into_iter().rev().enumerate() {
            state = state.advance(self, step)?;
            progress(i + 1);
        }
        Ok(state.finish())
    }

    /// Cooperative variant used by interactive applications. Yields browser
    /// tasks between DDIM steps without waiting for GPU completion.
    #[allow(clippy::too_many_arguments)]
    pub async fn sample_window_async(
        &self,
        x: Tensor<B, 3>,
        text: Tensor<B, 3>,
        heading: Tensor<B, 2>,
        observed: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        history_frames: usize,
        steps: usize,
        guidance: [f32; 2],
        mut progress: impl FnMut(usize),
    ) -> Result<Tensor<B, 3>> {
        let mut state =
            SampleState::new(x, text, heading, observed, mask, history_frames, guidance)?;
        for (i, step) in schedule(steps)?.into_iter().rev().enumerate() {
            state = state.advance(self, step)?;
            progress(i + 1);
            burn_human_inference::cooperative::yield_to_browser().await;
        }
        Ok(state.finish())
    }

    /// Generate a complete clip while retaining at most 160 history frames and
    /// 200 total conditioning frames. Only decoded motion leaves the GPU.
    /// Callback receives completed frames and can cancel between windows.
    pub async fn generate(
        &self,
        request: &MotionRequest,
        embedding: &TextEmbedding,
        progress: impl FnMut(usize) -> bool,
    ) -> Result<MotionClip> {
        Ok(self
            .generate_batch(
                std::slice::from_ref(request),
                std::slice::from_ref(embedding),
                progress,
            )
            .await?
            .remove(0))
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

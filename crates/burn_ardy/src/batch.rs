//! Bounded, heterogeneous text/waypoint batches with independent noise streams.
use crate::{
    Ardy,
    representation::{decode_clip, write_trajectory_window},
    sampler::NormalNoise,
};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_human_motion::{MotionClip, MotionRequest, TextEmbedding};
use std::collections::VecDeque;

pub(crate) fn validate_batch(
    requests: &[MotionRequest],
    embeddings: &[TextEmbedding],
) -> Result<()> {
    ensure!(
        (1..=8).contains(&requests.len()),
        "motion batch must contain 1..8 requests"
    );
    ensure!(
        requests.len() == embeddings.len(),
        "embedding batch mismatch"
    );
    let first = &requests[0];
    for (request, embedding) in requests.iter().zip(embeddings) {
        request.validate()?;
        embedding.validate(&request.prompt, 4096)?;
        ensure!(
            request.frames == first.frames
                && request.history_frames == first.history_frames
                && request.diffusion_steps == first.diffusion_steps
                && request.text_guidance == first.text_guidance
                && request.trajectory_guidance == first.trajectory_guidance,
            "motion batch requires matching window lengths, steps and guidance"
        );
    }
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn rejects_incompatible_batches_before_inference() {
        let request = MotionRequest::default();
        let embedding = TextEmbedding {
            prompt: request.prompt.clone(),
            encoder: "test".into(),
            revision: "test".into(),
            values: vec![0.0; 4096],
        };
        assert!(validate_batch(&[], &[]).is_err());
        assert!(validate_batch(std::slice::from_ref(&request), &[]).is_err());
        assert!(validate_batch(&vec![request.clone(); 9], &vec![embedding.clone(); 9]).is_err());
        let mut other = request.clone();
        other.frames += 4;
        assert!(
            validate_batch(
                &[request.clone(), other],
                &[embedding.clone(), embedding.clone()]
            )
            .is_err()
        );
        let mut stale = embedding.clone();
        stale.prompt = "different prompt".into();
        assert!(validate_batch(std::slice::from_ref(&request), &[stale]).is_err());
        let mut nonfinite = embedding.clone();
        nonfinite.values[10] = f32::NAN;
        assert!(validate_batch(std::slice::from_ref(&request), &[nonfinite]).is_err());
        assert!(validate_batch(&vec![request; 8], &vec![embedding; 8]).is_ok());
    }
}

#[cfg(all(test, feature = "wgpu", feature = "tools"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a GPU and the pinned ARDY model cache/download"]
    fn full_clip_batch_preserves_actor_order_and_cancellation() {
        type B = burn_human_inference::gpu::WgpuBackend;
        pollster::block_on(async {
            let device = burn::backend::wgpu::WgpuDevice::default();
            let model = Ardy::<B>::load_pretrained(&device, |_, _| {})
                .await
                .unwrap();
            // Numerical fixture only: these synthetic pooled vectors do not
            // claim text understanding or human motion quality.
            let requests: Vec<_> = [9, 81]
                .into_iter()
                .map(|seed| MotionRequest {
                    prompt: "numerical batch parity fixture".into(),
                    seed,
                    frames: 80,
                    history_frames: 40,
                    ..Default::default()
                })
                .collect();
            let embeddings: Vec<_> = requests
                .iter()
                .enumerate()
                .map(|(i, r)| TextEmbedding {
                    prompt: r.prompt.clone(),
                    encoder: "numerical-fixture".into(),
                    revision: "v1".into(),
                    values: (0..4096)
                        .map(|j| ((j + i * 7) as f32 * 0.17).sin() * 0.01)
                        .collect(),
                })
                .collect();
            let batch = model
                .generate_batch(&requests, &embeddings, |_| true)
                .await
                .unwrap();
            // Keep the batch shape fixed: changing it can change FSQ codes at
            // rounding boundaries. The real-checkpoint CLI separately gates
            // continuous batch/serial hooks and reports decoded-clip drift.
            let reversed = model
                .generate_batch(
                    &[requests[1].clone(), requests[0].clone()],
                    &[embeddings[1].clone(), embeddings[0].clone()],
                    |_| true,
                )
                .await
                .unwrap();
            for i in 0..2 {
                let single = &reversed[1 - i];
                let mut max_position = 0.0f32;
                let mut max_rotation = 0.0f32;
                for (a, b) in batch[i].frames.iter().zip(&single.frames) {
                    max_position =
                        max_position.max(a.root_translation.distance(b.root_translation));
                    for (a, b) in a.local_rotations.iter().zip(&b.local_rotations) {
                        let dot = a.as_dquat().normalize().dot(b.as_dquat().normalize());
                        max_rotation = max_rotation.max((2.0 * dot.abs().min(1.0).acos()) as f32);
                    }
                }
                println!(
                    "actor {i}: maximum root error {max_position} m, rotation error {max_rotation} rad"
                );
                assert!(max_position < 0.0001 && max_rotation < 0.00001);
            }
            assert_ne!(
                batch[0].frames[0].root_translation,
                batch[1].frames[0].root_translation
            );
            assert!(
                model
                    .generate_batch(&requests, &embeddings, |_| false)
                    .await
                    .is_err()
            );
            let mut invalid = requests.clone();
            invalid[1].frames = 120;
            assert!(
                model
                    .generate_batch(&invalid, &embeddings, |_| true)
                    .await
                    .is_err()
            );
        });
    }
}

impl<B: Backend> Ardy<B> {
    /// Generate 1..8 clips together. Length, history, DDIM steps and guidance must
    /// match; prompts, seeds and trajectories are independent. Histories stay on
    /// the device and are bounded to 160 frames per actor. Conditioning arrays
    /// cover at most 200 frames. Returned clips remain in request order.
    /// The callback receives completed frames per actor and can cancel before
    /// each window; its final notification does not cancel completed results.
    /// Keep batch size/backend fixed for seeded comparisons: floating-point
    /// differences can cross FSQ rounding boundaries and change decoded poses.
    pub async fn generate_batch(
        &self,
        requests: &[MotionRequest],
        embeddings: &[TextEmbedding],
        mut progress: impl FnMut(usize) -> bool,
    ) -> Result<Vec<MotionClip>> {
        validate_batch(requests, embeddings)?;
        ensure!(progress(0), "generation cancelled");
        let first = &requests[0];
        let batch = requests.len();
        let device = &self.weights.device;
        let text = Tensor::from_data(
            TensorData::new(
                embeddings
                    .iter()
                    .flat_map(|e| e.values.iter().copied())
                    .collect(),
                [batch, 1, 4096],
            ),
            device,
        );
        let mut results: Vec<Option<MotionClip>> = vec![None; batch];
        // CPU history contains only the five root features needed for window
        // centering. Decoded body history stays on the GPU; completed frames
        // are converted immediately to the public clip representation.
        let mut roots = vec![VecDeque::<[f32; 5]>::new(); batch];
        let mut random: Vec<_> = requests.iter().map(|r| NormalNoise::new(r.seed)).collect();
        let mut resident: Option<Tensor<B, 3>> = None;
        for window in 0..first.frames.div_ceil(40) {
            let generated = window * 40;
            if window > 0 {
                ensure!(progress(generated), "generation cancelled");
            }
            let history_frames = generated.min(first.history_frames);
            let total = (history_frames + first.frames.saturating_sub(generated).max(40))
                .min(200)
                .div_ceil(4)
                * 4;
            let mut translations = vec![0.0; batch * 5];
            let mut headings = vec![0.0; batch];
            let mut x = Tensor::zeros([batch, total / 4, 148], device);
            if history_frames > 0 {
                for b in 0..batch {
                    let last = roots[b].back().expect("previous root frame");
                    let first = roots[b].front().expect("previous root frame");
                    for d in [0, 2] {
                        // Preserve the single-request arithmetic order. The
                        // algebraic shortcut changes autoregressive rounding.
                        let center = last[d] * self.config.motion_stats.scale(d)
                            + self.config.motion_stats.mean[d];
                        translations[b * 5 + d] = center / self.config.motion_stats.scale(d);
                    }
                    let c = first[3] * self.config.motion_stats.scale(3)
                        + self.config.motion_stats.mean[3];
                    let s = first[4] * self.config.motion_stats.scale(4)
                        + self.config.motion_stats.mean[4];
                    headings[b] = s.atan2(c);
                }
                let history = resident.as_ref().expect("resident history").clone();
                let encoded = self.encode(history.clone())?;
                let root = (history.slice([0..batch, 0..history_frames, 0..5])
                    - Tensor::from_data(
                        TensorData::new(translations.clone(), [batch, 1, 5]),
                        device,
                    ))
                .reshape([batch, history_frames / 4, 20]);
                x = x.slice_assign(
                    [0..batch, 0..history_frames / 4, 0..148],
                    Tensor::cat(
                        vec![
                            root,
                            encoded.slice([0..batch, 0..history_frames / 4, 20..148]),
                        ],
                        2,
                    ),
                );
            }
            let noise: Vec<f32> = random
                .iter_mut()
                .flat_map(|r| (0..10 * 148).map(move |_| r.sample()))
                .collect();
            x = x.slice_assign(
                [
                    0..batch,
                    history_frames / 4..history_frames / 4 + 10,
                    0..148,
                ],
                Tensor::from_data(TensorData::new(noise, [batch, 10, 148]), device),
            );
            let mut obs = vec![0.0; batch * total * 330];
            let mut mask = vec![0.0; obs.len()];
            for b in 0..batch {
                let range = (b * total + history_frames) * 330..(b + 1) * total * 330;
                write_trajectory_window(
                    &self.config,
                    &requests[b],
                    generated,
                    &mut obs[range.clone()],
                    &mut mask[range],
                );
                for f in history_frames..total {
                    let at = (b * total + f) * 330;
                    for d in [0, 2] {
                        obs[at + d] -= translations[b * 5 + d] * mask[at + d];
                    }
                }
            }
            let hybrid = self
                .sample_window_async(
                    x,
                    text.clone(),
                    Tensor::from_data(TensorData::new(headings, [batch, 1]), device),
                    Tensor::from_data(TensorData::new(obs, [batch, total, 330]), device),
                    Tensor::from_data(TensorData::new(mask, [batch, total, 330]), device),
                    history_frames,
                    first.diffusion_steps,
                    [first.text_guidance, first.trajectory_guidance],
                    |_| {},
                )
                .await?;
            let frames = history_frames + 40;
            let root = hybrid
                .clone()
                .slice([0..batch, 0..frames / 4, 0..20])
                .reshape([batch, frames, 5])
                + Tensor::from_data(TensorData::new(translations, [batch, 1, 5]), device);
            let output = self
                .decode(Tensor::cat(
                    vec![
                        root.reshape([batch, frames / 4, 20]),
                        hybrid.slice([0..batch, 0..frames / 4, 20..148]),
                    ],
                    2,
                ))?
                .slice([0..batch, history_frames..frames, 0..330]);
            if first.history_frames > 0 {
                let history = resident.take().map_or_else(
                    || output.clone(),
                    |old| Tensor::cat(vec![old, output.clone()], 1),
                );
                let n = history.dims()[1];
                resident = Some(history.slice([
                    0..batch,
                    n.saturating_sub(first.history_frames)..n,
                    0..330,
                ]));
            }
            let data = output
                .into_data_async()
                .await
                .map_err(|e| anyhow::anyhow!("device read: {e}"))?
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            for (b, result) in results.iter_mut().enumerate() {
                let features = &data[b * 40 * 330..(b + 1) * 40 * 330];
                if first.history_frames > 0 {
                    for frame in features.as_chunks::<330>().0 {
                        roots[b].push_back(frame[..5].try_into().unwrap());
                        if roots[b].len() > first.history_frames {
                            roots[b].pop_front();
                        }
                    }
                }
                let keep = (first.frames - generated).min(40);
                let mut clip = decode_clip(&self.config, &features[..keep * 330], String::new())?;
                if let Some(result) = result {
                    result.frames.extend(clip.frames);
                } else {
                    let request = &requests[b];
                    clip.provenance = format!(
                        "{}@{}; seed={}; steps={}; prompt={}",
                        crate::config::MODEL_ID,
                        crate::config::MODEL_REVISION,
                        request.seed,
                        request.diffusion_steps,
                        request.prompt
                    );
                    *result = Some(clip);
                }
            }
        }
        progress(first.frames);
        Ok(results
            .into_iter()
            .map(|clip| clip.expect("generated window"))
            .collect())
    }
}

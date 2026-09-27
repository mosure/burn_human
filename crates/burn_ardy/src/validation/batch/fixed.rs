use crate::Ardy;
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_human_motion::{MotionRequest, TextEmbedding};
use serde_json::{Value, json};

async fn values<B: Backend>(tensor: Tensor<B, 3>) -> Result<Vec<f32>> {
    let data = tensor
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        data.iter().all(|v| v.is_finite()),
        "non-finite batch output"
    );
    Ok(data)
}

// Compare identical continuous inputs before FSQ rounding. Changed matmul batch
// shapes can cross a discrete code boundary; completed clips then diverge.
// Decoder parity uses identical hybrids so its inputs cannot diverge.
pub(super) async fn validate<B: Backend>(
    model: &Ardy<B>,
    requests: &[MotionRequest],
    embeddings: &[TextEmbedding],
) -> Result<Value> {
    let device = &model.weights.device;
    let batch = requests.len();
    let frames = requests[0].frames.min(200);
    let tokens = frames / 4;
    let mut noise = vec![0.0; batch * tokens * 148];
    let mut observed = vec![0.0; batch * frames * 330];
    let mut mask = vec![0.0; observed.len()];
    for (i, request) in requests.iter().enumerate() {
        let mut random = crate::sampler::NormalNoise::new(request.seed);
        for value in &mut noise[i * tokens * 148..i * tokens * 148 + 10 * 148] {
            *value = random.sample();
        }
        let (o, m) = crate::representation::trajectory_conditions(&model.config, request)?;
        observed[i * frames * 330..(i + 1) * frames * 330].copy_from_slice(&o[..frames * 330]);
        mask[i * frames * 330..(i + 1) * frames * 330].copy_from_slice(&m[..frames * 330]);
    }
    let x = Tensor::from_data(TensorData::new(noise, [batch, tokens, 148]), device);
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
    let observed = Tensor::from_data(TensorData::new(observed, [batch, frames, 330]), device);
    let mask = Tensor::from_data(TensorData::new(mask, [batch, frames, 330]), device);
    let request = &requests[0];
    let predicted = values(model.denoise(
        x.clone(),
        text.clone(),
        Tensor::zeros([batch, 1], device),
        observed.clone(),
        mask.clone(),
        0,
        7,
    )?)
    .await?;
    let hybrid = model
        .sample_window_async(
            x.clone(),
            text.clone(),
            Tensor::zeros([batch, 1], device),
            observed.clone(),
            mask.clone(),
            0,
            request.diffusion_steps,
            [request.text_guidance, request.trajectory_guidance],
            |_| {},
        )
        .await?;
    let batch_sample = values(hybrid.clone()).await?;
    let decoded = values(model.decode(hybrid.clone())?).await?;
    let mut sample_max = 0.0f32;
    let mut denoise_max = 0.0f32;
    let mut decode_max = 0.0f32;
    let mut changed_codes = 0;
    let mut max_code_delta = 0.0f32;
    for i in 0..batch {
        let single_prediction = values(model.denoise(
            x.clone().slice([i..i + 1, 0..tokens, 0..148]),
            text.clone().slice([i..i + 1, 0..1, 0..4096]),
            Tensor::zeros([1, 1], device),
            observed.clone().slice([i..i + 1, 0..frames, 0..330]),
            mask.clone().slice([i..i + 1, 0..frames, 0..330]),
            0,
            7,
        )?)
        .await?;
        for (a, b) in single_prediction
            .iter()
            .zip(&predicted[i * tokens * 148..(i + 1) * tokens * 148])
        {
            denoise_max = denoise_max.max((a - b).abs());
        }
        let single = model
            .sample_window_async(
                x.clone().slice([i..i + 1, 0..tokens, 0..148]),
                text.clone().slice([i..i + 1, 0..1, 0..4096]),
                Tensor::zeros([1, 1], device),
                observed.clone().slice([i..i + 1, 0..frames, 0..330]),
                mask.clone().slice([i..i + 1, 0..frames, 0..330]),
                0,
                request.diffusion_steps,
                [request.text_guidance, request.trajectory_guidance],
                |_| {},
            )
            .await?;
        let single = values(single).await?;
        let batched = &batch_sample[i * 10 * 148..(i + 1) * 10 * 148];
        for (j, (a, b)) in single.iter().zip(batched).enumerate() {
            sample_max = sample_max.max((a - b).abs());
            let d = j % 148;
            if d >= 20 {
                let stats = &model.config.latent_stats;
                let code = |v: f32| {
                    ((v * stats.scale(d - 20) + stats.mean[d - 20]).clamp(-1.0, 1.0) * 32.0).round()
                };
                let delta = (code(*a) - code(*b)).abs();
                changed_codes += usize::from(delta != 0.0);
                max_code_delta = max_code_delta.max(delta);
            }
        }
        let single_decode =
            values(model.decode(hybrid.clone().slice([i..i + 1, 0..10, 0..148]))?).await?;
        for (a, b) in single_decode
            .iter()
            .zip(&decoded[i * 40 * 330..(i + 1) * 40 * 330])
        {
            decode_max = decode_max.max((a - b).abs());
        }
    }
    // Same tolerances as the upstream checkpoint validator: the ten-step DDIM
    // result permits accumulation beyond a single denoiser/decoder call.
    ensure!(
        sample_max <= 0.02 && decode_max <= 0.002 && denoise_max <= 0.002,
        "fixed-input batch mismatch: denoised={denoise_max} sampled={sample_max} decoded={decode_max}"
    );
    eprintln!(
        "fixed inputs: denoised={denoise_max} sampled={sample_max} decoded={decode_max}; changed FSQ codes={changed_codes}"
    );
    Ok(
        json!({"denoised_max_abs":denoise_max,"sampled_max_abs":sample_max,"same_input_decoded_max_abs":decode_max,"tolerance":{"denoised":0.002,"sampled":0.02,"decoded":0.002},
        "changed_fsq_codes":changed_codes,"max_fsq_code_delta":max_code_delta,"fsq_code_count":batch*10*128}),
    )
}

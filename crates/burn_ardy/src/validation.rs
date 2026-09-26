//! Mandatory real-checkpoint parity runner. Missing artifacts are errors.
use crate::{Ardy, representation::decode_clip};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_human_motion::artifacts::Manifest;
use serde_json::json;
use web_time::Instant;

pub struct Fixture {
    pub data: Vec<u8>,
}
impl Fixture {
    fn values(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let safe = safetensors::SafeTensors::deserialize(&self.data)?;
        let v = safe.tensor(name)?;
        ensure!(v.dtype() == safetensors::Dtype::F32, "expected f32 fixture");
        Ok((
            v.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect(),
            v.shape().to_vec(),
        ))
    }
    fn tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let (data, shape) = self.values(name)?;
        Ok(Tensor::from_data(TensorData::new(data, shape), device))
    }
}

async fn compare<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
    fixture: &Fixture,
    tolerance: f32,
) -> Result<serde_json::Value> {
    let data = tensor
        .into_data_async()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual = data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (expected, shape) = fixture.values(name)?;
    ensure!(
        data.shape.as_slice() == shape && actual.iter().all(|v| v.is_finite()),
        "{name} shape/non-finite output"
    );
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rmse = (actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    eprintln!("{name}: max_abs={max:.7} rmse={rmse:.7} tolerance={tolerance}");
    ensure!(
        max <= tolerance,
        "{name} parity failed: {max} > {tolerance}"
    );
    Ok(json!({"hook":name,"max_abs":max,"rmse":rmse,"max_abs_tolerance":tolerance}))
}

pub async fn validate<B: Backend>(
    model: &Ardy<B>,
    manifest: &Manifest,
    fixture: Fixture,
    backend: &str,
    load_seconds: f64,
    benchmark_batch: bool,
) -> Result<serde_json::Value> {
    let device = model.weights.device.clone();
    let mut hooks = Vec::new();
    let x: Tensor<B, 3> = fixture.tensor("x", &device)?;
    hooks.push(
        compare(
            "local_root",
            model.local_root(
                x.clone().slice([0..1, 0..14, 0..20]).reshape([1, 56, 5]),
                48,
            ),
            &fixture,
            0.002,
        )
        .await?,
    );
    let denoise = || {
        model.denoise(
            x.clone(),
            fixture.tensor("text", &device)?,
            fixture.tensor("heading", &device)?,
            fixture.tensor("observed", &device)?,
            fixture.tensor("mask", &device)?,
            8,
            7,
        )
    };
    hooks.push(compare("denoised", denoise()?, &fixture, 0.002).await?);
    let sample = || {
        model.sample_window(
            x.clone(),
            fixture.tensor("text", &device)?,
            fixture.tensor("heading", &device)?,
            fixture.tensor("observed", &device)?,
            fixture.tensor("mask", &device)?,
            8,
            10,
            [2.0, 1.5],
            |_| {},
        )
    };
    hooks.push(compare("sampled", sample()?, &fixture, 0.02).await?);
    hooks.push(
        compare(
            "decoded",
            model.decode(fixture.tensor("sampled", &device)?)?,
            &fixture,
            0.002,
        )
        .await?,
    );
    hooks.push(
        compare(
            "encoded",
            model.encode(fixture.tensor("decoded", &device)?)?,
            &fixture,
            0.002,
        )
        .await?,
    );
    let clip = decode_clip(
        &model.config,
        &fixture.values("decoded")?.0,
        "reference".into(),
    )?;
    let expected = fixture.values("joints")?.0;
    let mut joint_max = 0.0f32;
    for (i, frame) in clip.frames.iter().enumerate() {
        let (positions, _) = clip.rig.forward(frame)?;
        for (j, p) in positions.iter().enumerate() {
            for (k, v) in p.to_array().iter().enumerate() {
                joint_max = joint_max.max((v - expected[(i * 27 + j) * 3 + k]).abs());
            }
        }
    }
    ensure!(joint_max < 0.00005, "FK parity failed {joint_max}");
    // Warm before timing, and synchronize by materializing each full output.
    let _ = sample()?.into_data_async().await;
    let mut timings = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = sample()?
            .into_data_async()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        timings.push(start.elapsed().as_secs_f64());
    }
    timings.sort_by(f64::total_cmp);
    let mut batch_report = serde_json::Value::Null;
    if benchmark_batch {
        let text: Tensor<B, 3> = fixture.tensor("text", &device)?;
        let heading: Tensor<B, 2> = fixture.tensor("heading", &device)?;
        let observed: Tensor<B, 3> = fixture.tensor("observed", &device)?;
        let mask: Tensor<B, 3> = fixture.tensor("mask", &device)?;
        let batch_x = Tensor::cat(vec![x.clone(); 4], 0);
        let text = Tensor::cat(vec![text; 4], 0);
        let heading = Tensor::cat(vec![heading; 4], 0);
        let observed = Tensor::cat(vec![observed; 4], 0);
        let mask = Tensor::cat(vec![mask; 4], 0);
        let sample_batch = || {
            model.sample_window(
                batch_x.clone(),
                text.clone(),
                heading.clone(),
                observed.clone(),
                mask.clone(),
                8,
                10,
                [2.0, 1.5],
                |_| {},
            )
        };
        let output = sample_batch()?;
        let mut parity = Vec::new();
        for i in 0..4 {
            parity.push(
                compare(
                    "sampled",
                    output.clone().slice([i..i + 1, 0..12, 0..148]),
                    &fixture,
                    0.02,
                )
                .await?,
            );
        }
        let mut batch_times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let _ = sample_batch()?
                .into_data_async()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            batch_times.push(start.elapsed().as_secs_f64());
        }
        batch_times.sort_by(f64::total_cmp);
        batch_report = json!({"requests":4,"cfg_batch":12,"warm_window_seconds":batch_times,"median_generated_frames_per_second":160.0/batch_times[1],"parity":parity,"includes_decoder":false,"includes_text_encoder":false});
    }
    Ok(
        json!({"backend":backend,"model_revision":manifest.model_revision,"source_revision":manifest.source_revision,"manifest_sha256":manifest.content_sha256,"fixture_sha256":burn_human_motion::artifacts::sha256(&fixture.data),"load_seconds":load_seconds,"parity":hooks,"fk_max_error_m":joint_max,"performance":{"warm_window_seconds":timings,"median_frames_per_second":40.0/timings[1],"frames_per_window":40,"history_frames":8,"future_frames":8,"diffusion_steps":10,"cfg_batch":3,"includes_decoder":false,"includes_text_encoder":false,"synchronization":"full_output_readback"},"batched_performance":batch_report,"semantic_quality":"not_measured_random_embeddings"}),
    )
}

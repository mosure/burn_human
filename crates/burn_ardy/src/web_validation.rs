use crate::{
    Ardy,
    transport::{ModelSource, read_bounded},
    validation::{Fixture, validate},
};
use burn::backend::wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async};
use burn_human_inference::gpu::WgpuBackend as Wgpu;
use wasm_bindgen::prelude::*;

/// Profile the production load/generate path without synchronous parity hooks.
/// Input is a batch Suite with actual embeddings; only its first case is used.
#[wasm_bindgen]
pub async fn profile_generation_webgpu(
    base: String,
    input: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, input: String, digest: String) -> anyhow::Result<String> {
        let start = web_time::Instant::now();
        let mut phases = Vec::new();
        let mut phase_start = 0.0;
        let mut mark = |name: &str| {
            let end = start.elapsed().as_secs_f64() * 1000.0;
            phases.push(serde_json::json!({"name":name,"start_ms":phase_start,"end_ms":end}));
            phase_start = end;
        };
        let device = WgpuDevice::default();
        let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        mark("device");
        let bytes = read_bounded(&input, 4 * 1024 * 1024).await?;
        let suite: crate::validation::batch::Suite = serde_json::from_slice(&bytes)?;
        let case = suite
            .cases
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty input suite"))?;
        let embeddings = case
            .embedding_indices
            .iter()
            .map(|&i| {
                suite
                    .embeddings
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("embedding index out of range"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        crate::batch::validate_batch(&case.requests, &embeddings)?;
        mark("input");
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest;
        let model = Ardy::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        mark("load");
        let mut timings = Vec::new();
        for i in 0..4 {
            let now = web_time::Instant::now();
            let clips = model
                .generate_batch(&case.requests, &embeddings, |_| true)
                .await?;
            timings.push(now.elapsed().as_secs_f64());
            for (clip, request) in clips.iter().zip(&case.requests) {
                clip.validate()?;
                anyhow::ensure!(clip.frames.len() == request.frames, "incomplete output");
            }
            mark(if i == 0 {
                "cold_generation"
            } else {
                "warm_generation"
            });
        }
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "passed":true,"phases":phases,"request_seconds":timings,
            "actors":case.requests.len(),"frames_per_actor":case.requests[0].frames,
            "backend":"browser-webgpu-f32","adapter":setup.adapter.get_info().name,
            "scope":"production load and decoded clips; text encoding and WASM module startup excluded",
            "model_manifest_sha256":artifact.sha256,
            "input_sha256":burn_human_motion::artifacts::sha256(&bytes),
        }))?)
    }
    run(base, input, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Full autoregressive batch parity and timing with the native validator's fixtures.
#[wasm_bindgen]
pub async fn validate_batch_webgpu(
    base: String,
    reference: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, reference: String, digest: String) -> anyhow::Result<String> {
        let device = WgpuDevice::default();
        let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest;
        let model = Ardy::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        let bytes = read_bounded(&reference, 32 * 1024 * 1024).await?;
        let mut report =
            crate::validation::batch::validate(&model, serde_json::from_slice(&bytes)?).await?;
        report["suite_sha256"] = burn_human_motion::artifacts::sha256(&bytes).into();
        report["model_manifest_sha256"] = artifact.sha256.into();
        report["backend"] = "browser-webgpu-f32".into();
        report["adapter"] = serde_json::json!(setup.adapter.get_info().name);
        Ok(serde_json::to_string_pretty(&report)?)
    }
    run(base, reference, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Identical parity hooks and timing protocol to ardy-validate, on real WebGPU.
#[wasm_bindgen]
pub async fn validate_webgpu(
    base: String,
    reference: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, reference: String, digest: String) -> anyhow::Result<String> {
        let device = WgpuDevice::default();
        let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        let source = ModelSource::new(base.clone());
        let manifest = source.manifest(Some(&digest)).await?;
        let start = web_time::Instant::now();
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest;
        let model = Ardy::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        let load_seconds = start.elapsed().as_secs_f64();
        let fixture = Fixture {
            data: read_bounded(&reference, 4 * 1024 * 1024).await?,
        };
        let mut report = validate(
            &model,
            &manifest,
            fixture,
            "browser-webgpu-f32",
            load_seconds,
            true,
        )
        .await?;
        report["adapter"] = serde_json::json!(setup.adapter.get_info().name);
        Ok(serde_json::to_string_pretty(&report)?)
    }
    run(base, reference, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

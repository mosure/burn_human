use crate::{
    Ardy,
    transport::{ModelSource, read_bounded},
    validation::{Fixture, validate},
};
use burn::backend::{
    Wgpu,
    wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async},
};
use wasm_bindgen::prelude::*;

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

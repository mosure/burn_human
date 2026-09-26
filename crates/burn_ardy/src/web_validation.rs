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
        let mut source = ModelSource::new(base);
        let manifest = source.manifest(Some(&digest)).await?;
        let start = web_time::Instant::now();
        let model = Ardy::<Wgpu>::load(&manifest, &mut source, &device, |_, _| {}).await?;
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

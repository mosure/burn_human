// The validation module is an explicit browser entry point. Keep startup here
// rather than in the shared library, so downstream Rust mains still start.
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn initialize_llama_validation_runtime() {
    burn_human_inference::initialize_browser_runtime();
}

use burn_human_inference::transport::read_bounded;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub async fn validate_text_webgpu(
    base: String,
    reference: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, reference: String, digest: String) -> anyhow::Result<String> {
        let (device, setup) = burn::tensor::Device::wgpu_options()
            .graphics_api(burn::tensor::wgpu::WgpuBackend::WebGpu)
            .init_with_setup_async()
            .await?;
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest.clone();
        let start = web_time::Instant::now();
        let mut model = crate::TextEncoder::load_artifact(&artifact, &device, |_, _| {}).await?;
        let load = start.elapsed().as_secs_f64();
        let reference = serde_json::from_slice(&read_bounded(&reference, 4 * 1024 * 1024).await?)?;
        let mut report = crate::validation::validate(&mut model, reference).await?;
        report["load_seconds"] = load.into();
        report["manifest"] = digest.into();
        report["adapter"] = setup.adapter.get_info().name.into();
        report["backend"] = "browser-webgpu-f32".into();
        Ok(serde_json::to_string(&report)?)
    }
    run(base, reference, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

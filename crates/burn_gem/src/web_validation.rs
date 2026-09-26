use burn::backend::{
    Wgpu,
    wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async},
};
use burn_human_inference::transport::read_bounded;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub async fn validate_gem_webgpu(
    suite: String,
    reference: String,
    image: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(suite: String, reference: String, image: String) -> anyhow::Result<String> {
        let device = WgpuDevice::default();
        let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        let artifacts = crate::pipeline::PipelineArtifacts::from_location(&suite).await?;
        let start = web_time::Instant::now();
        let model =
            crate::pipeline::Pipeline::<Wgpu>::load(&artifacts, &device, |_, _, _| {}).await?;
        let load = start.elapsed().as_secs_f64();
        let reference = read_bounded(&reference, 32 * 1024 * 1024).await?;
        let image = read_bounded(&image, 32 * 1024 * 1024).await?;
        let mut report = crate::validation::validate(&model, &image, &reference).await?;
        anyhow::ensure!(
            report["passed"] == true,
            "GEM WebGPU numerical parity failed: {report}"
        );
        report["performance"] = report["inference"]["timings"].clone();
        report["load_seconds"] = load.into();
        report["artifacts"] = serde_json::to_value(artifacts)?;
        report["adapter"] = setup.adapter.get_info().name.into();
        report["backend"] = "browser-webgpu-f32".into();
        Ok(serde_json::to_string(&report)?)
    }
    run(suite, reference, image)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

use burn::backend::wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async};
use burn_human_inference::gpu::WgpuBackend as Wgpu;
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

/// Measure image inference separately from the checkpoint reconstruction hooks.
#[wasm_bindgen]
pub async fn profile_gem_webgpu(
    suite: String,
    image: String,
    conditions: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(suite: String, image: String, conditions: String) -> anyhow::Result<String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Input {
            crop: crate::camera::Crop,
            camera: crate::camera::Camera,
        }
        let input: Input = serde_json::from_str(&conditions)?;
        input.crop.validate()?;
        input.camera.validate()?;
        let start = web_time::Instant::now();
        let mut phases = Vec::new();
        let mut phase_start = 0.0;
        let mut mark = |name: &str| {
            let end = start.elapsed().as_secs_f64() * 1000.0;
            phases.push(serde_json::json!({"name":name,"start_ms":phase_start,"end_ms":end}));
            phase_start = end;
        };
        let device = WgpuDevice::default();
        init_setup_async::<WebGpu>(&device, Default::default()).await;
        mark("device");
        let artifacts = crate::pipeline::PipelineArtifacts::from_location(&suite).await?;
        let model =
            crate::pipeline::Pipeline::<Wgpu>::load(&artifacts, &device, |_, _, _| {}).await?;
        mark("load");
        let bytes = read_bounded(&image, 32 * 1024 * 1024).await?;
        let image = crate::image_input::decode(&bytes)?;
        mark("image");
        let mut timings = Vec::new();
        for i in 0..4 {
            let start = web_time::Instant::now();
            let output = model
                .estimate_resident(&image, input.crop, input.camera, |_| {})
                .await?;
            let values = output.vertices.into_data_async().await?;
            timings.push(start.elapsed().as_secs_f64());
            anyhow::ensure!(
                values.shape.as_slice() == [1, 18056, 3],
                "Incomplete GEM-X output"
            );
            burn_human_inference::weights::ensure_finite(&values)?;
            mark(if i == 0 {
                "cold_inference"
            } else {
                "warm_inference"
            });
        }
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "passed":true,"phases":phases,"inference_seconds":timings,"artifacts":artifacts,
            "scope":"production load, image decode and synchronized image inference; WASM startup and checkpoint reconstruction hooks excluded"
        }))?)
    }
    run(suite, image, conditions)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

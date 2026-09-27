use burn::backend::wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async};
use burn_human_inference::gpu::WgpuBackend as Wgpu;
use burn_human_inference::transport::read_bounded;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub async fn validate_soma_webgpu(
    base: String,
    reference: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, reference: String, digest: String) -> anyhow::Result<String> {
        let device = WgpuDevice::default();
        let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest.clone();
        let start = web_time::Instant::now();
        let soma = crate::Soma::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        let load = start.elapsed().as_secs_f64();
        let bytes = read_bounded(&reference, 32 * 1024 * 1024).await?;
        let mut report = crate::validation::validate(&soma, &bytes).await?;
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

/// Measure the production load/identity/pose API without parsing reference meshes.
#[wasm_bindgen]
pub async fn profile_soma_webgpu(base: String, digest: String) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn run(base: String, digest: String) -> anyhow::Result<String> {
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
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest;
        let model = crate::Soma::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        mark("load");
        let identity = model.prepare_identity(Default::default()).await?;
        mark("identity");
        let pose = crate::SomaPose::default();
        let mut timings = Vec::new();
        for i in 0..4 {
            let start = web_time::Instant::now();
            let output = model.pose_batch(&identity, std::slice::from_ref(&pose))?;
            let values = output.vertices.into_data_async().await?;
            timings.push(start.elapsed().as_secs_f64());
            anyhow::ensure!(
                values.shape.as_slice() == [1, 18056, 3],
                "Incomplete SOMA output"
            );
            burn_human_inference::weights::ensure_finite(&values)?;
            mark(if i == 0 { "cold_pose" } else { "warm_pose" });
        }
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "passed":true,"phases":phases,"pose_seconds":timings,
            "model_manifest_sha256":artifact.sha256,
            "scope":"production load, identity and synchronized pose; WASM startup and reference parsing excluded"
        }))?)
    }
    run(base, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

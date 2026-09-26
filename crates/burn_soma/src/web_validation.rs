use burn::backend::{
    Wgpu,
    wgpu::{WgpuDevice, graphics::WebGpu, init_setup_async},
};
use burn_human_inference::transport::{ModelSource, read_bounded};
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
        let mut source = ModelSource::new(base);
        let manifest = source.manifest(Some(&digest)).await?;
        let start = web_time::Instant::now();
        let soma = crate::Soma::<Wgpu>::load(&manifest, &mut source, &device, |_, _| {}).await?;
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

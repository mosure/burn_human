// Append to src/web_validation.rs in the pinned baseline checkout. This adds a
// fixture export only; the published generation implementation stays unchanged.
#[wasm_bindgen]
pub async fn capture_serial_webgpu(
    base: String,
    input: String,
    digest: String,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    async fn capture(base: String, input: String, digest: String) -> anyhow::Result<String> {
        let device = WgpuDevice::default();
        let _setup = init_setup_async::<WebGpu>(&device, Default::default()).await;
        let mut artifact = crate::pretrained::DEFAULT.at_base(base);
        artifact.sha256 = digest;
        let model = Ardy::<Wgpu>::load_artifact(&artifact, &device, |_, _| {}).await?;
        let bytes = read_bounded(&input, 32 * 1024 * 1024).await?;
        let mut suite: serde_json::Value = serde_json::from_slice(&bytes)?;
        let embeddings: Vec<burn_human_motion::TextEmbedding> =
            serde_json::from_value(suite["embeddings"].clone())?;
        for case in suite["cases"].as_array_mut().unwrap() {
            let requests: Vec<burn_human_motion::MotionRequest> =
                serde_json::from_value(case["requests"].clone())?;
            let indices: Vec<usize> = serde_json::from_value(case["embedding_indices"].clone())?;
            let mut reference = Vec::new();
            for (request, i) in requests.iter().zip(indices) {
                reference.push(model.generate(request, &embeddings[i], |_| true).await?);
            }
            case["reference"] = serde_json::to_value(reference)?;
        }
        Ok(serde_json::to_string(&suite)?)
    }
    capture(base, input, digest)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

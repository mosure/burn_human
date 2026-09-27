//! Copy to crates/burn_ardy/examples/batch_reference.rs in the stated baseline
//! checkout, then run with --features tools,wgpu --example batch_reference.
//! No batch API is used: every reference comes from the baseline serial path.
use anyhow::{Result, ensure};
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn_ardy::Ardy;
use burn_human_motion::{MotionRequest, TextEmbedding};

fn main() -> Result<()> {
    pollster::block_on(async {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            args.len() == 3,
            "usage: batch_reference BUNDLE INPUT_JSON OUTPUT_JSON"
        );
        let mut suite: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let embeddings: Vec<TextEmbedding> = serde_json::from_value(suite["embeddings"].clone())?;
        let artifact = burn_ardy::pretrained::DEFAULT.at_base(&args[0]);
        let model =
            Ardy::<Wgpu>::load_artifact(&artifact, &WgpuDevice::default(), |_, _| {}).await?;
        for case in suite["cases"].as_array_mut().unwrap() {
            let requests: Vec<MotionRequest> = serde_json::from_value(case["requests"].clone())?;
            let indices: Vec<usize> = serde_json::from_value(case["embedding_indices"].clone())?;
            let mut reference = Vec::new();
            for (request, i) in requests.iter().zip(indices) {
                reference.push(model.generate(request, &embeddings[i], |_| true).await?);
            }
            case["reference"] = serde_json::to_value(reference)?;
            eprintln!("captured {}", case["name"]);
        }
        std::fs::write(&args[2], serde_json::to_vec(&suite)?)?;
        Ok(())
    })
}

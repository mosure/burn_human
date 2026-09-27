use anyhow::{Result, ensure};
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn_ardy::{
    Ardy,
    validation::batch::{Suite, validate},
};

fn main() -> Result<()> {
    pollster::block_on(async {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            args.len() == 3,
            "usage: ardy-batch-validate BUNDLE SUITE_JSON OUTPUT_JSON"
        );
        let suite_bytes = std::fs::read(&args[1])?;
        let suite: Suite = serde_json::from_slice(&suite_bytes)?;
        let artifact = burn_ardy::pretrained::DEFAULT.at_base(&args[0]);
        let model =
            Ardy::<Wgpu>::load_artifact(&artifact, &WgpuDevice::default(), |_, _| {}).await?;
        let mut report = validate(&model, suite).await?;
        report["suite_sha256"] = burn_human_motion::artifacts::sha256(&suite_bytes).into();
        report["model_manifest_sha256"] = artifact.sha256.into();
        report["backend"] = "native-wgpu-f32".into();
        std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    })
}

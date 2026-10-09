//! Mandatory real-checkpoint parity runner. Missing artifacts are errors.
use anyhow::{Result, ensure};
use burn_ardy::transport::ModelSource;
use burn_ardy::{
    Ardy,
    validation::{Fixture, validate},
};
use std::time::Instant;

async fn run(
    base: String,
    fixture: Fixture,
    device: burn::tensor::Device,
    backend: &str,
) -> Result<()> {
    let manifest = ModelSource::new(base.clone()).manifest(None).await?;
    let start = Instant::now();
    let mut artifact = burn_ardy::pretrained::DEFAULT.at_base(base);
    artifact.sha256 = manifest.content_sha256.clone();
    let model = Ardy::load_artifact(&artifact, &device, |_, _| {}).await?;
    let report = validate(
        &model,
        &manifest,
        fixture,
        backend,
        start.elapsed().as_secs_f64(),
        backend != "flex-f32",
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: ardy-validate BUNDLE_URL_OR_DIRECTORY REFERENCE_SAFETENSORS wgpu|cpu"
    );
    let base = args[0].clone();
    let fixture = Fixture {
        data: std::fs::read(&args[1])?,
    };
    match args[2].as_str() {
        "wgpu" => pollster::block_on(run(
            base,
            fixture,
            burn::tensor::Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
            "wgpu-f32",
        )),
        "cpu" | "ndarray" => {
            pollster::block_on(run(base, fixture, burn::tensor::Device::flex(), "flex-f32"))
        }
        _ => anyhow::bail!("unknown backend"),
    }
}

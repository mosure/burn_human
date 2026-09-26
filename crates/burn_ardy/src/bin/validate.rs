//! Mandatory real-checkpoint parity runner. Missing artifacts are errors.
use anyhow::{Result, ensure};
use burn::{
    backend::{NdArray, Wgpu, wgpu::WgpuDevice},
    prelude::Backend,
};
use burn_ardy::{
    Ardy,
    validation::{Fixture, validate},
};
use burn_human_motion::artifacts::{DirectoryReader, Manifest};
use std::{path::PathBuf, time::Instant};

async fn run<B: Backend>(
    base: PathBuf,
    fixture: Fixture,
    device: B::Device,
    backend: &str,
) -> Result<()> {
    let manifest = Manifest::from_bytes(&std::fs::read(base.join("manifest.json"))?, None)?;
    let start = Instant::now();
    let model = Ardy::<B>::load(&manifest, &mut DirectoryReader(base), &device, |_, _| {}).await?;
    let report = validate(
        &model,
        &manifest,
        fixture,
        backend,
        start.elapsed().as_secs_f64(),
        backend != "ndarray-f32",
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: ardy-validate BUNDLE_DIRECTORY REFERENCE_SAFETENSORS wgpu|ndarray"
    );
    let base = PathBuf::from(&args[0]);
    let fixture = Fixture {
        data: std::fs::read(&args[1])?,
    };
    match args[2].to_str() {
        Some("wgpu") => pollster::block_on(run::<Wgpu>(
            base,
            fixture,
            WgpuDevice::default(),
            "wgpu-f32",
        )),
        Some("ndarray") => pollster::block_on(run::<NdArray>(
            base,
            fixture,
            Default::default(),
            "ndarray-f32",
        )),
        _ => anyhow::bail!("unknown backend"),
    }
}

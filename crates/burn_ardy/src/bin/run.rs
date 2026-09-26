use anyhow::{Result, ensure};
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn_ardy::Ardy;
use burn_human_motion::{
    MotionRequest, TextEmbedding,
    artifacts::{DirectoryReader, Manifest},
};
use std::path::PathBuf;

fn main() -> Result<()> {
    pollster::block_on(run())
}
async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: ardy-run BUNDLE_DIRECTORY REQUEST_JSON EMBEDDING_JSON OUTPUT_CLIP_JSON"
    );
    let base = PathBuf::from(&args[0]);
    let manifest = Manifest::from_bytes(&std::fs::read(base.join("manifest.json"))?, None)?;
    let request: MotionRequest = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let embedding: TextEmbedding = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let device = WgpuDevice::default();
    let model = Ardy::<Wgpu>::load(&manifest, &mut DirectoryReader(base), &device, |i, n| {
        eprintln!("loading {i}/{n}")
    })
    .await?;
    let start = std::time::Instant::now();
    let clip = model
        .generate(&request, &embedding, |frames| {
            eprintln!("generated {frames}/{} frames", request.frames);
            true
        })
        .await?;
    let seconds = start.elapsed().as_secs_f64();
    std::fs::write(&args[3], serde_json::to_vec(&clip)?)?;
    println!(
        "{} frames in {seconds:.3}s ({:.2} generated frames/s, includes cold kernels)",
        clip.frames.len(),
        clip.frames.len() as f64 / seconds
    );
    Ok(())
}

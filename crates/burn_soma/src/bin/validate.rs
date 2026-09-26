use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_human_inference::transport::{ModelSource, read_bounded};
use burn_soma::{Soma, validation};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: soma-validate BUNDLE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let source = ModelSource::new(args[0].clone());
        let manifest = source.manifest(None).await?;
        let digest = manifest.content_sha256;
        let mut artifact = burn_soma::pretrained::DEFAULT.at_base(args[0].clone());
        artifact.sha256 = digest.clone();
        let start = web_time::Instant::now();
        let soma = Soma::<Wgpu>::load_artifact(&artifact, &Default::default(), |_, _| {}).await?;
        let load = start.elapsed().as_secs_f64();
        let bytes = read_bounded(&args[1], 32 * 1024 * 1024).await?;
        let mut report = validation::validate(&soma, &bytes).await?;
        report["manifest"] = digest.into();
        report["load_seconds"] = load.into();
        report["backend"] = "native-wgpu-f32".into();
        std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    })
}

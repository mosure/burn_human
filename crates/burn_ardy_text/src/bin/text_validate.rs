use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_ardy_text::{TextEncoder, validation};
use burn_human_inference::transport::{ModelSource, read_bounded};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        a.len() == 3,
        "usage: text-validate BUNDLE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let source = ModelSource::new(a[0].clone());
        let manifest = source.manifest(None).await?;
        let digest = manifest.content_sha256.clone();
        let start = web_time::Instant::now();
        let mut model =
            TextEncoder::<Wgpu>::load(manifest, source, &Default::default(), |_, _| {}).await?;
        let load = start.elapsed().as_secs_f64();
        let reference = serde_json::from_slice(&read_bounded(&a[1], 4 * 1024 * 1024).await?)?;
        let mut report = validation::validate(&mut model, reference).await?;
        report["manifest"] = digest.into();
        report["load_seconds"] = load.into();
        report["backend"] = "native-wgpu-f32".into();
        std::fs::write(&a[2], serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    })
}

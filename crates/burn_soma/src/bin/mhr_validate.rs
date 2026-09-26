use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_human_inference::transport::{ModelSource, read_bounded};
use burn_mhr::{Mhr, MhrInput};
use burn_soma::mhr::MhrSomaTransfer;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    revision: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    identity: Vec<f32>,
    scales: Vec<f32>,
    flex: [f32; 6],
    vertices: Vec<[f32; 3]>,
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        a.len() == 4,
        "usage: mhr-soma-validate MHR_BUNDLE TRANSFER_BUNDLE REFERENCE OUTPUT_JSON"
    );
    pollster::block_on(async {
        let device = Default::default();
        let mut source = ModelSource::new(a[0].clone());
        let manifest = source.manifest(None).await?;
        let model = Mhr::<Wgpu>::load(&manifest, &mut source, &device, |_, _| {}).await?;
        let mut source = ModelSource::new(a[1].clone());
        let manifest = source.manifest(None).await?;
        let transfer = MhrSomaTransfer::<Wgpu>::load(&manifest, &mut source, &device).await?;
        let fixture: Fixture =
            serde_json::from_slice(&read_bounded(&a[2], 16 * 1024 * 1024).await?)?;
        ensure!(
            fixture.revision == burn_soma::MODEL_REVISION,
            "Reference revision mismatch"
        );
        let mut records = vec![];
        for (i, case) in fixture.cases.into_iter().enumerate() {
            let mut input = MhrInput {
                identity: case.identity,
                ..Default::default()
            };
            input.parameters[130..136].copy_from_slice(&case.flex);
            input.parameters[136..].copy_from_slice(&case.scales);
            let output = transfer.transfer(model.evaluate(&[input])?.vertices)?;
            let vertices = output.into_data_async().await?.to_vec::<f32>().unwrap();
            let max = vertices
                .iter()
                .zip(case.vertices.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("case {i}: max transfer error {max}m");
            ensure!(max < 0.0002, "MHR to SOMA numerical parity failed");
            records.push(serde_json::json!({"case":i,"max_vertex_error_m":max}));
        }
        std::fs::write(
            &a[3],
            serde_json::to_vec_pretty(
                &serde_json::json!({"manifest":manifest.content_sha256,"records":records}),
            )?,
        )?;
        Ok(())
    })
}

use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_human_inference::transport::{ModelSource, read_bounded};
use burn_mhr::{Mhr, MhrInput, state_to_matrix};
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    checkpoint: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    identity: Vec<f32>,
    parameters: Vec<f32>,
    expression: Vec<f32>,
    correctives: bool,
    vertices: Vec<[f32; 3]>,
    skeleton: Vec<[f32; 8]>,
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        a.len() == 3,
        "usage: mhr-validate BUNDLE REFERENCE OUTPUT_JSON"
    );
    pollster::block_on(async {
        let mut source = ModelSource::new(a[0].clone());
        let manifest = source.manifest(None).await?;
        let model =
            Mhr::<Wgpu>::load(&manifest, &mut source, &Default::default(), |_, _| {}).await?;
        let fixture: Fixture =
            serde_json::from_slice(&read_bounded(&a[1], 32 * 1024 * 1024).await?)?;
        ensure!(
            fixture.checkpoint == burn_mhr::CHECKPOINT_SHA256,
            "Reference checkpoint mismatch"
        );
        let mut records = vec![];
        for (i, case) in fixture.cases.into_iter().enumerate() {
            let input = MhrInput {
                identity: case.identity,
                parameters: case.parameters,
                expression: case.expression,
                correctives: case.correctives,
            };
            let output = model.evaluate(std::slice::from_ref(&input))?;
            let vertices = output
                .vertices
                .into_data_async()
                .await?
                .to_vec::<f32>()
                .unwrap();
            let vertex_error = vertices
                .iter()
                .zip(case.vertices.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let skeleton_error = output.skeleton_world[0]
                .iter()
                .zip(&case.skeleton)
                .flat_map(|(a, b)| (*a - state_to_matrix(*b)).to_cols_array())
                .map(f32::abs)
                .fold(0f32, f32::max);
            println!("case {i}: vertices={vertex_error}cm skeleton={skeleton_error}");
            ensure!(
                vertex_error < 0.002 && skeleton_error < 0.002,
                "MHR numerical parity failed"
            );
            let mut times = vec![];
            for _ in 0..5 {
                let start = web_time::Instant::now();
                model
                    .evaluate(std::slice::from_ref(&input))?
                    .vertices
                    .into_data_async()
                    .await?;
                times.push(start.elapsed().as_secs_f64());
            }
            records.push(serde_json::json!({"case":i,"max_vertex_error_cm":vertex_error,"max_skeleton_error":skeleton_error,"warm_seconds":times}));
        }
        std::fs::write(
            &a[2],
            serde_json::to_vec_pretty(
                &serde_json::json!({"manifest":manifest.content_sha256,"records":records}),
            )?,
        )?;
        Ok(())
    })
}

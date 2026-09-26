use crate::{IdentityParameters, Soma, SomaPose};
use anyhow::{Result, ensure};
use burn::prelude::Backend;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    revision: String,
    source_revision: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    coefficients: Vec<f32>,
    scales: Vec<f32>,
    global_scale: f32,
    rotations: Vec<[f32; 3]>,
    translation: [f32; 3],
    correctives: bool,
    rest_vertices: Vec<[f32; 3]>,
    bind_world: Vec<[[f32; 4]; 4]>,
    vertices: Vec<[f32; 3]>,
    transforms: Vec<[[f32; 4]; 4]>,
}
fn error(a: impl IntoIterator<Item = f32>, b: impl IntoIterator<Item = f32>) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
pub async fn validate<B: Backend>(soma: &Soma<B>, reference: &[u8]) -> Result<serde_json::Value> {
    let fixture: Fixture = serde_json::from_slice(reference)?;
    ensure!(
        fixture.revision == crate::MODEL_REVISION
            && fixture.source_revision == crate::SOURCE_REVISION,
        "Reference revision mismatch"
    );
    let mut records = vec![];
    for case in fixture.cases {
        let start = web_time::Instant::now();
        let id = soma
            .prepare_identity(IdentityParameters {
                coefficients: case.coefficients,
                bone_scales: case.scales,
                global_scale: case.global_scale,
            })
            .await?;
        let identity_seconds = start.elapsed().as_secs_f64();
        let rest = id
            .rest_vertices
            .clone()
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap();
        let rest_error = error(rest, case.rest_vertices.into_iter().flatten());
        let bind_error = error(
            id.bind_world
                .iter()
                .flat_map(|m| m.transpose().to_cols_array()),
            case.bind_world.into_iter().flatten().flatten(),
        );
        let pose = SomaPose {
            rotations: case.rotations,
            translation: case.translation,
            apply_correctives: case.correctives,
            absolute_pose: false,
        };
        let output = soma.pose_batch(&id, std::slice::from_ref(&pose))?;
        let vertices = output
            .vertices
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap();
        let vertex_error = error(vertices, case.vertices.into_iter().flatten());
        let transform_error = error(
            output.transforms[0]
                .iter()
                .flat_map(|m| m.transpose().to_cols_array()),
            case.transforms.into_iter().flatten().flatten(),
        );
        println!(
            "{}: rest={rest_error}, bind={bind_error}, vertices={vertex_error}, transforms={transform_error}",
            case.name
        );
        ensure!(
            rest_error < 0.0002
                && bind_error < 0.001
                && vertex_error < 0.0002
                && transform_error < 0.001,
            "SOMA numerical parity failed for {}",
            case.name
        );
        let mut times = vec![];
        for batch in [1, 8, 32] {
            let poses = vec![pose.clone(); batch];
            soma.pose_batch(&id, &poses)?
                .vertices
                .into_data_async()
                .await?;
            let mut seconds = vec![];
            for _ in 0..5 {
                let start = web_time::Instant::now();
                soma.pose_batch(&id, &poses)?
                    .vertices
                    .into_data_async()
                    .await?;
                seconds.push(start.elapsed().as_secs_f64());
            }
            seconds.sort_by(f64::total_cmp);
            times.push(serde_json::json!({"batch":batch,"warm_seconds":seconds,"median_frames_per_second":batch as f64 / seconds[2]}));
        }
        records.push(serde_json::json!({"name":case.name,"rest_max_m":rest_error,"bind_max":bind_error,"vertex_max_m":vertex_error,"transform_max":transform_error,"identity_seconds":identity_seconds,"performance":times}));
    }
    Ok(
        serde_json::json!({"performance":records[1]["performance"],"records":records,"reference":"Official SOMA-X Torch, identity fit + canonical rebind + aligned twist + sparse correctives + LBS"}),
    )
}

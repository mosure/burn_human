use crate::{
    camera::{Camera, Crop},
    pipeline::Pipeline,
};
use anyhow::{Result, ensure};
use burn::prelude::Backend;
use burn_soma::BindConvention;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    crop: Crop,
    camera: Camera,
    keypoints: Vec<[f32; 3]>,
    features: Vec<f32>,
    pred_camera: Vec<f32>,
    vertices: Vec<[f32; 3]>,
    joints: Vec<[f32; 3]>,
}
fn errors(a: &[[f32; 3]], b: &[[f32; 3]]) -> Result<serde_json::Value> {
    ensure!(
        a.len() == b.len() && !a.is_empty(),
        "Reference output dimensions"
    );
    let mut max = 0f64;
    let mut sum = 0.0;
    let mut index = 0;
    let mut coordinate = 0;
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        for j in 0..3 {
            ensure!(a[j].is_finite() && b[j].is_finite(), "Non-finite output");
            let e = (a[j] as f64 - b[j] as f64).abs();
            sum += e * e;
            if e > max {
                max = e;
                index = i;
                coordinate = j;
            }
        }
    }
    Ok(
        serde_json::json!({"max":max,"rmse":(sum/(a.len()*3)as f64).sqrt(),"index":index,"coordinate":coordinate,"actual":a[index],"expected":b[index]}),
    )
}
pub async fn validate<B: Backend>(
    model: &Pipeline<B>,
    image_bytes: &[u8],
    reference: &[u8],
) -> Result<serde_json::Value> {
    let r: Fixture = serde_json::from_slice(reference)?;
    let image = crate::image_input::decode(image_bytes)?;
    // Isolate fitted rig reconstruction from network and image preprocessing.
    let (id, pose) = model.decode_pose(&r.features, &r.pred_camera, r.crop, r.camera)?;
    let prepared = model
        .transfer
        .prepare_identity_with_bind(&model.mhr, &model.soma, &id, BindConvention::Fitted)
        .await?;
    let output = model.soma.pose_batch(&prepared, &[pose])?;
    let points: Vec<[f32; 3]> = output
        .vertices
        .into_data_async()
        .await?
        .to_vec::<f32>()
        .unwrap()
        .as_chunks::<3>()
        .0
        .to_vec();
    let joints: Vec<[f32; 3]> = output.transforms[0]
        .iter()
        .skip(1)
        .map(|m| m.w_axis.truncate().to_array())
        .collect();
    let reconstruction = serde_json::json!({"vertices":errors(&points,&r.vertices)?,"joints":errors(&joints,&r.joints)?});
    println!("Reconstruction: {reconstruction}");
    let estimate = model
        .estimate(&image, r.crop, r.camera, |s| println!("{s}"))
        .await?;
    let inference = serde_json::json!({"vertices":errors(&estimate.vertices,&r.vertices)?,"joints":errors(&estimate.joints,&r.joints)?,"keypoints":errors(&estimate.keypoints_2d,&r.keypoints)?,"timings":estimate.timings});
    let passed = reconstruction["vertices"]["max"].as_f64().unwrap() < 0.001
        && inference["vertices"]["max"].as_f64().unwrap() < 0.005;
    let mut warm = Vec::new();
    for _ in 0..3 {
        warm.push(
            model
                .estimate(&image, r.crop, r.camera, |_| {})
                .await?
                .timings,
        );
    }
    Ok(
        serde_json::json!({"passed":passed,"reconstruction":reconstruction,"inference":inference,"warm_seconds_by_stage":warm}),
    )
}

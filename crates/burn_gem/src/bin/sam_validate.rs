use anyhow::{Result, ensure};
use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use burn_gem::{
    camera::{Camera, Crop},
    sam::SamBody,
};
use burn_human_inference::transport::ModelSource;
use burn_mhr::Mhr;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    embedding: Vec<f32>,
    crop: Crop,
    camera: Camera,
    token: Vec<f32>,
    ray_embedding: Vec<f32>,
    layers: Vec<Layer>,
}
#[derive(Deserialize)]
struct Layer {
    params: Vec<f32>,
    shape: Vec<f32>,
    keypoints: Vec<[f32; 3]>,
    camera: [f32; 3],
}
fn error(a: impl IntoIterator<Item = f32>, b: impl IntoIterator<Item = f32>) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "gem-sam-validate SAM_BUNDLE MHR_BUNDLE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let device = Default::default();
        let mut source = ModelSource::new(args[0].clone());
        let manifest = source.manifest(None).await?;
        let model = SamBody::<Wgpu>::load(&manifest, &mut source, &device, |_, _| {}).await?;
        let mut source = ModelSource::new(args[1].clone());
        let mhr_manifest = source.manifest(None).await?;
        let mhr = Mhr::load(&mhr_manifest, &mut source, &device, |_, _| {}).await?;
        let reference: Fixture = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        let image = Tensor::from_data(
            TensorData::new(reference.embedding, [1, 1280, 32, 32]),
            &device,
        );
        let conditioned = model
            .condition_image(image.clone(), reference.crop, reference.camera)?
            .swap_dims(1, 2)
            .reshape([1, 1280, 32, 32])
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap();
        let ray_max = error(conditioned, reference.ray_embedding);
        println!("ray_max: {ray_max}");
        let start = web_time::Instant::now();
        let output = model
            .forward(image.clone(), &mhr, reference.crop, reference.camera)
            .await?;
        let token = output
            .token
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap();
        let cold = start.elapsed().as_secs_f64();
        let max = error(token.clone(), reference.token.clone());
        let rmse = (token
            .iter()
            .zip(&reference.token)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / 1024.0)
            .sqrt();
        let mut layers = vec![];
        for (i, (a, b)) in output.layers.iter().zip(&reference.layers).enumerate() {
            let params = error(a.parameters.parameters.clone(), b.params.clone());
            let shape = error(a.parameters.identity.clone(), b.shape.clone());
            let keypoints = error(
                a.keypoints.iter().flatten().copied(),
                b.keypoints.iter().flatten().copied(),
            );
            let camera = error(a.camera, b.camera);
            println!(
                "layer {i}: params={params}, shape={shape}, keypoints={keypoints}, camera={camera}"
            );
            layers.push(serde_json::json!({"parameters_max":params,"shape_max":shape,"keypoints_max_m":keypoints,"camera_max":camera}));
        }
        let mut warm = vec![];
        for _ in 0..3 {
            let start = web_time::Instant::now();
            model
                .forward(image.clone(), &mhr, reference.crop, reference.camera)
                .await?
                .token
                .into_data_async()
                .await?;
            warm.push(start.elapsed().as_secs_f64());
        }
        let report = serde_json::json!({"manifest":manifest.content_sha256,"mhr_manifest":mhr_manifest.content_sha256,"backend":"native-wgpu-f32","ray_max":ray_max,"token_max":max,"token_rmse":rmse,"layers":layers,"cold_seconds":cold,"warm_seconds":warm});
        std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
        ensure!(
            max < 0.003 && rmse < 0.001 && ray_max < 0.0002,
            "SAM numerical parity failed"
        );
        Ok(())
    })
}

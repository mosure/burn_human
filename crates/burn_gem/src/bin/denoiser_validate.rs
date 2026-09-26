use anyhow::{Result, ensure};
use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use burn_gem::{
    camera::{Camera, Crop},
    denoiser::{Conditions, GemDenoiser},
};
use burn_human_inference::transport::ModelSource;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    batch: usize,
    frames: usize,
    observations: Vec<f32>,
    boxes: Vec<[f32; 3]>,
    cameras: Vec<[[f32; 3]; 3]>,
    features: Vec<f32>,
    angular: Vec<[f32; 6]>,
    prediction: Vec<f32>,
    camera: Vec<f32>,
}
fn max(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "gem-denoiser-validate BUNDLE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let device = Default::default();
        let mut source = ModelSource::new(args[0].clone());
        let manifest = source.manifest(None).await?;
        let model = GemDenoiser::<Wgpu>::load(&manifest, &mut source, &device, |_, _| {}).await?;
        let reference: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let mut records = vec![];
        let mut pass = true;
        for r in reference.cases {
            let cond = Conditions {
                batch: r.batch,
                frames: r.frames,
                observations: r.observations.as_chunks::<3>().0.to_vec(),
                crops: r
                    .boxes
                    .iter()
                    .map(|p| Crop {
                        center: [p[0], p[1]],
                        size: p[2],
                    })
                    .collect(),
                cameras: r
                    .cameras
                    .iter()
                    .map(|p| Camera {
                        focal: [p[0][0], p[1][1]],
                        center: [p[0][2], p[1][2]],
                    })
                    .collect(),
                angular: r.angular,
            };
            let features = Tensor::from_data(
                TensorData::new(r.features, [r.batch, r.frames, 1024]),
                &device,
            );
            let out = model.predict(&cond, features.clone())?;
            let x = out
                .features
                .into_data_async()
                .await?
                .to_vec::<f32>()
                .unwrap();
            let c = out.camera.into_data_async().await?.to_vec::<f32>().unwrap();
            let feature_max = max(&x, &r.prediction);
            let camera_max = max(&c, &r.camera);
            println!(
                "{}x{}: features={feature_max}, camera={camera_max}",
                r.batch, r.frames
            );
            pass &= feature_max < 0.001 && camera_max < 0.001;
            let start = web_time::Instant::now();
            model
                .predict(&cond, features)?
                .features
                .into_data_async()
                .await?;
            records.push(serde_json::json!({"batch":r.batch,"frames":r.frames,"feature_max":feature_max,"camera_max":camera_max,"warm_seconds":start.elapsed().as_secs_f64()}));
        }
        std::fs::write(
            &args[2],
            serde_json::to_vec_pretty(
                &serde_json::json!({"manifest":manifest.content_sha256,"backend":"native-wgpu-f32","records":records}),
            )?,
        )?;
        ensure!(pass, "GEM denoiser numerical parity failed");
        Ok(())
    })
}

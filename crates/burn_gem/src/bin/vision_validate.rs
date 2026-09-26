use anyhow::{Result, ensure};
use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use burn_gem::vision::Vision;
use burn_human_inference::transport::ModelSource;
use serde::Deserialize;
#[derive(Deserialize)]
struct Fixture {
    shape: [usize; 4],
    input: Vec<f32>,
    output_shape: [usize; 4],
    output: Vec<f32>,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "gem-vision-validate BUNDLE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let device = Default::default();
        let mut source = ModelSource::new(args[0].clone());
        let manifest = source.manifest(None).await?;
        let start = web_time::Instant::now();
        let model = Vision::<Wgpu>::load(&manifest, &mut source, &device, |i, n| {
            if i % 100 == 0 {
                println!("{i}/{n}");
            }
        })
        .await?;
        let load = start.elapsed().as_secs_f64();
        let reference: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let input = Tensor::from_data(TensorData::new(reference.input, reference.shape), &device);
        let start = web_time::Instant::now();
        let out = model.forward(input.clone())?;
        ensure!(
            out.dims() == reference.output_shape,
            "Output shape mismatch"
        );
        let values = out.into_data_async().await?.to_vec::<f32>().unwrap();
        let cold = start.elapsed().as_secs_f64();
        let mut max = 0f64;
        let mut square = 0.0;
        let mut dot = 0.0;
        let mut aa = 0.0;
        let mut bb = 0.0;
        for (a, b) in values.iter().zip(&reference.output) {
            let (a, b) = (*a as f64, *b as f64);
            ensure!(a.is_finite() && b.is_finite(), "Non-finite output");
            max = max.max((a - b).abs());
            square += (a - b).powi(2);
            dot += a * b;
            aa += a * a;
            bb += b * b;
        }
        let rmse = (square / values.len() as f64).sqrt();
        let cosine = dot / (aa * bb).sqrt();
        let mut warm = vec![];
        for _ in 0..3 {
            let start = web_time::Instant::now();
            model.forward(input.clone())?.into_data_async().await?;
            warm.push(start.elapsed().as_secs_f64());
        }
        let report = serde_json::json!({"manifest":manifest.content_sha256,"backend":"native-wgpu-f32","kind":model.config.kind,"max":max,"rmse":rmse,"cosine":cosine,"load_seconds":load,"cold_seconds":cold,"warm_seconds":warm});
        std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
        ensure!(
            cosine > 0.99999 && rmse < 0.001,
            "Vision numerical parity failed"
        );
        Ok(())
    })
}

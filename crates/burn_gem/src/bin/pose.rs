use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_gem::{
    camera::{Camera, Crop},
    image_input,
    pipeline::{Pipeline, PipelineArtifacts},
};
#[derive(serde::Deserialize)]
struct Request {
    crop: Crop,
    camera: Camera,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "gem-pose SUITE_JSON IMAGE CROP_CAMERA_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let artifacts = PipelineArtifacts::from_location(&args[0]).await?;
        let request: Request = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        let image = image_input::decode(&std::fs::read(&args[1])?)?;
        let start = web_time::Instant::now();
        let model = Pipeline::<Wgpu>::load(&artifacts, &Default::default(), |s, i, n| {
            if i == n || i % 100 == 0 {
                println!("{s} {i}/{n}");
            }
        })
        .await?;
        let load_seconds = start.elapsed().as_secs_f64();
        let pose = model
            .estimate(&image, request.crop, request.camera, |s| println!("{s}"))
            .await?;
        let mut warm = vec![];
        for _ in 0..3 {
            let start = web_time::Instant::now();
            model
                .estimate(&image, request.crop, request.camera, |_| {})
                .await?;
            warm.push(start.elapsed().as_secs_f64());
        }
        let mut output = serde_json::to_value(pose)?;
        output["load_seconds"] = load_seconds.into();
        output["warm_seconds"] = serde_json::to_value(warm)?;
        output["artifacts"] = serde_json::to_value(artifacts)?;
        output["backend"] = "native-wgpu-f32".into();
        std::fs::write(&args[3], serde_json::to_vec(&output)?)?;
        println!("Saved {}", args[3]);
        Ok(())
    })
}

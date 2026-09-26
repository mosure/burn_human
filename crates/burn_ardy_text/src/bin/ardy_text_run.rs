//! Text and motion inference in one native process, using the same portable APIs as Bevy.
use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_ardy::Ardy;
use burn_ardy_text::TextEncoder;
use burn_human_inference::transport::{ModelSource, read_bounded};
use burn_human_motion::MotionRequest;
use std::{collections::BTreeMap, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: ardy-text-run TEXT_BUNDLE MOTION_BUNDLE REQUESTS_JSON NEW_OUTPUT_DIRECTORY"
    );
    pollster::block_on(async {
        let requests: BTreeMap<String, MotionRequest> =
            serde_json::from_slice(&read_bounded(&args[2], 1024 * 1024).await?)?;
        ensure!(!requests.is_empty(), "No motion requests");
        for (label, request) in &requests {
            ensure!(
                !label.is_empty()
                    && label
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "Invalid request label"
            );
            request.validate()?;
        }
        let out = Path::new(&args[3]);
        std::fs::create_dir(out)?;
        let device = Default::default();
        let source = ModelSource::cached(args[0].clone());
        let text_manifest = source.manifest(None).await?;
        let text_digest = text_manifest.content_sha256.clone();
        let mut encoder =
            TextEncoder::<Wgpu>::load(text_manifest, source, &device, |_, _| {}).await?;
        let mut source = ModelSource::cached(args[1].clone());
        let motion_manifest = source.manifest(None).await?;
        let model = Ardy::<Wgpu>::load(&motion_manifest, &mut source, &device, |_, _| {}).await?;
        let mut records = Vec::new();
        for (label, request) in requests {
            let start = web_time::Instant::now();
            let embedding = encoder.encode(&request.prompt, |_, _| {}).await?;
            let text_seconds = start.elapsed().as_secs_f64();
            let start = web_time::Instant::now();
            let clip = model.generate(&request, &embedding, |_| true).await?;
            let motion_seconds = start.elapsed().as_secs_f64();
            std::fs::write(
                out.join(format!("embedding-{label}.json")),
                serde_json::to_vec(&embedding)?,
            )?;
            std::fs::write(
                out.join(format!("request-{label}.json")),
                serde_json::to_vec_pretty(&request)?,
            )?;
            std::fs::write(
                out.join(format!("clip-{label}.json")),
                serde_json::to_vec(&clip)?,
            )?;
            eprintln!("{label}: text {text_seconds:.3}s, motion {motion_seconds:.3}s");
            records.push(serde_json::json!({"case":label,"text_seconds":text_seconds,"motion_seconds":motion_seconds,"frames":clip.frames.len()}));
        }
        let report = serde_json::json!({"backend":"native-wgpu", "attention":"bidirectional", "text_manifest":text_digest,"motion_manifest":motion_manifest.content_sha256,"records":records,"timing_scope":"Includes decoder and readback; first request includes cold kernels; excludes model load"});
        std::fs::write(out.join("runs.json"), serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    })
}

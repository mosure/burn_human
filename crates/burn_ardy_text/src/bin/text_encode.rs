use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_ardy_text::{AttentionMode, TextEncoder};
use burn_human_inference::transport::ModelSource;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: text-encode BUNDLE OUTPUT_JSON PROMPT [causal]"
    );
    pollster::block_on(async {
        let source = ModelSource::new(args[0].clone());
        let manifest = source.manifest(None).await?;
        let start = web_time::Instant::now();
        let mut encoder =
            TextEncoder::<Wgpu>::load(manifest, source, &Default::default(), |i, n| {
                if i % 32 == 0 || i == n {
                    eprintln!("text weights {i}/{n}")
                }
            })
            .await?;
        if args.get(3).is_some_and(|m| m == "causal") {
            encoder.mode = AttentionMode::CausalExport;
        }
        let load = start.elapsed().as_secs_f64();
        let start = web_time::Instant::now();
        let embedding = encoder
            .encode(&args[2], |i, n| {
                if i % 8 == 0 {
                    eprintln!("text layer {i}/{n}")
                }
            })
            .await?;
        let cold = start.elapsed().as_secs_f64();
        std::fs::write(&args[1], serde_json::to_vec_pretty(&embedding)?)?;
        let mut times = vec![];
        for _ in 0..3 {
            let start = web_time::Instant::now();
            encoder.encode(&args[2], |_, _| {}).await?;
            times.push(start.elapsed().as_secs_f64());
        }
        eprintln!("load {load:.3}s, cold {cold:.3}s, warm {times:?}");
        Ok(())
    })
}

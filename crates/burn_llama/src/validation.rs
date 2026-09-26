//! Full checkpoint validation shared by the native and browser harnesses.
use crate::{AttentionMode, TextEncoder};
use anyhow::{Result, ensure};
use burn::prelude::Backend;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct Reference {
    model: String,
    revision: String,
    modes: BTreeMap<String, Vec<Record>>,
}
#[derive(Deserialize)]
struct Record {
    prompt: String,
    values: Vec<f32>,
    tokens: Tokens,
}
#[derive(Deserialize)]
struct Tokens {
    input_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    embed_mask: Vec<u8>,
}

pub async fn validate<B: Backend>(
    model: &mut TextEncoder<B>,
    reference: Reference,
) -> Result<serde_json::Value> {
    ensure!(
        reference.model == crate::MODEL_ID && reference.revision == crate::MODEL_REVISION,
        "Reference checkpoint mismatch"
    );
    let mut records = vec![];
    for (name, mode) in [
        ("causal_export", AttentionMode::CausalExport),
        ("bidirectional", AttentionMode::Bidirectional),
    ] {
        model.mode = mode;
        let refs = reference
            .modes
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Missing mask reference"))?;
        ensure!(refs.len() >= 5, "Require five independent prompt cases");
        for r in refs {
            let tokens = model.tokenize(&r.prompt)?;
            ensure!(
                tokens.ids.as_slice() == r.tokens.input_ids
                    && tokens
                        .attention
                        .iter()
                        .map(|v| u8::from(*v))
                        .collect::<Vec<_>>()
                        == r.tokens.attention_mask
                    && tokens.pool.iter().map(|v| *v as u8).collect::<Vec<_>>()
                        == r.tokens.embed_mask,
                "Tokenizer / pooling mismatch for {}",
                r.prompt
            );
            let start = web_time::Instant::now();
            let output = model.encode(&r.prompt, |_, _| {}).await?;
            let seconds = start.elapsed().as_secs_f64();
            ensure!(r.values.len() == 4096, "Reference embedding shape");
            let mut ab = 0f64;
            let mut aa = 0f64;
            let mut bb = 0f64;
            let mut sq = 0f64;
            let mut max = 0f64;
            for (&a, &b) in output.values.iter().zip(&r.values) {
                let a = a as f64;
                let b = b as f64;
                ab += a * b;
                aa += a * a;
                bb += b * b;
                sq += (a - b).powi(2);
                max = max.max((a - b).abs());
            }
            let cosine = ab / (aa * bb).sqrt();
            let rmse = (sq / 4096.0).sqrt();
            ensure!(
                cosine >= 0.99999 && rmse <= 0.01,
                "{name} / {}: cosine={cosine}, rmse={rmse}",
                r.prompt
            );
            records.push(serde_json::json!({"mode":name,"prompt":r.prompt,"cosine":cosine,"rmse":rmse,"max_abs":max,"seconds":seconds}));
        }
    }
    for bad in ["", " ", "<|eot_id|>"] {
        ensure!(model.tokenize(bad).is_err(), "Invalid prompt accepted");
    }
    ensure!(
        model.tokenize(&"walk ".repeat(100)).is_err(),
        "Overlong prompt silently truncated"
    );
    model.mode = AttentionMode::Bidirectional;
    let prompt = &reference.modes["bidirectional"][0].prompt;
    let mut warm = vec![];
    for _ in 0..5 {
        let start = web_time::Instant::now();
        model.encode(prompt, |_, _| {}).await?;
        warm.push(start.elapsed().as_secs_f64());
    }
    Ok(
        serde_json::json!({"records":records,"performance":{"warm_prompt_seconds":warm},"execution":"Packed Q4F weights; per-matrix expansion on GPU before matmul; F32 activations", "vocabulary_host_cache_limit_bytes":64*1024*1024,"pooling":"mean, excluding instruction header, including EOT", "reference":"Pinned ORT INT4/FP16 export, plus mask-only bidirectional variant"}),
    )
}

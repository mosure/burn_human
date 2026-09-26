//! Native HTTP/cache qualification for every released component, including lazy vocabulary pages.
use anyhow::{Result, ensure};
use burn_human_inference::{pretrained::ArtifactLocation, weights::read_tensors};
use burn_human_motion::artifacts::PartReader;
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Release {
    entries: BTreeMap<String, Entry>,
}
#[derive(Deserialize)]
struct Entry {
    bundle: String,
    sha256: String,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: human-cdn-check RELEASE_JSON HTTP_MODEL_ROOT NEW_CACHE_DIR OUTPUT_JSON"
    );
    ensure!(
        args[1].starts_with("http://") || args[1].starts_with("https://"),
        "qualification requires HTTP"
    );
    let release: Release = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let cache = PathBuf::from(&args[2]);
    ensure!(
        !cache.exists(),
        "qualification needs a fresh isolated cache"
    );
    std::fs::create_dir_all(&cache)?;
    let mut reports = BTreeMap::new();
    pollster::block_on(async {
        for (name, entry) in release.entries {
            let location = ArtifactLocation {
                base: format!("{}/{}", args[1].trim_end_matches('/'), entry.bundle),
                sha256: entry.sha256,
            };
            let mut runs = Vec::new();
            for run in ["cold", "warm", "corrupt"] {
                let start = Instant::now();
                let (manifest, mut source) = location.open().await?;
                source.cache = Some(cache.clone());
                let parts: BTreeMap<_, _> = manifest
                    .objects
                    .iter()
                    .flat_map(|o| &o.parts)
                    .map(|p| (p.sha256.clone(), p.clone()))
                    .collect();
                let corrupt = parts.values().next().unwrap();
                if run == "corrupt" {
                    std::fs::write(cache.join(corrupt.path()), b"damaged")?;
                }
                for asset in &manifest.assets {
                    source.asset(asset).await?;
                }
                // Decode/validate once on the cold path. Warm/corrupt paths still
                // hash every physical part and assert exact repair traffic.
                if run == "cold" {
                    for object in &manifest.objects {
                        drop(read_tensors(&mut source, object).await?);
                    }
                } else {
                    for part in parts.values() {
                        drop(source.read_part(part).await?);
                    }
                }
                let expected = match run {
                    "cold" => parts.values().map(|p| p.size).sum::<usize>(),
                    "warm" => 0,
                    _ => corrupt.size,
                };
                ensure!(
                    source.downloaded_bytes == expected,
                    "{name} {run} downloaded {}, expected {expected}",
                    source.downloaded_bytes
                );
                runs.push(serde_json::json!({"run":run,"downloaded_part_bytes":source.downloaded_bytes,"cache_hits":source.cache_hits,"seconds":start.elapsed().as_secs_f64()}));
            }
            reports.insert(
                name.clone(),
                serde_json::json!({"artifact":location,"runs":runs}),
            );
            println!("{name}: cold, zero-download warm, and single-part repair passed");
            // Qualification isolates each component so an 8 GiB FIFO cannot
            // obscure a cache miss. Remove only this tool's freshly created cache.
            std::fs::remove_dir_all(cache.join("parts"))?;
        }
        Ok::<_, anyhow::Error>(())
    })?;
    let report = serde_json::json!({"root":args[1],"passed":true,"scope":"all released components, each with an isolated cold cache; metadata is reauthenticated on every load", "components":reports});
    std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

//! Offline publication staging. Model identity and policy come from the release plan.
use anyhow::{Context, Result, ensure};
use burn_human_inference::{transport::ModelSource, weights::read_tensors};
use burn_human_motion::artifacts::{Asset, MAX_PART_BYTES, sha256};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    source: String,
    bundle: String,
    sha256: String,
    assets: BTreeMap<String, String>,
}
#[derive(Serialize)]
struct Entry {
    bundle: String,
    sha256: String,
    model: String,
    model_revision: String,
    source_revision: String,
    parts: usize,
    bytes: u64,
}

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: human-cdn-prepare PLAN_JSON NEW_OUTPUT_ROOT (run from repository root)"
    );
    let plan: BTreeMap<String, Input> = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let output = PathBuf::from(&args[1]);
    ensure!(
        !output.exists(),
        "output must be fresh; never overwrite an immutable release"
    );
    let tree = output.join("aberration.technology/model");
    std::fs::create_dir_all(&tree)?;
    let mut entries = BTreeMap::new();
    let mut checksums = BTreeMap::new();
    for (name, input) in plan {
        ensure!(
            input.bundle.split('/').all(|part| !part.is_empty()
                && part != "."
                && part != ".."
                && part.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')),
            "invalid bundle path"
        );
        let mut source = ModelSource::new(input.source.clone());
        let mut manifest = source
            .manifest(Some(&input.sha256))
            .await
            .context(name.clone())?;
        for asset in &manifest.assets {
            source.asset(asset).await?;
        }
        // Decode each bounded Burnpack, checking the exact tensor keyset, shape,
        // dtype, byte digest and finiteness. Never materialize a whole model.
        let mut parts = BTreeMap::new();
        for object in &manifest.objects {
            drop(
                read_tensors(&mut source, object)
                    .await
                    .context(object.stage.clone())?,
            );
            for part in &object.parts {
                ensure!(part.size <= MAX_PART_BYTES, "oversized CDN part");
                if let Some(size) = parts.insert(part.sha256.clone(), part.size) {
                    ensure!(size == part.size, "conflicting part size");
                }
            }
        }
        for (path, file) in &input.assets {
            ensure!(
                !manifest.assets.iter().any(|a| a.path == *path),
                "duplicate supplemental metadata {path}"
            );
            let bytes = std::fs::read(file).with_context(|| file.clone())?;
            manifest.assets.push(Asset {
                path: path.clone(),
                size: bytes.len(),
                sha256: sha256(&bytes),
            });
        }
        manifest.schema_version = 2;
        manifest.seal()?; // also validate metadata paths before writing them
        let bundle = input.bundle;
        let dest = tree.join(&bundle);
        ensure!(!dest.exists(), "bundle destination already exists");
        std::fs::create_dir_all(&dest)?;
        std::fs::create_dir(dest.join("parts"))?;
        for (digest, size) in &parts {
            let relative = format!("parts/{digest}.bin");
            let src = Path::new(&input.source).join(&relative);
            let target = dest.join(&relative);
            // Hardlink immutable weight bytes on one filesystem; support external
            // staging disks with a bounded copy fallback. Metadata is always copied.
            if std::fs::hard_link(&src, &target).is_err() {
                std::fs::copy(&src, &target)?;
            }
            ensure!(
                target.metadata()?.len() == *size as u64,
                "staged size mismatch"
            );
            checksums.insert(
                format!("aberration.technology/model/{bundle}/{relative}"),
                digest.clone(),
            );
        }
        for asset in &manifest.assets {
            let file = input
                .assets
                .get(&asset.path)
                .map(PathBuf::from)
                .unwrap_or_else(|| Path::new(&input.source).join(&asset.path));
            let bytes = std::fs::read(file)?;
            asset.verify(&bytes)?;
            let target = dest.join(&asset.path);
            std::fs::create_dir_all(target.parent().unwrap())?;
            std::fs::write(target, bytes)?;
            checksums.insert(
                format!("aberration.technology/model/{bundle}/{}", asset.path),
                asset.sha256.clone(),
            );
        }
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        checksums.insert(
            format!("aberration.technology/model/{bundle}/manifest.json"),
            sha256(&bytes),
        );
        std::fs::write(dest.join("manifest.json"), bytes)?;
        println!(
            "{name}: {} verified objects, {} parts -> {bundle}",
            manifest.objects.len(),
            parts.len()
        );
        entries.insert(
            name,
            Entry {
                bundle,
                sha256: manifest.content_sha256,
                model: manifest.model,
                model_revision: manifest.model_revision,
                source_revision: manifest.source_revision,
                parts: parts.len(),
                bytes: parts.values().map(|n| *n as u64).sum(),
            },
        );
    }
    let keys = [
        "vitpose",
        "sam_vision",
        "sam_decoder",
        "denoiser",
        "mhr",
        "soma",
        "transfer",
    ];
    let mut suite = BTreeMap::new();
    for key in keys {
        let entry = entries
            .get(key)
            .with_context(|| format!("missing GEM-X dependency {key}"))?;
        suite.insert(
            key,
            serde_json::json!({"base": relative_bundle("gemx/v1", &entry.bundle), "sha256": entry.sha256}),
        );
    }
    ensure!(
        entries.contains_key("ardy") && entries.contains_key("llama"),
        "missing motion model"
    );
    let bytes = serde_json::to_vec_pretty(&suite)?;
    let digest = sha256(&bytes);
    let suite_bundle = "gemx/v1";
    std::fs::create_dir_all(tree.join(suite_bundle))?;
    std::fs::write(tree.join(suite_bundle).join("suite.json"), bytes)?;
    checksums.insert(
        format!("aberration.technology/model/{suite_bundle}/suite.json"),
        digest.clone(),
    );
    let report = serde_json::json!({ "schema_version": 1, "cdn_root": "https://aberration.technology/model",
        "entries": entries, "suite": {"bundle": suite_bundle, "sha256": digest},
        "verified": "every part, logical Burnpack, tensor inventory/shape/dtype/SHA256/finiteness and metadata",
        "transport": {"part_target_bytes": 20*1024*1024, "part_limit_bytes": MAX_PART_BYTES,
            "object_limit_bytes": 64*1024*1024, "direct_weight_files": false},
        "deployment": "prepared; public CDN download validation awaits upload" });
    std::fs::write(
        output.join("release.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let sums = checksums
        .iter()
        .map(|(p, h)| format!("{h}  {p}\n"))
        .collect::<String>();
    std::fs::write(output.join("SHA256SUMS"), sums)?;
    std::fs::write(
        output.join("UPLOAD.md"),
        format!(
            "# Human model CDN upload\n\nUpload the contents of `aberration.technology/model/` to `https://aberration.technology/model/`, preserving all directory names. This is the full dependency closure: ARDY, Llama, SOMA and seven GEM-X components (SOMA is shared).\n\nModel families contain readable version/component directories. Version directories are immutable: use v2 for changed artifacts instead of replacing v1. Hashes and source revisions remain in manifests. Upload only to empty prefixes; upload `manifest.json` and `{suite_bundle}/suite.json` last after their parts/metadata and dependency bundles. Serve parts as `application/octet-stream`, JSON as `application/json`, with `Access-Control-Allow-Origin: *` and `Cache-Control: public, max-age=31536000, immutable`. GET must return exact uncompressed bytes, without authentication, HTML error wrappers or transformations. HEAD and byte ranges are useful but the current bounded loader uses GET. Largest physical part is 20 MiB; hard ceiling is 25,000,000 bytes.\n\nBefore upload run `sha256sum --check SHA256SUMS` from this directory. Do not upload `release.json`, this readme or SHA256SUMS into a model prefix. Shards are hardlinked to verified local source files when possible: treat both as immutable. No native binaries, source checkpoints, fixtures, raw tensor exports or direct logical Burnpacks are in the upload tree.\n\nSuite URL: https://aberration.technology/model/{suite_bundle}/suite.json\n\nThis bundle has passed local artifact verification. Public native/browser cold, warm and corrupted-cache tests must be run after upload; preparation is not deployment evidence. See the repository's docs/cdn.md for commands.\n"
        ),
    )?;
    println!("Upload tree: {}", tree.display());
    Ok(())
}

/// Portable URL paths, independent of the packer's operating system.
fn relative_bundle(parent: &str, target: &str) -> String {
    let parent: Vec<_> = parent.split('/').collect();
    let target: Vec<_> = target.split('/').collect();
    let common = parent
        .iter()
        .zip(&target)
        .take_while(|(a, b)| a == b)
        .count();
    std::iter::repeat_n("..", parent.len() - common)
        .chain(target[common..].iter().copied())
        .collect::<Vec<_>>()
        .join("/")
}

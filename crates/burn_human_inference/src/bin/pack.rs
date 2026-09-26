//! Package one verified raw tensor at a time; raw export is an offline interchange.
use anyhow::{Result, ensure};
use burn::{module::ParamId, tensor::TensorData};
use burn_human_motion::artifacts::{
    Asset, MAX_OBJECT_BYTES, Manifest, Object, PART_TARGET_BYTES, Part, TensorSpec, sha256,
};
use burn_store::{BurnpackWriter, TensorSnapshot};
use serde::Deserialize;
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAsset {
    path: String,
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Export {
    model: String,
    model_revision: String,
    source_revision: String,
    converter: String,
    license: String,
    config: serde_json::Value,
    tensors: Vec<RawTensor>,
    #[serde(default)]
    assets: Vec<RawAsset>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: human-model-pack RAW_EXPORT_DIRECTORY NEW_BUNDLE_DIRECTORY"
    );
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    ensure!(!output.exists(), "bundle destination must be new");
    let mut export: Export = serde_json::from_slice(&std::fs::read(input.join("export.json"))?)?;
    export.tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let mut manifest = Manifest {
        schema_version: 2,
        model: export.model,
        model_revision: export.model_revision,
        source_revision: export.source_revision,
        converter: export.converter,
        license: export.license,
        config: export.config,
        objects: vec![],
        assets: vec![],
        content_sha256: String::new(),
    };
    std::fs::create_dir_all(output.join("parts"))?;
    let mut names = BTreeSet::new();
    for t in export.tensors {
        ensure!(names.insert(t.name.clone()), "duplicate tensor {}", t.name);
        let spec = TensorSpec {
            name: t.name.clone(),
            shape: t.shape.clone(),
            dtype: t.dtype.clone(),
            sha256: t.sha256,
        };
        ensure!(
            spec.byte_len().is_some_and(|n| n < MAX_OBJECT_BYTES - 4096),
            "raw tensor exceeds object budget: {}",
            t.name
        );
        let file = std::fs::File::open(input.join(t.file))?;
        ensure!(
            Some(file.metadata()?.len() as usize) == spec.byte_len(),
            "raw tensor size mismatch"
        );
        use std::io::Read;
        let mut raw = Vec::new();
        file.take(MAX_OBJECT_BYTES as u64).read_to_end(&mut raw)?;
        ensure!(sha256(&raw) == spec.sha256, "raw tensor digest mismatch");
        let data = TensorData::from_bytes_vec(raw, t.shape, burn_human_inference::dtype(&t.dtype)?);
        burn_human_inference::weights::ensure_finite(&data)?;
        let id = u64::from_str_radix(&sha256(t.name.as_bytes())[..16], 16)?;
        let snapshot = TensorSnapshot::from_data(
            data,
            t.name.split('.').map(str::to_owned).collect(),
            vec![],
            ParamId::from(id),
        );
        let bytes = BurnpackWriter::new(vec![snapshot])
            .to_bytes()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            bytes.len() <= MAX_OBJECT_BYTES,
            "Burnpack exceeds object budget"
        );
        let mut parts = vec![];
        for chunk in bytes.chunks(PART_TARGET_BYTES) {
            let part = Part {
                sha256: sha256(chunk),
                size: chunk.len(),
            };
            let path = output.join(part.path());
            if !path.exists() {
                std::fs::write(path, chunk)?;
            }
            parts.push(part);
        }
        manifest.objects.push(Object {
            stage: t.name,
            sha256: sha256(&bytes),
            size: bytes.len(),
            parts,
            tensors: vec![spec],
        });
    }
    for asset in export.assets {
        let bytes = std::fs::read(input.join(asset.file))?;
        let bound = Asset {
            path: asset.path,
            size: bytes.len(),
            sha256: asset.sha256,
        };
        bound.verify(&bytes)?;
        manifest.assets.push(bound.clone());
        // Validate paths before writing anything outside the part directory.
        manifest.seal()?;
        let dest = output.join(&bound.path);
        std::fs::create_dir_all(dest.parent().unwrap())?;
        std::fs::write(dest, bytes)?;
    }
    manifest.seal()?;
    std::fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "{} objects, manifest {}",
        manifest.objects.len(),
        manifest.content_sha256
    );
    Ok(())
}

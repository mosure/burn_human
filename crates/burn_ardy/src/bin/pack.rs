//! Convert pinned safetensors into immutable bounded, part-only Burnpack artifacts.
use anyhow::{Result, ensure};
use burn::{
    module::ParamId,
    tensor::{DType, TensorData},
};
use burn_ardy::config::{ArdyConfig, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, expected_tensors};
use burn_human_motion::artifacts::{
    MAX_OBJECT_BYTES, Manifest, Object, PART_TARGET_BYTES, Part, TensorSpec, sha256,
};
use burn_store::{BurnpackWriter, TensorSnapshot};
use std::{collections::BTreeMap, path::PathBuf};

struct SourceTensor {
    name: String,
    shape: Vec<usize>,
    source_index: usize,
    source_name: String,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: ardy-pack CHECKPOINT_DIRECTORY CONFIG_JSON SOURCE_GIT_SHA NEW_OUTPUT_DIRECTORY"
    );
    let checkpoint = PathBuf::from(&args[0]);
    let config_bytes = std::fs::read(&args[1])?;
    ensure!(
        sha256(&config_bytes) == "80f786f0961a785a45a2e42a4bd4cd811cc4308e8e96ba8611e699a7ce698a54",
        "config does not match the pinned reference exporter output"
    );
    let config: ArdyConfig = serde_json::from_slice(&config_bytes)?;
    config.validate()?;
    let source_revision = args[2].to_string_lossy().to_string();
    ensure!(
        source_revision == SOURCE_REVISION,
        "unsupported source revision"
    );
    let output = PathBuf::from(&args[3]);
    ensure!(
        !output.exists(),
        "output must be a new immutable bundle directory"
    );
    let expected = expected_tensors();
    let mut groups: BTreeMap<String, Vec<SourceTensor>> = BTreeMap::new();
    let mut mappings = Vec::new();
    // mmap avoids materializing either source checkpoint as a second full host copy.
    for filename in ["denoiser.safetensors", "tokenizer.safetensors"] {
        let file = std::fs::File::open(checkpoint.join(filename))?;
        // SAFETY: read-only checkpoint files must not be modified during conversion.
        let mapped = unsafe { memmap2::Mmap::map(&file)? };
        let digest = if filename == "denoiser.safetensors" {
            "1019d0bf269cf8d1b3e3e9b4a384a58c112672959b071279ddb65814d77660cd"
        } else {
            "58a887e299a3a6779b5b4ff361b452c4349d45b513b5957f2a961d393623abcd"
        };
        ensure!(
            sha256(&mapped) == digest,
            "checkpoint differs from pinned Hugging Face LFS digest: {filename}"
        );
        let tensors = safetensors::SafeTensors::deserialize(&mapped)?;
        for (source, view) in tensors.tensors() {
            let name = source
                .strip_prefix("denoiser.backbone.")
                .or_else(|| source.strip_prefix("denoiser."))
                .or_else(|| source.strip_prefix("pose_net."))
                .ok_or_else(|| anyhow::anyhow!("unknown source prefix {source}"))?
                .to_owned();
            ensure!(
                expected.get(&name).is_some_and(|s| s == view.shape())
                    && view.dtype() == safetensors::Dtype::F32,
                "unexpected source tensor {name}"
            );
            let stage = if let Some((prefix, tail)) = name.split_once(".seqTransEncoder.layers.") {
                format!("{prefix}.layer.{}", tail.split('.').next().unwrap())
            } else {
                format!("{}.projections", name.split('.').next().unwrap())
            };
            // Keep offsets/names only. Materialize one logical stage at a time below.
            groups.entry(stage).or_default().push(SourceTensor {
                name,
                shape: view.shape().to_vec(),
                source_index: mappings.len(),
                source_name: source,
            });
        }
        mappings.push(mapped);
    }
    let actual: BTreeMap<_, _> = groups
        .values()
        .flatten()
        .map(|t| (t.name.clone(), t.shape.clone()))
        .collect();
    ensure!(
        actual == expected && groups.values().map(Vec::len).sum::<usize>() == expected.len(),
        "incomplete or duplicate checkpoint inventory"
    );
    std::fs::create_dir_all(output.join("parts"))?;
    let mut objects = Vec::new();
    for (stage, mut entries) in groups {
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let mut specs = Vec::new();
        let mut snapshots = Vec::new();
        for SourceTensor {
            name,
            shape,
            source_index,
            source_name,
        } in entries
        {
            let tensors = safetensors::SafeTensors::deserialize(&mappings[source_index])?;
            let view = tensors.tensor(&source_name)?;
            let bytes = view.data().to_vec();
            specs.push(TensorSpec {
                name: name.clone(),
                shape: shape.clone(),
                dtype: "f32".into(),
                sha256: sha256(&bytes),
            });
            let data = TensorData::from_bytes_vec(bytes, shape, DType::F32);
            // Deterministic IDs make the logical objects reproducible across conversions.
            let id = u64::from_str_radix(&sha256(name.as_bytes())[..16], 16)?;
            snapshots.push(TensorSnapshot::from_data(
                data,
                name.split('.').map(str::to_string).collect(),
                vec![],
                ParamId::from(id),
            ));
        }
        let bytes = BurnpackWriter::new(snapshots)
            .to_bytes()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            bytes.len() <= MAX_OBJECT_BYTES,
            "stage exceeds bounded object size: {stage}"
        );
        let mut parts = Vec::new();
        for data in bytes.chunks(PART_TARGET_BYTES) {
            let part = Part {
                sha256: sha256(data),
                size: data.len(),
            };
            let dest = output.join(part.path());
            if !dest.exists() {
                std::fs::write(dest, data)?;
            }
            parts.push(part);
        }
        objects.push(Object {
            stage,
            sha256: sha256(&bytes),
            size: bytes.len(),
            parts,
            tensors: specs,
        });
    }
    let mut manifest = Manifest {
        schema_version: 1,
        model: MODEL_ID.into(),
        model_revision: MODEL_REVISION.into(),
        source_revision,
        converter: "burn-ardy-pack-v1-f32".into(),
        license: "NVIDIA-Open-Model-License".into(),
        config: serde_json::to_value(config)?,
        objects,
        assets: Vec::new(),
        content_sha256: String::new(),
    };
    manifest.seal()?;
    std::fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    std::fs::copy(checkpoint.join("LICENSE"), output.join("LICENSE"))?;
    println!(
        "{} objects, {} tensors; manifest {}",
        manifest.objects.len(),
        expected.len(),
        manifest.content_sha256
    );
    Ok(())
}

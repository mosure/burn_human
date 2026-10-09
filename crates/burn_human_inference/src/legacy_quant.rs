//! Compatibility for the authenticated Q4F/F32/block-32 CDN packs made with Burn 0.21.
//! Only metadata is decoded here. Burnpack still validates offsets and owns the
//! shared payload views; no weight bytes are copied or rewritten.
use anyhow::{Result, ensure};
use burn::tensor::DType;
use serde::Deserialize;
use std::collections::BTreeMap;
#[derive(Deserialize)]
struct Metadata {
    tensors: BTreeMap<String, Descriptor>,
}
#[derive(Deserialize)]
struct Descriptor {
    dtype: serde_json::Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyScheme {
    value: String,
    param: String,
    store: serde_json::Value,
    level: LegacyLevel,
    mode: String,
}
#[derive(Deserialize)]
enum LegacyLevel {
    Block(Block),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Block {
    storage: [u8; 5],
    len: usize,
}
/// Reject old layouts other than the exact format used by the pinned catalog.
pub(crate) fn normalize(bytes: &[u8]) -> Result<BTreeMap<String, DType>> {
    ensure!(bytes.len() >= 10, "truncated Burnpack header");
    let len = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
    ensure!(len <= 1024 * 1024, "model Burnpack metadata exceeds budget");
    let metadata = bytes
        .get(10..10 + len)
        .ok_or_else(|| anyhow::anyhow!("truncated Burnpack metadata"))?;
    let metadata: Metadata = ciborium::from_reader(metadata)?;
    let mut formats = BTreeMap::new();
    for (name, descriptor) in metadata.tensors {
        let Some(scheme) = descriptor.dtype.get("QFloat") else {
            continue;
        };
        if scheme.get("level").is_none() {
            continue;
        }
        let scheme: LegacyScheme = serde_json::from_value(scheme.clone())?;
        let LegacyLevel::Block(block) = scheme.level;
        ensure!(
            scheme.value == "Q4F"
                && scheme.param == "F32"
                && scheme.mode == "Symmetric"
                && scheme.store == serde_json::json!({"PackedU32":0})
                && block.storage == [32, 1, 1, 1, 1]
                && block.len == 1,
            "unsupported legacy quantization: {name}"
        );
        formats.insert(name, crate::dtype("q4f32")?);
    }
    Ok(formats)
}

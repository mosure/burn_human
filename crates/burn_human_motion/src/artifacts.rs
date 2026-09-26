//! Immutable part-only CDN layout, matching burn_image's 20 MiB target / 25 MB
//! ceiling. A loader authenticates one bounded logical Burnpack at a time; it
//! never assembles the complete model in host or Wasm memory.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PART_TARGET_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_PART_BYTES: usize = 25_000_000;
pub const MAX_OBJECT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sha256: String,
}

/// Non-tensor payload sealed by the manifest, such as a tokenizer or rig table.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: String,
    pub size: usize,
    pub sha256: String,
}

impl Asset {
    pub fn verify(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() == self.size && sha256(bytes) == self.sha256,
            "asset size/digest mismatch: {}",
            self.path
        );
        Ok(())
    }
}

impl TensorSpec {
    /// Exact serialized tensor bytes, excluding the Burnpack header.
    pub fn byte_len(&self) -> Option<usize> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))?;
        match self.dtype.as_str() {
            "f32" | "i32" | "u32" => n.checked_mul(4),
            "f16" => n.checked_mul(2),
            "u8" => Some(n),
            "q4f32" if self.shape.len() == 2 && self.shape[1].is_multiple_of(32) => {
                (n / 2).checked_add((n / 32).checked_mul(4)?)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub sha256: String,
    pub size: usize,
}

impl Part {
    pub fn path(&self) -> String {
        format!("parts/{}.bin", self.sha256)
    }
    pub fn verify(&self, data: &[u8]) -> Result<()> {
        ensure!(
            data.len() == self.size && sha256(data) == self.sha256,
            "artifact part size/digest mismatch: {}",
            self.sha256
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Object {
    pub stage: String,
    pub sha256: String,
    pub size: usize,
    pub parts: Vec<Part>,
    pub tensors: Vec<TensorSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub model: String,
    pub model_revision: String,
    pub source_revision: String,
    pub converter: String,
    pub license: String,
    /// Small model configuration and normalization data, covered by the seal.
    pub config: serde_json::Value,
    pub objects: Vec<Object>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<Asset>,
    pub content_sha256: String,
}

impl Manifest {
    fn computed_digest(&self) -> Result<String> {
        let mut unsealed = self.clone();
        unsealed.content_sha256.clear();
        Ok(sha256(&serde_json::to_vec(&unsealed)?))
    }

    pub fn seal(&mut self) -> Result<()> {
        self.content_sha256 = self.computed_digest()?;
        self.validate()
    }

    pub fn from_bytes(bytes: &[u8], expected_digest: Option<&str>) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_MANIFEST_BYTES,
            "manifest exceeds byte limit"
        );
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        if let Some(expected) = expected_digest {
            ensure!(
                manifest.content_sha256 == expected,
                "manifest identity mismatch"
            );
        }
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.schema_version, 1 | 2),
            "unsupported artifact schema"
        );
        ensure!(
            self.model_revision.len() == 40
                && self.model_revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "model revision must be an immutable git SHA"
        );
        ensure!(
            self.source_revision.len() == 40
                && self.source_revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "source revision must be an immutable git SHA"
        );
        ensure!(
            !self.model.is_empty() && !self.converter.is_empty() && !self.license.is_empty(),
            "missing artifact provenance"
        );
        ensure!(
            valid_hash(&self.content_sha256) && self.computed_digest()? == self.content_sha256,
            "manifest seal mismatch"
        );
        ensure!(
            !self.objects.is_empty() && self.objects.len() <= 4096,
            "invalid object count"
        );
        let mut stages = BTreeSet::new();
        let mut tensors = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for asset in &self.assets {
            ensure!(
                self.schema_version == 2
                    && asset.path.starts_with("metadata/")
                    && asset.path.split('/').all(|v| !v.is_empty()
                        && v != "."
                        && v != ".."
                        && v.bytes()
                            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)))
                    && paths.insert(&asset.path)
                    && asset.size > 0
                    && asset.size <= MAX_PART_BYTES
                    && valid_hash(&asset.sha256),
                "invalid or duplicate metadata asset"
            );
        }
        for object in &self.objects {
            ensure!(
                !object.stage.is_empty() && stages.insert(&object.stage),
                "duplicate/empty stage"
            );
            ensure!(
                object.size > 0 && object.size <= MAX_OBJECT_BYTES && valid_hash(&object.sha256),
                "invalid logical object"
            );
            ensure!(!object.parts.is_empty(), "object has no transport parts");
            let mut size = 0usize;
            for part in &object.parts {
                ensure!(
                    part.size > 0 && part.size <= MAX_PART_BYTES && valid_hash(&part.sha256),
                    "invalid transport part"
                );
                size = size
                    .checked_add(part.size)
                    .ok_or_else(|| anyhow::anyhow!("part size overflow"))?;
            }
            ensure!(size == object.size, "transport/object size mismatch");
            ensure!(!object.tensors.is_empty(), "object has no tensor inventory");
            for tensor in &object.tensors {
                ensure!(
                    !tensor.name.is_empty() && tensors.insert(&tensor.name),
                    "duplicate/empty tensor"
                );
                ensure!(
                    (self.schema_version == 2 || tensor.dtype == "f32")
                        && valid_hash(&tensor.sha256),
                    "unsupported tensor dtype or digest"
                );
                ensure!(
                    !tensor.shape.is_empty()
                        && tensor.shape.len() <= 6
                        && !tensor.shape.contains(&0),
                    "invalid tensor rank"
                );
                ensure!(
                    tensor
                        .byte_len()
                        .is_some_and(|n| n > 0 && n <= MAX_OBJECT_BYTES),
                    "invalid tensor shape"
                );
            }
        }
        Ok(())
    }
}

/// Platform transport owns persistent cache policy. Implementations MUST enforce
/// the declared size while streaming, before allocating an unbounded response.
#[allow(async_fn_in_trait)]
pub trait PartReader {
    async fn read_part(&mut self, part: &Part) -> Result<Vec<u8>>;
}

pub async fn read_object(reader: &mut impl PartReader, object: &Object) -> Result<Vec<u8>> {
    ensure!(
        object.size <= MAX_OBJECT_BYTES,
        "logical object exceeds memory budget"
    );
    let mut bytes = Vec::with_capacity(object.size);
    for part in &object.parts {
        ensure!(
            part.size <= MAX_PART_BYTES,
            "transport part exceeds memory budget"
        );
        let data = reader.read_part(part).await?;
        part.verify(&data)?;
        ensure!(bytes.len() + data.len() <= object.size, "object overflow");
        bytes.extend_from_slice(&data);
    }
    ensure!(
        bytes.len() == object.size && sha256(&bytes) == object.sha256,
        "logical object size/digest mismatch"
    );
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct DirectoryReader(pub std::path::PathBuf);

#[cfg(not(target_arch = "wasm32"))]
impl PartReader for DirectoryReader {
    async fn read_part(&mut self, part: &Part) -> Result<Vec<u8>> {
        use std::io::Read;
        ensure!(
            valid_hash(&part.sha256) && part.size <= MAX_PART_BYTES,
            "invalid part"
        );
        let file = std::fs::File::open(self.0.join(part.path()))?;
        ensure!(
            file.metadata()?.len() == part.size as u64,
            "part size mismatch"
        );
        let mut bytes = Vec::with_capacity(part.size);
        file.take(part.size as u64 + 1).read_to_end(&mut bytes)?;
        part.verify(&bytes)?;
        Ok(bytes)
    }
}

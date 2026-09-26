//! Pinned model release and cached native/WebGPU loading.
use crate::TextEncoder;
use anyhow::Result;
use burn::prelude::Backend;
use burn_human_inference::pretrained::{ArtifactLocation, PretrainedArtifact};

pub const CDN_ROOT: &str = "https://aberration.technology/model";
/// Immutable model bundle; overriding the root preserves its SHA-256 trust anchor.
pub const DEFAULT: PretrainedArtifact = PretrainedArtifact {
    bundle: "llama/ardy-llm2vec-8b/v1",
    sha256: "09280ebcf02184708cff1a9f982b901166b433e504ff343f02c037cc52d8ee16",
};

impl<B: Backend> TextEncoder<B> {
    /// Load the pinned release from aberration CDN, using disk/CacheStorage caching.
    pub async fn load_pretrained(
        device: &B::Device,
        progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        Self::load_pretrained_from(CDN_ROOT, device, progress).await
    }

    /// Load the same pinned release from a local directory or alternate HTTP root.
    pub async fn load_pretrained_from(
        root: &str,
        device: &B::Device,
        progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        Self::load_artifact(&DEFAULT.at_root(root), device, progress).await
    }

    /// Load an explicitly pinned custom artifact. Checks metadata before downloading weights.
    pub async fn load_artifact(
        artifact: &ArtifactLocation,
        device: &B::Device,
        progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        let (manifest, source) = artifact.open().await?;
        Self::load(manifest, source, device, progress).await
    }
}

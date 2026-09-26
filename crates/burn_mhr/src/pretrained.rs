//! Pinned model release and cached native/WebGPU loading.
use crate::Mhr;
use anyhow::Result;
use burn::prelude::Backend;
use burn_human_inference::pretrained::{ArtifactLocation, PretrainedArtifact};

pub const CDN_ROOT: &str = "https://aberration.technology/model";
/// Immutable model bundle; overriding the root preserves its SHA-256 trust anchor.
pub const DEFAULT: PretrainedArtifact = PretrainedArtifact {
    bundle: "soma-x/v1/mhr",
    sha256: "c7d6e4fc479b8ca68ab1ba6736923bcd93d9f61252c3009ace1ebef6bb5eff65",
};

impl<B: Backend> Mhr<B> {
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
        let (manifest, mut source) = artifact.open().await?;
        Self::load(&manifest, &mut source, device, progress).await
    }
}

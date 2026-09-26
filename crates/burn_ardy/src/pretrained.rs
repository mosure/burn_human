//! Pinned model release and cached native/WebGPU loading.
use crate::Ardy;
use anyhow::Result;
use burn::prelude::Backend;
use burn_human_inference::pretrained::{ArtifactLocation, PretrainedArtifact};

pub const CDN_ROOT: &str = "https://aberration.technology/model";
/// Immutable model bundle; overriding the root preserves its SHA-256 trust anchor.
pub const DEFAULT: PretrainedArtifact = PretrainedArtifact {
    bundle: "ardy/core-rp-20fps-h40/v1",
    sha256: "6e03294854771ca3ed6641251c77ab75bd2542d778c02886ea795d8d77655cd8",
};

impl<B: Backend> Ardy<B> {
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

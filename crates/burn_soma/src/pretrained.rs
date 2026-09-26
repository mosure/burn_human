//! Pinned model release and cached native/WebGPU loading.
use crate::Soma;
use anyhow::Result;
use burn::prelude::Backend;
use burn_human_inference::pretrained::{ArtifactLocation, PretrainedArtifact};

pub const CDN_ROOT: &str = "https://aberration.technology/model";
/// Immutable model bundle; overriding the root preserves its SHA-256 trust anchor.
pub const DEFAULT: PretrainedArtifact = PretrainedArtifact {
    bundle: "soma-x/v1/body",
    sha256: "96aefe1f999063f2b6ce434efe785f7c030deab24c35fb36d2cf4e1f05d643f7",
};

impl<B: Backend> Soma<B> {
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

/// MHR-to-SOMA topology transfer, shared with GEM-X.
pub const MHR_TRANSFER: PretrainedArtifact = PretrainedArtifact {
    bundle: "soma-x/v1/transfer",
    sha256: "0eaabd1e7fd976c37c9f99e609c2d713b814f56497c2c903b854a7d1d3abb6f8",
};

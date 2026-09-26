//! Model-neutral immutable release references. Model crates own their catalogs.
use crate::transport::ModelSource;
use anyhow::{Result, ensure};
use burn_human_motion::artifacts::Manifest;
use serde::{Deserialize, Serialize};

/// A model-owned, compiled-in trust anchor. The root may be a CDN or a local mirror.
#[derive(Clone, Copy, Debug)]
pub struct PretrainedArtifact {
    pub bundle: &'static str,
    pub sha256: &'static str,
}

impl PretrainedArtifact {
    pub fn at_root(self, root: &str) -> ArtifactLocation {
        self.at_base(format!("{}/{}", root.trim_end_matches('/'), self.bundle))
    }

    /// Override just the bundle directory; retain the compiled-in manifest identity.
    pub fn at_base(self, base: impl Into<String>) -> ArtifactLocation {
        ArtifactLocation {
            base: base.into(),
            sha256: self.sha256.into(),
        }
    }
}

/// An explicitly pinned manifest location, portable between native and the browser.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactLocation {
    pub base: String,
    pub sha256: String,
}

impl ArtifactLocation {
    /// Authenticate the manifest before allowing model weights to be fetched.
    pub async fn open(&self) -> Result<(Manifest, ModelSource)> {
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "expected a 64-character lowercase manifest SHA-256"
        );
        let source = ModelSource::cached(self.base.clone());
        let manifest = source.manifest(Some(&self.sha256)).await?;
        Ok((manifest, source))
    }
}

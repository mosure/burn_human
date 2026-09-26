//! GEM-X owns its perception catalog and composes the MHR/SOMA releases.
use crate::pipeline::{Pipeline, PipelineArtifacts};
use anyhow::Result;
use burn::prelude::Backend;
use burn_human_inference::pretrained::PretrainedArtifact;

pub const CDN_ROOT: &str = "https://aberration.technology/model";
pub const VITPOSE: PretrainedArtifact = PretrainedArtifact {
    bundle: "gemx/v1/vitpose",
    sha256: "d868f3d06eb7ff2d7acf014c430b1b3a8b6bee269bc407e78a97127d77349e1d",
};
pub const SAM_VISION: PretrainedArtifact = PretrainedArtifact {
    bundle: "gemx/v1/sam-vision",
    sha256: "66fc185473d361d1934fbec7d427a3be4602bc410bec6427b4a7e3d48c18c90b",
};
pub const SAM_DECODER: PretrainedArtifact = PretrainedArtifact {
    bundle: "gemx/v1/sam-decoder",
    sha256: "35c5b33c1041f9950af97def56335e7d3929eec6a6ce6180989b8423fb3f4b3e",
};
pub const DENOISER: PretrainedArtifact = PretrainedArtifact {
    bundle: "gemx/v1/denoiser",
    sha256: "8849a57736ee9c5119a1598a5a931b8d0f6892f93eba340765553ac78b89b2f3",
};
pub const SUITE_URL: &str = "https://aberration.technology/model/gemx/v1/suite.json";
pub const SUITE_SHA256: &str = "c21010ed4da582e66fb975ecb2a775f4f0c3a28a7816e89f0275d468c0205952";

impl PipelineArtifacts {
    pub fn pretrained() -> Self {
        Self::pretrained_from(CDN_ROOT)
    }

    /// Resolve the complete dependency graph under a CDN or local model root.
    pub fn pretrained_from(root: &str) -> Self {
        Self {
            vitpose: VITPOSE.at_root(root),
            sam_vision: SAM_VISION.at_root(root),
            sam_decoder: SAM_DECODER.at_root(root),
            denoiser: DENOISER.at_root(root),
            mhr: burn_mhr::pretrained::DEFAULT.at_root(root),
            soma: burn_soma::pretrained::DEFAULT.at_root(root),
            transfer: burn_soma::pretrained::MHR_TRANSFER.at_root(root),
        }
    }
}

impl<B: Backend> Pipeline<B> {
    /// Load the compiled-in release, including every dependency, with verified caching.
    pub async fn load_pretrained(
        device: &B::Device,
        progress: impl FnMut(&str, usize, usize),
    ) -> Result<Self> {
        Self::load_pretrained_from(CDN_ROOT, device, progress).await
    }
    pub async fn load_pretrained_from(
        root: &str,
        device: &B::Device,
        progress: impl FnMut(&str, usize, usize),
    ) -> Result<Self> {
        Self::load(&PipelineArtifacts::pretrained_from(root), device, progress).await
    }
}

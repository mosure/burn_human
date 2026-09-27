//! Image -> keypoints + body token -> GEM-X -> fitted SOMA mesh.
use crate::{
    camera::{Camera, Crop},
    denoiser::{Conditions, GemDenoiser},
    image_input,
    sam::SamBody,
    vision::Vision,
};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use burn_mhr::Mhr;
use burn_soma::{
    BindConvention, Soma, SomaPose,
    mhr::{MhrIdentity, MhrSomaTransfer},
};
use glam::Quat;
use serde::{Deserialize, Serialize};

pub use burn_human_inference::pretrained::ArtifactLocation as Artifact;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineArtifacts {
    pub vitpose: Artifact,
    pub sam_vision: Artifact,
    pub sam_decoder: Artifact,
    pub denoiser: Artifact,
    pub mhr: Artifact,
    pub soma: Artifact,
    pub transfer: Artifact,
}
impl PipelineArtifacts {
    pub async fn from_location(location: &str) -> Result<Self> {
        let bytes = burn_human_inference::transport::read_bounded(location, 64 * 1024).await?;
        if location == crate::pretrained::SUITE_URL {
            ensure!(
                burn_human_motion::artifacts::sha256(&bytes) == crate::pretrained::SUITE_SHA256,
                "GEM-X suite identity mismatch"
            );
        }
        let mut suite: Self = serde_json::from_slice(&bytes)?;
        for a in [
            &mut suite.vitpose,
            &mut suite.sam_vision,
            &mut suite.sam_decoder,
            &mut suite.denoiser,
            &mut suite.mhr,
            &mut suite.soma,
            &mut suite.transfer,
        ] {
            a.base = resolve_component(location, &a.base)?;
        }
        Ok(suite)
    }
}

fn resolve_component(suite: &str, base: &str) -> Result<String> {
    if suite.starts_with("https://") || suite.starts_with("http://") {
        // Resolve dot segments before sending the request. Browsers do this
        // implicitly; native HTTP and object-storage origins need the exact key.
        return Ok(url::Url::parse(suite)?.join(base)?.to_string());
    }
    if base.contains(":/") || std::path::Path::new(base).is_absolute() {
        return Ok(base.into());
    }
    let parent = std::path::Path::new(suite)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    Ok(parent.join(base).to_string_lossy().into_owned())
}

#[cfg(test)]
mod location_tests {
    use super::resolve_component;
    #[test]
    fn grouped_cdn_suite_resolves_shared_dependencies_without_dot_segments() {
        let suite = "https://aberration.technology/model/gemx/v1/suite.json";
        assert_eq!(
            resolve_component(suite, "../../soma-x/v1/mhr").unwrap(),
            "https://aberration.technology/model/soma-x/v1/mhr"
        );
        assert_eq!(
            resolve_component(suite, "sam-decoder").unwrap(),
            "https://aberration.technology/model/gemx/v1/sam-decoder"
        );
        assert_eq!(
            resolve_component(&format!("{suite}?revision=1"), "/model/soma-x/v1/body").unwrap(),
            "https://aberration.technology/model/soma-x/v1/body"
        );
        assert_eq!(
            resolve_component(suite, "https://mirror.example/body").unwrap(),
            "https://mirror.example/body"
        );
    }
    #[test]
    fn local_suite_keeps_relative_filesystem_semantics() {
        let resolved = resolve_component("models/gemx/suite.json", "../soma").unwrap();
        assert_eq!(
            std::path::Path::new(&resolved),
            std::path::Path::new("models/gemx/../soma")
        );
    }
}
pub struct Pipeline<B: Backend> {
    pub vitpose: Vision<B>,
    pub sam_vision: Vision<B>,
    pub sam_decoder: SamBody<B>,
    pub denoiser: GemDenoiser<B>,
    pub mhr: Mhr<B>,
    pub soma: Soma<B>,
    pub transfer: MhrSomaTransfer<B>,
    device: B::Device,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoseEstimate {
    pub identity: MhrIdentity,
    pub pose: SomaPose,
    pub crop: Crop,
    pub camera: Camera,
    pub keypoints_2d: Vec<[f32; 3]>,
    /// Metres in camera coordinates (+X right, +Y down, +Z depth).
    pub vertices: Vec<[f32; 3]>,
    pub joints: Vec<[f32; 3]>,
    pub timings: std::collections::BTreeMap<String, f64>,
}

/// Image inference with the fitted mesh retained on its original device.
/// Renderers can consume `vertices` directly; exporters explicitly call
/// `into_host`. The prepared identity can be reused for subsequent pose edits.
pub struct ResidentPoseEstimate<B: Backend> {
    pub identity: MhrIdentity,
    pub pose: SomaPose,
    pub crop: Crop,
    pub camera: Camera,
    pub keypoints_2d: Vec<[f32; 3]>,
    pub vertices: Tensor<B, 3>,
    pub joints: Vec<[f32; 3]>,
    pub prepared: burn_soma::PreparedIdentity<B>,
    /// Enqueue time until a consumer synchronizes `vertices`.
    pub timings: std::collections::BTreeMap<String, f64>,
}
impl<B: Backend> ResidentPoseEstimate<B> {
    pub async fn into_host(mut self) -> Result<PoseEstimate> {
        let start = web_time::Instant::now();
        let vertices = self
            .vertices
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("SOMA output: {e}"))?
            .as_chunks::<3>()
            .0
            .to_vec();
        *self.timings.entry("soma".into()).or_default() += start.elapsed().as_secs_f64();
        Ok(PoseEstimate {
            identity: self.identity,
            pose: self.pose,
            crop: self.crop,
            camera: self.camera,
            keypoints_2d: self.keypoints_2d,
            vertices,
            joints: self.joints,
            timings: self.timings,
        })
    }
}
impl<B: Backend> Pipeline<B> {
    pub async fn load(
        artifacts: &PipelineArtifacts,
        device: &B::Device,
        mut progress: impl FnMut(&str, usize, usize),
    ) -> Result<Self> {
        macro_rules! load {
            ($field:ident,$ty:ty) => {{
                let a = &artifacts.$field;
                let (m, mut source) = a.open().await?;
                <$ty>::load(&m, &mut source, device, |i, n| {
                    progress(stringify!($field), i, n)
                })
                .await?
            }};
        }
        let vitpose = load!(vitpose, Vision<B>);
        let sam_vision = load!(sam_vision, Vision<B>);
        let sam_decoder = load!(sam_decoder, SamBody<B>);
        let denoiser = load!(denoiser, GemDenoiser<B>);
        let mhr = load!(mhr, Mhr<B>);
        let soma = load!(soma, Soma<B>);
        let a = &artifacts.transfer;
        let (m, mut source) = a.open().await?;
        let transfer = MhrSomaTransfer::load(&m, &mut source, device).await?;
        Ok(Self {
            vitpose,
            sam_vision,
            sam_decoder,
            denoiser,
            mhr,
            soma,
            transfer,
            device: device.clone(),
        })
    }
    pub async fn estimate(
        &self,
        image: &image::RgbImage,
        crop: Crop,
        camera: Camera,
        progress: impl FnMut(&str),
    ) -> Result<PoseEstimate> {
        self.estimate_resident(image, crop, camera, progress)
            .await?
            .into_host()
            .await
    }

    pub async fn estimate_resident(
        &self,
        image: &image::RgbImage,
        crop: Crop,
        camera: Camera,
        mut progress: impl FnMut(&str),
    ) -> Result<ResidentPoseEstimate<B>> {
        crop.validate()?;
        camera.validate()?;
        // Image decoding in the caller may already have consumed this task.
        burn_human_inference::cooperative::yield_to_browser().await;
        let mut timings = std::collections::BTreeMap::new();
        let start = web_time::Instant::now();
        progress("2D keypoints");
        let input = image_input::crop_rgb(image, crop, false)?;
        let input =
            Tensor::<B, 4>::from_data(TensorData::new(input, [1, 3, 256, 192]), &self.device);
        let flipped = input.clone().flip([3]);
        let heatmaps = self
            .vitpose
            .forward_async(Tensor::cat(vec![input, flipped], 0))
            .await?
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Heatmaps: {e}"))?;
        let observations = image_input::keypoints(&heatmaps, crop, true)?;
        timings.insert("keypoints".into(), start.elapsed().as_secs_f64());
        let start = web_time::Instant::now();
        // SAM's transform pads by 1.25, expands to its 3:4 prior aspect,
        // then expands to the square DINO input (fix_square is false).
        progress("Image body features");
        let sam_crop = Crop {
            size: crop.size * 1.25 / 0.75,
            ..crop
        };
        let input = image_input::crop_rgb(image, sam_crop, true)?;
        let input =
            Tensor::<B, 4>::from_data(TensorData::new(input, [1, 3, 512, 512]), &self.device);
        let embedding = self.sam_vision.forward_async(input).await?;
        let sam = self
            .sam_decoder
            .forward_token(embedding, &self.mhr, sam_crop, camera)
            .await?;
        timings.insert("body_features".into(), start.elapsed().as_secs_f64());
        let start = web_time::Instant::now();
        progress("GEM-X pose prediction");
        let conditions = Conditions {
            batch: 1,
            frames: 1,
            observations: observations.clone(),
            crops: vec![crop],
            cameras: vec![camera],
            angular: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        };
        let prediction = self.denoiser.predict(&conditions, sam)?;
        let values = Tensor::cat(
            vec![
                prediction.features.reshape([585]),
                prediction.camera.reshape([3]),
            ],
            0,
        )
        .into_data_async()
        .await?
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("GEM prediction: {e}"))?;
        let (features, pred_camera) = values.split_at(585);
        let (identity, pose) = self.decode_pose(features, pred_camera, crop, camera)?;
        timings.insert("gem".into(), start.elapsed().as_secs_f64());
        let start = web_time::Instant::now();
        progress("SOMA identity and skinning");
        let prepared = self
            .transfer
            .prepare_identity_with_bind(&self.mhr, &self.soma, &identity, BindConvention::Fitted)
            .await?;
        let output = self
            .soma
            .pose_batch(&prepared, std::slice::from_ref(&pose))?;
        let joints = output.transforms[0]
            .iter()
            .skip(1)
            .map(|m| m.w_axis.truncate().to_array())
            .collect();
        timings.insert("soma".into(), start.elapsed().as_secs_f64());
        Ok(ResidentPoseEstimate {
            identity,
            pose,
            crop,
            camera,
            keypoints_2d: observations,
            vertices: output.vertices,
            joints,
            timings,
            prepared,
        })
    }
    pub fn decode_pose(
        &self,
        features: &[f32],
        pred_camera: &[f32],
        crop: Crop,
        camera: Camera,
    ) -> Result<(MhrIdentity, SomaPose)> {
        ensure!(
            features.len() == 585
                && pred_camera.len() == 3
                && features.iter().chain(pred_camera).all(|x| x.is_finite()),
            "Invalid GEM output"
        );
        let features: Vec<f32> = features
            .iter()
            .zip(&self.denoiser.decode.mean)
            .zip(&self.denoiser.decode.std)
            .map(|((x, mean), std)| x * std + mean)
            .collect();
        let axis = |p: &[f32]| {
            let rotation = crate::sam_head::rotation6(p).transpose();
            let q = Quat::from_mat3(&rotation).normalize();
            let q = if q.w < 0.0 { -q } else { q };
            q.to_scaled_axis().to_array()
        };
        let mut rotations = vec![axis(&features[570..576])];
        rotations.extend(features[..456].as_chunks::<6>().0.iter().map(|p| axis(p)));
        let identity = MhrIdentity {
            coefficients: features[456..501].to_vec(),
            scales: features[502..570].to_vec(),
            global_scale: features[501].clamp(0.7, 1.0),
            ..Default::default()
        };
        let sb = pred_camera[0] * crop.size + 1e-9;
        ensure!(sb.is_finite() && sb > 0.0, "Invalid GEM camera scale");
        let translation = [
            pred_camera[1] + 2.0 * (crop.center[0] - camera.center[0]) / sb,
            pred_camera[2] + 2.0 * (crop.center[1] - camera.center[1]) / sb,
            2.0 * camera.focal[0] / sb,
        ];
        // Match the demo's temporal SOMA wrapper, which disables correctives.
        Ok((
            identity,
            SomaPose {
                rotations,
                translation,
                apply_correctives: false,
                absolute_pose: false,
            },
        ))
    }
}

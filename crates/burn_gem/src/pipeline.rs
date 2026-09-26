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
use burn_human_inference::transport::ModelSource;
use burn_mhr::Mhr;
use burn_soma::{
    BindConvention, Soma, SomaPose,
    mhr::{MhrIdentity, MhrSomaTransfer},
};
use glam::Quat;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub base: String,
    pub sha256: String,
}
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
        let mut suite: Self = serde_json::from_slice(&bytes)?;
        let parent = location.rsplit_once('/').map_or(".", |p| p.0).to_string();
        #[cfg(not(target_arch = "wasm32"))]
        let parent = if location.starts_with("http://") || location.starts_with("https://") {
            parent
        } else {
            std::path::Path::new(location)
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_string_lossy()
                .into_owned()
        };
        for a in [
            &mut suite.vitpose,
            &mut suite.sam_vision,
            &mut suite.sam_decoder,
            &mut suite.denoiser,
            &mut suite.mhr,
            &mut suite.soma,
            &mut suite.transfer,
        ] {
            let absolute = a.base.starts_with('/') || a.base.contains(":/");
            #[cfg(not(target_arch = "wasm32"))]
            let absolute = absolute || std::path::Path::new(&a.base).is_absolute();
            if !absolute {
                a.base = format!("{parent}/{}", a.base);
            }
        }
        Ok(suite)
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
impl<B: Backend> Pipeline<B> {
    pub async fn load(
        artifacts: &PipelineArtifacts,
        device: &B::Device,
        mut progress: impl FnMut(&str, usize, usize),
    ) -> Result<Self> {
        macro_rules! load {
            ($field:ident,$ty:ty) => {{
                let a = &artifacts.$field;
                let mut source = ModelSource::cached(a.base.clone());
                let m = source.manifest(Some(&a.sha256)).await?;
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
        let mut source = ModelSource::cached(a.base.clone());
        let m = source.manifest(Some(&a.sha256)).await?;
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
        mut progress: impl FnMut(&str),
    ) -> Result<PoseEstimate> {
        crop.validate()?;
        camera.validate()?;
        let mut timings = std::collections::BTreeMap::new();
        let start = web_time::Instant::now();
        progress("2D keypoints");
        let input = image_input::crop_rgb(image, crop, false)?;
        let input =
            Tensor::<B, 4>::from_data(TensorData::new(input, [1, 3, 256, 192]), &self.device);
        let flipped = input.clone().flip([3]);
        let heatmaps = self
            .vitpose
            .forward(Tensor::cat(vec![input, flipped], 0))?
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
        let embedding = self.sam_vision.forward(input)?;
        let sam = self
            .sam_decoder
            .forward(embedding, &self.mhr, sam_crop, camera)
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
        let prediction = self.denoiser.predict(&conditions, sam.token)?;
        let features = prediction
            .features
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("GEM prediction: {e}"))?;
        let pred_camera = prediction
            .camera
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("GEM camera: {e}"))?;
        let (identity, pose) = self.decode_pose(&features, &pred_camera, crop, camera)?;
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
        let vertices = output
            .vertices
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("SOMA output: {e}"))?
            .as_chunks::<3>()
            .0
            .to_vec();
        let joints = output.transforms[0]
            .iter()
            .skip(1)
            .map(|m| m.w_axis.truncate().to_array())
            .collect();
        timings.insert("soma".into(), start.elapsed().as_secs_f64());
        Ok(PoseEstimate {
            identity,
            pose,
            crop,
            camera,
            keypoints_2d: observations,
            vertices,
            joints,
            timings,
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

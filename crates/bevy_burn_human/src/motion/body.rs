//! SOMA/GEM model ownership and background evaluation, separate from ARDY playback.
use super::{MotionRuntime, MotionStatus, runtime::spawn_job};
use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_ardy::transport::ModelSource;
use burn_gem::{
    camera::{Camera, Crop},
    pipeline::{Pipeline, PipelineArtifacts, PoseEstimate},
};
use burn_soma::{
    BindConvention, IdentityParameters, PreparedIdentity, Soma, SomaPose, mhr::MhrIdentity,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BodyIdentity {
    Native(IdentityParameters),
    Image(MhrIdentity),
}
pub struct Surface {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub joints: Vec<[f32; 3]>,
    pub parents: Vec<usize>,
    pub camera_space: bool,
}
#[derive(Default)]
pub(super) struct BodyState {
    pub soma: Option<Soma<Wgpu>>,
    pub gem: Option<Pipeline<Wgpu>>,
    pub prepared: Option<(String, PreparedIdentity<Wgpu>)>,
    pub surface: Option<Surface>,
    pub estimate: Option<PoseEstimate>,
    pub joint_names: Vec<String>,
    pub scale_names: Vec<String>,
    pub revision: u64,
}

pub(super) fn load_soma(runtime: &MotionRuntime, base: String, digest: String) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(device) = state.device.clone() else {
        return;
    };
    state.status = MotionStatus::Working("Loading SOMA".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let mut source = ModelSource::cached(base);
        let m = source
            .manifest((!digest.trim().is_empty()).then_some(digest.trim()))
            .await?;
        let soma = Soma::load(&m, &mut source, &device, |i, n| {
            shared.lock().unwrap().status = MotionStatus::Working(format!("SOMA weights {i}/{n}"))
        })
        .await?;
        let id = soma.prepare_identity(IdentityParameters::default()).await?;
        let surface = evaluate(&soma, &id, &SomaPose::default(), false).await?;
        let mut state = shared.lock().unwrap();
        state.body.joint_names = soma.rig.public_names[1..].to_vec();
        state.body.scale_names = soma.rig.scale_names.clone();
        state.body.soma = Some(soma);
        state.body.prepared = None;
        state.body.surface = Some(surface);
        state.body.estimate = None;
        state.body.revision += 1;
        state.status = MotionStatus::Ready;
        Ok(())
    });
}
pub(super) fn load_gem(runtime: &MotionRuntime, location: String) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(device) = state.device.clone() else {
        return;
    };
    state.status = MotionStatus::Working("Loading image pose models".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let artifacts = PipelineArtifacts::from_location(&location).await?;
        let gem = Pipeline::load(&artifacts, &device, |s, i, n| {
            let stage = match s {
                "vitpose" => "2D pose model",
                "sam_vision" => "image features",
                "sam_decoder" => "body features",
                "denoiser" => "pose predictor",
                "mhr" => "body identity",
                "soma" => "SOMA rig",
                _ => "identity transfer",
            };
            shared.lock().unwrap().status =
                MotionStatus::Working(format!("Loading {stage} {i}/{n}"))
        })
        .await?;
        let mut state = shared.lock().unwrap();
        state.body.joint_names = gem.soma.rig.public_names[1..].to_vec();
        state.body.scale_names = gem.soma.rig.scale_names.clone();
        state.body.gem = Some(gem);
        state.status = MotionStatus::Ready;
        Ok(())
    });
}
pub(super) fn estimate(runtime: &MotionRuntime, crop: Crop, camera: Camera) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let Some(input) = state.image.clone() else {
        return;
    };
    let Some(gem) = state.body.gem.take() else {
        return;
    };
    state.status = MotionStatus::Working("Estimating image pose".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let result = async {
            let rgba = image::RgbaImage::from_raw(input.width, input.height, input.rgba)
                .ok_or_else(|| anyhow::anyhow!("Invalid input image"))?;
            let rgb = image::DynamicImage::ImageRgba8(rgba).into_rgb8();
            gem.estimate(&rgb, crop, camera, |s| {
                shared.lock().unwrap().status = MotionStatus::Working(s.into())
            })
            .await
        }
        .await;
        let mut state = shared.lock().unwrap();
        match result {
            Ok(pose) => {
                let mut joints = vec![[0.0; 3]];
                joints.extend(pose.joints.clone());
                state.body.surface = Some(Surface {
                    vertices: pose.vertices.clone(),
                    faces: gem.soma.faces.clone(),
                    joints,
                    parents: gem.soma.rig.public_parents.clone(),
                    camera_space: true,
                });
                state.body.estimate = Some(pose);
                state.body.prepared = None;
                state.body.revision += 1;
                state.status = MotionStatus::Ready;
            }
            Err(e) => state.status = MotionStatus::Failed(e.to_string()),
        }
        state.body.gem = Some(gem);
        Ok(())
    });
}
pub(super) fn apply(runtime: &MotionRuntime, identity: BodyIdentity, pose: SomaPose) {
    let mut state = runtime.0.lock().unwrap();
    if state.status.busy() {
        return;
    }
    let soma = state.body.soma.take();
    let gem = state.body.gem.take();
    let prepared = state.body.prepared.take();
    state.status = MotionStatus::Working("Evaluating SOMA controls".into());
    drop(state);
    let shared = runtime.0.clone();
    spawn_job(runtime.clone(), move || async move {
        let mut cached = prepared;
        let result = async {
            let model = soma
                .as_ref()
                .or_else(|| gem.as_ref().map(|g| &g.soma))
                .ok_or_else(|| anyhow::anyhow!("Load SOMA or GEM-X first"))?;
            let key = serde_json::to_string(&identity)?;
            ensure!(
                !matches!(identity, BodyIdentity::Image(_)) || !pose.apply_correctives,
                "Use canonical SOMA identity for pose correctives"
            );
            if cached.as_ref().is_none_or(|(k, _)| *k != key) {
                let id = match &identity {
                    BodyIdentity::Native(p) => model.prepare_identity(p.clone()).await?,
                    BodyIdentity::Image(p) => {
                        let gem = gem.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Load GEM-X for MHR identity controls")
                        })?;
                        ensure!(
                            !pose.apply_correctives,
                            "Use canonical SOMA identity for pose correctives"
                        );
                        gem.transfer
                            .prepare_identity_with_bind(&gem.mhr, model, p, BindConvention::Fitted)
                            .await?
                    }
                };
                cached = Some((key, id));
            }
            evaluate(
                model,
                &cached.as_ref().unwrap().1,
                &pose,
                matches!(identity, BodyIdentity::Image(_)),
            )
            .await
        }
        .await;
        let mut state = shared.lock().unwrap();
        state.body.soma = soma;
        state.body.gem = gem;
        state.body.prepared = cached;
        match result {
            Ok(surface) => {
                state.body.surface = Some(surface);
                state.status = MotionStatus::Ready;
            }
            Err(e) => state.status = MotionStatus::Failed(e.to_string()),
        }
        Ok(())
    });
}
async fn evaluate(
    soma: &Soma<Wgpu>,
    id: &PreparedIdentity<Wgpu>,
    pose: &SomaPose,
    camera_space: bool,
) -> Result<Surface> {
    let output = soma.pose_batch(id, std::slice::from_ref(pose))?;
    let vertices = output
        .vertices
        .into_data_async()
        .await?
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("SOMA vertices: {e}"))?
        .as_chunks::<3>()
        .0
        .to_vec();
    Ok(Surface {
        vertices,
        faces: soma.faces.clone(),
        joints: output.transforms[0]
            .iter()
            .map(|m| m.w_axis.truncate().to_array())
            .collect(),
        parents: soma.rig.public_parents.clone(),
        camera_space,
    })
}

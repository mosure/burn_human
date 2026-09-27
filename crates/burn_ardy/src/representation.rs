use crate::{Ardy, config::ArdyConfig};
use anyhow::{Result, ensure};
use burn::{prelude::Backend, tensor::Tensor};
use burn_human_motion::{MotionClip, MotionRequest, PoseFrame, rig::rotation_from_6d};
use glam::{Quat, Vec3};

impl<B: Backend> Ardy<B> {
    pub fn unnormalize_latent(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x * self.weights.constants.latent_scale.clone() + self.weights.constants.latent_mean.clone()
    }
    pub fn normalize_latent(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        (x - self.weights.constants.latent_mean.clone())
            / self.weights.constants.latent_scale.clone()
    }
    pub fn unnormalize_root(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x * self.weights.constants.root_scale.clone() + self.weights.constants.root_mean.clone()
    }
    pub fn normalize_root(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        (x - self.weights.constants.root_mean.clone()) / self.weights.constants.root_scale.clone()
    }
    pub fn local_root(&self, root: Tensor<B, 3>, valid_frames: usize) -> Tensor<B, 3> {
        let [batch, frames, _] = root.dims();
        let root = self.unnormalize_root(root);
        let slice = |t: std::ops::Range<usize>, d: std::ops::Range<usize>| {
            root.clone().slice([0..batch, t, d])
        };
        let angle = slice(0..frames, 4..5).atan2(slice(0..frames, 3..4));
        let delta = angle.clone().slice([0..batch, 1..frames, 0..1])
            - angle.slice([0..batch, 0..frames - 1, 0..1]);
        let angular = delta.clone().sin().atan2(delta.cos()) * 20.0;
        let dx = (slice(1..frames, 0..1) - slice(0..frames - 1, 0..1)) * 20.0;
        let dz = (slice(1..frames, 2..3) - slice(0..frames - 1, 2..3)) * 20.0;
        let velocity = Tensor::cat(vec![angular, dx, dz], 2);
        let last = velocity
            .clone()
            .slice([0..batch, valid_frames - 2..valid_frames - 1, 0..3]);
        let velocity = Tensor::cat(
            vec![velocity, Tensor::zeros([batch, 1, 3], &self.weights.device)],
            1,
        )
        .slice_assign([0..batch, valid_frames - 1..valid_frames, 0..3], last);
        let local = Tensor::cat(vec![velocity, slice(0..frames, 1..2)], 2);
        (local - self.weights.constants.local_mean.clone())
            / self.weights.constants.local_scale.clone()
    }
}

/// Convert normalized ARDY features into portable Core27 rotations and positions.
pub fn decode_clip(
    config: &ArdyConfig,
    features: &[f32],
    provenance: String,
) -> Result<MotionClip> {
    config.validate()?;
    ensure!(
        !features.is_empty() && features.len().is_multiple_of(330),
        "invalid motion feature shape"
    );
    ensure!(
        features.iter().all(|v| v.is_finite()),
        "non-finite model output"
    );
    let mut frames = Vec::with_capacity(features.len() / 330);
    for values in features.as_chunks::<330>().0.iter() {
        let raw: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let j = if i < 5 { i } else { i + 4 };
                v * config.motion_stats.scale(j) + config.motion_stats.mean[j]
            })
            .collect();
        let mut global = Vec::with_capacity(27);
        let mut local = Vec::with_capacity(27);
        for i in 0..27 {
            let rotation = rotation_from_6d(raw[83 + i * 6..89 + i * 6].try_into().unwrap())?;
            let parent: Quat = config.skeleton.joints[i]
                .parent
                .map_or(Quat::IDENTITY, |p| global[p]);
            global.push(rotation);
            local.push((parent.inverse() * rotation).normalize());
        }
        frames.push(PoseFrame {
            root_translation: Vec3::from_slice(&raw[..3]),
            local_rotations: local,
            foot_contacts: std::array::from_fn(|i| raw[326 + i] > 0.5),
        });
    }
    let clip = MotionClip {
        schema_version: 1,
        rig: config.skeleton.clone(),
        fps: 20.0,
        frames,
        provenance,
        identity: None,
    };
    clip.validate()?;
    Ok(clip)
}

/// Root path features in world coordinates, normalized only where observed.
/// Sparse waypoints constrain their own frames. Dense mode fills intervals;
/// nothing is extrapolated before the first or after the last waypoint.
pub fn trajectory_conditions(
    config: &ArdyConfig,
    request: &MotionRequest,
) -> Result<(Vec<f32>, Vec<f32>)> {
    request.validate()?;
    let mut observed = vec![0.0; request.frames * 330];
    let mut mask = vec![0.0; observed.len()];
    write_trajectory_window(config, request, 0, &mut observed, &mut mask);
    Ok((observed, mask))
}

/// Fill an already-zeroed window without allocating a full-clip condition array.
/// Callers validate the request once before autoregressive generation starts.
pub(crate) fn write_trajectory_window(
    config: &ArdyConfig,
    request: &MotionRequest,
    first_frame: usize,
    observed: &mut [f32],
    mask: &mut [f32],
) {
    debug_assert_eq!(observed.len(), mask.len());
    debug_assert!(observed.len().is_multiple_of(330));
    let end_frame = first_frame + observed.len() / 330;
    let mut write = |frame: usize, pos: Vec3, heading: Option<f32>, height: bool| {
        if !(first_frame..end_frame).contains(&frame) {
            return;
        }
        let frame = frame - first_frame;
        let mut set = |d: usize, v: f32| {
            observed[frame * 330 + d] =
                (v - config.motion_stats.mean[d]) / config.motion_stats.scale(d);
            mask[frame * 330 + d] = 1.0;
        };
        set(0, pos.x);
        set(2, pos.z);
        if height {
            set(1, pos.y);
        }
        if let Some(angle) = heading {
            set(3, angle.cos());
            set(4, angle.sin());
        }
    };
    for waypoint in &request.waypoints {
        write(
            waypoint.frame,
            waypoint.position,
            waypoint.heading,
            waypoint.constrain_height,
        );
    }
    if request.dense_trajectory {
        for pair in request.waypoints.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            for frame in (a.frame + 1).max(first_frame)..b.frame.min(end_frame) {
                let t = (frame - a.frame) as f32 / (b.frame - a.frame) as f32;
                let angle = a
                    .heading
                    .zip(b.heading)
                    .map(|(a, b)| a + (b - a).sin().atan2((b - a).cos()) * t);
                write(
                    frame,
                    a.position.lerp(b.position, t),
                    angle,
                    a.constrain_height && b.constrain_height,
                );
            }
        }
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::config::Stats;
    use burn_human_motion::{RigDefinition, Waypoint};

    #[test]
    fn window_conditions_preserve_boundaries_and_wrapped_headings() {
        let config = ArdyConfig {
            architecture: "unused".into(),
            fps: 20,
            horizon: 40,
            frames_per_token: 4,
            motion_stats: Stats {
                mean: vec![0.25; 334],
                std: vec![2.0; 334],
            },
            latent_stats: Stats {
                mean: vec![],
                std: vec![],
            },
            skeleton: RigDefinition {
                id: "unused".into(),
                joints: vec![],
            },
        };
        let mut request = MotionRequest {
            frames: 12000,
            waypoints: vec![
                Waypoint {
                    frame: 4,
                    position: Vec3::ZERO,
                    heading: Some(179f32.to_radians()),
                    constrain_height: true,
                },
                Waypoint {
                    frame: 20,
                    position: Vec3::new(2.0, 1.0, 4.0),
                    heading: Some(-179f32.to_radians()),
                    constrain_height: true,
                },
            ],
            ..Default::default()
        };
        let mut observed = vec![0.0; 24 * 330];
        let mut mask = vec![0.0; observed.len()];
        write_trajectory_window(&config, &request, 8, &mut observed, &mut mask);
        let denorm = |index| {
            observed[4 * 330 + index] * config.motion_stats.scale(index)
                + config.motion_stats.mean[index]
        };
        assert!((denorm(0) - 1.0).abs() < 1e-6);
        assert!((denorm(1) - 0.5).abs() < 1e-6);
        assert!((denorm(2) - 2.0).abs() < 1e-6);
        assert!(denorm(3) < -0.999);
        assert_eq!(mask[12 * 330], 1.0); // endpoint at global frame 20
        assert_eq!(mask[13 * 330], 0.0); // no extrapolation
        request.dense_trajectory = false;
        observed.fill(0.0);
        mask.fill(0.0);
        write_trajectory_window(&config, &request, 8, &mut observed, &mut mask);
        assert_eq!(mask[4 * 330], 0.0);
        assert_eq!(mask[12 * 330], 1.0);
    }
}

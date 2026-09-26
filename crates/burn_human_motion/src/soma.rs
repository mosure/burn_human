//! SOMA-X interchange. A SOMA rig must be supplied with its real bind transforms;
//! ARDY Core joints are never relabelled as SOMA joints. Identity/corrective model
//! evaluation belongs to a future SOMA inference backend.

use crate::{MotionClip, PoseFrame, RigDefinition, rig::RigIdentity};
use anyhow::{Result, ensure};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SomaAnimation {
    pub rig: RigDefinition,
    pub fps: f32,
    pub identity_model_type: String,
    pub identity_coeffs: Vec<f32>,
    #[serde(default)]
    pub scale_params: Vec<f32>,
    #[serde(default)]
    pub global_scale: Option<f32>,
    /// SOMA axis-angle rotations, radians, joint-name order from the supplied rig.
    pub poses: Vec<Vec<Vec3>>,
    pub translations: Vec<Vec3>,
    pub absolute_pose: bool,
    /// Required for relative poses: world-space T-pose joint orientations.
    /// These are distinct from identity-fitted bind rotations.
    pub joint_orient: Option<Vec<Quat>>,
    pub unit: String,
    pub source_revision: String,
}

impl SomaAnimation {
    pub fn into_clip(self) -> Result<MotionClip> {
        self.rig.validate()?;
        ensure!(
            self.rig.id.starts_with("soma-"),
            "SOMA animation requires a SOMA rig"
        );
        ensure!(
            self.poses.len() == self.translations.len() && !self.poses.is_empty(),
            "SOMA frame count mismatch"
        );
        ensure!(
            self.identity_coeffs.iter().all(|v| v.is_finite()),
            "invalid identity coefficients"
        );
        let scale = match self.unit.as_str() {
            "m" | "meters" => 1.0,
            "cm" | "centimeters" => 0.01,
            "mm" | "millimeters" => 0.001,
            _ => anyhow::bail!("unsupported SOMA length unit"),
        };
        let mut rig = self.rig;
        if !self.absolute_pose {
            let orient = self
                .joint_orient
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("relative SOMA poses require joint_orient"))?;
            ensure!(
                orient.len() == rig.joints.len()
                    && orient
                        .iter()
                        .all(|q| q.is_finite() && (q.length_squared() - 1.0).abs() < 0.005),
                "invalid SOMA joint_orient"
            );
        }
        for joint in &mut rig.joints {
            joint.offset *= scale;
        }
        let mut frames = Vec::with_capacity(self.poses.len());
        for (pose, translation) in self.poses.into_iter().zip(self.translations) {
            ensure!(
                pose.len() == rig.joints.len() && pose.iter().all(|v| v.is_finite()),
                "SOMA pose shape/values mismatch"
            );
            let rotations = pose
                .into_iter()
                .zip(&rig.joints)
                .enumerate()
                .map(|(i, (axis, j))| {
                    let q = Quat::from_scaled_axis(axis);
                    if self.absolute_pose {
                        q
                    } else {
                        let orient = self.joint_orient.as_ref().unwrap();
                        let parent = j.parent.map_or(Quat::IDENTITY, |p| orient[p]);
                        (parent.inverse() * q * orient[i]).normalize()
                    }
                })
                .collect();
            frames.push(PoseFrame {
                root_translation: translation * scale,
                local_rotations: rotations,
                foot_contacts: [false; 4],
            });
        }
        let clip = MotionClip {
            schema_version: 1,
            rig,
            fps: self.fps,
            frames,
            provenance: format!(
                "SOMA-X {}; identity {}",
                self.source_revision, self.identity_model_type
            ),
            identity: Some(RigIdentity {
                model: self.identity_model_type,
                coefficients: self.identity_coeffs,
                scale_parameters: self.scale_params,
                global_scale: self.global_scale,
            }),
        };
        clip.validate()?;
        Ok(clip)
    }
}

use anyhow::{Result, ensure};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A parent-before-child hierarchy. Bind rotations use XYZW quaternions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigDefinition {
    pub id: String,
    pub joints: Vec<RigJoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigJoint {
    pub name: String,
    pub parent: Option<usize>,
    pub offset: Vec3,
    pub bind_rotation: Quat,
}

impl RigDefinition {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "rig id is empty");
        ensure!(
            !self.joints.is_empty() && self.joints.len() <= 4096,
            "invalid joint count"
        );
        let mut names = BTreeSet::new();
        for (i, joint) in self.joints.iter().enumerate() {
            ensure!(
                !joint.name.is_empty() && names.insert(&joint.name),
                "duplicate/empty joint name"
            );
            ensure!(
                if i == 0 {
                    joint.parent.is_none()
                } else {
                    joint.parent.is_some_and(|p| p < i)
                },
                "rig must have one root and parent-before-child order"
            );
            ensure!(
                joint.offset.is_finite() && valid_rotation(joint.bind_rotation),
                "invalid bind transform at {}",
                joint.name
            );
        }
        Ok(())
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }

    /// FK with local *absolute* rotations (including the bind orientation).
    pub fn forward(&self, frame: &PoseFrame) -> Result<(Vec<Vec3>, Vec<Quat>)> {
        self.validate()?;
        frame.validate(self.joints.len())?;
        let mut positions = Vec::with_capacity(self.joints.len());
        let mut rotations: Vec<Quat> = Vec::with_capacity(self.joints.len());
        for (i, joint) in self.joints.iter().enumerate() {
            let local = frame.local_rotations[i];
            if let Some(parent) = joint.parent {
                positions.push(positions[parent] + rotations[parent] * joint.offset);
                rotations.push((rotations[parent] * local).normalize());
            } else {
                positions.push(frame.root_translation);
                rotations.push(local);
            }
        }
        Ok((positions, rotations))
    }

    pub fn rest_frame(&self) -> PoseFrame {
        PoseFrame {
            root_translation: self.joints[0].offset,
            local_rotations: self.joints.iter().map(|j| j.bind_rotation).collect(),
            foot_contacts: [false; 4],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseFrame {
    pub root_translation: Vec3,
    pub local_rotations: Vec<Quat>,
    #[serde(default)]
    pub foot_contacts: [bool; 4],
}

impl PoseFrame {
    pub fn validate(&self, joints: usize) -> Result<()> {
        ensure!(
            self.root_translation.is_finite(),
            "non-finite root translation"
        );
        ensure!(
            self.local_rotations.len() == joints,
            "pose/rig joint count mismatch"
        );
        ensure!(
            self.local_rotations.iter().copied().all(valid_rotation),
            "invalid pose rotation"
        );
        Ok(())
    }

    pub fn interpolate(&self, next: &Self, fraction: f32) -> Self {
        Self {
            root_translation: self.root_translation.lerp(next.root_translation, fraction),
            local_rotations: self
                .local_rotations
                .iter()
                .zip(&next.local_rotations)
                .map(|(a, b)| a.slerp(*b, fraction).normalize())
                .collect(),
            foot_contacts: if fraction < 0.5 {
                self.foot_contacts
            } else {
                next.foot_contacts
            },
        }
    }
}

fn valid_rotation(q: Quat) -> bool {
    q.is_finite() && (q.length_squared() - 1.0).abs() < 0.005
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionClip {
    pub schema_version: u32,
    pub rig: RigDefinition,
    pub fps: f32,
    pub frames: Vec<PoseFrame>,
    pub provenance: String,
    /// Source identity metadata. A supplied rig already contains its fitted offsets;
    /// consumers must not apply these scales again when retargeting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<RigIdentity>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RigIdentity {
    pub model: String,
    pub coefficients: Vec<f32>,
    pub scale_parameters: Vec<f32>,
    pub global_scale: Option<f32>,
}

impl MotionClip {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported motion schema");
        self.rig.validate()?;
        ensure!(
            self.fps.is_finite() && self.fps > 0.0 && self.fps <= 1000.0,
            "invalid frame rate"
        );
        ensure!(!self.frames.is_empty(), "empty motion clip");
        if let Some(identity) = &self.identity {
            ensure!(
                !identity.model.is_empty()
                    && identity
                        .coefficients
                        .iter()
                        .chain(&identity.scale_parameters)
                        .all(|v| v.is_finite())
                    && identity
                        .global_scale
                        .is_none_or(|v| v.is_finite() && v > 0.0),
                "invalid rig identity metadata"
            );
        }
        for frame in &self.frames {
            frame.validate(self.rig.joints.len())?;
        }
        Ok(())
    }

    pub fn duration(&self) -> f32 {
        self.frames.len().saturating_sub(1) as f32 / self.fps
    }

    /// Sample a validated clip without wrapping the last frame to the first.
    pub fn sample(&self, seconds: f32) -> PoseFrame {
        let t = (seconds.max(0.0) * self.fps).min((self.frames.len() - 1) as f32);
        let i = t.floor() as usize;
        self.frames[i].interpolate(&self.frames[(i + 1).min(self.frames.len() - 1)], t.fract())
    }
}

/// Explicit semantic correspondence. Missing joints retain their bind-local rotation.
/// Retargeting transfers world-space rotation deltas, so intermediate/twist joints
/// do not accidentally apply an ancestor's motion twice.
#[derive(Clone, Debug)]
pub struct RigMapping {
    pub source_for_target: Vec<Option<usize>>,
    pub root_scale: f32,
    pub basis: Quat,
}

impl RigMapping {
    pub fn retarget(
        &self,
        source: &RigDefinition,
        target: &RigDefinition,
        frame: &PoseFrame,
    ) -> Result<PoseFrame> {
        source.validate()?;
        target.validate()?;
        ensure!(
            self.source_for_target.len() == target.joints.len(),
            "mapping/target mismatch"
        );
        ensure!(
            self.root_scale.is_finite() && self.root_scale > 0.0 && valid_rotation(self.basis),
            "invalid retarget scale/basis"
        );
        let (_, source_global) = source.forward(frame)?;
        let (_, source_rest) = source.forward(&source.rest_frame())?;
        let (_, target_rest) = target.forward(&target.rest_frame())?;
        let mut global: Vec<Quat> = Vec::with_capacity(target.joints.len());
        let mut local = Vec::with_capacity(target.joints.len());
        for (i, joint) in target.joints.iter().enumerate() {
            let parent = joint.parent.map_or(Quat::IDENTITY, |p| global[p]);
            let rotation = if let Some(s) = self.source_for_target[i] {
                ensure!(s < source.joints.len(), "mapping source out of range");
                let delta =
                    self.basis * source_global[s] * source_rest[s].inverse() * self.basis.inverse();
                delta * target_rest[i]
            } else {
                parent * joint.bind_rotation
            };
            global.push(rotation.normalize());
            local.push((parent.inverse() * rotation).normalize());
        }
        Ok(PoseFrame {
            root_translation: self.basis * frame.root_translation * self.root_scale,
            local_rotations: local,
            foot_contacts: frame.foot_contacts,
        })
    }
}

/// ARDY encodes the first two **columns**, not rows, of a rotation matrix.
pub fn rotation_from_6d(v: [f32; 6]) -> Result<Quat> {
    let x = Vec3::from_slice(&v[..3]);
    let y = Vec3::from_slice(&v[3..]);
    ensure!(
        x.is_finite() && y.is_finite() && x.length_squared() > 1e-12,
        "degenerate 6D rotation"
    );
    let x = x.normalize();
    let z = x.cross(y);
    ensure!(z.length_squared() > 1e-12, "collinear 6D rotation");
    let z = z.normalize();
    Ok(Quat::from_mat3(&Mat3::from_cols(x, z.cross(x), z)).normalize())
}

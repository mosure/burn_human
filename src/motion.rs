//! Bind-aware conversion from portable Y-up motion to Anny's Z-up rig and
//! root_relative_world pose matrices. Build once per phenotype, reuse per frame.
use crate::{AnnyBody, AnnyInput};
use anyhow::{Result, ensure};
use burn_human_motion::{PoseFrame, RigDefinition, RigJoint, RigMapping};
use glam::{Mat4, Quat, Vec3};

pub struct AnnyMotionBinding {
    pub target: RigDefinition,
    pub mapping: RigMapping,
    /// Optional vertical root scale for fitting a source motion to Anny's leg
    /// length. X/Z world trajectory is unchanged. Exact root transfer remains
    /// the default of pose_parameters; the viewer exposes this separately.
    pub root_height_scale: f32,
    rest_local_rotations: Vec<Quat>,
}

impl AnnyMotionBinding {
    pub fn new(body: &AnnyBody, phenotypes: &[f64], source: &RigDefinition) -> Result<Self> {
        source.validate()?;
        ensure!(
            matches!(source.id.as_str(), "ardy-core27" | "soma-77" | "soma-30"),
            "no semantic mapping for rig {}",
            source.id
        );
        let output = body.forward(AnnyInput {
            phenotype_inputs: Some(phenotypes),
            ..Default::default()
        })?;
        let meta = &body.metadata().metadata;
        let globals: Vec<Mat4> = output
            .rest_bone_poses
            .data
            .as_chunks::<16>()
            .0
            .iter()
            .map(|v| Mat4::from_cols_array(&std::array::from_fn(|i| v[(i % 4) * 4 + i / 4] as f32)))
            .collect();
        ensure!(
            globals.len() == meta.bone_labels.len(),
            "Anny binding requires a single phenotype"
        );
        let mut joints = Vec::new();
        let mut indices = Vec::new();
        for (i, name) in meta.bone_labels.iter().enumerate() {
            let parent = usize::try_from(meta.bone_parents[i]).ok();
            let local = parent.map_or(globals[i], |p| globals[p].inverse() * globals[i]);
            let (_, rotation, position) = local.to_scale_rotation_translation();
            joints.push(RigJoint {
                name: name.clone(),
                parent,
                offset: position,
                bind_rotation: rotation.normalize(),
            });
            indices.push(semantic_source(name, source));
        }
        ensure!(indices[0].is_some(), "source rig has no Hips joint");
        let mut target = RigDefinition {
            id: "anny-fullbody".into(),
            joints,
        };
        target.validate()?;
        let rest_local_rotations = target.joints.iter().map(|j| j.bind_rotation).collect();
        let mapping = RigMapping {
            source_for_target: indices,
            root_scale: 1.0,
            basis: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        };
        calibrate_reference_pose(&mut target, source, &mapping)?;
        let (source_rest, _) = source.forward(&source.rest_frame())?;
        let (target_rest, _) = target.forward(&target.rest_frame())?;
        let source_feet = ["LeftToeBase", "RightToeBase"].map(|n| source.index(n));
        let target_feet = ["toe3-1.L", "toe3-1.R"].map(|n| target.index(n));
        let mut root_height_scale = 1.0;
        if let ([Some(sl), Some(sr)], [Some(tl), Some(tr)]) = (source_feet, target_feet) {
            let source_height = source_rest[0].y - source_rest[sl].y.min(source_rest[sr].y);
            let target_height = target_rest[0].z - target_rest[tl].z.min(target_rest[tr].z);
            if source_height > 1e-5 && target_height > 1e-5 {
                root_height_scale = target_height / source_height;
            }
        }
        Ok(Self {
            target,
            mapping,
            rest_local_rotations,
            root_height_scale,
        })
    }

    /// Row-major [1,J,4,4] pose parameters consumed by AnnyBody and BurnHumanInput.
    pub fn pose_parameters(&self, source: &RigDefinition, frame: &PoseFrame) -> Result<Vec<f64>> {
        let pose = self.mapping.retarget(source, &self.target, frame)?;
        let mut matrices = Vec::with_capacity(self.target.joints.len() * 16);
        for (i, q) in pose.local_rotations.iter().enumerate() {
            let (delta, translation) = if i == 0 {
                (
                    *q * self.rest_local_rotations[0].inverse(),
                    pose.root_translation,
                )
            } else {
                (self.rest_local_rotations[i].inverse() * *q, Vec3::ZERO)
            };
            let m = Mat4::from_rotation_translation(delta.normalize(), translation).to_cols_array();
            matrices.extend((0..16).map(|j| m[(j % 4) * 4 + j / 4] as f64));
        }
        Ok(matrices)
    }
}

/// Match the source reference limb directions before transferring motion deltas.
/// Anny's bind mesh has an A-pose; Core uses a T-pose. Transferring deltas onto
/// the uncalibrated A-pose would bias both arms by about forty degrees. Keep the
/// actual bind rotations separately for Anny's skinning parameterization.
fn calibrate_reference_pose(
    target: &mut RigDefinition,
    source: &RigDefinition,
    mapping: &RigMapping,
) -> Result<()> {
    let (source_positions, _) = source.forward(&source.rest_frame())?;
    let (target_positions, target_global) = target.forward(&target.rest_frame())?;
    let mut corrections = vec![None; target.joints.len()];
    for side in ["L", "R"] {
        for (a, b) in [
            ("upperarm01", "lowerarm01"),
            ("lowerarm01", "wrist"),
            ("upperleg01", "lowerleg01"),
            ("lowerleg01", "foot"),
            ("foot", "toe3-1"),
            ("wrist", "finger3-1"),
        ] {
            let Some(i) = target.index(&format!("{a}.{side}")) else {
                continue;
            };
            let Some(j) = target.index(&format!("{b}.{side}")) else {
                continue;
            };
            let Some(s) = mapping.source_for_target[i] else {
                continue;
            };
            let end = if a == "wrist" {
                let prefix = if side == "L" { "Left" } else { "Right" };
                source
                    .index(&format!("{prefix}HandEnd"))
                    .or_else(|| source.index(&format!("{prefix}HandMiddle1")))
            } else {
                mapping.source_for_target[j]
            };
            let Some(t) = end else {
                continue;
            };
            let from = target_positions[j] - target_positions[i];
            let to = mapping.basis * (source_positions[t] - source_positions[s]);
            ensure!(
                from.length_squared() > 1e-10 && to.length_squared() > 1e-10,
                "degenerate reference limb {a}.{side}"
            );
            corrections[i] = Some(Quat::from_rotation_arc(from.normalize(), to.normalize()));
        }
    }
    let mut reference_global = Vec::with_capacity(target.joints.len());
    for (i, joint) in target.joints.iter_mut().enumerate() {
        let parent = joint.parent.map_or(Quat::IDENTITY, |p| reference_global[p]);
        let global = corrections[i]
            .map_or(parent * joint.bind_rotation, |q| q * target_global[i])
            .normalize();
        joint.bind_rotation = (parent.inverse() * global).normalize();
        reference_global.push(global);
    }
    Ok(())
}

fn semantic_source(name: &str, source: &RigDefinition) -> Option<usize> {
    let soma = source.id.starts_with("soma-");
    let key = match name {
        "root" => "Hips",
        "spine05" => {
            if soma {
                "Spine1"
            } else {
                "Spine"
            }
        }
        "spine04" => {
            if soma {
                "Spine2"
            } else {
                "Spine1"
            }
        }
        "spine03" | "spine02" => {
            if soma {
                "Chest"
            } else {
                "Spine2"
            }
        }
        "spine01" => {
            if soma {
                "Chest"
            } else {
                "Spine3"
            }
        }
        "neck01" => {
            if soma {
                "Neck1"
            } else {
                "Neck"
            }
        }
        "head" => "Head",
        "jaw" if soma => "Jaw",
        "eye.L" if soma => "LeftEye",
        "eye.R" if soma => "RightEye",
        "clavicle.L" | "shoulder01.L" => "LeftShoulder",
        "clavicle.R" | "shoulder01.R" => "RightShoulder",
        "upperarm01.L" => "LeftArm",
        "upperarm01.R" => "RightArm",
        "lowerarm01.L" => "LeftForeArm",
        "lowerarm01.R" => "RightForeArm",
        "wrist.L" => "LeftHand",
        "wrist.R" => "RightHand",
        "upperleg01.L" => {
            if soma {
                "LeftLeg"
            } else {
                "LeftUpLeg"
            }
        }
        "upperleg01.R" => {
            if soma {
                "RightLeg"
            } else {
                "RightUpLeg"
            }
        }
        "lowerleg01.L" => {
            if soma {
                "LeftShin"
            } else {
                "LeftLeg"
            }
        }
        "lowerleg01.R" => {
            if soma {
                "RightShin"
            } else {
                "RightLeg"
            }
        }
        "foot.L" => "LeftFoot",
        "foot.R" => "RightFoot",
        "toe1-1.L" | "toe2-1.L" | "toe3-1.L" | "toe4-1.L" | "toe5-1.L" => "LeftToeBase",
        "toe1-1.R" | "toe2-1.R" | "toe3-1.R" | "toe4-1.R" | "toe5-1.R" => "RightToeBase",
        _ => return None,
    };
    source.index(key)
}

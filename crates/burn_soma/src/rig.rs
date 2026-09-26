// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use crate::geometry::*;
use anyhow::{Result, ensure};
use glam::{Mat3, Mat4, Quat, Vec3};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct CorrectiveGroup {
    pub joint: usize,
    pub columns: usize,
}
#[derive(Deserialize)]
pub struct RigData {
    pub public_names: Vec<String>,
    pub target_names: Vec<String>,
    pub public_parents: Vec<usize>,
    pub target_parents: Vec<usize>,
    pub public_indices: Vec<usize>,
    pub bind_world: Vec<[[f32; 4]; 4]>,
    pub bind_local: Vec<[[f32; 4]; 4]>,
    pub fit_bind_world: Vec<[[f32; 4]; 4]>,
    pub fit_bind_local: Vec<[[f32; 4]; 4]>,
    pub fit_skinned: Vec<Vec<usize>>,
    pub fit_frozen: Vec<usize>,
    pub fit_skip_endjoints: bool,
    pub fit_skip_inverse_lbs: bool,
    pub procedural: Procedural,
    pub corrective_bind: Vec<[[f32; 3]; 3]>,
    pub corrective_tanh: bool,
    pub corrective_channels: usize,
    pub corrective_groups: Vec<CorrectiveGroup>,
    pub scale_names: Vec<String>,
    pub scale_public_indices: Vec<usize>,
    pub scale_target_map: Vec<usize>,
}

#[derive(Deserialize)]
pub struct Procedural {
    control_source_ids: Vec<usize>,
    control_target_ids: Vec<usize>,
    twist_target_ids: Vec<usize>,
    twist_parent_target_ids: Vec<usize>,
    twist_axis_ids: Vec<usize>,
    twist_axis_signs: Vec<f32>,
    target_t_pose_local_rotations: Vec<[[f32; 3]; 3]>,
    source_joint_orient: Vec<[[f32; 3]; 3]>,
    source_joint_orient_parent_t: Vec<[[f32; 3]; 3]>,
    source_bind_quaternions: Vec<[f32; 4]>,
    segment_bind_align_quaternions: Vec<[f32; 4]>,
    segment_start_source_ids: Vec<usize>,
    segment_end_source_ids: Vec<usize>,
    segment_reverse_mask: Vec<bool>,
    aligned_virtual_segment_ids: Vec<usize>,
    aligned_virtual_joint_ids: Vec<usize>,
    rotation_parameter_matrix: Vec<Vec<f32>>,
    translation_parameter_matrix: Vec<Vec<f32>>,
}

impl RigData {
    pub fn validate(&self, vertices: usize) -> Result<()> {
        ensure!(
            self.public_names.len() == 78
                && self.target_names.len() == 110
                && self.public_indices.len() == 78,
            "SOMA joint inventory"
        );
        for (p, n) in [(&self.public_parents, 78), (&self.target_parents, 110)] {
            ensure!(
                p.len() == n && p[0] == 0 && p.iter().enumerate().skip(1).all(|(i, p)| *p < i),
                "Rig parents must be topologically ordered"
            );
        }
        ensure!(
            self.bind_world.len() == 110
                && self.bind_local.len() == 110
                && self.fit_bind_world.len() == 78
                && self.fit_bind_local.len() == 78,
            "Rig bind shapes"
        );
        ensure!(
            self.public_indices.iter().all(|i| *i < 110)
                && self.fit_skinned.len() == 78
                && self.fit_skinned.iter().flatten().all(|i| *i < vertices),
            "Rig index bounds"
        );
        ensure!(
            self.scale_names.len() == 60
                && self.scale_public_indices.len() == 60
                && self.scale_public_indices.iter().all(|i| *i < 78)
                && self.scale_target_map.len() == 110
                && self.scale_target_map.iter().all(|i| *i < 78),
            "Scale control inventory"
        );
        ensure!(
            self.corrective_bind.len() == 78
                && self.corrective_channels == 24
                && self
                    .corrective_groups
                    .iter()
                    .all(|g| g.joint < 78 && g.columns > 0 && g.columns <= vertices * 3),
            "Corrective inventory"
        );
        ensure!(
            self.fit_frozen.iter().all(|i| *i < 78),
            "Frozen joint index"
        );
        for matrices in [
            &self.bind_world,
            &self.bind_local,
            &self.fit_bind_world,
            &self.fit_bind_local,
        ] {
            ensure!(
                matrices.iter().flatten().flatten().all(|x| x.is_finite()),
                "Non-finite bind matrix"
            );
        }
        ensure!(
            self.corrective_bind
                .iter()
                .flatten()
                .flatten()
                .all(|x| x.is_finite()),
            "Non-finite corrective bind"
        );
        ensure!(
            self.public_indices
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == 78,
            "Duplicate public joint index"
        );
        self.procedural.validate()
    }

    pub fn fit(&self, positions: &[Vec3], vertices: &[Vec3], template: &[Vec3]) -> Vec<Mat4> {
        let bind: Vec<_> = self.fit_bind_world.iter().copied().map(matrix4).collect();
        let mut rotations: Vec<_> = bind.iter().copied().map(rotation).collect();
        for i in 1..78 {
            let children: Vec<_> = (1..78).filter(|j| self.public_parents[*j] == i).collect();
            if children.is_empty() && self.fit_skip_endjoints {
                rotations[i] = rotations[self.public_parents[i]];
                continue;
            }
            if self.fit_frozen.contains(&i) {
                rotations[i] =
                    rotations[self.public_parents[i]] * rotation(matrix4(self.fit_bind_local[i]));
                continue;
            }
            let initial = if self.fit_skip_inverse_lbs {
                Mat3::IDENTITY
            } else {
                let source: Vec<_> = self.fit_skinned[i]
                    .iter()
                    .map(|j| template[*j] - position(bind[i]))
                    .collect();
                let target: Vec<_> = self.fit_skinned[i]
                    .iter()
                    .map(|j| vertices[*j] - positions[i])
                    .collect();
                align(&target, &source)
            };
            let child_rotation = if children.is_empty() {
                Mat3::IDENTITY
            } else {
                let source: Vec<_> = children
                    .iter()
                    .map(|j| initial * (position(bind[*j]) - position(bind[i])))
                    .collect();
                let target: Vec<_> = children
                    .iter()
                    .map(|j| positions[*j] - positions[i])
                    .collect();
                align(&target, &source)
            };
            rotations[i] = child_rotation * initial * rotations[i];
        }
        rotations
            .into_iter()
            .zip(positions)
            .map(|(r, &t)| transform(r, t))
            .collect()
    }

    pub fn expand_bind(&self, public: &[Mat4]) -> Vec<Mat4> {
        let mut target: Vec<_> = self.bind_world.iter().copied().map(matrix4).collect();
        for (&i, &m) in self.public_indices.iter().zip(public) {
            target[i] = m;
        }
        let positions: Vec<_> = target.iter().copied().map(position).collect();
        for (i, row) in self
            .procedural
            .translation_parameter_matrix
            .iter()
            .enumerate()
        {
            let t = row
                .iter()
                .zip(&positions)
                .fold(Vec3::ZERO, |v, (w, t)| v + *t * *w);
            target[i].w_axis = t.extend(1.0);
        }
        target
    }
}

impl Procedural {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.control_source_ids.len() == 78
                && self.control_target_ids.len() == 78
                && self.control_source_ids.iter().all(|i| *i < 78)
                && self.control_target_ids.iter().all(|i| *i < 110),
            "Procedural controls"
        );
        ensure!(
            self.twist_target_ids.len() == 32
                && self.twist_parent_target_ids.len() == 32
                && self
                    .twist_target_ids
                    .iter()
                    .chain(&self.twist_parent_target_ids)
                    .all(|i| *i < 110),
            "Twist topology"
        );
        ensure!(
            self.twist_axis_ids.len() == 32
                && self.twist_axis_ids.iter().all(|i| *i < 3)
                && self.twist_axis_signs.len() == 32,
            "Twist axes"
        );
        ensure!(
            self.target_t_pose_local_rotations.len() == 110
                && self.source_joint_orient.len() == 78
                && self.source_joint_orient_parent_t.len() == 78
                && self.source_bind_quaternions.len() == 78,
            "Procedural orientations"
        );
        ensure!(
            self.segment_bind_align_quaternions.len() == 8
                && self.segment_start_source_ids.len() == 8
                && self.segment_end_source_ids.len() == 8
                && self.segment_reverse_mask.len() == 8,
            "Twist segments"
        );
        ensure!(
            self.segment_start_source_ids
                .iter()
                .chain(&self.segment_end_source_ids)
                .all(|i| *i < 78)
                && self.aligned_virtual_segment_ids.len() == 24
                && self.aligned_virtual_segment_ids.iter().all(|i| *i < 8)
                && self.aligned_virtual_joint_ids.len() == 24
                && self.aligned_virtual_joint_ids.iter().all(|i| *i < 78),
            "Aligned twist indices"
        );
        ensure!(
            self.rotation_parameter_matrix.len() == 32
                && self.rotation_parameter_matrix.iter().all(|r| r.len() == 78)
                && self.translation_parameter_matrix.len() == 110
                && self
                    .translation_parameter_matrix
                    .iter()
                    .all(|r| r.len() == 110),
            "Procedural parameter matrices"
        );
        ensure!(
            self.rotation_parameter_matrix
                .iter()
                .flatten()
                .chain(self.translation_parameter_matrix.iter().flatten())
                .chain(&self.twist_axis_signs)
                .all(|x| x.is_finite()),
            "Non-finite procedural parameters"
        );
        for matrices in [
            &self.target_t_pose_local_rotations,
            &self.source_joint_orient,
            &self.source_joint_orient_parent_t,
        ] {
            ensure!(
                matrices.iter().flatten().flatten().all(|x| x.is_finite()),
                "Non-finite procedural orientation"
            );
        }
        ensure!(
            self.source_bind_quaternions
                .iter()
                .chain(&self.segment_bind_align_quaternions)
                .all(|q| q.iter().all(|x| x.is_finite())
                    && (q.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.001),
            "Invalid procedural quaternion"
        );
        Ok(())
    }
    pub fn orient(&self, relative: &[Mat3]) -> Vec<Mat3> {
        relative
            .iter()
            .enumerate()
            .map(|(i, r)| {
                matrix3(self.source_joint_orient_parent_t[i])
                    * *r
                    * matrix3(self.source_joint_orient[i])
            })
            .collect()
    }
    pub fn expand(
        &self,
        world: &[Mat4],
        target_translations: &[Vec3],
        target_local: &[Mat4],
        parents: &[usize],
    ) -> Vec<Mat4> {
        let virtuals: Vec<_> = self
            .aligned_virtual_joint_ids
            .iter()
            .zip(&self.aligned_virtual_segment_ids)
            .map(|(&j, &s)| {
                (Quat::from_mat3(&rotation(world[j]))
                    * Quat::from_array(self.source_bind_quaternions[j]).conjugate()
                    * Quat::from_array(self.segment_bind_align_quaternions[s]))
                .normalize()
            })
            .collect();
        let mut channels = [0f32; 78];
        for s in 0..8 {
            channels[self.segment_end_source_ids[s]] =
                twist_x(virtuals[s + 8].conjugate() * virtuals[s]);
            if self.segment_reverse_mask[s] {
                channels[self.segment_start_source_ids[s]] =
                    twist_x(virtuals[s + 16].conjugate() * virtuals[s + 8]);
            }
        }
        let mut out = vec![Mat4::IDENTITY; 110];
        let mut assigned = [false; 110];
        for (&s, &t) in self.control_source_ids.iter().zip(&self.control_target_ids) {
            out[t] = world[s];
            assigned[t] = true;
        }
        for (k, (&j, &parent)) in self
            .twist_target_ids
            .iter()
            .zip(&self.twist_parent_target_ids)
            .enumerate()
        {
            let angle = self.rotation_parameter_matrix[k]
                .iter()
                .zip(channels)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * self.twist_axis_signs[k];
            let axis = [Vec3::X, Vec3::Y, Vec3::Z][self.twist_axis_ids[k]];
            let r = rotation(target_local[j]) * Mat3::from_quat(Quat::from_axis_angle(axis, angle));
            out[j] = out[parent] * transform(r, target_translations[j]);
            assigned[j] = true;
        }
        for j in 1..110 {
            if !assigned[j] {
                out[j] =
                    out[parents[j]] * transform(rotation(target_local[j]), target_translations[j]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fitted_twist_helpers_retain_identity_bind_orientation() {
        // At zero twist, a helper must keep its fitted local bind rotation.
        // Replacing this with a canonical T-pose rotation changes the mesh even
        // though every public FK joint is still correct.
        let identity = Mat3::IDENTITY.to_cols_array_2d();
        let p = Procedural {
            control_source_ids: (0..78).collect(),
            control_target_ids: (0..78).collect(),
            twist_target_ids: vec![80],
            twist_parent_target_ids: vec![3],
            twist_axis_ids: vec![0],
            twist_axis_signs: vec![1.0],
            target_t_pose_local_rotations: vec![identity; 110],
            source_joint_orient: vec![identity; 78],
            source_joint_orient_parent_t: vec![identity; 78],
            source_bind_quaternions: vec![[0.0, 0.0, 0.0, 1.0]; 78],
            segment_bind_align_quaternions: vec![[0.0, 0.0, 0.0, 1.0]; 8],
            segment_start_source_ids: vec![0; 8],
            segment_end_source_ids: vec![1; 8],
            segment_reverse_mask: vec![false; 8],
            aligned_virtual_segment_ids: vec![0; 24],
            aligned_virtual_joint_ids: vec![0; 24],
            rotation_parameter_matrix: vec![vec![0.0; 78]],
            translation_parameter_matrix: vec![],
        };
        let world = vec![Mat4::IDENTITY; 78];
        let mut local = vec![Mat4::IDENTITY; 110];
        local[80] = Mat4::from_rotation_y(0.42);
        let mut translations = vec![Vec3::ZERO; 110];
        translations[80] = Vec3::new(0.1, 0.2, 0.3);
        let expanded = p.expand(&world, &translations, &local, &[0; 110]);
        assert!((rotation(expanded[80]) * Vec3::X - rotation(local[80]) * Vec3::X).length() < 1e-6);
        assert_eq!(position(expanded[80]), translations[80]);
    }
}

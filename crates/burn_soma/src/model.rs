// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use crate::{geometry::*, rig::RigData};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{IndexingUpdateOp, Int, Tensor, TensorData, activation},
};
use burn_human_inference::{
    transport::ModelSource,
    weights::{TensorBank, read_tensors},
};
use burn_human_motion::artifacts::Manifest;
use glam::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityParameters {
    pub coefficients: Vec<f32>,
    pub bone_scales: Vec<f32>,
    pub global_scale: f32,
}
impl Default for IdentityParameters {
    fn default() -> Self {
        Self {
            coefficients: vec![0.0; 128],
            bone_scales: vec![1.0; 60],
            global_scale: 1.0,
        }
    }
}
impl IdentityParameters {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.coefficients.len() == 128
                && self
                    .coefficients
                    .iter()
                    .all(|x| x.is_finite() && x.abs() <= 64.0),
            "SOMA requires 128 finite PCA coefficients"
        );
        ensure!(
            self.bone_scales.len() == 60
                && self
                    .bone_scales
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0 && *v <= 10.0),
            "SOMA requires 60 positive bone scale ratios"
        );
        ensure!(
            self.global_scale.is_finite() && self.global_scale > 0.0 && self.global_scale <= 10.0,
            "Invalid global scale"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SomaPose {
    /// 77 axis-angle rotations in public joint order, Hips first.
    pub rotations: Vec<[f32; 3]>,
    pub translation: [f32; 3],
    pub apply_correctives: bool,
    /// False: T-pose-relative joint rotations. True: absolute local rotations.
    pub absolute_pose: bool,
}
impl Default for SomaPose {
    fn default() -> Self {
        Self {
            rotations: vec![[0.0; 3]; 77],
            translation: [0.0; 3],
            apply_correctives: true,
            absolute_pose: false,
        }
    }
}

pub struct PreparedIdentity<B: Backend> {
    pub rest_vertices: Tensor<B, 2>,
    pub bind_world: Vec<Mat4>,
    public_local: Vec<Mat4>,
    target_local: Vec<Mat4>,
    pub parameters: IdentityParameters,
}
pub struct SomaOutput<B: Backend> {
    pub vertices: Tensor<B, 3>,
    /// Includes the virtual root, in public SOMA order; one list per batch item.
    pub transforms: Vec<Vec<Mat4>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BindConvention {
    #[default]
    Canonical,
    Fitted,
}

pub struct Soma<B: Backend> {
    pub rig: RigData,
    weights: TensorBank<B>,
    template: Vec<Vec3>,
    corrective_indices: BTreeMap<usize, Tensor<B, 1, Int>>,
    pub faces: Vec<[u32; 3]>,
}

impl<B: Backend> Soma<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == "nvidia/SOMA-X"
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.source_revision == crate::SOURCE_REVISION,
            "Unsupported SOMA checkpoint"
        );
        ensure!(
            manifest.config
                == serde_json::json!({"vertices":18056,"public_joints":78,"skin_joints":110,"identity_coefficients":128,"scale_parameters":60,"lod":"mid","unit":"meters","up_axis":"Y","forward_axis":"Z"}),
            "Unsupported SOMA architecture"
        );
        let asset = manifest
            .assets
            .iter()
            .find(|a| a.path == "metadata/rig.json")
            .ok_or_else(|| anyhow::anyhow!("Missing SOMA rig metadata"))?;
        let rig: RigData = serde_json::from_slice(&source.asset(asset).await?)?;
        rig.validate(18056)?;
        let mut expected: BTreeMap<String, (Vec<usize>, &str)> = [
            ("identity.mean", vec![54168], "f32"),
            ("identity.directions", vec![54168, 128], "f32"),
            ("identity.stddev", vec![128], "f32"),
            ("fit.regressor", vec![78, 18056], "f32"),
            ("fit.vertices", vec![18056, 3], "f32"),
            ("skin.weights", vec![18056, 110], "f32"),
            ("skin.public_weights", vec![18056, 78], "f32"),
            ("corrective.input.weight", vec![1872, 468], "f32"),
        ]
        .into_iter()
        .map(|(n, s, d)| (n.into(), (s, d)))
        .collect();
        for group in &rig.corrective_groups {
            expected.insert(
                format!("corrective.output.{}.weight", group.joint),
                (vec![group.columns, 24], "f32"),
            );
            expected.insert(
                format!("corrective.output.{}.indices", group.joint),
                (vec![group.columns], "i32"),
            );
        }
        let mut actual = BTreeMap::new();
        for object in &manifest.objects {
            for t in &object.tensors {
                if t.name == "mesh.faces" {
                    ensure!(
                        t.dtype == "i32"
                            && t.shape.len() == 2
                            && t.shape[1] == 3
                            && t.shape[0] < 100000,
                        "Mesh face shape"
                    );
                } else {
                    ensure!(
                        actual
                            .insert(t.name.clone(), (t.shape.clone(), t.dtype.as_str()))
                            .is_none(),
                        "Duplicate SOMA tensor"
                    );
                }
            }
        }
        ensure!(actual == expected, "SOMA tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        let mut template = vec![];
        let mut faces = vec![];
        let mut corrective_indices = BTreeMap::new();
        for (i, object) in manifest.objects.iter().enumerate() {
            for (name, data) in read_tensors(source, object).await? {
                if name == "fit.vertices" {
                    let flat = data
                        .to_vec::<f32>()
                        .map_err(|e| anyhow::anyhow!("vertices: {e}"))?;
                    template = flat
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| Vec3::from_array(*p))
                        .collect();
                } else if name == "mesh.faces" {
                    let flat = data
                        .to_vec::<i32>()
                        .map_err(|e| anyhow::anyhow!("indices: {e}"))?;
                    ensure!(
                        flat.iter().all(|v| *v >= 0 && *v < 18056),
                        "Mesh face index out of bounds"
                    );
                    faces = flat
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                        .collect();
                } else if name.ends_with(".indices") {
                    let joint = name.split('.').nth(2).unwrap().parse()?;
                    let indices = data
                        .to_vec::<i32>()
                        .map_err(|e| anyhow::anyhow!("indices: {e}"))?;
                    ensure!(
                        indices.iter().all(|v| *v >= 0 && *v < 54168)
                            && indices.windows(2).all(|v| v[0] < v[1]),
                        "Corrective scatter indices"
                    );
                    corrective_indices.insert(joint, Tensor::from_data(data, device));
                } else {
                    weights.insert(name, data)?;
                }
            }
            progress(i + 1, manifest.objects.len());
        }
        ensure!(!faces.is_empty(), "Missing mesh topology");
        Ok(Self {
            rig,
            weights,
            template,
            corrective_indices,
            faces,
        })
    }

    pub async fn prepare_identity(
        &self,
        parameters: IdentityParameters,
    ) -> Result<PreparedIdentity<B>> {
        parameters.validate()?;
        let coeff = Tensor::<B, 2>::from_data(
            TensorData::new(parameters.coefficients.clone(), [1, 128]),
            &self.weights.device,
        );
        let weighted = coeff * self.weights.tensor::<1>("identity.stddev").unsqueeze();
        let vertices = (self.weights.linear("identity.directions", weighted)
            + self.weights.tensor::<1>("identity.mean").unsqueeze())
            * parameters.global_scale;
        self.prepare_rest_shape(vertices.reshape([18056, 3]), parameters)
            .await
    }

    /// Fit and bind a SOMA-topology rest mesh, also used by the MHR identity adapter.
    pub async fn prepare_rest_shape(
        &self,
        vertices: Tensor<B, 2>,
        parameters: IdentityParameters,
    ) -> Result<PreparedIdentity<B>> {
        self.prepare_rest_shape_with_bind(vertices, parameters, BindConvention::Canonical)
            .await
    }

    /// GEM-X's reference wrapper retains the identity-fitted bind orientations.
    /// Native SOMA controls use the canonical bind convention by default.
    pub async fn prepare_rest_shape_with_bind(
        &self,
        vertices: Tensor<B, 2>,
        parameters: IdentityParameters,
        convention: BindConvention,
    ) -> Result<PreparedIdentity<B>> {
        parameters.validate()?;
        ensure!(
            vertices.dims() == [18056, 3],
            "Rest mesh topology differs from SOMA mid LOD"
        );
        let positions = self
            .weights
            .tensor::<2>("fit.regressor")
            .matmul(vertices.clone());
        let pos = positions
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("joint positions: {e}"))?;
        let host = vertices
            .clone()
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("rest vertices: {e}"))?;
        ensure!(
            host.iter().all(|v| v.is_finite()),
            "Non-finite identity mesh"
        );
        let pos: Vec<_> = pos
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| Vec3::from_array(*p))
            .collect();
        let host: Vec<_> = host
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| Vec3::from_array(*p))
            .collect();
        let fitted = self.rig.fit(&pos, &host, &self.template);
        if convention == BindConvention::Fitted {
            let bind_world = self.rig.expand_bind(&fitted);
            let public_local = local(&fitted, &self.rig.public_parents);
            let target_local = local(&bind_world, &self.rig.target_parents);
            return Ok(PreparedIdentity {
                rest_vertices: vertices,
                bind_world,
                public_local,
                target_local,
                parameters,
            });
        }
        let fitted_local = local(&fitted, &self.rig.public_parents);
        // Repose to the canonical bind orientation, retaining fitted bone lengths.
        let rotations: Vec<_> = self
            .rig
            .public_indices
            .iter()
            .map(|i| rotation(matrix4(self.rig.bind_local[*i])))
            .collect();
        let mut translations: Vec<_> = fitted_local.iter().copied().map(position).collect();
        translations[1].x = 0.0;
        translations[1].z = 0.0;
        let mut public = fk(&rotations, &translations, &self.rig.public_parents);
        let floor = public
            .iter()
            .map(|m| m.w_axis.y)
            .fold(f32::INFINITY, f32::min);
        for m in &mut public {
            m.w_axis.y -= floor;
        }
        let rest_vertices = self
            .skin(
                vertices.unsqueeze(),
                &[public.clone()],
                &fitted,
                "skin.public_weights",
            )
            .reshape([18056, 3]);
        public[0] = Mat4::IDENTITY;
        let bind_world = self.rig.expand_bind(&public);
        let public_local = local(&public, &self.rig.public_parents);
        let target_local = local(&bind_world, &self.rig.target_parents);
        Ok(PreparedIdentity {
            rest_vertices,
            bind_world,
            public_local,
            target_local,
            parameters,
        })
    }

    fn skin(
        &self,
        vertices: Tensor<B, 3>,
        transforms: &[Vec<Mat4>],
        bind: &[Mat4],
        weights: &str,
    ) -> Tensor<B, 3> {
        let batch = transforms.len();
        let joints = bind.len();
        let mut flat = Vec::with_capacity(batch * joints * 12);
        let inverses: Vec<_> = bind.iter().map(|m| m.inverse()).collect();
        for frames in transforms {
            for (pose, inverse) in frames.iter().zip(&inverses) {
                let rows = (*pose * *inverse).transpose().to_cols_array();
                flat.extend_from_slice(&rows[..12]);
            }
        }
        let delta = Tensor::<B, 3>::from_data(
            TensorData::new(flat, [batch, joints, 12]),
            &self.weights.device,
        );
        let blended = self
            .weights
            .tensor::<2>(weights)
            .unsqueeze::<3>()
            .expand([batch, 18056, joints])
            .matmul(delta)
            .reshape([batch, 18056, 3, 4]);
        let homogeneous = Tensor::cat(
            vec![
                vertices,
                Tensor::ones([batch, 18056, 1], &self.weights.device),
            ],
            2,
        )
        .unsqueeze_dim::<4>(2);
        (blended * homogeneous)
            .sum_dim(3)
            .reshape([batch, 18056, 3])
    }

    /// Evaluate a batch of poses using GPU correctives and GPU linear blend skinning.
    pub fn pose_batch(
        &self,
        identity: &PreparedIdentity<B>,
        poses: &[SomaPose],
    ) -> Result<SomaOutput<B>> {
        ensure!(
            !poses.is_empty() && poses.len() <= 256,
            "SOMA pose batch must contain 1..256 poses"
        );
        let batch = poses.len();
        let mut full_scales = [1f32; 78];
        for (&i, &scale) in self
            .rig
            .scale_public_indices
            .iter()
            .zip(&identity.parameters.bone_scales)
        {
            full_scales[i] = scale;
        }
        let mut public_translations: Vec<_> = identity
            .public_local
            .iter()
            .enumerate()
            .map(|(i, m)| position(*m) * full_scales[i])
            .collect();
        let target_translations: Vec<_> = identity
            .target_local
            .iter()
            .enumerate()
            .map(|(i, m)| position(*m) * full_scales[self.rig.scale_target_map[i]])
            .collect();
        let mut outputs = vec![];
        let mut skin_transforms = vec![];
        let mut corrective_input = Vec::with_capacity(batch * 468);
        for pose in poses {
            ensure!(
                pose.rotations.len() == 77
                    && pose
                        .rotations
                        .iter()
                        .flatten()
                        .chain(&pose.translation)
                        .all(|v| v.is_finite()),
                "Invalid SOMA pose values"
            );
            let mut rotations = vec![Mat3::IDENTITY];
            rotations.extend(
                pose.rotations
                    .iter()
                    .map(|r| Mat3::from_quat(Quat::from_scaled_axis(Vec3::from_array(*r)))),
            );
            let absolute = if pose.absolute_pose {
                rotations
            } else {
                self.rig.procedural.orient(&rotations)
            };
            public_translations[1] = Vec3::from_array(pose.translation);
            let world = fk(&absolute, &public_translations, &self.rig.public_parents);
            skin_transforms.push(self.rig.procedural.expand(
                &world,
                &target_translations,
                &identity.target_local,
                &self.rig.target_parents,
            ));
            outputs.push(world);
            for (j, r) in absolute.iter().enumerate() {
                let mut residual = matrix3(self.rig.corrective_bind[j]).transpose() * *r;
                residual.x_axis.x -= 1.0;
                residual.y_axis.y -= 1.0;
                for row in 0..3 {
                    for col in 0..2 {
                        corrective_input.push(if pose.apply_correctives {
                            residual.col(col)[row]
                        } else {
                            0.0
                        });
                    }
                }
            }
        }
        let mut vertices = identity
            .rest_vertices
            .clone()
            .unsqueeze::<3>()
            .expand([batch, 18056, 3]);
        if poses.iter().any(|p| p.apply_correctives) {
            let input = Tensor::<B, 2>::from_data(
                TensorData::new(corrective_input, [batch, 468]),
                &self.weights.device,
            );
            let mut z = activation::relu(self.weights.linear("corrective.input.weight", input));
            if self.rig.corrective_tanh {
                z = z.tanh();
            }
            let mut offsets = Tensor::<B, 2>::zeros([batch, 54168], &self.weights.device);
            for group in &self.rig.corrective_groups {
                let start = group.joint * 24;
                let values = self.weights.linear(
                    &format!("corrective.output.{}.weight", group.joint),
                    z.clone().slice([0..batch, start..start + 24]),
                );
                offsets = offsets.select_assign(
                    1,
                    self.corrective_indices[&group.joint].clone(),
                    values,
                    IndexingUpdateOp::Add,
                );
            }
            vertices =
                vertices + offsets.reshape([batch, 18056, 3]) * identity.parameters.global_scale;
        }
        let vertices = self.skin(
            vertices,
            &skin_transforms,
            &identity.bind_world,
            "skin.weights",
        );
        Ok(SomaOutput {
            vertices,
            transforms: outputs,
        })
    }
}

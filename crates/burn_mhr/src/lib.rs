//! Momentum Human Rig evaluation on Burn, using the public SOMA-X LOD1 checkpoint.
//! Native MHR coordinates are centimetres, +Y up. SOMA adapters convert at their boundary.
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData, activation},
};
use burn_human_inference::{
    transport::ModelSource,
    weights::{TensorBank, read_tensors},
};
use burn_human_motion::artifacts::Manifest;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHECKPOINT_SHA256: &str =
    "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc";
pub const MODEL_REVISION: &str = "104578ed58857f6faa7592fb83d0a2dad43c36fa";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MhrInput {
    pub identity: Vec<f32>,
    pub parameters: Vec<f32>,
    pub expression: Vec<f32>,
    pub correctives: bool,
}
impl Default for MhrInput {
    fn default() -> Self {
        Self {
            identity: vec![0.0; 45],
            parameters: vec![0.0; 204],
            expression: vec![0.0; 72],
            correctives: true,
        }
    }
}
impl MhrInput {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.identity.len() == 45
                && self.parameters.len() == 204
                && self.expression.len() == 72,
            "Invalid MHR parameter counts"
        );
        ensure!(
            self.identity
                .iter()
                .chain(&self.parameters)
                .chain(&self.expression)
                .all(|v| v.is_finite() && v.abs() < 100.0),
            "Invalid MHR parameter values"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rig {
    parents: Vec<i32>,
    translations: Vec<[f32; 3]>,
    prerotations: Vec<[f32; 4]>,
    inverse_bind: Vec<[f32; 8]>,
}

/// Convert the checkpoint's [translation XYZ, quaternion XYZW, uniform scale].
pub fn state_to_matrix(state: [f32; 8]) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::splat(state[7]),
        Quat::from_xyzw(state[3], state[4], state[5], state[6]),
        Vec3::new(state[0], state[1], state[2]),
    )
}
pub struct MhrOutput<B: Backend> {
    pub vertices: Tensor<B, 3>,
    pub skeleton_world: Vec<Vec<Mat4>>,
}
pub struct Mhr<B: Backend> {
    weights: TensorBank<B>,
    rig: Rig,
    parameters: Vec<f32>,
    inverse_bind: Vec<Mat4>,
    pub faces: Vec<[u32; 3]>,
}
impl<B: Backend> Mhr<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == "facebookresearch/MHR:SOMA-X-lod1"
                && manifest.model_revision == MODEL_REVISION
                && manifest.source_revision == MODEL_REVISION,
            "Unsupported MHR checkpoint"
        );
        ensure!(
            manifest.config
                == serde_json::json!({"vertices":18439,"joints":127,"identity":45,"expression":72,"parameters":204,"corrective_hidden":3000,"corrective_rows_per_object":4096,"unit":"centimeters"}),
            "Unsupported MHR architecture"
        );
        let asset = manifest
            .assets
            .iter()
            .find(|a| a.path == "metadata/rig.json")
            .ok_or_else(|| anyhow::anyhow!("Missing MHR skeleton"))?;
        let rig: Rig = serde_json::from_slice(&source.asset(asset).await?)?;
        ensure!(
            rig.parents.len() == 127
                && rig.parents[0] == -1
                && rig
                    .parents
                    .iter()
                    .enumerate()
                    .skip(1)
                    .all(|(i, p)| *p >= 0 && (*p as usize) < i)
                && rig.translations.len() == 127
                && rig.prerotations.len() == 127
                && rig.inverse_bind.len() == 127,
            "Invalid MHR rig topology"
        );
        ensure!(
            rig.translations
                .iter()
                .flatten()
                .chain(rig.prerotations.iter().flatten())
                .chain(rig.inverse_bind.iter().flatten())
                .all(|v| v.is_finite()),
            "Non-finite MHR skeleton"
        );
        let mut expected: BTreeMap<String, (Vec<usize>, &str)> = [
            ("identity.mean", vec![55317], "f32"),
            ("identity.directions", vec![55317, 45], "f32"),
            ("expression.directions", vec![55317, 72], "f32"),
            ("joint.parameters", vec![889, 204], "f32"),
            ("skin.weights", vec![18439, 127], "f32"),
            ("mesh.faces", vec![36874, 3], "i32"),
            ("corrective.input.weight", vec![3000, 750], "f32"),
        ]
        .into_iter()
        .map(|(n, s, d)| (n.into(), (s, d)))
        .collect();
        for i in 0..14 {
            expected.insert(
                format!("corrective.output.{i}.weight"),
                (vec![(55317 - i * 4096).min(4096), 3000], "f32"),
            );
        }
        let mut actual = BTreeMap::new();
        for object in &manifest.objects {
            for t in &object.tensors {
                ensure!(
                    actual
                        .insert(t.name.clone(), (t.shape.clone(), t.dtype.as_str()))
                        .is_none(),
                    "Duplicate MHR tensor"
                );
            }
        }
        ensure!(actual == expected, "MHR tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        let mut parameters = vec![];
        let mut faces = vec![];
        for (i, object) in manifest.objects.iter().enumerate() {
            for (name, data) in read_tensors(source, object).await? {
                if name == "joint.parameters" {
                    parameters = data
                        .to_vec::<f32>()
                        .map_err(|e| anyhow::anyhow!("parameters: {e}"))?;
                } else if name == "mesh.faces" {
                    let values = data
                        .to_vec::<i32>()
                        .map_err(|e| anyhow::anyhow!("faces: {e}"))?;
                    ensure!(
                        values.iter().all(|i| *i >= 0 && *i < 18439),
                        "MHR face index bounds"
                    );
                    faces = values
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                        .collect();
                } else {
                    weights.insert(name, data)?;
                }
            }
            progress(i + 1, manifest.objects.len());
        }
        let inverse_bind = rig
            .inverse_bind
            .iter()
            .copied()
            .map(state_to_matrix)
            .collect();
        Ok(Self {
            weights,
            rig,
            parameters,
            inverse_bind,
            faces,
        })
    }

    pub fn evaluate(&self, inputs: &[MhrInput]) -> Result<MhrOutput<B>> {
        ensure!(
            !inputs.is_empty() && inputs.len() <= 128,
            "MHR batch must contain 1..128 subjects"
        );
        for input in inputs {
            input.validate()?;
        }
        let batch = inputs.len();
        let device = &self.weights.device;
        let identity = Tensor::<B, 2>::from_data(
            TensorData::new(
                inputs
                    .iter()
                    .flat_map(|p| p.identity.clone())
                    .collect::<Vec<_>>(),
                [batch, 45],
            ),
            device,
        );
        let expressions = Tensor::<B, 2>::from_data(
            TensorData::new(
                inputs
                    .iter()
                    .flat_map(|p| p.expression.clone())
                    .collect::<Vec<_>>(),
                [batch, 72],
            ),
            device,
        );
        let mut vertices = self.weights.linear("identity.directions", identity)
            + self.weights.tensor::<1>("identity.mean").unsqueeze()
            + self.weights.linear("expression.directions", expressions);
        let mut features = Vec::with_capacity(batch * 750);
        let mut skin = Vec::with_capacity(batch * 127 * 12);
        let mut skeleton_world = vec![];
        for input in inputs {
            let params: Vec<f32> = self
                .parameters
                .as_chunks::<204>()
                .0
                .iter()
                .map(|row| row.iter().zip(&input.parameters).map(|(a, b)| a * b).sum())
                .collect();
            let mut world = vec![Mat4::IDENTITY; 127];
            for j in 0..127 {
                let p = &params[j * 7..j * 7 + 7];
                let r = Quat::from_rotation_z(p[5])
                    * Quat::from_rotation_y(p[4])
                    * Quat::from_rotation_x(p[3]);
                let r = Quat::from_array(self.rig.prerotations[j]) * r;
                let t = Vec3::from_slice(&p[..3]) + Vec3::from_array(self.rig.translations[j]);
                let s = (p[6] * std::f32::consts::LN_2).exp();
                let local = Mat4::from_scale_rotation_translation(Vec3::splat(s), r, t);
                world[j] = if j == 0 {
                    local
                } else {
                    world[self.rig.parents[j] as usize] * local
                };
                let delta = (world[j] * self.inverse_bind[j])
                    .transpose()
                    .to_cols_array();
                skin.extend_from_slice(&delta[..12]);
                if j >= 2 {
                    let (sx, cx) = p[3].sin_cos();
                    let (sy, cy) = p[4].sin_cos();
                    let (sz, cz) = p[5].sin_cos();
                    let values = [
                        cy * cz - 1.0,
                        cy * sz,
                        -sy,
                        -cx * sz + sx * sy * cz,
                        cx * cz + sx * sy * sz - 1.0,
                        sx * cy,
                    ];
                    features.extend(values.map(|v| if input.correctives { v } else { 0.0 }));
                }
            }
            ensure!(world.iter().all(|m| m.is_finite()), "MHR skeleton overflow");
            skeleton_world.push(world);
        }
        if inputs.iter().any(|p| p.correctives) {
            let input = Tensor::<B, 2>::from_data(TensorData::new(features, [batch, 750]), device);
            let z = activation::relu(self.weights.linear("corrective.input.weight", input));
            let chunks = (0..14)
                .map(|i| {
                    self.weights
                        .linear(&format!("corrective.output.{i}.weight"), z.clone())
                })
                .collect();
            vertices = vertices + Tensor::cat(chunks, 1);
        }
        let vertices = vertices.reshape([batch, 18439, 3]);
        let skin = Tensor::<B, 3>::from_data(TensorData::new(skin, [batch, 127, 12]), device);
        let blended = self
            .weights
            .tensor::<2>("skin.weights")
            .unsqueeze::<3>()
            .expand([batch, 18439, 127])
            .matmul(skin)
            .reshape([batch, 18439, 3, 4]);
        let homogeneous = Tensor::cat(vec![vertices, Tensor::ones([batch, 18439, 1], device)], 2)
            .unsqueeze_dim::<4>(2);
        let vertices = (blended * homogeneous)
            .sum_dim(3)
            .reshape([batch, 18439, 3]);
        Ok(MhrOutput {
            vertices,
            skeleton_world,
        })
    }
}

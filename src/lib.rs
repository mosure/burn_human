//! burn_human: parametric human forward model (Anny) for Burn.
//!
//! This crate is currently a scaffold. It mirrors the module layout needed
//! to port the Python implementation stage by stage while keeping tests and
//! assets aligned with burn_depth / burn_dino conventions.

pub mod data;
pub mod model;
pub mod util;

use std::collections::HashMap;

use crate::data::reference::{
    ReferenceBundle, ReferenceCase, TensorData, load_reference_bundle,
    load_reference_bundle_from_bytes,
};
use anyhow::{Context, Result, bail};
use model::{kinematics, phenotype::PhenotypeEvaluator, skinning};

/// Lightweight placeholder for the eventual loaded model.
#[derive(Debug, Clone, Default)]
pub struct AnnyBodyPlaceholder;

impl AnnyBodyPlaceholder {
    /// Construct an empty placeholder instance.
    pub fn new() -> Self {
        Self
    }

    /// Stub forward pass. This will be replaced by the full pipeline once the
    /// data format and reference tests are in place.
    pub fn forward(&self) {
        // no-op placeholder
    }
}

/// Inference-only model backed by safetensors reference data.
#[derive(Debug, Clone)]
pub struct AnnyReference {
    bundle: ReferenceBundle,
    cases_by_name: HashMap<String, ReferenceCase>,
    phenotype_eval: PhenotypeEvaluator,
}

#[derive(Debug, Clone)]
pub struct AnnyOutput {
    pub rest_vertices: TensorData<f64>,
    pub posed_vertices: TensorData<f64>,
    pub rest_bone_poses: TensorData<f64>,
    pub bone_poses: TensorData<f64>,
    pub bone_heads: TensorData<f64>,
    pub bone_tails: TensorData<f64>,
    pub blendshape_weights: TensorData<f64>,
    pub phenotype_inputs: TensorData<f64>,
    pub pose_parameters: TensorData<f64>,
}

impl AnnyReference {
    /// Load reference safetensors + metadata from disk.
    pub fn from_paths(
        tensor_path: impl AsRef<std::path::Path>,
        meta_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let bundle = load_reference_bundle(tensor_path, meta_path)?;
        Self::new(bundle)
    }

    /// Load reference safetensors + metadata from in-memory bytes (useful for wasm).
    pub fn from_bytes(tensor_bytes: &[u8], meta_bytes: &[u8]) -> Result<Self> {
        let bundle = load_reference_bundle_from_bytes(tensor_bytes, meta_bytes)?;
        Self::new(bundle)
    }

    fn new(bundle: ReferenceBundle) -> Result<Self> {
        let mut cases_by_name = HashMap::new();
        for case in bundle.cases.iter() {
            cases_by_name.insert(case.name.clone(), case.clone());
        }
        let phenotype_eval = PhenotypeEvaluator {
            phenotype_labels: bundle.metadata.phenotype_labels.clone(),
            macrodetail_keys: bundle.metadata.macrodetail_keys.clone(),
            anchors: bundle.metadata.phenotype_anchors.clone(),
            variations: bundle.metadata.phenotype_variations.clone(),
            mask: bundle.static_data.blendshape_mask.clone(),
        };
        Ok(Self {
            bundle,
            cases_by_name,
            phenotype_eval,
        })
    }

    /// Run a forward pass for a known reference case by name.
    pub fn forward_case(&self, name: &str) -> Result<AnnyOutput> {
        self.forward(AnnyInput::case(name))
    }

    /// Forward pass with custom parameters or reference cases.
    pub fn forward(&self, input: AnnyInput<'_>) -> Result<AnnyOutput> {
        let case = if let Some(name) = input.case_name {
            Some(
                self.cases_by_name
                    .get(name)
                    .context("reference case not found")?,
            )
        } else {
            None
        };

        let blend_count = self.bundle.static_data.blendshapes.shape[0];
        let bone_count = self.bundle.static_data.template_bone_heads.shape[0];
        let phenotype_len = self.bundle.metadata.phenotype_labels.len();

        let phenotype_inputs = if let Some(vals) = input.phenotype_inputs {
            ensure_len(vals.len(), phenotype_len, "phenotype_inputs")?;
            TensorData {
                shape: vec![1, phenotype_len],
                data: vals.to_vec(),
            }
        } else if let Some(case) = case {
            case.phenotype_inputs.clone()
        } else {
            TensorData {
                shape: vec![1, phenotype_len],
                data: vec![0.5; phenotype_len],
            }
        };

        let mut blendshape_weights = match input.blendshape_weights {
            Some(weights) => {
                ensure_len(weights.len(), blend_count, "blendshape_weights")?;
                TensorData {
                    shape: vec![1, blend_count],
                    data: weights.to_vec(),
                }
            }
            None => match (&input.phenotype_inputs, &case) {
                (_, None) | (Some(_), _) => self.phenotype_eval.weights(&phenotype_inputs)?,
                (None, Some(c)) => c.blendshape_coeffs.clone(),
            },
        };
        if let Some(delta) = input.blendshape_delta {
            ensure_len(
                delta.len(),
                blendshape_weights.data.len(),
                "blendshape_delta",
            )?;
            for (w, d) in blendshape_weights.data.iter_mut().zip(delta.iter()) {
                *w = (*w + d).clamp(0.0, 1.0);
            }
        }

        let mut pose_parameters = if let Some(pose) = input.pose_parameters {
            ensure_len(pose.len(), bone_count * 16, "pose_parameters")?;
            TensorData {
                shape: vec![1, bone_count, 4, 4],
                data: pose.to_vec(),
            }
        } else if let Some(case) = case {
            case.pose_parameters.clone()
        } else {
            identity_pose_parameters(bone_count)
        };
        if let Some(delta) = input.pose_parameters_delta {
            ensure_len(
                delta.len(),
                pose_parameters.data.len(),
                "pose_parameters_delta",
            )?;
            for (p, d) in pose_parameters.data.iter_mut().zip(delta.iter()) {
                *p += d;
            }
        }
        if let Some(delta_t) = input.root_translation_delta {
            // indices 3,7,11 in row-major 4x4 for translation
            let base = 0;
            if pose_parameters.data.len() >= 12 {
                pose_parameters.data[base + 3] += delta_t[0];
                pose_parameters.data[base + 7] += delta_t[1];
                pose_parameters.data[base + 11] += delta_t[2];
            }
        }

        let rest_vertices = model::blendshape::apply_blendshapes(
            &self.bundle.static_data.template_vertices,
            &self.bundle.static_data.blendshapes,
            &blendshape_weights,
        )?;
        let rest_bone_heads = model::blendshape::apply_bone_blendshapes(
            &self.bundle.static_data.template_bone_heads,
            &self.bundle.static_data.bone_heads_blendshapes,
            &blendshape_weights,
        )?;
        let rest_bone_tails = model::blendshape::apply_bone_blendshapes(
            &self.bundle.static_data.template_bone_tails,
            &self.bundle.static_data.bone_tails_blendshapes,
            &blendshape_weights,
        )?;
        let rest_bone_poses = kinematics::rest_bone_poses_from_heads_tails(
            &rest_bone_heads,
            &rest_bone_tails,
            &self.bundle.static_data.bone_rolls_rotmat,
        )?;
        let (bone_poses, bone_transforms) = kinematics::forward_root_relative_world(
            &rest_bone_poses,
            &pose_parameters,
            &self.bundle.metadata.bone_parents,
        )?;
        let posed_vertices = skinning::linear_blend_skinning(
            &rest_vertices,
            &bone_poses,
            &rest_bone_poses,
            &self.bundle.static_data.vertex_bone_indices,
            &self.bundle.static_data.vertex_bone_weights,
        )?;
        let bone_heads = transform_points(&rest_bone_heads, &bone_transforms)?;
        let bone_tails = transform_points(&rest_bone_tails, &bone_transforms)?;
        Ok(AnnyOutput {
            rest_vertices,
            posed_vertices,
            rest_bone_poses,
            bone_poses,
            bone_heads,
            bone_tails,
            blendshape_weights,
            phenotype_inputs,
            pose_parameters,
        })
    }

    /// Access metadata (e.g., case names) for driving tests.
    pub fn case_names(&self) -> impl Iterator<Item = &str> {
        self.bundle.metadata.case_names.iter().map(|s| s.as_str())
    }

    pub fn metadata(&self) -> &ReferenceBundle {
        &self.bundle
    }

    pub fn phenotype_evaluator(&self) -> &PhenotypeEvaluator {
        &self.phenotype_eval
    }
}

/// Public-facing inference-only model (backed by reference data for now).
#[derive(Debug, Clone)]
pub struct AnnyBody {
    reference: AnnyReference,
}

#[derive(Debug, Clone, Default)]
pub struct AnnyInput<'a> {
    /// Select a precomputed reference case by name (from metadata.case_names).
    pub case_name: Option<&'a str>,
    /// Override phenotype inputs [P]; if None, use case (if provided) or mid anchors.
    pub phenotype_inputs: Option<&'a [f64]>,
    /// Directly provide blendshape weights [N]; if None, derived from phenotype or case.
    pub blendshape_weights: Option<&'a [f64]>,
    /// Blendshape deltas added to the base weights and clamped to [0, 1].
    pub blendshape_delta: Option<&'a [f64]>,
    /// Pose parameters [J*16] row-major; if None, use case (if provided) or identity.
    pub pose_parameters: Option<&'a [f64]>,
    /// Pose deltas added to the base pose parameters.
    pub pose_parameters_delta: Option<&'a [f64]>,
    /// Root translation delta applied to pose parameters.
    pub root_translation_delta: Option<[f64; 3]>,
}

impl<'a> AnnyInput<'a> {
    pub fn case(name: &'a str) -> Self {
        Self {
            case_name: Some(name),
            ..Default::default()
        }
    }
}

impl AnnyBody {
    /// Load an inference-only model from reference safetensors + metadata.
    pub fn from_reference_paths(
        tensor_path: impl AsRef<std::path::Path>,
        meta_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        Ok(Self {
            reference: AnnyReference::from_paths(tensor_path, meta_path)?,
        })
    }

    /// Load an inference-only model from reference bytes (for embedded/wasm usage).
    pub fn from_reference_bytes(
        tensor_bytes: &[u8],
        meta_bytes: &[u8],
    ) -> Result<Self> {
        Ok(Self {
            reference: AnnyReference::from_bytes(tensor_bytes, meta_bytes)?,
        })
    }

    /// Forward pass using reference cases or custom inputs.
    pub fn forward(&self, input: AnnyInput<'_>) -> Result<AnnyOutput> {
        self.reference.forward(input)
    }

    /// Forward pass with optional blendshape and root translation offsets.
    pub fn forward_with_offsets(
        &self,
        case_name: &str,
        blendshape_delta: Option<&[f64]>,
        root_translation_delta: Option<[f64; 3]>,
        pose_parameters_delta: Option<&[f64]>,
    ) -> Result<AnnyOutput> {
        self.reference.forward(AnnyInput {
            case_name: Some(case_name),
            blendshape_delta,
            root_translation_delta,
            pose_parameters_delta,
            ..Default::default()
        })
    }

    /// Backwards-compatible convenience for case-only forward passes.
    pub fn forward_case(&self, case_name: &str) -> Result<AnnyOutput> {
        self.reference.forward_case(case_name)
    }

    /// Quad faces (topology helper).
    pub fn faces_quads(&self) -> &TensorData<i64> {
        &self.reference.bundle.static_data.faces_quads
    }

    /// Reference case names (for testing/benching).
    pub fn case_names(&self) -> impl Iterator<Item = &str> {
        self.reference.case_names()
    }

    /// Vertex bone indices/weights (skinning helper).
    pub fn skinning_bindings(&self) -> (&TensorData<i64>, &TensorData<f64>) {
        (
            &self.reference.bundle.static_data.vertex_bone_indices,
            &self.reference.bundle.static_data.vertex_bone_weights,
        )
    }

    /// Bone labels (matching joint order) and parent indices (-1 for root).
    pub fn bone_hierarchy(&self) -> (&[String], &[i64]) {
        (
            &self.reference.bundle.metadata.bone_labels,
            &self.reference.bundle.metadata.bone_parents,
        )
    }

    /// Template/rest vertices from static data.
    pub fn template_vertices(&self) -> &TensorData<f64> {
        &self.reference.bundle.static_data.template_vertices
    }

    /// Full reference bundle (static data + metadata) access.
    pub fn metadata(&self) -> &ReferenceBundle {
        self.reference.metadata()
    }

    pub fn phenotype_evaluator(&self) -> &PhenotypeEvaluator {
        self.reference.phenotype_evaluator()
    }
}

fn ensure_len(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual != expected {
        bail!(
            "{} length mismatch: expected {}, got {}",
            label,
            expected,
            actual
        );
    }
    Ok(())
}

fn identity_pose_parameters(bones: usize) -> TensorData<f64> {
    let mut data = vec![0.0; bones * 16];
    for b in 0..bones {
        let base = b * 16;
        data[base] = 1.0;
        data[base + 5] = 1.0;
        data[base + 10] = 1.0;
        data[base + 15] = 1.0;
    }
    TensorData {
        shape: vec![1, bones, 4, 4],
        data,
    }
}

fn transform_points(
    points: &TensorData<f64>,     // [B,J,3]
    transforms: &TensorData<f64>, // [B,J,4,4]
) -> Result<TensorData<f64>> {
    if points.shape.len() != 3 || points.shape[2] != 3 {
        bail!("points must be [B,J,3]");
    }
    if transforms.shape.len() != 4
        || transforms.shape[0] != points.shape[0]
        || transforms.shape[1] != points.shape[1]
        || transforms.shape[2] != 4
        || transforms.shape[3] != 4
    {
        bail!("transforms must be [B,J,4,4] matching points batch/bones");
    }
    let batch = points.shape[0];
    let bones = points.shape[1];
    let mut out = TensorData {
        shape: points.shape.clone(),
        data: vec![0.0; points.data.len()],
    };
    for b in 0..batch {
        for j in 0..bones {
            let p_idx = (b * bones + j) * 3;
            let px = points.data[p_idx];
            let py = points.data[p_idx + 1];
            let pz = points.data[p_idx + 2];
            let t_idx = (b * bones + j) * 16;
            let m = &transforms.data[t_idx..t_idx + 16];
            let x = m[0] * px + m[1] * py + m[2] * pz + m[3];
            let y = m[4] * px + m[5] * py + m[6] * pz + m[7];
            let z = m[8] * px + m[9] * py + m[10] * pz + m[11];
            out.data[p_idx] = x;
            out.data[p_idx + 1] = y;
            out.data[p_idx + 2] = z;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::AnnyBodyPlaceholder;

    #[test]
    fn placeholder_constructs() {
        let _model = AnnyBodyPlaceholder::new();
    }
}

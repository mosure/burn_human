use anyhow::{Context, Result, bail};
use safetensors::{SafeTensors, tensor::Dtype, tensor::TensorView};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceMetadata {
    pub seed: u64,
    pub rig: String,
    pub topology: String,
    pub case_names: Vec<String>,
    pub bone_labels: Vec<String>,
    pub bone_parents: Vec<i64>,
    pub phenotype_labels: Vec<String>,
    pub pose_parameterization: String,
    pub phenotype_variations: std::collections::HashMap<String, Vec<String>>,
    pub macrodetail_keys: Vec<String>,
    pub phenotype_anchors: std::collections::HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct ReferenceCase {
    pub name: String,
    pub phenotype_inputs: TensorData<f64>,
    pub blendshape_coeffs: TensorData<f64>,
    pub pose_parameters: TensorData<f64>,
    pub rest_vertices: TensorData<f64>,
    pub posed_vertices: TensorData<f64>,
    pub rest_bone_poses: TensorData<f64>,
    pub bone_poses: TensorData<f64>,
    pub bone_heads: TensorData<f64>,
    pub bone_tails: TensorData<f64>,
}

#[derive(Debug, Clone)]
pub struct ReferenceStatic {
    pub faces_quads: TensorData<i64>,
    pub face_texture_coordinate_indices: TensorData<i64>,
    pub template_vertices: TensorData<f64>,
    pub texture_coordinates: TensorData<f64>,
    pub vertex_bone_indices: TensorData<i64>,
    pub vertex_bone_weights: TensorData<f64>,
    pub blendshapes: TensorData<f64>,
    pub blendshape_mask: TensorData<f64>,
    pub template_bone_heads: TensorData<f64>,
    pub bone_heads_blendshapes: TensorData<f64>,
    pub template_bone_tails: TensorData<f64>,
    pub bone_tails_blendshapes: TensorData<f64>,
    pub bone_rolls_rotmat: TensorData<f64>,
}

#[derive(Debug, Clone)]
pub struct ReferenceBundle {
    pub metadata: ReferenceMetadata,
    pub static_data: ReferenceStatic,
    pub cases: Vec<ReferenceCase>,
}

fn tensor_to_vec<T, const N: usize, F>(t: TensorView, dtype: Dtype, convert: F) -> Result<TensorData<T>>
where
    F: Fn([u8; N]) -> T,
{
    if t.dtype() != dtype {
        bail!("expected {dtype:?} tensor, got {:?}", t.dtype());
    }
    let mut chunks = t.data().chunks_exact(N);
    let data = chunks
        .by_ref()
        .map(|chunk| {
            let mut bytes = [0u8; N];
            bytes.copy_from_slice(chunk);
            convert(bytes)
        })
        .collect();
    if !chunks.remainder().is_empty() {
        bail!("tensor byte length not divisible by {}", N);
    }
    Ok(TensorData {
        shape: t.shape().to_vec(),
        data,
    })
}

fn tensor_to_vec_f64(t: TensorView) -> Result<TensorData<f64>> {
    tensor_to_vec(t, Dtype::F64, f64::from_le_bytes)
}

fn tensor_to_vec_i64(t: TensorView) -> Result<TensorData<i64>> {
    tensor_to_vec(t, Dtype::I64, i64::from_le_bytes)
}

fn load_metadata(meta_path: &Path) -> Result<ReferenceMetadata> {
    let bytes = std::fs::read(meta_path).with_context(|| format!("reading {:?}", meta_path))?;
    let meta: ReferenceMetadata =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {:?}", meta_path))?;
    Ok(meta)
}

fn load_safetensors(path: &Path) -> Result<SafeTensors<'static>> {
    let data = std::fs::read(path).with_context(|| format!("reading {:?}", path))?;
    let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
    SafeTensors::deserialize(leaked).context("deserializing safetensors")
}

fn load_metadata_from_bytes(bytes: &'static [u8]) -> Result<ReferenceMetadata> {
    let meta: ReferenceMetadata =
        serde_json::from_slice(bytes).context("parsing in-memory metadata bytes")?;
    Ok(meta)
}

fn load_safetensors_from_bytes(bytes: &'static [u8]) -> Result<SafeTensors<'static>> {
    SafeTensors::deserialize(bytes).context("deserializing in-memory safetensors")
}

pub fn load_reference_bundle(
    tensor_path: impl AsRef<Path>,
    meta_path: impl AsRef<Path>,
) -> Result<ReferenceBundle> {
    let tensor_path = tensor_path.as_ref();
    let meta_path = meta_path.as_ref();
    let meta = load_metadata(meta_path)?;
    let safes = load_safetensors(tensor_path)?;
    load_reference_bundle_impl(meta, safes)
}

pub fn load_reference_bundle_from_bytes(
    tensor_bytes: &'static [u8],
    meta_bytes: &'static [u8],
) -> Result<ReferenceBundle> {
    let meta = load_metadata_from_bytes(meta_bytes)?;
    let safes = load_safetensors_from_bytes(tensor_bytes)?;
    load_reference_bundle_impl(meta, safes)
}

fn load_reference_bundle_impl(
    meta: ReferenceMetadata,
    safes: SafeTensors<'static>,
) -> Result<ReferenceBundle> {
    let static_data = ReferenceStatic {
        faces_quads: tensor_to_vec_i64(
            safes
                .tensor("faces_quads")
                .context("faces_quads tensor missing")?,
        )?,
        face_texture_coordinate_indices: tensor_to_vec_i64(
            safes
                .tensor("face_texture_coordinate_indices")
                .context("face_texture_coordinate_indices tensor missing")?,
        )?,
        template_vertices: tensor_to_vec_f64(
            safes
                .tensor("template_vertices")
                .context("template_vertices tensor missing")?,
        )?,
        texture_coordinates: tensor_to_vec_f64(
            safes
                .tensor("texture_coordinates")
                .context("texture_coordinates tensor missing")?,
        )?,
        vertex_bone_indices: tensor_to_vec_i64(
            safes
                .tensor("vertex_bone_indices")
                .context("vertex_bone_indices tensor missing")?,
        )?,
        vertex_bone_weights: tensor_to_vec_f64(
            safes
                .tensor("vertex_bone_weights")
                .context("vertex_bone_weights tensor missing")?,
        )?,
        blendshapes: tensor_to_vec_f64(
            safes
                .tensor("blendshapes")
                .context("blendshapes tensor missing")?,
        )?,
        blendshape_mask: tensor_to_vec_f64(
            safes
                .tensor("blendshape_mask")
                .context("blendshape_mask tensor missing")?,
        )?,
        template_bone_heads: tensor_to_vec_f64(
            safes
                .tensor("template_bone_heads")
                .context("template_bone_heads tensor missing")?,
        )?,
        bone_heads_blendshapes: tensor_to_vec_f64(
            safes
                .tensor("bone_heads_blendshapes")
                .context("bone_heads_blendshapes tensor missing")?,
        )?,
        template_bone_tails: tensor_to_vec_f64(
            safes
                .tensor("template_bone_tails")
                .context("template_bone_tails tensor missing")?,
        )?,
        bone_tails_blendshapes: tensor_to_vec_f64(
            safes
                .tensor("bone_tails_blendshapes")
                .context("bone_tails_blendshapes tensor missing")?,
        )?,
        bone_rolls_rotmat: tensor_to_vec_f64(
            safes
                .tensor("bone_rolls_rotmat")
                .context("bone_rolls_rotmat tensor missing")?,
        )?,
    };

    let mut cases = Vec::with_capacity(meta.case_names.len());
    for name in &meta.case_names {
        let mk = |suffix: &str| format!("{name}__{suffix}");
        let case = ReferenceCase {
            name: name.clone(),
            phenotype_inputs: tensor_to_vec_f64(
                safes
                    .tensor(&mk("phenotype_inputs"))
                    .context("phenotype_inputs missing")?,
            )?,
            blendshape_coeffs: tensor_to_vec_f64(
                safes
                    .tensor(&mk("blendshape_coeffs"))
                    .context("blendshape_coeffs missing")?,
            )?,
            pose_parameters: tensor_to_vec_f64(
                safes
                    .tensor(&mk("pose_parameters"))
                    .context("pose_parameters missing")?,
            )?,
            rest_vertices: tensor_to_vec_f64(
                safes
                    .tensor(&mk("rest_vertices"))
                    .context("rest_vertices missing")?,
            )?,
            posed_vertices: tensor_to_vec_f64(
                safes
                    .tensor(&mk("posed_vertices"))
                    .context("posed_vertices missing")?,
            )?,
            rest_bone_poses: tensor_to_vec_f64(
                safes
                    .tensor(&mk("rest_bone_poses"))
                    .context("rest_bone_poses missing")?,
            )?,
            bone_poses: tensor_to_vec_f64(
                safes
                    .tensor(&mk("bone_poses"))
                    .context("bone_poses missing")?,
            )?,
            bone_heads: tensor_to_vec_f64(
                safes
                    .tensor(&mk("bone_heads"))
                    .context("bone_heads missing")?,
            )?,
            bone_tails: tensor_to_vec_f64(
                safes
                    .tensor(&mk("bone_tails"))
                    .context("bone_tails missing")?,
            )?,
        };
        cases.push(case);
    }

    Ok(ReferenceBundle {
        metadata: meta,
        static_data,
        cases,
    })
}

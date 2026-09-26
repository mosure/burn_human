//! The actual MHR identity backend used by GEM-X, including topology transfer
//! and the facial Laplacian solve. MHR coefficients are never treated as SOMA PCA.
use crate::{BindConvention, IdentityParameters, PreparedIdentity, Soma};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{IndexingUpdateOp, Int, Tensor},
};
use burn_human_inference::{
    transport::ModelSource,
    weights::{TensorBank, read_tensors},
};
use burn_human_motion::artifacts::Manifest;
use burn_mhr::{Mhr, MhrInput};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MhrIdentity {
    pub coefficients: Vec<f32>,
    pub scales: Vec<f32>,
    pub bone_length_flexibles: [f32; 6],
    pub global_scale: f32,
}
impl Default for MhrIdentity {
    fn default() -> Self {
        Self {
            coefficients: vec![0.0; 45],
            scales: vec![0.0; 68],
            bone_length_flexibles: [0.0; 6],
            global_scale: 1.0,
        }
    }
}
pub struct MhrSomaTransfer<B: Backend> {
    weights: TensorBank<B>,
    indices: BTreeMap<String, Tensor<B, 1, Int>>,
}
impl<B: Backend> MhrSomaTransfer<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == "nvidia/SOMA-X:MHR-transfer"
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.source_revision == crate::SOURCE_REVISION,
            "Unsupported MHR/SOMA transfer"
        );
        ensure!(
            manifest.config
                == serde_json::json!({"source_vertices":18439,"target_vertices":18056,"boundary":124,"unknown":691,"normal_scale":"area","output_unit":"meters"}),
            "Unsupported topology transfer"
        );
        let expected: BTreeMap<String, (Vec<usize>, &str)> = [
            ("transfer.source.0", vec![18056], "i32"),
            ("transfer.source.1", vec![18056], "i32"),
            ("transfer.source.2", vec![18056], "i32"),
            ("transfer.barycentric", vec![18056, 4], "f32"),
            ("transfer.boundary", vec![124], "i32"),
            ("transfer.unknown", vec![691], "i32"),
            ("transfer.solve", vec![691, 124], "f32"),
            ("transfer.bias", vec![691, 3], "f32"),
        ]
        .into_iter()
        .map(|(n, s, d)| (n.into(), (s, d)))
        .collect();
        let mut actual = BTreeMap::new();
        for o in &manifest.objects {
            for t in &o.tensors {
                ensure!(
                    actual
                        .insert(t.name.clone(), (t.shape.clone(), t.dtype.as_str()))
                        .is_none(),
                    "Duplicate transfer tensor"
                );
            }
        }
        ensure!(actual == expected, "Transfer tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        let mut indices = BTreeMap::new();
        for object in &manifest.objects {
            for (name, data) in read_tensors(source, object).await? {
                if data.dtype == burn::tensor::DType::I32 {
                    let values = data
                        .to_vec::<i32>()
                        .map_err(|e| anyhow::anyhow!("indices: {e}"))?;
                    let limit = if name.starts_with("transfer.source.") {
                        18439
                    } else {
                        18056
                    };
                    ensure!(
                        values.iter().all(|v| *v >= 0 && *v < limit),
                        "Topology index out of bounds"
                    );
                    if !name.starts_with("transfer.source.") {
                        ensure!(
                            values.windows(2).all(|v| v[0] < v[1]),
                            "Transfer indices must be sorted and unique"
                        );
                    }
                    indices.insert(name, Tensor::from_data(data, device));
                } else {
                    weights.insert(name, data)?;
                }
            }
        }
        Ok(Self { weights, indices })
    }

    /// MHR centimetres to SOMA metres; no fit-dependent coefficient is cached.
    pub fn transfer(&self, source: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, vertices, dim] = source.dims();
        ensure!(vertices == 18439 && dim == 3, "MHR source mesh shape");
        let p0 = source
            .clone()
            .select(1, self.indices["transfer.source.0"].clone());
        let p1 = source
            .clone()
            .select(1, self.indices["transfer.source.1"].clone());
        let p2 = source.select(1, self.indices["transfer.source.2"].clone());
        let a = p1.clone() - p0.clone();
        let b = p2.clone() - p0.clone();
        let yzx = Tensor::<B, 1, Int>::from_data([1i32, 2, 0], &self.weights.device);
        let zxy = Tensor::<B, 1, Int>::from_data([2i32, 0, 1], &self.weights.device);
        let normal = a.clone().select(2, yzx.clone()) * b.clone().select(2, zxy.clone())
            - a.select(2, zxy) * b.select(2, yzx);
        let p3 = p0.clone() + normal;
        let bary = self
            .weights
            .tensor::<2>("transfer.barycentric")
            .unsqueeze::<3>();
        let coefficient = |i| bary.clone().slice([0..1, 0..18056, i..i + 1]);
        let target =
            p0 * coefficient(0) + p1 * coefficient(1) + p2 * coefficient(2) + p3 * coefficient(3);
        let boundary = target
            .clone()
            .select(1, self.indices["transfer.boundary"].clone());
        let solved = self
            .weights
            .tensor::<2>("transfer.solve")
            .unsqueeze::<3>()
            .expand([batch, 691, 124])
            .matmul(boundary)
            + self.weights.tensor::<2>("transfer.bias").unsqueeze::<3>();
        let residual = solved
            - target
                .clone()
                .select(1, self.indices["transfer.unknown"].clone());
        Ok(target.select_assign(
            1,
            self.indices["transfer.unknown"].clone(),
            residual,
            IndexingUpdateOp::Add,
        ) * 0.01)
    }

    pub async fn prepare_identity(
        &self,
        model: &Mhr<B>,
        soma: &Soma<B>,
        identity: &MhrIdentity,
    ) -> Result<PreparedIdentity<B>> {
        self.prepare_identity_with_bind(model, soma, identity, BindConvention::Canonical)
            .await
    }

    pub async fn prepare_identity_with_bind(
        &self,
        model: &Mhr<B>,
        soma: &Soma<B>,
        identity: &MhrIdentity,
        convention: BindConvention,
    ) -> Result<PreparedIdentity<B>> {
        ensure!(
            identity.coefficients.len() == 45 && identity.scales.len() == 68,
            "MHR identity dimensions"
        );
        let mut input = MhrInput {
            identity: identity.coefficients.clone(),
            ..Default::default()
        };
        input.parameters[130..136].copy_from_slice(&identity.bone_length_flexibles);
        input.parameters[136..].copy_from_slice(&identity.scales);
        let shape = self.transfer(model.evaluate(&[input])?.vertices)? * identity.global_scale;
        let parameters = IdentityParameters {
            global_scale: identity.global_scale,
            ..Default::default()
        };
        soma.prepare_rest_shape_with_bind(shape.reshape([18056, 3]), parameters, convention)
            .await
    }
}

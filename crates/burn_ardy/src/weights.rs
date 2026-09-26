use crate::config::{ArdyConfig, MODEL_ID, MODEL_REVISION, expected_tensors};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Bytes, DType, Tensor, TensorData},
};
use burn_human_motion::artifacts::{Manifest, PartReader, read_object, sha256};
use burn_store::{BurnpackStore, ModuleStore};
use std::collections::{BTreeMap, BTreeSet};

pub struct Weights<B: Backend> {
    tensors: BTreeMap<String, (Tensor<B, 1>, Vec<usize>)>,
    pub device: B::Device,
}

impl<B: Backend> Weights<B> {
    pub async fn load(
        manifest: &Manifest,
        reader: &mut impl PartReader,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(Self, ArdyConfig)> {
        manifest.validate()?;
        ensure!(
            manifest.model == MODEL_ID && manifest.model_revision == MODEL_REVISION,
            "unsupported model identity"
        );
        let config: ArdyConfig = serde_json::from_value(manifest.config.clone())?;
        config.validate()?;
        let expected = expected_tensors();
        let declared: BTreeMap<_, _> = manifest
            .objects
            .iter()
            .flat_map(|o| &o.tensors)
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        ensure!(
            declared == expected,
            "checkpoint tensor inventory does not match ARDY Core architecture"
        );
        let mut tensors = BTreeMap::new();
        for (i, object) in manifest.objects.iter().enumerate() {
            let bytes = read_object(reader, object).await?;
            let mut pack = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
            let snapshots = pack
                .get_all_snapshots()
                .map_err(|e| anyhow::anyhow!("Burnpack snapshots: {e}"))?;
            let mut seen = BTreeSet::new();
            for snapshot in snapshots.values() {
                let name = snapshot
                    .path_stack
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("unnamed tensor"))?
                    .join(".");
                let spec = object
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| anyhow::anyhow!("unknown tensor {name}"))?;
                ensure!(
                    seen.insert(name.clone()) && !tensors.contains_key(&name),
                    "duplicate tensor {name}"
                );
                let data = snapshot
                    .to_data()
                    .map_err(|e| anyhow::anyhow!("tensor data: {e}"))?;
                ensure!(
                    data.dtype == DType::F32 && data.shape.as_slice() == spec.shape,
                    "tensor shape/dtype mismatch: {name}"
                );
                ensure!(
                    sha256(&data.bytes) == spec.sha256,
                    "tensor hash mismatch: {name}"
                );
                ensure!(
                    data.as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                        .iter()
                        .all(|v| v.is_finite()),
                    "non-finite weights: {name}"
                );
                let shape = spec.shape.clone();
                let size: usize = shape.iter().product();
                let flat = TensorData::from_bytes(data.bytes, [size], DType::F32);
                tensors.insert(name, (Tensor::from_data(flat, device), shape));
            }
            ensure!(
                seen.len() == object.tensors.len(),
                "missing tensors in {}",
                object.stage
            );
            // Each object, its snapshots and decoded host tensors drop before fetching the next.
            progress(i + 1, manifest.objects.len());
        }
        Ok((
            Self {
                tensors,
                device: device.clone(),
            },
            config,
        ))
    }

    pub fn tensor<const D: usize>(&self, name: &str) -> Tensor<B, D> {
        let (tensor, shape) = &self.tensors[name];
        let dims: [usize; D] = shape.as_slice().try_into().expect("validated tensor rank");
        tensor.clone().reshape(dims)
    }

    pub fn linear(&self, prefix: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear_named(&format!("{prefix}.weight"), &format!("{prefix}.bias"), x)
    }

    pub fn linear_named(&self, weight: &str, bias: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let w: Tensor<B, 2> = self.tensor(weight);
        let b: Tensor<B, 1> = self.tensor(bias);
        let [batch, time, dim] = x.dims();
        let output = w.dims()[0];
        (x.reshape([batch * time, dim]).matmul(w.transpose()) + b.unsqueeze_dim(0))
            .reshape([batch, time, output])
    }

    pub fn norm(&self, prefix: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let weight: Tensor<B, 1> = self.tensor(&format!("{prefix}.weight"));
        let bias: Tensor<B, 1> = self.tensor(&format!("{prefix}.bias"));
        centered / (var + 1e-5).sqrt() * weight.unsqueeze() + bias.unsqueeze()
    }
}

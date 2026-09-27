use crate::config::{ArdyConfig, MODEL_ID, MODEL_REVISION, expected_tensors};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{DType, Tensor, TensorData},
};
use burn_human_motion::artifacts::{Manifest, PartReader};
use std::collections::BTreeMap;

pub struct Weights<B: Backend> {
    tensors: BTreeMap<String, (Tensor<B, 1>, Vec<usize>)>,
    pub device: B::Device,
    pub(crate) constants: crate::constants::Constants<B>,
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
        let mut uploads = burn_human_inference::weights::UploadBudget::default();
        for (i, object) in manifest.objects.iter().enumerate() {
            for (name, data) in burn_human_inference::weights::read_tensors(reader, object).await? {
                let bytes = data.bytes.len();
                ensure!(
                    data.dtype == DType::F32,
                    "ARDY requires f32 weights: {name}"
                );
                ensure!(!tensors.contains_key(&name), "duplicate tensor {name}");
                let shape = data.shape.to_vec();
                let size: usize = shape.iter().product();
                let flat = TensorData::from_bytes(data.bytes, [size], DType::F32);
                tensors.insert(name, (Tensor::from_data(flat, device), shape));
                uploads.record::<B>(device, bytes).await?;
                burn_human_inference::cooperative::yield_to_browser().await;
            }
            // Each object, its snapshots and decoded host tensors drop before fetching the next.
            progress(i + 1, manifest.objects.len());
        }
        Ok((
            Self {
                tensors,
                device: device.clone(),
                constants: crate::constants::Constants::new(&config, device),
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
        (crate::ops::matmul(x.reshape([batch * time, dim]), w.transpose()) + b.unsqueeze_dim(0))
            .reshape([batch, time, output])
    }

    pub fn norm(&self, prefix: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = crate::ops::mean_dim(x.clone(), 2);
        let centered = x - mean;
        let var = crate::ops::mean_dim(centered.clone().powf_scalar(2.0), 2);
        let weight: Tensor<B, 1> = self.tensor(&format!("{prefix}.weight"));
        let bias: Tensor<B, 1> = self.tensor(&format!("{prefix}.bias"));
        centered / (var + 1e-5).sqrt() * weight.unsqueeze() + bias.unsqueeze()
    }
}

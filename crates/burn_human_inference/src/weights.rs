use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Bytes, DType, Tensor, TensorData},
};
use burn_human_motion::artifacts::{Object, PartReader, read_object, sha256};
use burn_store::{BurnpackStore, ModuleStore};
use std::collections::{BTreeMap, BTreeSet};

/// Authenticate one object and its exact tensor inventory before uploading it.
pub async fn read_tensors(
    reader: &mut impl PartReader,
    object: &Object,
) -> Result<BTreeMap<String, TensorData>> {
    let bytes = read_object(reader, object).await?;
    let mut pack = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes)));
    let snapshots = pack
        .get_all_snapshots()
        .map_err(|e| anyhow::anyhow!("Burnpack: {e}"))?;
    let mut seen = BTreeSet::new();
    let mut tensors = BTreeMap::new();
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
        ensure!(seen.insert(name.clone()), "duplicate tensor {name}");
        let data = snapshot
            .to_data()
            .map_err(|e| anyhow::anyhow!("tensor: {e}"))?;
        ensure!(
            data.dtype == crate::dtype(&spec.dtype)?
                && data.shape.as_slice() == spec.shape
                && Some(data.bytes.len()) == spec.byte_len()
                && sha256(&data.bytes) == spec.sha256,
            "tensor shape/dtype/digest mismatch: {name}"
        );
        ensure_finite(&data)?;
        tensors.insert(name, data);
    }
    ensure!(
        seen.len() == object.tensors.len(),
        "missing tensor in {}",
        object.stage
    );
    Ok(tensors)
}

pub fn ensure_finite(data: &TensorData) -> Result<()> {
    let bytes = &data.bytes;
    let finite = match data.dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|b| f32::from_le_bytes(*b).is_finite()),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| half::f16::from_bits(u16::from_le_bytes(*b)).is_finite()),
        DType::QFloat(_) => {
            let n: usize = data.shape.iter().product();
            bytes[n / 2..]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|b| f32::from_le_bytes(*b).is_finite())
        }
        _ => true,
    };
    ensure!(finite, "non-finite tensor data");
    Ok(())
}

pub struct TensorBank<B: Backend> {
    floats: BTreeMap<String, (Tensor<B, 1>, Vec<usize>)>,
    quantized: BTreeMap<String, Tensor<B, 2>>,
    pub device: B::Device,
}

impl<B: Backend> TensorBank<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            floats: BTreeMap::new(),
            quantized: BTreeMap::new(),
            device: device.clone(),
        }
    }

    pub fn insert(&mut self, name: String, data: TensorData) -> Result<()> {
        ensure!(!self.contains(&name), "duplicate tensor {name}");
        match data.dtype {
            DType::QFloat(_) => {
                ensure!(data.shape.len() == 2, "quantized matrix rank");
                self.quantized
                    .insert(name, Tensor::from_data(data, &self.device));
            }
            DType::F32 | DType::F16 => {
                let shape = data.shape.to_vec();
                let count: usize = shape.iter().product();
                let flat = TensorData::from_bytes(data.bytes, [count], data.dtype);
                self.floats
                    .insert(name, (Tensor::from_data(flat, &self.device), shape));
            }
            _ => anyhow::bail!("non-floating tensor {name} cannot enter the weight bank"),
        }
        Ok(())
    }

    pub async fn load_object(
        &mut self,
        reader: &mut impl PartReader,
        object: &Object,
    ) -> Result<()> {
        for (name, data) in read_tensors(reader, object).await? {
            self.insert(name, data)?;
        }
        Ok(())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.floats.contains_key(name) || self.quantized.contains_key(name)
    }

    pub fn tensor<const D: usize>(&self, name: &str) -> Tensor<B, D> {
        let (data, shape) = &self.floats[name];
        let dims: [usize; D] = shape
            .as_slice()
            .try_into()
            .expect("validated model tensor rank");
        data.clone().reshape(dims)
    }

    /// Checkpoint matrices use [output,input]. Quantized weights remain packed
    /// between calls; one matrix is expanded on-device for portable matmul.
    /// Burn 0.21's packed block transpose/mixed matmul fails the independent
    /// Q4F parity probe, so dequantize BEFORE transposing. No dependency patch.
    pub fn linear<const D: usize>(&self, name: &str, input: Tensor<B, D>) -> Tensor<B, D> {
        let dims = input.dims();
        let k = dims[D - 1];
        let rows = dims.iter().product::<usize>() / k;
        let weight = self
            .quantized
            .get(name)
            .cloned()
            .map(|q| q.dequantize())
            .unwrap_or_else(|| self.tensor::<2>(name));
        let output_dim = weight.dims()[0];
        let output = input.reshape([rows, k]).matmul(weight.transpose());
        let mut shape = dims;
        shape[D - 1] = output_dim;
        output.reshape(shape)
    }

    pub fn affine(&self, prefix: &str, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear(&format!("{prefix}.weight"), input)
            + self.tensor::<1>(&format!("{prefix}.bias")).unsqueeze()
    }
}

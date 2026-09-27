use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{AllocationProperty, Bytes, DType, Tensor, TensorData},
};
use burn_human_motion::artifacts::{Object, PartReader, read_object};
use burn_store::{BurnpackStore, ModuleStore};
use std::collections::{BTreeMap, BTreeSet};

/// Authenticate one object and its exact tensor inventory before uploading it.
pub async fn read_tensors(
    reader: &mut impl PartReader,
    object: &Object,
) -> Result<BTreeMap<String, TensorData>> {
    let bytes = read_object(reader, object).await?;
    crate::cooperative::yield_to_browser().await;
    // Each tensor borrows its authenticated object's allocation. The reader
    // and tensor data retain it through shared ownership until upload completes.
    let mut pack = BurnpackStore::from_bytes(Some(Bytes::from_shared(
        bytes::Bytes::from(bytes),
        AllocationProperty::Native,
    )))
    .zero_copy(true);
    let snapshots = pack
        .get_all_snapshots()
        .map_err(|e| anyhow::anyhow!("Burnpack: {e}"))?;
    crate::cooperative::yield_to_browser().await;
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
                && reader.digest(&data.bytes).await? == spec.sha256,
            "tensor shape/dtype/digest mismatch: {name}"
        );
        ensure_finite_async(&data).await?;
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
    let (bytes, dtype) = finite_values(data)?;
    ensure!(finite_bytes(bytes, dtype), "non-finite tensor data");
    Ok(())
}

fn finite_values(data: &TensorData) -> Result<(&[u8], DType)> {
    if matches!(data.dtype, DType::QFloat(_)) {
        let n = data
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| anyhow::anyhow!("quantized shape overflow"))?;
        Ok((
            data.bytes
                .get(n / 2..)
                .ok_or_else(|| anyhow::anyhow!("truncated quantized scales"))?,
            DType::F32,
        ))
    } else {
        Ok((&data.bytes, data.dtype))
    }
}

fn finite_bytes(bytes: &[u8], dtype: DType) -> bool {
    match dtype {
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
        _ => true,
    }
}

async fn ensure_finite_async(data: &TensorData) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ensure_finite(data)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (bytes, dtype) = finite_values(data)?;
        // Large vision matrices contain millions of scalars. Keep validation
        // bounded per task without copying data or weakening the finite check.
        for chunk in bytes.chunks(1024 * 1024) {
            ensure!(finite_bytes(chunk, dtype), "non-finite tensor data");
            crate::cooperative::yield_to_browser().await;
        }
        Ok(())
    }
}

/// Bound pending browser GPU uploads so a later tiny write does not flush
/// gigabytes of queued weights in one synchronous WebGPU call. This is loading
/// backpressure only; it adds no tensor readback or per-layer inference fence.
#[derive(Default)]
pub struct UploadBudget {
    #[cfg(all(target_arch = "wasm32", feature = "wgpu"))]
    pending: usize,
}

impl UploadBudget {
    pub async fn record<B: Backend>(&mut self, device: &B::Device, bytes: usize) -> Result<()> {
        #[cfg(all(target_arch = "wasm32", feature = "wgpu"))]
        {
            use burn::{
                backend::wgpu::{WgpuDevice, WgpuRuntime},
                cubecl::Runtime,
            };
            use std::any::Any;
            self.pending += bytes;
            if self.pending >= 32 * 1024 * 1024 {
                if let Some(device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
                    WgpuRuntime::client(device)
                        .sync()
                        .await
                        .map_err(|e| anyhow::anyhow!("GPU upload completion: {e}"))?;
                }
                self.pending = 0;
            }
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "wgpu")))]
        let _ = (device, bytes);
        Ok(())
    }
}

pub struct TensorBank<B: Backend> {
    floats: BTreeMap<String, (Tensor<B, 1>, Vec<usize>)>,
    quantized: BTreeMap<String, Tensor<B, 2>>,
    pub device: B::Device,
    uploads: UploadBudget,
}

impl<B: Backend> TensorBank<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            floats: BTreeMap::new(),
            quantized: BTreeMap::new(),
            device: device.clone(),
            uploads: UploadBudget::default(),
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
            self.insert_async(name, data).await?;
            crate::cooperative::yield_to_browser().await;
        }
        Ok(())
    }

    /// Upload a verified tensor with bounded pending browser transfers.
    pub async fn insert_async(&mut self, name: String, data: TensorData) -> Result<()> {
        let bytes = data.bytes.len();
        self.insert(name, data)?;
        self.uploads.record::<B>(&self.device, bytes).await
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

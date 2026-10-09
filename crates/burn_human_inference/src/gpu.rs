//! Same-device tensor handoff without a host readback or GPU completion wait.
use anyhow::{Result, ensure};
use burn::{
    backend::wgpu::{AutoCompiler, CubeBackend},
    tensor::{DType, Tensor},
};
use std::{num::NonZeroU64, sync::Arc};

/// Raw CubeCL primitive type used by the unfused WGPU render bridge.
/// Burn 0.22 selects frontend dispatch through `Device`; enabling fusion changes
/// that dispatch and causes this bridge to reject the resulting fused primitive.
pub type WgpuBackend = CubeBackend;

/// A buffer view and its allocator lease. Keep this value alive until every
/// external GPU command using the view has completed; a raw buffer clone alone
/// does not prevent Burn's suballocator from reusing its region.
#[derive(Clone)]
pub struct TensorBuffer {
    buffer: wgpu::Buffer,
    offset: u64,
    bytes: u64,
    _allocation: Arc<dyn Send + Sync>,
}
impl TensorBuffer {
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: NonZeroU64::new(self.bytes),
        })
    }
    pub fn byte_len(&self) -> u64 {
        self.bytes
    }
}

pub fn export<const D: usize>(tensor: Tensor<D>) -> Result<TensorBuffer> {
    ensure!(
        tensor.dtype() == DType::F32,
        "Render buffers require f32 tensors"
    );
    let bytes = tensor.dims().iter().product::<usize>() as u64 * 4;
    ensure!(bytes > 0, "Cannot export an empty tensor");
    let primitive = tensor
        .try_into_primitive::<CubeBackend>()
        .map_err(|e| anyhow::anyhow!("Render handoff needs an unfused CubeCL tensor: {e:?}"))?;
    // Elementwise copy can preserve pitched/permuted storage in Burn 0.22.
    // Export a row-major view, including untile when the backend used tiled
    // storage. Already-contiguous tensors retain their original allocation.
    let primitive = burn_cubecl::kernel::into_contiguous(primitive);
    let resource = primitive
        .client
        .get_resource::<burn::cubecl::wgpu::WgpuServer<AutoCompiler>>(primitive.handle.clone())?;
    // Submits inference before a renderer uses this view on the shared queue.
    // flush is deliberately not sync: it never waits for GPU completion.
    primitive.client.flush()?;
    let view = resource.resource();
    ensure!(
        view.size >= bytes,
        "Tensor buffer is smaller than its logical shape"
    );
    Ok(TensorBuffer {
        buffer: view.buffer.clone(),
        offset: view.offset,
        bytes,
        _allocation: Arc::new(resource),
    })
}

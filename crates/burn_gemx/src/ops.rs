use burn::tensor::{Tensor, module, ops::AttentionModuleOptions};
use burn_human_inference::weights::TensorBank;

pub fn norm(bank: &TensorBank, prefix: &str, x: Tensor<3>, eps: f32) -> Tensor<3> {
    let center = x.clone() - x.mean_dim(2);
    let var = (center.clone() * center.clone()).mean_dim(2);
    center / (var + eps).sqrt() * bank.tensor::<1>(&format!("{prefix}.weight")).unsqueeze()
        + bank.tensor::<1>(&format!("{prefix}.bias")).unsqueeze()
}

pub fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    // Let the backend select attention kernels. WGPU autotuning can use fused
    // attention without materializing an N x N matrix; other backends retain
    // Burn's portable fallback.
    module::attention(q, k, v, None, None, AttentionModuleOptions::default())
}

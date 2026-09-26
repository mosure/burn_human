use burn::{
    prelude::Backend,
    tensor::{Tensor, activation},
};
use burn_human_inference::weights::TensorBank;

pub fn norm<B: Backend>(
    bank: &TensorBank<B>,
    prefix: &str,
    x: Tensor<B, 3>,
    eps: f32,
) -> Tensor<B, 3> {
    let center = x.clone() - x.mean_dim(2);
    let var = (center.clone() * center.clone()).mean_dim(2);
    center / (var + eps).sqrt() * bank.tensor::<1>(&format!("{prefix}.weight")).unsqueeze()
        + bank.tensor::<1>(&format!("{prefix}.bias")).unsqueeze()
}

pub fn attention<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let scale = (q.dims()[3] as f32).sqrt().recip();
    activation::softmax(q.matmul(k.swap_dims(2, 3)) * scale, 3).matmul(v)
}

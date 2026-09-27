//! Small checkpoint-derived tensors retained on device across diffusion steps.
use crate::{
    config::{ArdyConfig, Stats},
    network::position_encoding,
};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};

pub(crate) struct Constants<B: Backend> {
    pub root_mean: Tensor<B, 3>,
    pub root_scale: Tensor<B, 3>,
    pub local_mean: Tensor<B, 3>,
    pub local_scale: Tensor<B, 3>,
    pub latent_mean: Tensor<B, 3>,
    pub latent_scale: Tensor<B, 3>,
    positions: Tensor<B, 3>,
    codec_positions: Tensor<B, 3>,
}
impl<B: Backend> Constants<B> {
    pub fn new(config: &ArdyConfig, device: &B::Device) -> Self {
        let stats = |s: &Stats, range: std::ops::Range<usize>, scale: bool| {
            let values: Vec<f32> = range
                .map(|i| if scale { s.scale(i) } else { s.mean[i] })
                .collect();
            let n = values.len();
            Tensor::from_data(TensorData::new(values, [1, 1, n]), device)
        };
        Self {
            root_mean: stats(&config.motion_stats, 0..5, false),
            root_scale: stats(&config.motion_stats, 0..5, true),
            local_mean: stats(&config.motion_stats, 5..9, false),
            local_scale: stats(&config.motion_stats, 5..9, true),
            latent_mean: stats(&config.latent_stats, 0..128, false),
            latent_scale: stats(&config.latent_stats, 0..128, true),
            positions: position_encoding(99, 1024, -49, device),
            codec_positions: position_encoding(50, 512, 0, device),
        }
    }
    pub fn position(&self, length: usize, dim: usize, origin: isize) -> Tensor<B, 3> {
        if dim == 512 {
            assert_eq!(origin, 0);
            return self
                .codec_positions
                .clone()
                .slice([0..1, 0..length, 0..512]);
        }
        assert_eq!(dim, 1024);
        let start = (origin + 49) as usize;
        self.positions
            .clone()
            .slice([0..1, start..start + length, 0..1024])
    }
}

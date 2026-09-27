//! Fixed kernel selection for numerically sensitive motion code prediction.
//!
//! Shape-dependent autotuning can change accumulation order enough to cross an
//! FSQ boundary. Pin the unfused WGPU backend's matmul and reduction strategies;
//! other Burn backends retain their own dispatch. These downcasts move tensor
//! handles, never tensor contents. Clones below only retain allocation leases.
use burn::{prelude::Backend, tensor::Tensor};

pub(crate) async fn attention<B: Backend, const COOPERATIVE: bool>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, burn::tensor::Bool>>,
    causal: bool,
) -> Tensor<B, 4> {
    let [_, _, seq_q, dim] = query.dims();
    let seq_k = key.dims()[2];
    let mut scores = matmul(query, key.transpose()) * (1.0 / (dim as f64).sqrt());
    if let Some(mask) = mask {
        scores = scores.mask_fill(mask, f32::NEG_INFINITY);
    }
    if causal {
        let mask = Tensor::<B, 2, burn::tensor::Bool>::tril_mask(
            [seq_q, seq_k],
            seq_k as i64 - seq_q as i64,
            &scores.device(),
        )
        .unsqueeze::<4>();
        scores = scores.mask_fill(mask, f32::NEG_INFINITY);
    }
    if COOPERATIVE {
        burn_human_inference::cooperative::yield_to_browser().await;
    }
    let finfo = scores.dtype().finfo().expect("floating attention scores");
    let max = reduce_dim(scores.clone(), 3, Reduction::Max).clamp_min(finfo.min);
    let numerator = (scores - max).exp();
    let denominator = sum_dim(numerator.clone(), 3).clamp_min(finfo.min_positive);
    if COOPERATIVE {
        burn_human_inference::cooperative::yield_to_browser().await;
    }
    matmul(numerator / denominator, value)
}

pub(crate) fn mean_dim<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    reduce_dim(input, dim, Reduction::Mean)
}

pub(crate) fn sum_dim<B: Backend, const D: usize>(input: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    reduce_dim(input, dim, Reduction::Sum)
}

#[derive(Clone, Copy)]
enum Reduction {
    Mean,
    Sum,
    Max,
}

fn reduce_dim<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    dim: usize,
    operation: Reduction,
) -> Tensor<B, D> {
    #[cfg(feature = "wgpu")]
    {
        use burn::{
            backend::wgpu::{CubeTensor, WgpuRuntime},
            tensor::TensorPrimitive,
        };
        use burn_cubecl::kernel::reduce::{KernelReduceStrategy, reduce_dim};
        use cubek::reduce::{
            components::instructions::ReduceOperationConfig,
            launch::{ReduceStrategy, RoutineStrategy, VectorizationStrategy},
            routines::{BlueprintStrategy, plane::PlaneStrategy},
        };
        use std::any::{Any, TypeId};
        if TypeId::of::<B::FloatTensorPrimitive>() == TypeId::of::<CubeTensor<WgpuRuntime>>() {
            let input: Box<dyn Any> = Box::new(input.into_primitive().tensor());
            let input = *input.downcast::<CubeTensor<WgpuRuntime>>().unwrap();
            let operation = match operation {
                Reduction::Mean => ReduceOperationConfig::Mean,
                Reduction::Sum => ReduceOperationConfig::Sum,
                Reduction::Max => ReduceOperationConfig::Max,
            };
            let strategy = KernelReduceStrategy::Specific(ReduceStrategy {
                routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                    independent: true,
                })),
                vectorization: VectorizationStrategy {
                    parallel_output_vectorization: false,
                },
            });
            // Some WebGPU adapters lack subgroup support. Keep their portable
            // fallback deterministic too: Unspecified does not benchmark kernels.
            let output = reduce_dim(input.clone(), None, dim, strategy, operation)
                .or_else(|_| {
                    reduce_dim(
                        input,
                        None,
                        dim,
                        KernelReduceStrategy::Unspecified,
                        operation,
                    )
                })
                .expect("ARDY fixed reduction");
            let output: Box<dyn Any> = Box::new(output);
            return Tensor::from_primitive(TensorPrimitive::Float(
                *output.downcast::<B::FloatTensorPrimitive>().unwrap(),
            ));
        }
    }
    match operation {
        Reduction::Mean => input.mean_dim(dim),
        Reduction::Sum => input.sum_dim(dim),
        Reduction::Max => input.max_dim(dim),
    }
}

pub(crate) fn matmul<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    #[cfg(feature = "wgpu")]
    {
        use burn::{
            backend::wgpu::{CubeTensor, WgpuRuntime},
            tensor::TensorPrimitive,
        };
        use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
        use std::any::{Any, TypeId};
        if TypeId::of::<B::FloatTensorPrimitive>() == TypeId::of::<CubeTensor<WgpuRuntime>>() {
            let dtype = lhs.dtype();
            let lhs: Box<dyn Any> = Box::new(lhs.into_primitive().tensor());
            let rhs: Box<dyn Any> = Box::new(rhs.into_primitive().tensor());
            let lhs = *lhs.downcast::<CubeTensor<WgpuRuntime>>().unwrap();
            let rhs = *rhs.downcast::<CubeTensor<WgpuRuntime>>().unwrap();
            let output = if D == 2 {
                use cubek::{
                    matmul::{
                        definition::{MatmulElems, MatmulGlobalElems},
                        launch::Strategy,
                        routines::{
                            BlueprintStrategy, TileSizeSelection,
                            double_unit::DoubleUnitSelectionArgs,
                        },
                    },
                    std::InputBinding,
                };
                let output = burn_cubecl::kernel::matmul::init_matmul_output(&lhs, &rhs, dtype);
                let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
                    lhs: dtype.into(),
                    rhs: dtype.into(),
                    out: dtype.into(),
                });
                let strategy =
                    Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                        tile_size: TileSizeSelection::MinTileSize,
                    }));
                // Dense projections keep a fixed small tile as actor count
                // changes. Attention's batched rank-four products use Cube.
                let launched = cubek::matmul::launch::launch_ref(
                    &strategy,
                    &output.client,
                    InputBinding::new(lhs.clone().binding(), dtype.into()),
                    InputBinding::new(rhs.clone().binding(), dtype.into()),
                    output.clone().binding(),
                    &mut dtypes,
                );
                if launched.is_ok() {
                    output
                } else {
                    matmul(lhs, rhs, None, MatmulStrategy::Cube, dtype)
                        .expect("ARDY portable matmul")
                }
            } else {
                matmul(lhs, rhs, None, MatmulStrategy::Cube, dtype).expect("ARDY fixed matmul")
            };
            let output: Box<dyn Any> = Box::new(output);
            return Tensor::from_primitive(TensorPrimitive::Float(
                *output.downcast::<B::FloatTensorPrimitive>().unwrap(),
            ));
        }
    }
    lhs.matmul(rhs)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        tensor::{Bool, TensorData},
    };

    #[test]
    fn causal_attention_keeps_the_diagonal_and_handles_fully_masked_rows() {
        type B = NdArray<f32>;
        let device = Default::default();
        let q = Tensor::<B, 4>::zeros([1, 1, 3, 2], &device);
        let v = Tensor::from_data(
            TensorData::new(vec![10.0f32, 20.0, 30.0], [1, 1, 3, 1]),
            &device,
        );
        let result = pollster::block_on(attention::<B, false>(
            q.clone(),
            q.clone(),
            v.clone(),
            None,
            true,
        ));
        assert_eq!(
            result.into_data().to_vec::<f32>().unwrap(),
            [10.0, 15.0, 20.0]
        );
        let mask = Tensor::<B, 4, Bool>::from_data(
            TensorData::new(vec![true, false, false], [1, 1, 1, 3]),
            &device,
        );
        let result = pollster::block_on(attention::<B, false>(
            q.clone(),
            q.clone(),
            v.clone(),
            Some(mask.clone()),
            true,
        ));
        assert_eq!(
            result.into_data().to_vec::<f32>().unwrap(),
            [0.0, 20.0, 25.0]
        );
        let result = pollster::block_on(attention::<B, false>(q.clone(), q, v, Some(mask), false));
        assert_eq!(
            result.into_data().to_vec::<f32>().unwrap(),
            [25.0, 25.0, 25.0]
        );
    }
}

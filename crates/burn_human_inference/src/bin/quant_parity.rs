use anyhow::{Result, ensure};
use burn::{
    backend::Wgpu,
    tensor::{DType, Tensor, TensorData},
};

fn main() -> Result<()> {
    pollster::block_on(async {
        let device = Default::default();
        // ONNX has an implicit zero-point of 8; XOR maps both nibbles to signed Q4F.
        let mut bytes = vec![];
        let mut dense = vec![];
        for n in 0..64 {
            for k in (0..64).step_by(2) {
                let a = ((n + k) % 16) as u8;
                let b = ((n + k + 1) % 16) as u8;
                bytes.push((a | b << 4) ^ 0x88);
                let scale = if n % 2 == 0 { 0.25 } else { -0.125 };
                dense.extend([(a as f32 - 8.0) * scale, (b as f32 - 8.0) * scale]);
            }
        }
        for n in 0..64 {
            for _ in 0..2 {
                bytes.extend_from_slice(
                    &(if n % 2 == 0 { 0.25f32 } else { -0.125f32 }).to_le_bytes(),
                );
            }
        }
        let q: Tensor<Wgpu, 2> = Tensor::from_data(
            TensorData::from_bytes_vec(bytes, [64, 64], burn_human_inference::dtype("q4f32")?),
            &device,
        );
        ensure!(
            matches!(q.dtype(), DType::QFloat(_)),
            "quantized weights were widened during loading"
        );
        let expected: Tensor<Wgpu, 2> =
            Tensor::from_data(TensorData::new(dense, [64, 64]), &device);
        let input: Tensor<Wgpu, 2> = Tensor::from_data(
            TensorData::new(
                (0..64 * 64)
                    .map(|i| (i % 29) as f32 * 0.03)
                    .collect::<Vec<_>>(),
                [64, 64],
            ),
            &device,
        );
        let dequantized = q.clone().dequantize();
        let dequant_error = (dequantized.clone() - expected.clone())
            .abs()
            .max()
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap()[0];
        ensure!(
            dequant_error == 0.0,
            "Q4F unpacking changed a checkpoint value"
        );
        let output = input.clone().matmul(dequantized.transpose());
        let error = (output - input.matmul(expected.transpose()))
            .abs()
            .max()
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .unwrap()[0];
        println!("Q4F32 packed storage / per-matrix GPU expansion max error {error}");
        ensure!(error < 0.002, "quantized matmul parity failed");
        Ok(())
    })
}

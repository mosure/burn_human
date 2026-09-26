//! Shared, model-neutral loading and tensor operations. No Bevy or model policy.
#[cfg(feature = "transport")]
pub mod transport;
pub mod weights;

use burn::tensor::{
    DType,
    quantization::{QuantLevel, QuantScheme, QuantValue},
};

pub fn dtype(name: &str) -> anyhow::Result<DType> {
    Ok(match name {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "i32" => DType::I32,
        "u32" => DType::U32,
        "u8" => DType::U8,
        "q4f32" => DType::QFloat(
            QuantScheme::default()
                .with_value(QuantValue::Q4F)
                .with_level(QuantLevel::block([32])),
        ),
        _ => anyhow::bail!("unsupported tensor dtype {name}"),
    })
}

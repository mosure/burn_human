//! Shared, model-neutral loading and tensor operations. No Bevy or model policy.
#![recursion_limit = "256"]
#[cfg(feature = "wgpu")]
pub mod gpu;
mod legacy_quant;
#[cfg(feature = "transport")]
pub mod pretrained;
#[cfg(feature = "transport")]
pub mod transport;
pub mod weights;

/// Initialize static GPU/compiler registrations once per browser module.
///
/// Call before initializing Burn or Bevy in a browser entry point. The model
/// validators and studio do this automatically; native calls are harmless.
/// A reachable constructor call gives LLD reactor-style linkage, avoiding
/// thousands of inventory registrations on every Rust/JavaScript crossing.
pub fn initialize_browser_runtime() {
    #[cfg(all(target_arch = "wasm32", feature = "cooperative"))]
    {
        unsafe extern "C" {
            fn __wasm_call_ctors();
        }
        static INITIALIZED: std::sync::Once = std::sync::Once::new();
        INITIALIZED.call_once(|| {
            // SAFETY: LLD supplies this module's static constructors. This runs
            // once, before any compiler registry is accessed.
            unsafe { __wasm_call_ctors() };
        });
    }
}

use burn::tensor::{
    DType,
    quantization::{QuantScheme, QuantValue, ScaleDtype},
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
                .per_block([32], ScaleDtype::F32),
        ),
        _ => anyhow::bail!("unsupported tensor dtype {name}"),
    })
}
pub mod cooperative;

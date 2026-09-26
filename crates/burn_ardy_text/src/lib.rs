//! Llama 3 / LLM2Vec text inference implemented with portable Burn operations.
pub mod model;
pub mod tokenizer;
pub mod validation;
#[cfg(all(target_arch = "wasm32", feature = "web-validation"))]
mod web_validation;
pub use model::{AttentionMode, TextEncoder};
pub const MODEL_ID: &str = "TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX";
pub const MODEL_REVISION: &str = "7aa52a05d54c2fd9177366aeb3f88e9e7f3c5766";

//! GEM-X perception, regression and SOMA reconstruction, entirely in Rust/Burn.
pub mod camera;
pub mod denoiser;
pub mod image_input;
mod ops;
pub mod pipeline;
pub mod sam;
mod sam_head;
pub mod validation;
pub mod vision;
#[cfg(all(target_arch = "wasm32", feature = "web-validation"))]
mod web_validation;
pub const MODEL_REVISION: &str = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd";
pub const SOURCE_REVISION: &str = "32992550dba114c62243fb55e361311972dce8f9";

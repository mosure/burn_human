//! ARDY Core RP 20 FPS / Horizon40 inference, ported from NVIDIA's Apache-2.0
//! implementation. See THIRD_PARTY.md for provenance and model-weight terms.
pub mod config;
pub mod network;
pub mod representation;
pub mod sampler;
#[cfg(feature = "transport")]
pub mod transport;
#[cfg(feature = "validation")]
pub mod validation;
#[cfg(all(feature = "web-validation", target_arch = "wasm32"))]
mod web_validation;
pub mod weights;

pub use config::ArdyConfig;
pub use network::Ardy;

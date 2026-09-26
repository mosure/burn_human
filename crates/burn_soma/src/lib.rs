//! SOMA-X evaluation in meters, +Y up and +Z forward. No Python runtime.
mod geometry;
mod model;
mod rig;
pub mod validation;
#[cfg(all(target_arch = "wasm32", feature = "web-validation"))]
mod web_validation;
pub use rig::RigData;
#[cfg(feature = "mhr")]
pub mod mhr;
pub use model::{BindConvention, IdentityParameters, PreparedIdentity, Soma, SomaOutput, SomaPose};
pub const MODEL_REVISION: &str = "104578ed58857f6faa7592fb83d0a2dad43c36fa";
pub const SOURCE_REVISION: &str = "cc1f3967755f8e36d187d2e26114633dbd651cd5";

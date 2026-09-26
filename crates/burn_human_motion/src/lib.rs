//! Model-independent motion contracts. Coordinates are right-handed, Y-up, in metres.
//!
//! Inference crates own their representations and weights; this crate owns validated
//! interchange data, rigging, conditioning and immutable artifact transport.

pub mod artifacts;
pub mod conditioning;
pub mod rig;
pub mod soma;

pub use conditioning::{ImageCondition, MotionRequest, TextEmbedding, Waypoint};
pub use rig::{MotionClip, PoseFrame, RigDefinition, RigJoint, RigMapping};

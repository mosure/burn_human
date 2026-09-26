use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crop {
    pub center: [f32; 2],
    pub size: f32,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub focal: [f32; 2],
    pub center: [f32; 2],
}
impl Crop {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.center
                .iter()
                .all(|x| x.is_finite() && x.abs() < 100000.0)
                && self.size.is_finite()
                && (1.0..=32768.0).contains(&self.size),
            "Invalid person crop"
        );
        Ok(())
    }
    pub fn condition(&self, camera: Camera) -> [f32; 3] {
        [
            (self.center[0] - camera.center[0]) / camera.focal[0],
            (self.center[1] - camera.center[1]) / camera.focal[0],
            self.size / camera.focal[0],
        ]
    }
}
impl Camera {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.center
                .iter()
                .all(|x| x.is_finite() && x.abs() < 100000.0)
                && self
                    .focal
                    .iter()
                    .all(|x| x.is_finite() && *x > 1.0 && *x < 100000.0),
            "Invalid camera intrinsics"
        );
        Ok(())
    }
}

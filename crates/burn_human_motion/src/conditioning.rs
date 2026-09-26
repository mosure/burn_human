use anyhow::{Result, ensure};
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Waypoint {
    pub frame: usize,
    /// World position in metres. Root trajectory uses X and Z; height is optional.
    pub position: Vec3,
    pub heading: Option<f32>,
    #[serde(default)]
    pub constrain_height: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionRequest {
    pub prompt: String,
    pub seed: u64,
    pub frames: usize,
    pub history_frames: usize,
    pub diffusion_steps: usize,
    pub text_guidance: f32,
    pub trajectory_guidance: f32,
    pub waypoints: Vec<Waypoint>,
    pub dense_trajectory: bool,
}

impl Default for MotionRequest {
    fn default() -> Self {
        Self {
            prompt: "A person walks forward.".into(),
            seed: 42,
            frames: 120,
            history_frames: 40,
            diffusion_steps: 10,
            text_guidance: 2.0,
            trajectory_guidance: 2.0,
            waypoints: Vec::new(),
            dense_trajectory: true,
        }
    }
}

impl MotionRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.prompt.trim().is_empty() && self.prompt.len() <= 16384,
            "Enter a nonempty motion prompt of at most 16384 UTF-8 bytes"
        );
        ensure!(
            (40..=12000).contains(&self.frames) && self.frames.is_multiple_of(4),
            "frames must be 40..12000 and divisible by four"
        );
        ensure!(
            self.history_frames <= 160 && self.history_frames.is_multiple_of(4),
            "history must be 0..160 frames and divisible by four"
        );
        ensure!(
            (1..=10).contains(&self.diffusion_steps),
            "diffusion steps must be 1..10"
        );
        ensure!(
            [self.text_guidance, self.trajectory_guidance]
                .iter()
                .all(|x| x.is_finite() && (0.0..=10.0).contains(x)),
            "guidance must be finite and 0..10"
        );
        let mut last = None;
        for w in &self.waypoints {
            ensure!(
                w.frame < self.frames && last.is_none_or(|v| w.frame > v),
                "waypoint frames must be unique, sorted and inside the clip"
            );
            ensure!(
                w.position.is_finite() && w.heading.is_none_or(f32::is_finite),
                "non-finite waypoint"
            );
            last = Some(w.frame);
        }
        Ok(())
    }
}

/// Exact pooled LLM2Vec output. The prompt and encoder identity prevent stale-cache reuse.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEmbedding {
    pub prompt: String,
    pub encoder: String,
    pub revision: String,
    pub values: Vec<f32>,
}

impl TextEmbedding {
    pub fn validate(&self, prompt: &str, dimensions: usize) -> Result<()> {
        ensure!(
            self.prompt == prompt,
            "embedding belongs to a different prompt"
        );
        ensure!(
            !self.encoder.is_empty() && !self.revision.is_empty(),
            "missing encoder provenance"
        );
        ensure!(
            self.values.len() == dimensions && self.values.iter().all(|v| v.is_finite()),
            "invalid embedding shape/values"
        );
        Ok(())
    }
}

/// Camera-aware image condition for pose estimation. It cannot be submitted
/// to ARDY, which has no image encoder. Crop and camera metadata survive interchange.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageCondition {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub crop_xywh: Option<[u32; 4]>,
    pub focal_length_px: Option<[f32; 2]>,
}

impl ImageCondition {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0 && self.height > 0 && self.width <= 8192 && self.height <= 8192,
            "invalid image dimensions"
        );
        ensure!(
            self.rgba.len() as u64 == u64::from(self.width) * u64::from(self.height) * 4,
            "RGBA byte count mismatch"
        );
        if let Some([x, y, w, h]) = self.crop_xywh {
            ensure!(
                w > 0
                    && h > 0
                    && x.checked_add(w).is_some_and(|v| v <= self.width)
                    && y.checked_add(h).is_some_and(|v| v <= self.height),
                "invalid image crop"
            );
        }
        if let Some(f) = self.focal_length_px {
            ensure!(
                f.iter().all(|v| v.is_finite() && *v > 0.0),
                "invalid focal length"
            );
        }
        Ok(())
    }
}

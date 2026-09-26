use anyhow::{Result, ensure};
use burn_human_motion::RigDefinition;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MODEL_ID: &str = "nvidia/ARDY-Core-RP-20FPS-Horizon40";
pub const MODEL_REVISION: &str = "abe6c43beb28c867c950acb824b9c4ef3d63fb76";
pub const SOURCE_REVISION: &str = "693f74d13b3d04a0a22ce127ee79c929dd89756b";
pub const MOTION_DIM: usize = 330;
pub const BODY_DIM: usize = 325;
pub const HYBRID_DIM: usize = 148;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}
impl Stats {
    pub fn validate(&self, size: usize) -> Result<()> {
        ensure!(
            self.mean.len() == size && self.std.len() == size,
            "normalization shape mismatch"
        );
        ensure!(
            self.mean.iter().all(|v| v.is_finite())
                && self.std.iter().all(|v| v.is_finite() && *v >= 0.0),
            "invalid normalization data"
        );
        Ok(())
    }
    pub fn scale(&self, i: usize) -> f32 {
        (self.std[i] * self.std[i] + 1e-5).sqrt()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArdyConfig {
    pub architecture: String,
    pub fps: usize,
    pub horizon: usize,
    pub frames_per_token: usize,
    pub motion_stats: Stats,
    pub latent_stats: Stats,
    pub skeleton: RigDefinition,
}

impl ArdyConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.architecture == "ardy-core-rp-v1"
                && self.fps == 20
                && self.horizon == 40
                && self.frames_per_token == 4,
            "unsupported ARDY architecture; expected Core RP 20FPS Horizon40"
        );
        self.motion_stats.validate(334)?;
        self.latent_stats.validate(128)?;
        self.skeleton.validate()?;
        ensure!(
            self.skeleton.id == "ardy-core27" && self.skeleton.joints.len() == 27,
            "ARDY Core requires the Core27 rig"
        );
        Ok(())
    }
}

/// Strict architecture inventory. Conversion and runtime both reject extra,
/// missing, or shape-incompatible weights before any inference can run.
pub fn expected_tensors() -> BTreeMap<String, Vec<usize>> {
    fn linear(out: &mut BTreeMap<String, Vec<usize>>, p: &str, input: usize, output: usize) {
        out.insert(format!("{p}.weight"), vec![output, input]);
        out.insert(format!("{p}.bias"), vec![output]);
    }
    let mut out = BTreeMap::new();
    for (name, input) in [
        ("global_root_hybrid_proj", 148),
        ("global_root_hybrid_constraints_proj", 2768),
        ("local_root_hybrid_proj", 144),
        ("local_root_hybrid_constraints_proj", 2764),
        ("future_constraints_proj", 2640),
    ] {
        linear(&mut out, name, input, 1024);
    }
    for (name, output) in [("root_model", 20), ("body_model", 128)] {
        linear(&mut out, &format!("{name}.embed_text"), 4096, 1024);
        linear(
            &mut out,
            &format!("{name}.embed_timestep.time_embed.0"),
            1024,
            1024,
        );
        linear(
            &mut out,
            &format!("{name}.embed_timestep.time_embed.2"),
            1024,
            1024,
        );
        linear(
            &mut out,
            &format!("{name}.linear_first_heading_angle"),
            2,
            1024,
        );
        linear(&mut out, &format!("{name}.output_linear"), 1024, output);
        out.insert(
            format!("{name}.learned_prefix_embedding.embedding.weight"),
            vec![3, 1024],
        );
    }
    linear(&mut out, "encoder.input_proj", 1300, 512);
    linear(&mut out, "encoder.output_proj", 512, 128);
    linear(&mut out, "decoder.input_proj", 128, 512);
    linear(&mut out, "decoder.output_proj", 512, 1316);
    linear(&mut out, "decoder.external_cond_blocks.0", 528, 512);
    linear(&mut out, "decoder.target_cond_blocks.0", 325, 128);
    for (model, dim) in [
        ("root_model", 1024),
        ("body_model", 1024),
        ("encoder", 512),
        ("decoder", 512),
    ] {
        for i in 0..8 {
            let p = format!("{model}.seqTransEncoder.layers.{i}");
            linear(&mut out, &format!("{p}.linear1"), dim, dim * 2);
            linear(&mut out, &format!("{p}.linear2"), dim * 2, dim);
            linear(&mut out, &format!("{p}.self_attn.out_proj"), dim, dim);
            out.insert(format!("{p}.self_attn.in_proj_weight"), vec![dim * 3, dim]);
            out.insert(format!("{p}.self_attn.in_proj_bias"), vec![dim * 3]);
            for norm in ["norm1", "norm2"] {
                out.insert(format!("{p}.{norm}.weight"), vec![dim]);
                out.insert(format!("{p}.{norm}.bias"), vec![dim]);
            }
        }
    }
    out
}

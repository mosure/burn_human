//! GEM-X's released 12-layer model, including image/keypoint conditioning.
use crate::{
    camera::{Camera, Crop},
    ops::{attention, norm},
};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData, activation},
};
use burn_human_inference::{transport::ModelSource, weights::TensorBank};
use burn_human_motion::artifacts::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decode {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub observation_joints: Vec<usize>,
}
pub struct GemDenoiser<B: Backend> {
    weights: TensorBank<B>,
    pub decode: Decode,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conditions {
    pub batch: usize,
    pub frames: usize,
    pub observations: Vec<[f32; 3]>,
    pub crops: Vec<Crop>,
    pub cameras: Vec<Camera>,
    pub angular: Vec<[f32; 6]>,
}
pub struct Prediction<B: Backend> {
    pub features: Tensor<B, 3>,
    pub camera: Tensor<B, 3>,
}
fn linear(map: &mut BTreeMap<String, Vec<usize>>, name: &str, i: usize, o: usize) {
    map.insert(format!("{name}.weight"), vec![o, i]);
    map.insert(format!("{name}.bias"), vec![o]);
}
fn ln(map: &mut BTreeMap<String, Vec<usize>>, name: &str, n: usize) {
    map.insert(format!("{name}.weight"), vec![n]);
    map.insert(format!("{name}.bias"), vec![n]);
}
impl Conditions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=32).contains(&self.batch) && (1..=120).contains(&self.frames),
            "GEM batch/frame limit"
        );
        let n = self.batch * self.frames;
        ensure!(
            self.observations.len() == n * 77
                && self.crops.len() == n
                && self.cameras.len() == n
                && self.angular.len() == n,
            "GEM conditioning dimensions"
        );
        ensure!(
            self.observations
                .iter()
                .flatten()
                .chain(self.angular.iter().flatten())
                .all(|v| v.is_finite() && v.abs() < 1e6),
            "Non-finite GEM conditioning"
        );
        for (crop, cam) in self.crops.iter().zip(&self.cameras) {
            crop.validate()?;
            cam.validate()?;
        }
        Ok(())
    }
}
impl<B: Backend> GemDenoiser<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == "nvidia/GEM-X:denoiser"
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.source_revision == crate::SOURCE_REVISION,
            "Unsupported GEM checkpoint"
        );
        ensure!(
            manifest.config
                == serde_json::json!({"layers":12,"hidden":512,"heads":8,"features":585,"observation_joints":33,"image_features":1024,"scale_components":28,"max_frames":120}),
            "Unsupported GEM architecture"
        );
        let asset = manifest
            .assets
            .iter()
            .find(|a| a.path == "metadata/decode.json")
            .ok_or_else(|| anyhow::anyhow!("Missing GEM decode metadata"))?;
        let decode: Decode = serde_json::from_slice(&source.asset(asset).await?)?;
        ensure!(
            decode.mean.len() == 585
                && decode.std.len() == 585
                && decode.mean.iter().all(|v| v.is_finite())
                && decode.std.iter().all(|v| v.is_finite() && *v >= 1.0)
                && decode.observation_joints.len() == 33
                && decode.observation_joints.iter().all(|i| *i < 77),
            "Invalid GEM decode metadata"
        );
        let mut expected: BTreeMap<String, Vec<usize>> = [
            ("learned_pos_params", vec![33, 32]),
            ("scale_mean", vec![69]),
            ("scale_comps", vec![28, 69]),
            ("time.positions", vec![1000, 512]),
        ]
        .into_iter()
        .map(|(n, s)| (n.into(), s))
        .collect();
        for (name, i, o) in [
            ("learned_pos_linear", 2, 32),
            ("embed_noisyobs.fc1", 1056, 1024),
            ("embed_noisyobs.fc2", 1024, 512),
            ("cliffcam_embedder.0", 3, 512),
            ("cliffcam_embedder.3", 512, 512),
            ("cam_angvel_embedder.0", 6, 512),
            ("cam_angvel_embedder.3", 512, 512),
            ("imgseq_embedder.1", 1024, 512),
            ("embed_timestep.time_embed.0", 512, 512),
            ("embed_timestep.time_embed.2", 512, 512),
            ("add_cond_linear", 1097, 512),
            ("final_layer.fc1", 512, 512),
            ("final_layer.fc2", 512, 585),
            ("pred_cam_head.fc1", 512, 512),
            ("pred_cam_head.fc2", 512, 3),
        ] {
            linear(&mut expected, name, i, o);
        }
        ln(&mut expected, "imgseq_embedder.0", 1024);
        for name in ["f_cliffcam", "f_imgseq", "f_cam_angvel"] {
            linear(
                &mut expected,
                &format!("cond_exists_embedder.{name}.0"),
                513,
                512,
            );
            linear(
                &mut expected,
                &format!("cond_exists_embedder.{name}.2"),
                512,
                512,
            );
        }
        for i in 0..12 {
            let p = format!("blocks.{i}");
            for name in ["gate_msa", "gate_mlp"] {
                expected.insert(format!("{p}.{name}"), vec![1, 1, 512]);
            }
            for name in ["norm1", "norm2"] {
                ln(&mut expected, &format!("{p}.{name}"), 512);
            }
            for name in ["query", "key", "value", "proj"] {
                linear(&mut expected, &format!("{p}.attn.{name}"), 512, 512);
            }
            linear(&mut expected, &format!("{p}.mlp.fc1"), 512, 2048);
            linear(&mut expected, &format!("{p}.mlp.fc2"), 2048, 512);
        }
        let mut actual = BTreeMap::new();
        for o in &manifest.objects {
            for t in &o.tensors {
                ensure!(
                    t.dtype == "f32" && actual.insert(t.name.clone(), t.shape.clone()).is_none(),
                    "Invalid GEM tensor"
                );
            }
        }
        ensure!(actual == expected, "GEM tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        for (i, o) in manifest.objects.iter().enumerate() {
            weights.load_object(source, o).await?;
            progress(i + 1, manifest.objects.len());
        }
        Ok(Self { weights, decode })
    }
    fn mlp(&self, p: &str, x: Tensor<B, 3>, approx: bool) -> Tensor<B, 3> {
        let x = self.weights.affine(&format!("{p}.fc1"), x);
        let x = if approx {
            let u = x.clone() + x.clone().powf_scalar(3.0) * 0.044715;
            x * 0.5 * ((u * (2.0 / std::f32::consts::PI).sqrt()).tanh() + 1.0)
        } else {
            activation::gelu(x)
        };
        self.weights.affine(&format!("{p}.fc2"), x)
    }
    fn exists(&self, name: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let z = Tensor::cat(vec![x, Tensor::ones([b, l, 1], &self.weights.device)], 2);
        self.weights.affine(
            &format!("cond_exists_embedder.{name}.2"),
            activation::silu(
                self.weights
                    .affine(&format!("cond_exists_embedder.{name}.0"), z),
            ),
        )
    }
    pub fn condition(&self, input: &Conditions, features: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        input.validate()?;
        let (b, l) = (input.batch, input.frames);
        ensure!(
            features.dims() == [b, l, 1024],
            "GEM image feature dimensions"
        );
        let w = &self.weights;
        let device = &w.device;
        let mut xy = vec![];
        let mut valid = vec![];
        let mut cliff = vec![];
        let mut angular = vec![];
        for i in 0..b * l {
            let crop = input.crops[i];
            cliff.extend(crop.condition(input.cameras[i]));
            for j in &self.decode.observation_joints {
                let p = input.observations[i * 77 + j];
                let v = p[2] > 0.5
                    && (p[0] - crop.center[0]).abs() <= crop.size / 2.0
                    && (p[1] - crop.center[1]).abs() <= crop.size / 2.0;
                xy.extend(if v {
                    [
                        (p[0] - crop.center[0]) * 2.0 / crop.size,
                        (p[1] - crop.center[1]) * 2.0 / crop.size,
                    ]
                } else {
                    [0.0, 0.0]
                });
                valid.push(if v { 1.0 } else { 0.0 });
            }
            for (j, value) in input.angular[i].iter().enumerate() {
                let mean = if j == 0 || j == 4 { 1.0 } else { 0.0 };
                let std = if j == 0 || j == 4 { 0.001 } else { 0.1 };
                angular.push((value - mean) / std);
            }
        }
        let xy = Tensor::<B, 3>::from_data(TensorData::new(xy, [b * l, 33, 2]), device);
        let valid = Tensor::<B, 3>::from_data(TensorData::new(valid, [b * l, 33, 1]), device);
        let embedded = w.affine("learned_pos_linear", xy) * valid.clone()
            + w.tensor::<2>("learned_pos_params").unsqueeze::<3>() * (valid.neg() + 1.0);
        let mut cond = self.mlp("embed_noisyobs", embedded.reshape([b, l, 1056]), false);
        let cliff = Tensor::<B, 3>::from_data(TensorData::new(cliff, [b, l, 3]), device);
        let cliff = w.affine(
            "cliffcam_embedder.3",
            activation::silu(w.affine("cliffcam_embedder.0", cliff)),
        );
        cond = cond + self.exists("f_cliffcam", cliff);
        let image = w.affine(
            "imgseq_embedder.1",
            norm(w, "imgseq_embedder.0", features, 1e-5),
        );
        cond = cond + self.exists("f_imgseq", image);
        let angular = Tensor::<B, 3>::from_data(TensorData::new(angular, [b, l, 6]), device);
        let angular = w.affine(
            "cam_angvel_embedder.3",
            activation::silu(w.affine("cam_angvel_embedder.0", angular)),
        );
        Ok(cond + self.exists("f_cam_angvel", angular))
    }
    /// Denoise a latent at an explicit training timestep. The public fast image
    /// path uses zeros at 999, exactly as the released ONNX graph does.
    pub fn denoise(
        &self,
        condition: Tensor<B, 3>,
        latent: Tensor<B, 3>,
        timestep: usize,
    ) -> Result<Prediction<B>> {
        let [b, l, c] = condition.dims();
        ensure!(
            (1..=32).contains(&b)
                && (1..=120).contains(&l)
                && c == 512
                && latent.dims() == [b, l, 585]
                && timestep < 1000,
            "GEM latent dimensions or timestep"
        );
        let w = &self.weights;
        let time = w
            .tensor::<2>("time.positions")
            .slice([timestep..timestep + 1, 0..512])
            .reshape([1, 1, 512]);
        let time = w.affine(
            "embed_timestep.time_embed.2",
            activation::silu(w.affine("embed_timestep.time_embed.0", time)),
        );
        let mut x = w.affine(
            "add_cond_linear",
            Tensor::cat(vec![condition + time, latent], 2),
        );
        let mut angles = vec![];
        for i in 0..l {
            for j in 0..64 {
                angles.push(i as f32 / 10000f32.powf((j / 2 * 2) as f32 / 64.0));
            }
        }
        let angles = Tensor::<B, 4>::from_data(TensorData::new(angles, [1, 1, l, 64]), &w.device);
        let sin = angles.clone().sin();
        let cos = angles.cos();
        let rope = |x: Tensor<B, 4>| {
            let pairs = x.clone().reshape([b, 8, l, 32, 2]);
            let a = pairs.clone().slice([0..b, 0..8, 0..l, 0..32, 0..1]);
            let z = pairs.slice([0..b, 0..8, 0..l, 0..32, 1..2]);
            let half = Tensor::cat(vec![-z, a], 4).reshape([b, 8, l, 64]);
            x * cos.clone() + half * sin.clone()
        };
        for i in 0..12 {
            let p = format!("blocks.{i}");
            let z = norm(w, &format!("{p}.norm1"), x.clone(), 1e-6);
            let head = |name| {
                w.affine(&format!("{p}.attn.{name}"), z.clone())
                    .reshape([b, l, 8, 64])
                    .swap_dims(1, 2)
            };
            let a = attention(rope(head("query")), rope(head("key")), head("value"))
                .swap_dims(1, 2)
                .reshape([b, l, 512]);
            x = x + w.affine(&format!("{p}.attn.proj"), a)
                * w.tensor::<3>(&format!("{p}.gate_msa"));
            let z = norm(w, &format!("{p}.norm2"), x.clone(), 1e-6);
            x = x + self.mlp(&format!("{p}.mlp"), z, true)
                * w.tensor::<3>(&format!("{p}.gate_mlp"));
        }
        let out = self.mlp("final_layer", x.clone(), false);
        let id = out
            .clone()
            .slice([0..b, 0..l, 456..501])
            .mean_dim(1)
            .expand([b, l, 45]);
        let scale = out
            .clone()
            .slice([0..b, 0..l, 501..529])
            .mean_dim(1)
            .reshape([b, 28])
            .matmul(w.tensor("scale_comps"))
            + w.tensor::<1>("scale_mean").unsqueeze();
        let scale = scale.reshape([b, 1, 69]).expand([b, l, 69]);
        let features = Tensor::cat(
            vec![
                out.clone().slice([0..b, 0..l, 0..456]),
                id,
                scale,
                out.slice([0..b, 0..l, 570..585]),
            ],
            2,
        );
        let camera = self.mlp("pred_cam_head", x, false)
            * Tensor::<B, 1>::from_data([0.1784, 0.0956, 0.0764], &w.device).unsqueeze()
            + Tensor::<B, 1>::from_data([1.0606, -0.0027, 0.2702], &w.device).unsqueeze();
        let first = camera.clone().slice([0..b, 0..l, 0..1]).clamp_min(0.25);
        let camera = camera.slice_assign([0..b, 0..l, 0..1], first);
        Ok(Prediction { features, camera })
    }
    pub fn predict(&self, input: &Conditions, features: Tensor<B, 3>) -> Result<Prediction<B>> {
        let condition = self.condition(input, features)?;
        self.denoise(
            condition,
            Tensor::zeros([input.batch, input.frames, 585], &self.weights.device),
            999,
        )
    }
}

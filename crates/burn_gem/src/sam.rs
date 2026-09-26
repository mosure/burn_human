//! SAM3D legacy body-token contract used by GEM-X (six iterative MHR updates).
//! This module implements inference, not the optional interactive SAM hand detector.
use crate::{
    camera::{Camera, Crop},
    ops::{attention, norm},
    sam_head::Head,
};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, activation},
};
use burn_human_inference::{transport::ModelSource, weights::TensorBank};
use burn_human_motion::artifacts::Manifest;
use burn_mhr::{Mhr, MhrInput};
use std::collections::BTreeMap;

pub struct SamBody<B: Backend> {
    weights: TensorBank<B>,
    head: Head,
}
#[derive(serde::Serialize)]
pub struct SamLayer {
    pub parameters: MhrInput,
    pub keypoints: Vec<[f32; 3]>,
    pub camera: [f32; 3],
}
pub struct SamOutput<B: Backend> {
    pub token: Tensor<B, 3>,
    pub layers: Vec<SamLayer>,
}
fn linear(map: &mut BTreeMap<String, Vec<usize>>, name: &str, input: usize, output: usize) {
    map.insert(format!("{name}.weight"), vec![output, input]);
    map.insert(format!("{name}.bias"), vec![output]);
}
fn ln(map: &mut BTreeMap<String, Vec<usize>>, name: &str, dim: usize) {
    map.insert(format!("{name}.weight"), vec![dim]);
    map.insert(format!("{name}.bias"), vec![dim]);
}
fn ffn(map: &mut BTreeMap<String, Vec<usize>>, name: &str, input: usize, output: usize) {
    linear(map, &format!("{name}.layers.0.0"), input, 1024);
    linear(map, &format!("{name}.layers.1"), 1024, output);
}
impl<B: Backend> SamBody<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == "nvidia/GEM-X:sam-decoder"
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.source_revision == crate::SOURCE_REVISION,
            "Unsupported SAM checkpoint"
        );
        ensure!(
            manifest.config
                == serde_json::json!({"layers":6,"hidden":1024,"heads":8,"head_dim":64,"image_height":32,"image_width":32,"image_channels":1280,"pose_dimensions":519,"keypoints":70}),
            "Unsupported SAM architecture"
        );
        let asset = manifest
            .assets
            .iter()
            .find(|a| a.path == "metadata/head.json")
            .ok_or_else(|| anyhow::anyhow!("Missing SAM parameter metadata"))?;
        let head: Head = serde_json::from_slice(&source.asset(asset).await?)?;
        head.validate()?;
        let mut expected: BTreeMap<String, Vec<usize>> = [
            ("init_pose.weight", vec![1, 519]),
            ("init_camera.weight", vec![1, 3]),
            ("keypoint_embedding.weight", vec![70, 1024]),
            ("keypoint3d_embedding.weight", vec![70, 1024]),
            ("ray_cond_emb.conv.weight", vec![1280, 1379]),
            ("prompt.invalid", vec![1, 1, 1280]),
            ("prompt.image_pe", vec![1, 1024, 1280]),
            ("prompt.no_mask", vec![1280]),
            ("head_pose.keypoint_mapping", vec![70, 18566]),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
        for (name, i, o) in [
            ("init_to_token_mhr", 525, 1024),
            ("prev_to_token_mhr", 522, 1024),
            ("prompt_to_token", 1280, 1024),
            ("keypoint_feat_linear", 1280, 1024),
        ] {
            linear(&mut expected, name, i, o);
        }
        for (name, i, o) in [
            ("head_pose.proj", 1024, 519),
            ("head_camera.proj", 1024, 3),
            ("keypoint_posemb_linear", 2, 1024),
            ("keypoint3d_posemb_linear", 3, 1024),
        ] {
            ffn(&mut expected, name, i, o);
        }
        ln(&mut expected, "ray_cond_emb.norm", 1280);
        ln(&mut expected, "decoder.norm_final", 1024);
        for i in 0..6 {
            let p = format!("decoder.layers.{i}");
            for (name, n) in [
                ("ln_pe_1", 1024),
                ("ln_pe_2", 1280),
                ("ln1", 1024),
                ("ln2_1", 1024),
                ("ln2_2", 1280),
                ("ln3", 1024),
            ] {
                ln(&mut expected, &format!("{p}.{name}"), n);
            }
            for (attn, context) in [("self_attn", 1024), ("cross_attn", 1280)] {
                for (name, n) in [("q_proj", 1024), ("k_proj", context), ("v_proj", context)] {
                    linear(&mut expected, &format!("{p}.{attn}.{name}"), n, 512);
                }
                linear(&mut expected, &format!("{p}.{attn}.proj"), 512, 1024);
            }
            ffn(&mut expected, &format!("{p}.ffn"), 1024, 1024);
        }
        let mut actual = BTreeMap::new();
        for o in &manifest.objects {
            for t in &o.tensors {
                ensure!(
                    t.dtype == "f32" && actual.insert(t.name.clone(), t.shape.clone()).is_none(),
                    "Invalid SAM tensor"
                );
            }
        }
        ensure!(actual == expected, "SAM tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        for (i, o) in manifest.objects.iter().enumerate() {
            weights.load_object(source, o).await?;
            progress(i + 1, manifest.objects.len());
        }
        Ok(Self { weights, head })
    }
    fn mlp(&self, prefix: &str, x: Tensor<B, 3>, gelu: bool) -> Tensor<B, 3> {
        let x = self.weights.affine(&format!("{prefix}.layers.0.0"), x);
        let x = if gelu {
            activation::gelu(x)
        } else {
            activation::relu(x)
        };
        self.weights.affine(&format!("{prefix}.layers.1"), x)
    }
    fn attn(
        &self,
        prefix: &str,
        q: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let n = q.dims()[1];
        let m = k.dims()[1];
        let w = &self.weights;
        let q = w
            .affine(&format!("{prefix}.q_proj"), q)
            .reshape([1, n, 8, 64])
            .swap_dims(1, 2);
        let k = w
            .affine(&format!("{prefix}.k_proj"), k)
            .reshape([1, m, 8, 64])
            .swap_dims(1, 2);
        let v = w
            .affine(&format!("{prefix}.v_proj"), v)
            .reshape([1, m, 8, 64])
            .swap_dims(1, 2);
        w.affine(
            &format!("{prefix}.proj"),
            attention(q, k, v).swap_dims(1, 2).reshape([1, n, 512]),
        )
    }
    pub fn condition_image(
        &self,
        embedding: Tensor<B, 4>,
        crop: Crop,
        camera: Camera,
    ) -> Result<Tensor<B, 3>> {
        crop.validate()?;
        camera.validate()?;
        ensure!(
            embedding.dims() == [1, 1280, 32, 32],
            "SAM image feature shape"
        );
        // Antialiased bilinear downsampling of an affine ray field is separable.
        // Compute each filter's exact first moment, including renormalized edges.
        let centers: Vec<f32> = (0..32)
            .map(|o| {
                let center = (o as f32 + 0.5) * 16.0 - 0.5;
                let mut sum = 0.0;
                let mut weight = 0.0;
                for i in 0..512 {
                    let w = (1.0 - ((i as f32 - center) / 16.0).abs()).max(0.0);
                    sum += i as f32 * w;
                    weight += w;
                }
                sum / weight
            })
            .collect();
        let mut rays = Vec::with_capacity(1024 * 99);
        for y in 0..32 {
            for x in 0..32 {
                let pos = [
                    ((centers[x] / 512.0 - 0.5) * crop.size + crop.center[0] - camera.center[0])
                        / camera.focal[0],
                    ((centers[y] / 512.0 - 0.5) * crop.size + crop.center[1] - camera.center[1])
                        / camera.focal[1],
                    1.0,
                ];
                rays.extend(pos);
                for cosine in [false, true] {
                    for p in pos {
                        for i in 0..16 {
                            let a = p * (1.0 + i as f32 * 31.0 / 15.0) * std::f32::consts::PI;
                            rays.push(if cosine { a.cos() } else { a.sin() });
                        }
                    }
                }
            }
        }
        let rays =
            Tensor::<B, 3>::from_data(TensorData::new(rays, [1, 1024, 99]), &self.weights.device);
        let image = embedding.reshape([1, 1280, 1024]).swap_dims(1, 2)
            + self.weights.tensor::<1>("prompt.no_mask").unsqueeze();
        let out = self.weights.linear(
            "ray_cond_emb.conv.weight",
            Tensor::cat(vec![image, rays], 2),
        );
        Ok(norm(&self.weights, "ray_cond_emb.norm", out, 1e-6))
    }
    /// One person's image crop. Five intermediate MHR projections update the
    /// following layer's 2D/3D keypoint tokens; the sixth supplies final diagnostics.
    pub async fn forward(
        &self,
        embedding: Tensor<B, 4>,
        mhr: &Mhr<B>,
        crop: Crop,
        camera: Camera,
    ) -> Result<SamOutput<B>> {
        let w = &self.weights;
        let device = &w.device;
        let image = self.condition_image(embedding, crop, camera)?;
        let init = Tensor::cat(
            vec![
                w.tensor::<2>("init_pose.weight"),
                w.tensor::<2>("init_camera.weight"),
            ],
            1,
        )
        .unsqueeze::<3>();
        let condition = Tensor::<B, 3>::from_data(
            TensorData::new(crop.condition(camera).to_vec(), [1, 1, 3]),
            device,
        );
        let primary = w.affine(
            "init_to_token_mhr",
            Tensor::cat(vec![condition, init.clone()], 2),
        );
        let prev = w.affine("prev_to_token_mhr", init);
        let prompt = w.affine("prompt_to_token", w.tensor("prompt.invalid"));
        let mut x = Tensor::cat(
            vec![
                primary,
                prev.clone(),
                prompt.clone(),
                w.tensor::<2>("keypoint_embedding.weight").unsqueeze(),
                w.tensor::<2>("keypoint3d_embedding.weight").unsqueeze(),
            ],
            1,
        );
        let mut pe = Tensor::cat(
            vec![
                Tensor::zeros([1, 1, 1024], device),
                prev,
                prompt,
                Tensor::zeros([1, 140, 1024], device),
            ],
            1,
        );
        let image_pe = w.tensor::<3>("prompt.image_pe");
        let mut layers = vec![];
        for i in 0..6 {
            let p = format!("decoder.layers.{i}");
            let xp = norm(w, &format!("{p}.ln_pe_1"), pe.clone(), 1e-6);
            let ip = norm(w, &format!("{p}.ln_pe_2"), image_pe.clone(), 1e-6);
            let z = norm(w, &format!("{p}.ln1"), x.clone(), 1e-6);
            let q = if i == 0 {
                z.clone()
            } else {
                z.clone() + xp.clone()
            };
            x = x + self.attn(&format!("{p}.self_attn"), q.clone(), q, z);
            let q = norm(w, &format!("{p}.ln2_1"), x.clone(), 1e-6) + xp;
            let v = norm(w, &format!("{p}.ln2_2"), image.clone(), 1e-6);
            x = x + self.attn(&format!("{p}.cross_attn"), q, v.clone() + ip, v);
            let z = norm(w, &format!("{p}.ln3"), x.clone(), 1e-6);
            x = x + self.mlp(&format!("{p}.ffn"), z, true);
            let token = norm(w, "decoder.norm_final", x.clone(), 1e-6).slice([0..1, 0..1, 0..1024]);
            let raw = self.mlp("head_pose.proj", token.clone(), false)
                + w.tensor::<2>("init_pose.weight").unsqueeze();
            let raw = raw
                .into_data_async()
                .await?
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("SAM head: {e}"))?;
            let parameters = self.head.parameters(&raw)?;
            let cam = self.mlp("head_camera.proj", token, false)
                + w.tensor::<2>("init_camera.weight").unsqueeze();
            let cam = cam
                .into_data_async()
                .await?
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("SAM camera: {e}"))?;
            let cam: [f32; 3] = cam.try_into().unwrap();
            let output = mhr.evaluate(std::slice::from_ref(&parameters))?;
            let joints: Vec<f32> = output.skeleton_world[0]
                .iter()
                .flat_map(|m| m.w_axis.truncate().to_array())
                .collect();
            let joints = Tensor::<B, 3>::from_data(TensorData::new(joints, [1, 127, 3]), device);
            let mesh = Tensor::cat(vec![output.vertices, joints], 1) * 0.01;
            let kp = w
                .tensor::<2>("head_pose.keypoint_mapping")
                .unsqueeze::<3>()
                .matmul(mesh);
            let values = kp
                .into_data_async()
                .await?
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("SAM keypoints: {e}"))?;
            let keypoints: Vec<[f32; 3]> = values
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| [p[0], -p[1], -p[2]])
                .collect();
            if i < 5 {
                let bs = -crop.size * cam[0] + 1e-8;
                ensure!(bs.is_finite() && bs.abs() > 1e-8, "Degenerate SAM camera");
                let t = [
                    cam[1] + 2.0 * (crop.center[0] - camera.center[0]) / bs,
                    -cam[2] + 2.0 * (crop.center[1] - camera.center[1]) / bs,
                    2.0 * camera.focal[0] / bs,
                ];
                let mut xy = vec![];
                let mut valid = vec![];
                let mut ij = [vec![], vec![], vec![], vec![]];
                let mut factors = [vec![], vec![], vec![], vec![]];
                for kp in &keypoints {
                    let depth = kp[2] + t[2];
                    let coords = [
                        ((kp[0] + t[0]) / depth * camera.focal[0] + camera.center[0]
                            - crop.center[0])
                            / crop.size,
                        ((kp[1] + t[1]) / depth * camera.focal[1] + camera.center[1]
                            - crop.center[1])
                            / crop.size,
                    ];
                    ensure!(
                        coords.iter().all(|v| v.is_finite()),
                        "Non-finite SAM projection"
                    );
                    let ok = depth >= 1e-5 && coords.iter().all(|v| (-0.5..=0.5).contains(v));
                    xy.extend(coords);
                    valid.push(if ok { 1.0 } else { 0.0 });
                    let px = (coords[0] + 0.5) * 32.0 - 0.5;
                    let py = (coords[1] + 0.5) * 32.0 - 0.5;
                    let (ix, iy) = (px.floor() as i32, py.floor() as i32);
                    let (fx, fy) = (px - px.floor(), py - py.floor());
                    for (j, (dx, dy, weight)) in [
                        (0, 0, (1.0 - fx) * (1.0 - fy)),
                        (1, 0, fx * (1.0 - fy)),
                        (0, 1, (1.0 - fx) * fy),
                        (1, 1, fx * fy),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let (a, b) = (ix.saturating_add(dx), iy.saturating_add(dy));
                        let inside = ok && (0..32).contains(&a) && (0..32).contains(&b);
                        ij[j].push(if inside { b * 32 + a } else { 0 });
                        factors[j].push(if inside { weight } else { 0.0 });
                    }
                }
                let xy = Tensor::<B, 3>::from_data(TensorData::new(xy, [1, 70, 2]), device);
                let valid = Tensor::<B, 3>::from_data(TensorData::new(valid, [1, 70, 1]), device);
                let mut sampled = Tensor::zeros([1, 70, 1280], device);
                for (indices, scale) in ij.into_iter().zip(factors) {
                    let indices =
                        Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [70]), device);
                    let scale =
                        Tensor::<B, 3>::from_data(TensorData::new(scale, [1, 70, 1]), device);
                    sampled = sampled + image.clone().select(1, indices) * scale;
                }
                let old = x.clone().slice([0..1, 3..73, 0..1024]);
                x = x.slice_assign(
                    [0..1, 3..73, 0..1024],
                    old + w.affine("keypoint_feat_linear", sampled),
                );
                pe = pe.slice_assign(
                    [0..1, 3..73, 0..1024],
                    self.mlp("keypoint_posemb_linear", xy, false) * valid,
                );
                let pelvis: [f32; 3] =
                    std::array::from_fn(|j| (keypoints[9][j] + keypoints[10][j]) * 0.5);
                let centered: Vec<f32> = keypoints
                    .iter()
                    .flat_map(|p| (0..3).map(|j| p[j] - pelvis[j]))
                    .collect();
                let centered =
                    Tensor::<B, 3>::from_data(TensorData::new(centered, [1, 70, 3]), device);
                pe = pe.slice_assign(
                    [0..1, 73..143, 0..1024],
                    self.mlp("keypoint3d_posemb_linear", centered, false),
                );
            }
            layers.push(SamLayer {
                parameters,
                keypoints,
                camera: cam,
            });
        }
        Ok(SamOutput {
            token: norm(w, "decoder.norm_final", x, 1e-6).slice([0..1, 0..1, 0..1024]),
            layers,
        })
    }
}

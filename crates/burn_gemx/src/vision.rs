//! DINOv3 H+ backbones and the ViTPose heatmap head from the GEM-X release.
//! RoPE tables preserve the source export's numerical convention.
use crate::ops::{attention, norm};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{
        Tensor, activation, module,
        ops::{ConvOptions, ConvTransposeOptions},
    },
};
use burn_human_inference::{transport::ModelSource, weights::TensorBank};
use burn_human_motion::artifacts::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    pub kind: String,
    pub height: usize,
    pub width: usize,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub patch: usize,
}
pub struct Vision<B: Backend> {
    weights: TensorBank<B>,
    pub config: VisionConfig,
}
impl<B: Backend> Vision<B> {
    pub async fn load(
        manifest: &Manifest,
        source: &mut ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        let config: VisionConfig = serde_json::from_value(manifest.config.clone())?;
        ensure!(
            manifest.model == format!("nvidia/GEM-X:{}", config.kind)
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.source_revision == crate::SOURCE_REVISION,
            "Unsupported vision checkpoint"
        );
        ensure!(
            config.layers == 32
                && config.hidden == 1280
                && config.heads == 20
                && config.patch == 16
                && ((config.kind == "vitpose" && config.height == 256 && config.width == 192)
                    || (config.kind == "sam-body" && config.height == 512 && config.width == 512)),
            "Unsupported vision architecture"
        );
        let mut expected: BTreeMap<String, Vec<usize>> = [
            ("cls_token", vec![1, 1, 1280]),
            ("storage_tokens", vec![1, 4, 1280]),
            ("patch_embed.proj.weight", vec![1280, 3, 16, 16]),
            ("patch_embed.proj.bias", vec![1280]),
            (
                "rope.angles",
                vec![config.height / 16 * config.width / 16, 64],
            ),
            ("norm.weight", vec![1280]),
            ("norm.bias", vec![1280]),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
        for i in 0..32 {
            let p = format!("blocks.{i}");
            for n in [
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "ls1.gamma",
                "ls2.gamma",
                "attn.proj.bias",
                "mlp.w3.bias",
            ] {
                expected.insert(format!("{p}.{n}"), vec![1280]);
            }
            expected.insert(format!("{p}.attn.qkv.weight"), vec![3840, 1280]);
            expected.insert(format!("{p}.attn.qkv.bias"), vec![3840]);
            expected.insert(format!("{p}.attn.proj.weight"), vec![1280, 1280]);
            for n in ["w1", "w2"] {
                expected.insert(format!("{p}.mlp.{n}.weight"), vec![5120, 1280]);
                expected.insert(format!("{p}.mlp.{n}.bias"), vec![5120]);
            }
            expected.insert(format!("{p}.mlp.w3.weight"), vec![1280, 5120]);
        }
        if config.kind == "vitpose" {
            for (n, s) in [
                ("head.deconv.0.weight", vec![1280, 256, 4, 4]),
                ("head.deconv.0.bias", vec![256]),
                ("head.deconv.1.weight", vec![256, 256, 4, 4]),
                ("head.deconv.1.bias", vec![256]),
                ("head.final.weight", vec![77, 256, 1, 1]),
                ("head.final.bias", vec![77]),
            ] {
                expected.insert(n.into(), s);
            }
        }
        let mut actual = BTreeMap::new();
        for o in &manifest.objects {
            for t in &o.tensors {
                ensure!(
                    t.dtype == "f32" && actual.insert(t.name.clone(), t.shape.clone()).is_none(),
                    "Invalid vision tensor"
                );
            }
        }
        ensure!(actual == expected, "Vision tensor inventory mismatch");
        let mut weights = TensorBank::new(device);
        for (i, o) in manifest.objects.iter().enumerate() {
            weights.load_object(source, o).await?;
            progress(i + 1, manifest.objects.len());
        }
        Ok(Self { weights, config })
    }

    /// ImageNet-normalized RGB. Batches are bounded because full attention is quadratic.
    pub fn forward(&self, input: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
        let [batch, c, h, w] = input.dims();
        ensure!(
            (1..=8).contains(&batch) && c == 3 && h == self.config.height && w == self.config.width,
            "Vision input dimensions"
        );
        let (h, w) = (h / 16, w / 16);
        let count = h * w;
        let n = count + 5;
        let bank = &self.weights;
        let patch = module::conv2d(
            input,
            bank.tensor("patch_embed.proj.weight"),
            Some(bank.tensor("patch_embed.proj.bias")),
            ConvOptions::new([16, 16], [0, 0], [1, 1], 1),
        )
        .reshape([batch, 1280, count])
        .swap_dims(1, 2);
        let mut x = Tensor::cat(
            vec![
                bank.tensor::<3>("cls_token").expand([batch, 1, 1280]),
                bank.tensor::<3>("storage_tokens").expand([batch, 4, 1280]),
                patch,
            ],
            1,
        );
        let angles = bank.tensor::<2>("rope.angles").reshape([1, 1, count, 64]);
        let cos = angles.clone().cos();
        let sin = angles.sin();
        let rope = |t: Tensor<B, 4>| {
            let prefix = t.clone().slice([0..batch, 0..20, 0..5, 0..64]);
            let body = t.slice([0..batch, 0..20, 5..n, 0..64]);
            let half = Tensor::cat(
                vec![
                    -body.clone().slice([0..batch, 0..20, 0..count, 32..64]),
                    body.clone().slice([0..batch, 0..20, 0..count, 0..32]),
                ],
                3,
            );
            Tensor::cat(vec![prefix, body * cos.clone() + half * sin.clone()], 2)
        };
        for i in 0..32 {
            let p = format!("blocks.{i}");
            let z = norm(bank, &format!("{p}.norm1"), x.clone(), 1e-6);
            let qkv = bank
                .affine(&format!("{p}.attn.qkv"), z)
                .reshape([batch, n, 3, 20, 64]);
            let head = |i| {
                qkv.clone()
                    .slice([0..batch, 0..n, i..i + 1, 0..20, 0..64])
                    .reshape([batch, n, 20, 64])
                    .swap_dims(1, 2)
            };
            let out = attention(rope(head(0)), rope(head(1)), head(2))
                .swap_dims(1, 2)
                .reshape([batch, n, 1280]);
            x = x + bank.affine(&format!("{p}.attn.proj"), out)
                * bank.tensor::<1>(&format!("{p}.ls1.gamma")).unsqueeze();
            let z = norm(bank, &format!("{p}.norm2"), x.clone(), 1e-6);
            let z = activation::silu(bank.affine(&format!("{p}.mlp.w1"), z.clone()))
                * bank.affine(&format!("{p}.mlp.w2"), z);
            x = x + bank.affine(&format!("{p}.mlp.w3"), z)
                * bank.tensor::<1>(&format!("{p}.ls2.gamma")).unsqueeze();
        }
        let x = norm(bank, "norm", x, 1e-6)
            .slice([0..batch, 5..n, 0..1280])
            .reshape([batch, h, w, 1280])
            .permute([0, 3, 1, 2]);
        if self.config.kind != "vitpose" {
            return Ok(x);
        }
        let mut x = x;
        for i in 0..2 {
            x = activation::relu(module::conv_transpose2d(
                x,
                bank.tensor(&format!("head.deconv.{i}.weight")),
                Some(bank.tensor(&format!("head.deconv.{i}.bias"))),
                ConvTransposeOptions::new([2, 2], [1, 1], [0, 0], [1, 1], 1),
            ));
        }
        Ok(module::conv2d(
            x,
            bank.tensor("head.final.weight"),
            Some(bank.tensor("head.final.bias")),
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        ))
    }
}

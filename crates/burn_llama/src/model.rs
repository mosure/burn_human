//! Llama 3 8B with merged LLM2Vec adapters, grouped query attention and paged vocabulary.
use crate::tokenizer::{EncodedPrompt, PromptTokenizer, SEQUENCE};
use anyhow::{Result, ensure};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData, activation, module::attention, ops::AttentionModuleOptions},
};
use burn_human_inference::{
    transport::ModelSource,
    weights::{TensorBank, read_tensors},
};
use burn_human_motion::{TextEmbedding, artifacts::Manifest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// ARDY's LLM2Vec contract is bidirectional. The public ONNX export contains a
/// causal mask; select CausalExport only to reproduce that export explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMode {
    #[default]
    Bidirectional,
    CausalExport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub page_rows: usize,
    pub max_sequence: usize,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
}

impl TextConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden == 4096
                && self.intermediate == 14336
                && self.layers == 32
                && self.heads == 32
                && self.kv_heads == 8
                && self.vocab == 128256
                && self.page_rows == 512
                && self.max_sequence == SEQUENCE
                && self.rms_epsilon == 1e-5
                && self.rope_theta == 500000.0,
            "Unsupported Llama architecture"
        );
        Ok(())
    }
    fn inventory(&self) -> BTreeMap<String, (Vec<usize>, &'static str)> {
        let mut out = BTreeMap::new();
        out.insert("model.norm.weight".into(), (vec![self.hidden], "f16"));
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            for n in ["input_layernorm", "post_attention_layernorm"] {
                out.insert(format!("{p}.{n}.weight"), (vec![self.hidden], "f16"));
            }
            for (n, output, input) in [
                ("self_attn.q_proj", 4096, 4096),
                ("self_attn.k_proj", 1024, 4096),
                ("self_attn.v_proj", 1024, 4096),
                ("self_attn.o_proj", 4096, 4096),
                ("mlp.gate_proj", 14336, 4096),
                ("mlp.up_proj", 14336, 4096),
                ("mlp.down_proj", 4096, 14336),
            ] {
                out.insert(format!("{p}.{n}.weight"), (vec![output, input], "q4f32"));
            }
        }
        for page in 0..self.vocab.div_ceil(self.page_rows) {
            out.insert(
                page_name(page),
                (
                    vec![
                        (self.vocab - page * self.page_rows).min(self.page_rows),
                        self.hidden,
                    ],
                    "f16",
                ),
            );
        }
        out
    }
}

fn page_name(page: usize) -> String {
    format!("model.embed_tokens.page.{page:04}")
}

pub struct TextEncoder<B: Backend> {
    pub config: TextConfig,
    pub mode: AttentionMode,
    weights: TensorBank<B>,
    tokenizer: PromptTokenizer,
    manifest: Manifest,
    source: ModelSource,
    // At most 64 MiB of vocabulary pages; only selected rows reach the GPU.
    pages: VecDeque<(usize, Vec<u8>)>,
}

impl<B: Backend> TextEncoder<B> {
    pub async fn load(
        manifest: Manifest,
        mut source: ModelSource,
        device: &B::Device,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        manifest.validate()?;
        ensure!(
            manifest.model == crate::MODEL_ID
                && manifest.model_revision == crate::MODEL_REVISION
                && manifest.schema_version == 2,
            "Unrecognized ARDY text checkpoint"
        );
        let config: TextConfig = serde_json::from_value(manifest.config.clone())?;
        config.validate()?;
        let expected = config.inventory();
        let mut actual = BTreeMap::new();
        for object in &manifest.objects {
            for t in &object.tensors {
                ensure!(
                    actual
                        .insert(t.name.clone(), (t.shape.clone(), t.dtype.as_str()))
                        .is_none(),
                    "duplicate model tensor"
                );
            }
        }
        ensure!(
            actual == expected,
            "Text checkpoint tensor inventory differs from the pinned architecture"
        );
        let asset = manifest
            .assets
            .iter()
            .find(|a| a.path == "metadata/tokenizer.json")
            .ok_or_else(|| anyhow::anyhow!("Missing tokenizer"))?;
        let tokenizer = PromptTokenizer::from_bytes(&source.asset(asset).await?)?;
        let mut weights = TensorBank::new(device);
        let objects: Vec<_> = manifest
            .objects
            .iter()
            .filter(|o| !o.stage.starts_with("model.embed_tokens.page."))
            .collect();
        for (i, object) in objects.iter().enumerate() {
            weights.load_object(&mut source, object).await?;
            progress(i + 1, objects.len());
        }
        Ok(Self {
            config,
            mode: AttentionMode::default(),
            weights,
            tokenizer,
            manifest,
            source,
            pages: VecDeque::new(),
        })
    }

    pub fn tokenize(&self, prompt: &str) -> Result<EncodedPrompt> {
        self.tokenizer.encode(prompt)
    }

    async fn embeddings(&mut self, tokens: &EncodedPrompt) -> Result<Tensor<B, 3>> {
        let mut values = vec![0f32; SEQUENCE * self.config.hidden];
        for (position, &id) in tokens.ids.iter().enumerate() {
            let id = id as usize;
            ensure!(id < self.config.vocab, "Token outside vocabulary");
            let page = id / self.config.page_rows;
            if let Some(index) = self.pages.iter().position(|(p, _)| *p == page) {
                let entry = self.pages.remove(index).unwrap();
                self.pages.push_back(entry);
            } else {
                let name = page_name(page);
                let object = self
                    .manifest
                    .objects
                    .iter()
                    .find(|o| o.stage == name)
                    .ok_or_else(|| anyhow::anyhow!("Missing vocabulary page"))?;
                let data = read_tensors(&mut self.source, object)
                    .await?
                    .remove(&name)
                    .unwrap();
                if self.pages.len() == 16 {
                    self.pages.pop_front();
                }
                self.pages.push_back((page, data.bytes.to_vec()));
            }
            let bytes = &self.pages.back().unwrap().1;
            let start = (id % self.config.page_rows) * self.config.hidden * 2;
            for (out, raw) in values
                [position * self.config.hidden..(position + 1) * self.config.hidden]
                .iter_mut()
                .zip(
                    bytes[start..start + self.config.hidden * 2]
                        .as_chunks::<2>()
                        .0
                        .iter(),
                )
            {
                *out = half::f16::from_bits(u16::from_le_bytes(*raw)).to_f32();
            }
        }
        Ok(Tensor::from_data(
            TensorData::new(values, [1, SEQUENCE, self.config.hidden]),
            &self.weights.device,
        ))
    }

    fn norm(&self, name: &str, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let inverse = (x.clone().square().mean_dim(2) + self.config.rms_epsilon)
            .sqrt()
            .recip();
        x * inverse * self.weights.tensor::<1>(name).unsqueeze()
    }

    fn rope(&self, x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, t, d] = x.dims();
        let rotated = Tensor::cat(
            vec![
                x.clone().slice([0..b, 0..h, 0..t, d / 2..d]).neg(),
                x.clone().slice([0..b, 0..h, 0..t, 0..d / 2]),
            ],
            3,
        );
        x * cos + rotated * sin
    }

    /// Encode locally on the selected Burn device. No runtime Python or HTTP service.
    pub async fn encode(
        &mut self,
        prompt: &str,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<TextEmbedding> {
        let tokens = self.tokenize(prompt)?;
        let mut x = self.embeddings(&tokens).await?;
        let device = &self.weights.device;
        let d = self.config.hidden / self.config.heads;
        let mut cos = Vec::with_capacity(SEQUENCE * d);
        let mut sin = Vec::with_capacity(SEQUENCE * d);
        for t in 0..SEQUENCE {
            for k in 0..d {
                let theta = t as f32
                    / self
                        .config
                        .rope_theta
                        .powf((2 * (k % (d / 2))) as f32 / d as f32);
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let cos = Tensor::<B, 4>::from_data(TensorData::new(cos, [1, 1, SEQUENCE, d]), device);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sin, [1, 1, SEQUENCE, d]), device);
        let mut bias = Vec::with_capacity(SEQUENCE * SEQUENCE);
        for q in 0..SEQUENCE {
            for (k, &visible) in tokens.attention.iter().enumerate() {
                bias.push(
                    if visible && (self.mode == AttentionMode::Bidirectional || k <= q) {
                        0.0
                    } else {
                        -65504.0
                    },
                );
            }
        }
        let bias =
            Tensor::<B, 4>::from_data(TensorData::new(bias, [1, 1, SEQUENCE, SEQUENCE]), device);
        for i in 0..self.config.layers {
            let p = format!("model.layers.{i}");
            let normalized = self.norm(&format!("{p}.input_layernorm.weight"), x.clone());
            let project = |suffix: &str, heads| {
                self.weights
                    .linear(
                        &format!("{p}.self_attn.{suffix}_proj.weight"),
                        normalized.clone(),
                    )
                    .reshape([1, SEQUENCE, heads, d])
                    .swap_dims(1, 2)
            };
            let q = self.rope(project("q", self.config.heads), cos.clone(), sin.clone());
            let k = self.rope(project("k", self.config.kv_heads), cos.clone(), sin.clone());
            let v = project("v", self.config.kv_heads);
            let repeat = |v: Tensor<B, 4>| {
                v.unsqueeze_dim::<5>(2)
                    .expand([
                        1,
                        self.config.kv_heads,
                        self.config.heads / self.config.kv_heads,
                        SEQUENCE,
                        d,
                    ])
                    .reshape([1, self.config.heads, SEQUENCE, d])
            };
            let attended = attention(
                q,
                repeat(k),
                repeat(v),
                None,
                Some(bias.clone()),
                AttentionModuleOptions::default(),
            )
            .swap_dims(1, 2)
            .reshape([1, SEQUENCE, self.config.hidden]);
            x = x + self
                .weights
                .linear(&format!("{p}.self_attn.o_proj.weight"), attended);
            let normalized = self.norm(&format!("{p}.post_attention_layernorm.weight"), x.clone());
            let gate = activation::silu(
                self.weights
                    .linear(&format!("{p}.mlp.gate_proj.weight"), normalized.clone()),
            );
            let up = self
                .weights
                .linear(&format!("{p}.mlp.up_proj.weight"), normalized);
            x = x + self
                .weights
                .linear(&format!("{p}.mlp.down_proj.weight"), gate * up);
            progress(i + 1, self.config.layers);
        }
        let pool = Tensor::<B, 3>::from_data(
            TensorData::new(tokens.pool.to_vec(), [1, SEQUENCE, 1]),
            device,
        );
        let pooled =
            (self.norm("model.norm.weight", x) * pool).sum_dim(1) / tokens.pool.iter().sum::<f32>();
        let values = pooled
            .into_data_async()
            .await?
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("embedding: {e}"))?;
        let result = TextEmbedding {
            prompt: prompt.into(),
            encoder: format!("{}:burn:{:?}", crate::MODEL_ID, self.mode),
            revision: crate::MODEL_REVISION.into(),
            values,
        };
        result.validate(prompt, self.config.hidden)?;
        Ok(result)
    }
}

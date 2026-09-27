//! Incremental initialization of the pinned Llama BPE vocabulary and merge ranks.
//!
//! Tokenizers still handles normalization, pretokenization, added tokens and the
//! chat postprocessor. Only its eager BPE table construction is replaced. The
//! rank/leftmost merge policy matches Hugging Face tokenizers (Apache-2.0).
use anyhow::{Result, ensure};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::value::RawValue;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    path::{Path, PathBuf},
};
use tokenizers::{
    AddedToken, Model, Token, TokenizerImpl, decoders::DecoderWrapper, models::bpe::BpeTrainer,
    normalizers::NormalizerWrapper, pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,
};

type Tokenizer = TokenizerImpl<
    Bpe,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;
pub(super) struct CooperativeTokenizer(Tokenizer);

#[derive(Deserialize)]
struct Definition<'a> {
    version: String,
    #[serde(borrow)]
    model: DefinitionBpe<'a>,
    added_tokens: Vec<Added>,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: Option<PreTokenizerWrapper>,
    post_processor: Option<PostProcessorWrapper>,
    decoder: Option<DecoderWrapper>,
}
#[derive(Deserialize)]
struct Added {
    id: u32,
    #[serde(flatten)]
    token: AddedToken,
}
#[derive(Deserialize)]
struct DefinitionBpe<'a> {
    #[serde(rename = "type")]
    kind: String,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    #[serde(borrow)]
    vocab: &'a RawValue,
    #[serde(borrow)]
    merges: &'a RawValue,
}

impl CooperativeTokenizer {
    pub(super) async fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Asset authentication may have consumed most of the current task.
        burn_human_inference::cooperative::yield_to_browser().await;
        // Borrow the two large JSON containers instead of materializing an
        // untagged Value tree and converting it into another owned map.
        let definition: Definition<'_> = serde_json::from_slice(bytes)?;
        let model = definition.model;
        ensure!(
            definition.version == "1.0" && model.kind == "BPE",
            "Unsupported tokenizer format"
        );
        ensure!(
            model.dropout.is_none()
                && model.unk_token.is_none()
                && model.continuing_subword_prefix.is_none()
                && model.end_of_word_suffix.is_none()
                && !model.fuse_unk
                && !model.byte_fallback,
            "Unsupported Llama BPE options"
        );
        burn_human_inference::cooperative::yield_to_browser().await;
        let mut vocab = HashMap::with_capacity(128000);
        let mut reverse = vec![String::new(); 128000];
        let mut input = model.vocab.get().as_bytes();
        delimiter(&mut input, b'{')?;
        while !end(&mut input, b'}') {
            if !vocab.is_empty() {
                delimiter(&mut input, b',')?;
            }
            let token: String = take(&mut input)?;
            delimiter(&mut input, b':')?;
            let id: u32 = take(&mut input)?;
            ensure!(
                id < 128000 && !token.is_empty(),
                "Invalid Llama vocabulary entry"
            );
            ensure!(reverse[id as usize].is_empty(), "Duplicate vocabulary id");
            reverse[id as usize] = token.clone();
            ensure!(
                vocab.insert(token, id).is_none(),
                "Duplicate vocabulary token"
            );
            if vocab.len().is_multiple_of(2048) {
                burn_human_inference::cooperative::yield_to_browser().await;
            }
        }
        ensure!(vocab.len() == 128000, "Incomplete Llama vocabulary");
        let mut merges = HashMap::with_capacity(280147);
        let mut input = model.merges.get().as_bytes();
        delimiter(&mut input, b'[')?;
        let mut rank = 0u32;
        while !end(&mut input, b']') {
            if rank != 0 {
                delimiter(&mut input, b',')?;
            }
            // This pinned export uses the original string-pair representation.
            let pair: String = take(&mut input)?;
            let (left, right) = pair
                .split_once(' ')
                .ok_or_else(|| anyhow::anyhow!("Invalid BPE merge"))?;
            let lookup = |token: &str| {
                vocab
                    .get(token)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("BPE merge outside vocabulary"))
            };
            let key = (lookup(left)?, lookup(right)?);
            let value = (rank, lookup(&format!("{left}{right}"))?);
            ensure!(merges.insert(key, value).is_none(), "Duplicate BPE merge");
            rank += 1;
            ensure!(rank <= 1_000_000, "BPE merge limit");
            if rank.is_multiple_of(2048) {
                burn_human_inference::cooperative::yield_to_browser().await;
            }
        }
        let mut tokenizer = Tokenizer::new(Bpe {
            vocab,
            reverse,
            merges,
            ignore_merges: model.ignore_merges,
        });
        tokenizer.with_normalizer(definition.normalizer);
        tokenizer.with_pre_tokenizer(definition.pre_tokenizer);
        tokenizer.with_post_processor(definition.post_processor);
        tokenizer.with_decoder(definition.decoder);
        let (ids, tokens): (Vec<_>, Vec<_>) = definition
            .added_tokens
            .into_iter()
            .map(|a| (a.id, a.token))
            .unzip();
        tokenizer.add_tokens(&tokens);
        for (id, token) in ids.into_iter().zip(tokens) {
            ensure!(
                tokenizer.token_to_id(&token.content) == Some(id),
                "Added token id differs from checkpoint"
            );
        }
        Ok(Self(tokenizer))
    }

    pub(super) fn encode(&self, text: String) -> Result<Vec<u32>> {
        Ok(self
            .0
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            .get_ids()
            .to_vec())
    }
}

fn whitespace(input: &mut &[u8]) {
    *input = input.trim_ascii_start();
}
fn end(input: &mut &[u8], byte: u8) -> bool {
    whitespace(input);
    input.first() == Some(&byte)
}
fn delimiter(input: &mut &[u8], byte: u8) -> Result<()> {
    whitespace(input);
    ensure!(
        input.first() == Some(&byte),
        "Invalid tokenizer JSON delimiter"
    );
    *input = &input[1..];
    Ok(())
}
fn take<T: DeserializeOwned>(input: &mut &[u8]) -> Result<T> {
    let mut stream = serde_json::Deserializer::from_slice(input).into_iter::<T>();
    let value = stream
        .next()
        .ok_or_else(|| anyhow::anyhow!("Truncated tokenizer JSON"))??;
    *input = &input[stream.byte_offset()..];
    Ok(value)
}

#[derive(Clone)]
struct Bpe {
    vocab: HashMap<String, u32>,
    reverse: Vec<String>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    ignore_merges: bool,
}
#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    previous: Option<usize>,
    next: Option<usize>,
    start: usize,
    end: usize,
    live: bool,
}
impl Model for Bpe {
    type Trainer = BpeTrainer;
    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(sequence)
        {
            return Ok(vec![Token::new(id, sequence.into(), (0, sequence.len()))]);
        }
        let mut symbols = Vec::new();
        for (start, c) in sequence.char_indices() {
            let end = start + c.len_utf8();
            let id = *self
                .vocab
                .get(&sequence[start..end])
                .ok_or("Missing byte-alphabet token")?;
            let i = symbols.len();
            symbols.push(Symbol {
                id,
                previous: i.checked_sub(1),
                next: Some(i + 1),
                start,
                end,
                live: true,
            });
        }
        if let Some(last) = symbols.last_mut() {
            last.next = None;
        }
        let mut queue = BinaryHeap::with_capacity(symbols.len());
        let enqueue = |i: usize, symbols: &[Symbol], queue: &mut BinaryHeap<_>| {
            if let Some(j) = symbols[i].next
                && let Some(&(rank, id)) = self.merges.get(&(symbols[i].id, symbols[j].id))
            {
                queue.push(Reverse((rank, i, id)));
            }
        };
        for i in 0..symbols.len() {
            enqueue(i, &symbols, &mut queue);
        }
        while let Some(Reverse((_, i, id))) = queue.pop() {
            if !symbols[i].live {
                continue;
            }
            let Some(j) = symbols[i].next else {
                continue;
            };
            if self
                .merges
                .get(&(symbols[i].id, symbols[j].id))
                .is_none_or(|&(_, new)| new != id)
            {
                continue;
            }
            let right = symbols[j];
            symbols[i].id = id;
            symbols[i].end = right.end;
            symbols[i].next = right.next;
            symbols[j].live = false;
            if let Some(next) = right.next {
                symbols[next].previous = Some(i);
            }
            if let Some(previous) = symbols[i].previous {
                enqueue(previous, &symbols, &mut queue);
            }
            enqueue(i, &symbols, &mut queue);
        }
        Ok(symbols
            .into_iter()
            .filter(|s| s.live)
            .map(|s| Token::new(s.id, self.reverse[s.id as usize].clone(), (s.start, s.end)))
            .collect())
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.reverse.get(id as usize).cloned()
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
    fn save(&self, _: &Path, _: Option<&str>) -> tokenizers::Result<Vec<PathBuf>> {
        Err("Inference-only Llama BPE".into())
    }
    fn get_trainer(&self) -> Self::Trainer {
        unreachable!("private inference-only BPE model")
    }
}

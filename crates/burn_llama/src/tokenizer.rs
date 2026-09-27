//! The upstream LLM2Vec chat template, left padding, and instruction-excluding pooling.
use anyhow::{Result, ensure};
mod bpe;

pub const SEQUENCE: usize = 64;
pub const PAD: u32 = 128009;

#[derive(Clone, Debug)]
pub struct EncodedPrompt {
    pub ids: [u32; SEQUENCE],
    pub attention: [bool; SEQUENCE],
    pub pool: [f32; SEQUENCE],
}

enum Engine {
    Standard(Box<tokenizers::Tokenizer>),
    Cooperative(Box<bpe::CooperativeTokenizer>),
}
pub struct PromptTokenizer(Engine);

impl PromptTokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(Self(Engine::Standard(Box::new(tokenizer))))
    }

    /// Initialize the pinned Llama tokenizer in bounded browser tasks. Native
    /// callers use the same tables and merge policy, with no scheduler overhead.
    pub async fn from_bytes_async(bytes: &[u8]) -> Result<Self> {
        Ok(Self(Engine::Cooperative(Box::new(
            bpe::CooperativeTokenizer::from_bytes(bytes).await?,
        ))))
    }

    pub fn encode(&self, prompt: &str) -> Result<EncodedPrompt> {
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= 16384,
            "Enter a nonempty prompt of at most 16384 UTF-8 bytes"
        );
        // Reject control tokens in user text: they would change the chat boundary.
        ensure!(
            !prompt.contains("<|"),
            "Prompt must not contain tokenizer control tokens"
        );
        let formatted = format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            prompt.trim()
        );
        let ids = match &self.0 {
            Engine::Standard(tokenizer) => tokenizer
                .encode(formatted, true)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
                .get_ids()
                .to_vec(),
            Engine::Cooperative(tokenizer) => tokenizer.encode(formatted)?,
        };
        Self::pad(&ids)
    }

    fn pad(ids: &[u32]) -> Result<EncodedPrompt> {
        ensure!(
            ids.starts_with(&[128000, 128006, 882, 128007, 271]),
            "Unexpected Llama chat header"
        );
        ensure!(
            ids.len() >= 7 && ids.last() == Some(&PAD) && ids.iter().all(|id| *id < 128256),
            "Invalid Llama prompt token boundaries"
        );
        ensure!(
            ids.len() <= SEQUENCE,
            "Prompt needs {} tokens; the encoder allows {SEQUENCE} including the chat header",
            ids.len()
        );
        let mut output = EncodedPrompt {
            ids: [PAD; SEQUENCE],
            attention: [false; SEQUENCE],
            pool: [0.0; SEQUENCE],
        };
        let offset = SEQUENCE - ids.len();
        output.ids[offset..].copy_from_slice(ids);
        output.attention[offset..].fill(true);
        output.pool[offset + 5..].fill(1.0);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn left_padding_excludes_header_but_pools_end_of_turn() {
        let ids = [128000, 128006, 882, 128007, 271, 1234, PAD];
        let p = PromptTokenizer::pad(&ids).unwrap();
        assert!(p.attention[..57].iter().all(|v| !*v));
        assert!(p.attention[57..].iter().all(|v| *v));
        assert_eq!(p.pool.iter().sum::<f32>(), 2.0);
        assert_eq!(&p.pool[62..], &[1.0, 1.0]);
        assert_eq!(&p.ids[57..], &ids);
        let mut long = ids[..5].to_vec();
        long.extend([1234; 60]);
        long.push(PAD);
        assert!(PromptTokenizer::pad(&long).is_err());
        assert!(PromptTokenizer::pad(&ids[..6]).is_err());
        assert!(PromptTokenizer::pad(&[128000, 128006, 882, 128007, 271, PAD]).is_err());
    }
    #[cfg(feature = "tools")]
    #[test]
    #[ignore = "requires the pinned external tokenizer asset"]
    fn incremental_bpe_matches_hugging_face_on_unicode_and_adversarial_prompts() {
        let path = std::env::var("BURN_LLAMA_TOKENIZER").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/cdn-upload-human-v1/aberration.technology/model/llama/ardy-llm2vec-8b/v1/metadata/tokenizer.json")
        });
        let bytes = std::fs::read(path).unwrap();
        let reference = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
        let candidate = pollster::block_on(bpe::CooperativeTokenizer::from_bytes(&bytes)).unwrap();
        let atoms = [
            "walk",
            "walking",
            "123456789012345",
            "don't",
            "we'll",
            "?!",
            "_long_name_",
            "café",
            "cafe\u{301}",
            "你好世界",
            "こんにちは",
            "مرحبا",
            "🙂👩‍💻",
            "\r\n",
            "\t",
            "  ",
            "abcdefghijklmno",
            "https://example.test/a?q=x",
            "x",
        ];
        let mut prompts = vec![
            "walk forward calmly".to_string(),
            "sneak forward cautiously".into(),
            "a".repeat(16384),
            "🙂".repeat(4096),
        ];
        let mut state = 17u64;
        for i in 0..1024 {
            let mut text = String::new();
            for _ in 0..1 + i % 24 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                text.push_str(atoms[(state >> 32) as usize % atoms.len()]);
                if state & 1 != 0 {
                    text.push(' ');
                }
            }
            prompts.push(text);
        }
        for prompt in prompts {
            let text = format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                prompt.trim()
            );
            let expected = reference.encode(text.clone(), true).unwrap();
            assert_eq!(
                candidate.encode(text).unwrap(),
                expected.get_ids(),
                "prompt {prompt:?}"
            );
        }
    }
}

//! The upstream LLM2Vec chat template, left padding, and instruction-excluding pooling.
use anyhow::{Result, ensure};

pub const SEQUENCE: usize = 64;
pub const PAD: u32 = 128009;

#[derive(Clone, Debug)]
pub struct EncodedPrompt {
    pub ids: [u32; SEQUENCE],
    pub attention: [bool; SEQUENCE],
    pub pool: [f32; SEQUENCE],
}

pub struct PromptTokenizer(tokenizers::Tokenizer);

impl PromptTokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(Self(tokenizer))
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
        let tokens = self
            .0
            .encode(formatted, true)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Self::pad(tokens.get_ids())
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
}

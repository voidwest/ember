use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub type TokenOffsets = Vec<(usize, usize)>;

/// wraps the huggingface `tokenizers` crate for text-token id conversion.
pub struct EmberTokenizer {
    /// wrapped huggingface tokenizers instance
    inner: Tokenizer,
}

impl EmberTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(anyhow::Error::msg)
            .context("failed to load tokenizer")?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(anyhow::Error::msg)
            .context("encode failed")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, TokenOffsets)> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(anyhow::Error::msg)
            .context("encode failed")?;
        Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(anyhow::Error::msg)
            .context("decode failed")
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// return the end-of-sequence token id, if the tokenizer defines one.
    ///
    /// tries the common token strings across architectures, in order:
    /// `<|end_of_text|>` (llama-3), `<|eot_id|>` (llama-3 alt),
    /// `<|endoftext|>` (gpt-2). returns `None` if none are found.
    pub fn eos_token_id(&self) -> Option<u32> {
        for token_str in &["<|end_of_text|>", "<|eot_id|>", "<|endoftext|>"] {
            if let Some(id) = self.inner.token_to_id(token_str) {
                return Some(id);
            }
        }
        None
    }
}

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// wraps the huggingface `tokenizers` crate for text ↔ token id conversion.
/// currently hardcoded for gpt-2's tokenizer (vocab size 50257).
pub struct EmberTokenizer {
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

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(anyhow::Error::msg)
            .context("decode failed")
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

pub fn download_gpt2_tokenizer_blocking() -> Result<EmberTokenizer> {
    let url = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json";
    let response = reqwest::blocking::get(url)?.bytes()?;
    std::fs::write("tokenizer.json", &response)?;
    EmberTokenizer::from_file("tokenizer.json")
}

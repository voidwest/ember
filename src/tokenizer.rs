use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub struct EmberTokenizer {
    inner: Tokenizer,
}

impl EmberTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let inner = Tokenizer::from_file(path).context("failed to load tokenizer")?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, true).context("encode failed")?;
        Ok(encoding.get_idts().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true).context("decode failed")
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

pub fn download_gpt2_tokenizer() -> Result<EmberTokenizer> {
    anyhow::bail!(
        "auto-download not implemented — download tokenizer.json manually from HuggingFace"
    );
}

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
        Ok(self.ensure_bos(encoding.get_ids().to_vec()))
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<bos>")
    }

    pub fn encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, TokenOffsets)> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(anyhow::Error::msg)
            .context("encode failed")?;
        let ids = encoding.get_ids().to_vec();
        let offsets = encoding.get_offsets().to_vec();
        Ok(self.ensure_bos_with_offsets(ids, offsets))
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

    /// return all end-of-sequence token ids defined by the tokenizer.
    ///
    /// checks for `<|eot_id|>` (llama-3 end-of-turn), `<|end_of_text|>`
    /// (llama-3 end-of-sequence), `<|endoftext|>` (gpt-2), and `<eos>`
    /// (Gemma-family tokenizers).
    /// models typically predict `<|eot_id|>` at the end of an assistant
    /// turn; stopping there prevents the model from looping on header tokens.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for token_str in &["<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<eos>"] {
            if let Some(id) = self.inner.token_to_id(token_str) {
                ids.push(id);
            }
        }
        ids
    }

    fn ensure_bos(&self, ids: Vec<u32>) -> Vec<u32> {
        let Some(bos) = self.bos_token_id() else {
            return ids;
        };
        if ids.first() == Some(&bos) {
            return ids;
        }

        let mut with_bos = Vec::with_capacity(ids.len() + 1);
        with_bos.push(bos);
        with_bos.extend(ids);
        with_bos
    }

    fn ensure_bos_with_offsets(
        &self,
        ids: Vec<u32>,
        offsets: TokenOffsets,
    ) -> (Vec<u32>, TokenOffsets) {
        let Some(bos) = self.bos_token_id() else {
            return (ids, offsets);
        };
        if ids.first() == Some(&bos) {
            return (ids, offsets);
        }

        let mut with_bos = Vec::with_capacity(ids.len() + 1);
        with_bos.push(bos);
        with_bos.extend(ids);

        let mut with_offsets = Vec::with_capacity(offsets.len() + 1);
        with_offsets.push((0, 0));
        with_offsets.extend(offsets);

        (with_bos, with_offsets)
    }
}

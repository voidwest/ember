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

    /// Load a tokenizer directly from a serialized `tokenizer.json` payload.
    ///
    /// This avoids materializing embedded tokenizers in a shared temporary
    /// path, where stale files or concurrent processes could change which
    /// tokenizer a run actually used.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(anyhow::Error::msg)
            .context("failed to load tokenizer from bytes")?;
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
        anyhow::ensure!(
            ids.len() == offsets.len(),
            "tokenizer returned {} IDs but {} offsets",
            ids.len(),
            offsets.len()
        );
        let character_count = text.chars().count();
        let mut previous = None;
        for (index, &(start, end)) in offsets.iter().enumerate() {
            anyhow::ensure!(
                start <= end && end <= character_count,
                "tokenizer offset {index} ({start}, {end}) is invalid for {character_count} Unicode characters"
            );
            if start == end {
                continue;
            }
            if let Some((previous_start, previous_end)) = previous {
                anyhow::ensure!(
                    start >= previous_start && end >= previous_end,
                    "tokenizer offsets are not monotonic at token {index}: ({start}, {end}) follows ({previous_start}, {previous_end})"
                );
            }
            previous = Some((start, end));
        }
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

    /// Largest token ID known to the tokenizer, including added tokens.
    /// Tokenizer vocabularies are not required to be densely numbered, so
    /// this is a stronger embedding-compatibility check than `vocab_size()`.
    pub fn max_token_id(&self) -> Option<u32> {
        self.inner.get_vocab(true).into_values().max()
    }

    /// Whether the tokenizer can decode a model-emitted token ID.
    pub fn contains_token_id(&self, id: u32) -> bool {
        self.inner.id_to_token(id).is_some()
    }

    /// The vocabulary piece for a token ID, when known.
    pub fn token_piece(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Validate that every ID this tokenizer can emit is addressable by the
    /// model embedding table.
    pub fn validate_model_vocab(&self, model_vocab_size: usize) -> Result<()> {
        anyhow::ensure!(model_vocab_size > 0, "model vocabulary is empty");
        anyhow::ensure!(
            model_vocab_size as u128 <= u32::MAX as u128 + 1,
            "model vocabulary size {model_vocab_size} exceeds the u32 token-ID space"
        );
        let max_id = self
            .max_token_id()
            .context("tokenizer vocabulary is empty")?;
        anyhow::ensure!(
            (max_id as usize) < model_vocab_size,
            "tokenizer contains token ID {max_id}, but model vocabulary has only {model_vocab_size} rows"
        );
        Ok(())
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
        for token_str in &[
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<eos>",
        ] {
            if let Some(id) = self.inner.token_to_id(token_str) {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
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

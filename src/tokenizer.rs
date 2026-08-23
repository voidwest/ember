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

    /// Encode without the tokenizer's automatic special tokens (the
    /// equivalent of HF's `add_special_tokens=False`). Assemblers that
    /// embed structural special tokens in the rendered text themselves
    /// (e.g. Llama-3 templates carrying their own `<|begin_of_text|>`)
    /// must use this so the template is not double-encoded.
    pub fn encode_no_special(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(anyhow::Error::msg)
            .context("encode failed")?;
        Ok(encoding.get_ids().to_vec())
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
        // The `tokenizers` crate reports byte offsets relative to the
        // original string. Validate against the byte length.
        let byte_count = text.len();
        let mut previous = None;
        for (index, &(start, end)) in offsets.iter().enumerate() {
            anyhow::ensure!(
                start <= end && end <= byte_count,
                "tokenizer offset {index} ({start}, {end}) is invalid for {byte_count} bytes"
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

    /// UTF-8-safe incremental detokenizer for streaming output.
    ///
    /// Byte-level BPE vocabularies contain tokens that are FRAGMENTS of a
    /// multi-byte code point (the first bytes of an Arabic word, an emoji,
    /// any non-ASCII text). Decoding such a token in isolation yields
    /// U+FFFD replacement characters — per-token streaming corrupts every
    /// non-Latin script. This decoder accumulates ids, decodes the running
    /// sequence (prefix-stable for concat-style decoders), and releases
    /// only text up to the last complete character boundary.
    ///
    /// Contract: concatenating the returned pieces over a push sequence
    /// equals `decode(&all_ids)` up to trailing bytes not yet released;
    /// `finish()` flushes them.
    pub fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            ids: Vec::new(),
            released_chars: 0,
            prev: String::new(),
        }
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

    /// The token ID for a vocabulary piece (including added tokens), when
    /// known. Used by multimodal assemblers to resolve image/special tokens.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
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

/// Streaming detokenizer: see [`EmberTokenizer::incremental_decoder`].
pub struct IncrementalDecoder<'t> {
    tokenizer: &'t EmberTokenizer,
    ids: Vec<u32>,
    /// characters already handed to the consumer
    released_chars: usize,
    /// last full decode of all ids so far
    prev: String,
}

impl<'t> IncrementalDecoder<'t> {
    /// Push one generated token; returns the text newly available as
    /// complete characters (possibly empty — e.g. mid-code-point tokens).
    ///
    /// A trailing run of U+FFFD in the running decode marks an INCOMPLETE
    /// multi-byte sequence, not real text, so it is never released; once
    /// later tokens complete the sequence the true characters flow out.
    pub fn push(&mut self, id: u32) -> Result<String> {
        self.ids.push(id);
        let new = self.tokenizer.decode(&self.ids)?;
        let piece = advance_released(&mut self.released_chars, &self.prev, &new);
        self.prev = new;
        Ok(piece)
    }

    /// Flush any remainder (call once the generation is over). A trailing
    /// U+FFFD at cut-off is emitted as-is: there is nothing better to send.
    pub fn finish(&mut self) -> Result<String> {
        let total = self.prev.chars().count();
        if total > self.released_chars {
            let start_idx = split_at_char(&self.prev, self.released_chars).len();
            self.released_chars = total;
            Ok(self.prev[start_idx..].to_string())
        } else {
            Ok(String::new())
        }
    }

    /// Ids pushed so far.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True when no ids have been pushed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Length in chars of the longest common character prefix.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

fn split_at_char(s: &str, chars: usize) -> &str {
    match s.char_indices().nth(chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn ends_with_replacement(s: &str) -> bool {
    s.ends_with('\u{FFFD}')
}

/// Release newly stable characters between the previous and current full
/// decodes.
///
/// Concat-style byte decoders evolve by appending bytes or resolving a
/// trailing U+FFFD into the completed code points — interiors never change.
/// A character is releasable once (a) it is not part of a trailing U+FFFD
/// run (incomplete-sequence placeholder) and (b) it agrees with the previous
/// decode (the first push has no history, so `safe` alone gates it).
fn advance_released(released: &mut usize, prev: &str, new: &str) -> String {
    let mut safe = new.chars().count();
    while safe > *released && ends_with_replacement(split_at_char(new, safe)) {
        safe -= 1;
    }
    let confirmed = if prev.is_empty() {
        safe
    } else {
        common_prefix_chars(prev, new)
    };
    let upto = safe.min(confirmed).max(*released);
    if upto > *released {
        let start_idx = split_at_char(new, *released).len();
        let end_idx = split_at_char(new, upto).len();
        *released = upto;
        new[start_idx..end_idx].to_string()
    } else {
        String::new()
    }
}

// (incremental-detokenizer behavior is covered by tests/arabic_streaming.rs)

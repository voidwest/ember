use anyhow::{Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;

/// Maximum tokenizer JSON payload accepted by the parser.
///
/// Real tokenizer files are substantially smaller than this bound. Keeping a
/// finite limit prevents an attacker-controlled path or byte payload from
/// forcing an unbounded read/parse allocation before tokenization starts.
pub const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;

pub type TokenOffsets = Vec<(usize, usize)>;

/// wraps the huggingface `tokenizers` crate for text-token id conversion.
pub struct EmberTokenizer {
    /// wrapped huggingface tokenizers instance
    inner: Tokenizer,
}

impl EmberTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let path = path.as_ref();
        // Reject symlinks and bind the path to one opened descriptor before
        // checking its size. A metadata-then-open sequence otherwise permits a
        // replacement race that can select a different tokenizer.
        let path_metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("failed to stat tokenizer {:?}", path))?;
        anyhow::ensure!(
            path_metadata.file_type().is_file(),
            "tokenizer {:?} is not a regular file",
            path
        );
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to read tokenizer {:?}", path))?;
        let initial_metadata = file
            .metadata()
            .with_context(|| format!("failed to stat tokenizer {:?}", path))?;
        anyhow::ensure!(
            initial_metadata.file_type().is_file()
                && initial_metadata.len() == path_metadata.len()
                && initial_metadata.modified().ok() == path_metadata.modified().ok(),
            "tokenizer file changed while opening {:?}",
            path
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            anyhow::ensure!(
                initial_metadata.dev() == path_metadata.dev()
                    && initial_metadata.ino() == path_metadata.ino(),
                "tokenizer file changed while opening {:?}",
                path
            );
        }
        let length = initial_metadata.len();
        anyhow::ensure!(
            length <= MAX_TOKENIZER_BYTES,
            "tokenizer file {:?} is {length} bytes, exceeding the {} byte limit",
            path,
            MAX_TOKENIZER_BYTES
        );
        let capacity =
            usize::try_from(length).context("tokenizer file length exceeds address space")?;
        let max_bytes = usize::try_from(MAX_TOKENIZER_BYTES)
            .context("tokenizer byte limit exceeds address space")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|error| anyhow::anyhow!("failed to reserve tokenizer buffer: {error}"))?;
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut chunk)
                .with_context(|| format!("failed to read tokenizer {:?}", path))?;
            if read == 0 {
                break;
            }
            anyhow::ensure!(
                bytes.len() <= max_bytes.saturating_sub(read),
                "tokenizer file {:?} grew beyond the {} byte limit while reading",
                path,
                MAX_TOKENIZER_BYTES
            );
            bytes
                .try_reserve_exact(read)
                .map_err(|error| anyhow::anyhow!("failed to grow tokenizer buffer: {error}"))?;
            bytes.extend_from_slice(&chunk[..read]);
        }
        let final_metadata = file
            .metadata()
            .with_context(|| format!("failed to stat tokenizer {:?} after reading", path))?;
        let final_path_metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("failed to stat tokenizer {:?} after reading", path))?;
        anyhow::ensure!(
            final_metadata.len() == length
                && final_metadata.modified().ok() == initial_metadata.modified().ok()
                && final_path_metadata.file_type().is_file()
                && final_path_metadata.len() == initial_metadata.len()
                && final_path_metadata.modified().ok() == initial_metadata.modified().ok(),
            "tokenizer file changed while reading {:?}",
            path
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            anyhow::ensure!(
                final_path_metadata.dev() == initial_metadata.dev()
                    && final_path_metadata.ino() == initial_metadata.ino(),
                "tokenizer file changed while reading {:?}",
                path
            );
        }
        Self::from_bytes(bytes)
    }

    /// Load a tokenizer directly from a serialized `tokenizer.json` payload.
    ///
    /// This avoids materializing embedded tokenizers in a shared temporary
    /// path, where stale files or concurrent processes could change which
    /// tokenizer a run actually used. The JSON is checked before handing it
    /// to the upstream parser because some tokenizers deserializers panic on
    /// malformed fields instead of returning an error.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = bytes.as_ref();
        let length = u64::try_from(bytes.len()).context("tokenizer payload length exceeds u64")?;
        anyhow::ensure!(
            length <= MAX_TOKENIZER_BYTES,
            "tokenizer payload is {length} bytes, exceeding the {MAX_TOKENIZER_BYTES} byte limit"
        );
        std::str::from_utf8(bytes)
            .map_err(|error| anyhow::anyhow!("tokenizer JSON is not valid UTF-8: {error}"))?;
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        serde::de::IgnoredAny::deserialize(&mut deserializer)
            .map_err(|error| anyhow::anyhow!("tokenizer JSON is malformed: {error}"))?;
        deserializer
            .end()
            .map_err(|error| anyhow::anyhow!("tokenizer JSON has trailing content: {error}"))?;
        let inner = parse_tokenizer(bytes)?;
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

/// Parse through the upstream tokenizers crate behind an unwind boundary.
///
/// `tokenizers` has historically used `expect` in a few deserialization paths;
/// malformed attacker-controlled JSON must become a normal load error rather
/// than aborting a process that is serving other requests.
fn parse_tokenizer(bytes: &[u8]) -> Result<Tokenizer> {
    let parsed = std::panic::catch_unwind(|| Tokenizer::from_bytes(bytes)).map_err(|payload| {
        let detail = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
            })
            .unwrap_or_default();
        if detail.is_empty() {
            anyhow::anyhow!("tokenizers crate panicked while parsing tokenizer JSON")
        } else {
            anyhow::anyhow!("tokenizers crate panicked while parsing tokenizer JSON: {detail}")
        }
    })?;
    parsed
        .map_err(anyhow::Error::msg)
        .context("failed to load tokenizer from bytes")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_file_rejects_symlinks_and_keeps_path_identity() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ember-tokenizer-path-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).expect("create tokenizer fixture dir");
        let target = dir.join("tokenizer.json");
        std::fs::write(&target, b"{}").expect("write tokenizer fixture");
        #[cfg(unix)]
        {
            let link = dir.join("tokenizer-link.json");
            std::os::unix::fs::symlink(&target, &link).expect("create tokenizer symlink");
            let error = EmberTokenizer::from_file(&link)
                .err()
                .expect("symlinked tokenizer paths must be rejected");
            assert!(error.to_string().contains("regular file"), "{error}");
        }
        // A regular file passes the path/identity checks and reaches parsing,
        // which rejects the non-tokenizer payload.
        let error = EmberTokenizer::from_file(&target)
            .err()
            .expect("a regular file must pass path checks and fail at parse time");
        assert!(!error.to_string().contains("regular file"), "{error}");
        std::fs::remove_dir_all(&dir).expect("remove tokenizer fixture dir");
    }
}

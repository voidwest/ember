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
    /// non-Latin script. The upstream stream decoder retains only the token
    /// window needed to resolve byte fragments and decoder context; completed
    /// text does not get decoded again on every subsequent token.
    ///
    /// Contract: concatenating the returned pieces over a push sequence
    /// equals `decode(&all_ids)` up to trailing bytes not yet released;
    /// `finish()` flushes them.
    pub fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            read_index: 0,
            token_count: 0,
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
    /// Upstream decode window, including enough previously emitted context
    /// to preserve whitespace/byte-decoder behavior at the left boundary.
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    read_index: usize,
    token_count: usize,
}

impl<'t> IncrementalDecoder<'t> {
    /// Push one generated token; returns the text newly available as
    /// complete characters (possibly empty — e.g. mid-code-point tokens).
    ///
    /// A trailing U+FFFD may represent an incomplete byte sequence or literal
    /// text. It stays pending until more tokens disambiguate it or `finish()`
    /// flushes the final decode; literal replacement characters are preserved.
    pub fn push(&mut self, id: u32) -> Result<String> {
        // Use the public stateful helper rather than DecodeStream itself:
        // DecodeStream keeps its pending IDs private and has no EOF flush.
        let previous_prefix_end = self.prefix_index;
        let discarded_tokens = self.read_index;
        let piece = tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer.inner,
            id,
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
            &mut self.read_index,
        )
        .map_err(anyhow::Error::msg)
        .context("incremental decode failed")?;
        if piece.is_some() {
            // tokenizers 0.20.4 updates these indices in pre-drain coordinates,
            // which can underflow after repeated emissions. Rebase both to the
            // retained window: old prefix end becomes the new chunk's start;
            // the whole window now belongs to the emitted prefix.
            self.read_index = previous_prefix_end - discarded_tokens;
            self.prefix_index = self.ids.len();
        }
        self.token_count += 1;
        Ok(piece.unwrap_or_default())
    }

    /// Flush any remainder (call once the generation is over). A trailing
    /// U+FFFD at cut-off is emitted as-is: there is nothing better to send.
    pub fn finish(&mut self) -> Result<String> {
        let decoded = self.tokenizer.decode(&self.ids)?;
        let tail = decoded
            .strip_prefix(&self.prefix)
            .context("incremental decode changed already emitted text")?
            .to_string();
        self.prefix = decoded;
        Ok(tail)
    }

    /// Ids pushed so far.
    pub fn len(&self) -> usize {
        self.token_count
    }

    /// True when no ids have been pushed.
    pub fn is_empty(&self) -> bool {
        self.token_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_tokenizer() -> EmberTokenizer {
        use tokenizers::{models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel, AddedToken};

        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, character)| (character.to_string(), id as u32))
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut inner = Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        inner.with_decoder(Some(ByteLevel::new(false, false, false)));
        inner.add_special_tokens(&[AddedToken::from("<eos>", true)]);
        EmberTokenizer { inner }
    }

    #[test]
    fn incremental_decode_preserves_unicode_special_tokens_and_every_eof_boundary() {
        let tokenizer = byte_tokenizer();
        for text in ["الْعَرَبِيَّةُ 👋🌟", "a\u{fffd}b\u{fffd}", " hello  world! "]
        {
            let mut ids = tokenizer.encode_no_special(text).unwrap();
            // A skipped special token must not split a pending UTF-8 sequence.
            ids.insert(1, tokenizer.token_to_id("<eos>").unwrap());
            for end in 0..=ids.len() {
                let mut decoder = tokenizer.incremental_decoder();
                let mut actual = String::new();
                for &id in &ids[..end] {
                    actual.push_str(&decoder.push(id).unwrap());
                }
                assert_eq!(decoder.len(), end);
                assert_eq!(decoder.is_empty(), end == 0);
                actual.push_str(&decoder.finish().unwrap());
                assert_eq!(
                    actual,
                    tokenizer.decode(&ids[..end]).unwrap(),
                    "{text:?} at {end}"
                );
                assert!(
                    decoder.finish().unwrap().is_empty(),
                    "EOF flush is idempotent"
                );
            }
        }
    }

    #[test]
    fn incremental_decode_emits_complete_characters_without_growing_history() {
        let tokenizer = byte_tokenizer();
        let text = "لغة 👋 abc ".repeat(1024);
        let ids = tokenizer.encode_no_special(&text).unwrap();
        let mut decoder = tokenizer.incremental_decoder();
        let mut actual = String::new();
        for &id in &ids {
            let piece = decoder.push(id).unwrap();
            assert!(
                !piece.contains('\u{fffd}'),
                "incomplete UTF-8 escaped: {piece:?}"
            );
            actual.push_str(&piece);
            assert!(
                decoder.ids.len() < 32,
                "completed text retained in decode window"
            );
        }
        actual.push_str(&decoder.finish().unwrap());
        assert_eq!(actual, text);

        let mut decoder = tokenizer.incremental_decoder();
        for (id, expected) in tokenizer
            .encode_no_special("abc")
            .unwrap()
            .into_iter()
            .zip(["a", "b", "c"])
        {
            assert_eq!(
                decoder.push(id).unwrap(),
                expected,
                "ASCII must stream immediately"
            );
        }
    }

    #[test]
    fn incremental_decode_preserves_wordpiece_cleanup_and_metaspace_context() {
        for (decoder, tokens, expected) in [
            (
                serde_json::json!({"type":"WordPiece", "prefix":"##", "cleanup":true}),
                vec!["I", "'m", "play", "##ing", "."],
                "I'm playing.",
            ),
            (
                serde_json::json!({"type":"Metaspace", "replacement":"▁", "prepend_scheme":"always", "split":true}),
                vec!["▁This", "▁is", "▁a", "▁test", "!"],
                "This is a test!",
            ),
        ] {
            let vocab: serde_json::Map<_, _> = tokens
                .iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), serde_json::json!(id)))
                .chain(std::iter::once((
                    "[UNK]".into(),
                    serde_json::json!(tokens.len()),
                )))
                .collect();
            let tokenizer = EmberTokenizer::from_bytes(
                serde_json::to_vec(&serde_json::json!({
                    "version":"1.0", "truncation":null, "padding":null,
                    "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
                    "post_processor":null, "decoder":decoder,
                    "model":{"type":"WordLevel", "vocab":vocab, "unk_token":"[UNK]"},
                }))
                .unwrap(),
            )
            .unwrap();
            let ids: Vec<_> = (0..tokens.len() as u32).collect();
            assert_eq!(tokenizer.decode(&ids).unwrap(), expected);
            let ids = ids.repeat(8);
            for end in 0..=ids.len() {
                let mut stream = tokenizer.incremental_decoder();
                let mut actual = String::new();
                for &id in &ids[..end] {
                    actual.push_str(&stream.push(id).unwrap());
                }
                actual.push_str(&stream.finish().unwrap());
                assert_eq!(
                    actual,
                    tokenizer.decode(&ids[..end]).unwrap(),
                    "{tokens:?} at {end}"
                );
            }
        }
    }

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

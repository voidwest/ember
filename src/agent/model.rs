//! The session-engine seam (Tracks F/G): what the agent loop needs from a
//! model, and the real Llama-family implementation.
//!
//! [`ChatModelEngine`] is deliberately narrow: commit rendered messages
//! into persistent context, run ONE assistant generation as an atomic
//! transaction (speculative scaffold rolled back on cancellation), report
//! timings. Implementations:
//!
//! - [`LlamaChatModel`] — real inference over `crate::llama::Llama` with
//!   a live KV cache (llama/qwen2/qwen3 architectures);
//! - `ScriptedModel` in [`crate::agent::testkit`] — deterministic test
//!   double; basic agent correctness never depends on a GGUF file.
//!
//! Semantics pinned here (mirroring the proven `VoiceSession` path):
//!
//! - **Stop strings bound the parsed text only.** Raw tokens stay in the
//!   committed context exactly as emitted — the official Qwen/Llama
//!   templates expect the `<tool_call>...</tool_call>` span to remain part
//!   of the assistant message. Parsing operates on the bounded text.
//! - **Terminal policy**: if generation stopped on eos, that token IS the
//!   terminal (nothing was decoded from it) and is forwarded alone;
//!   otherwise the last generated token plus the protocol suffix are
//!   forwarded together. A zero-token turn commits just the suffix.
//! - **Cancellation** truncates the KV cursor back to the pre-turn
//!   boundary — a cancelled generation never leaves half-committed state.

use anyhow::{ensure, Context as _, Result};
use rand::SeedableRng;
use std::time::Instant;

use crate::backend::Backend;
use crate::llama::Llama;
use crate::tokenizer::EmberTokenizer;

use super::tool::CancelFlag;

/// Provenance identity of the model behind an engine (Track K).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelIdentity {
    pub model_path: String,
    /// SHA-256 of the GGUF file when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    /// Ember architecture family (`llama`, `qwen3`, ...).
    pub architecture: String,
    /// Quantization label when known (GGUF `general.file_type` name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    pub n_layers: usize,
    pub embed_dim: usize,
    pub vocab_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_sha256: Option<String>,
    /// Maximum context the engine will accept.
    pub context_len: usize,
}

/// Sampling knobs for one generation. Greedy at temperature 0 (the ember
/// research default); a seed makes temperature sampling reproducible.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Deterministic sampling seed (temperature > 0 only).
    pub seed: Option<u64>,
    /// Strings that end generation immediately when they appear in the
    /// decoded text; they bound the parsed action, not the committed ids.
    pub stop_strings: Vec<String>,
    /// Additional special-token literals treated as eos for this turn.
    pub extra_eos_tokens: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            stop_strings: Vec::new(),
            extra_eos_tokens: Vec::new(),
        }
    }
}

/// Why a finished generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnStop {
    /// A tokenizer/protocol eos token was emitted.
    Eos,
    /// A protocol stop string appeared in the text.
    StopString(String),
    /// Token budget exhausted for this turn.
    MaxTokens,
}

/// One completed (committed) or cancelled assistant generation.
#[derive(Debug)]
pub struct GeneratedTurn {
    /// Continuation text as parsed (stop strings excluded).
    pub text: String,
    /// Token ids now part of the committed context (scaffold + content +
    /// terminal policy tokens). Empty when cancelled.
    pub committed_ids: Vec<u32>,
    pub stop: Option<TurnStop>,
    /// True when cancelled before anything committed (KV rolled back).
    pub cancelled: bool,
    // timing / volume metadata for traces (Track M)
    pub prompt_tokens_prefilled: usize,
    pub decode_evaluations: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

impl GeneratedTurn {
    pub fn output_tokens(&self) -> usize {
        self.committed_ids.len()
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_ms <= 0.0 {
            return 0.0;
        }
        self.decode_evaluations as f64 / (self.decode_ms / 1000.0)
    }
}

/// The seam between the agent state machine and inference.
///
/// Implementations own their conversation state (KV cache or equivalent)
/// and enforce the transaction contract: [`ChatModelEngine::generate_turn`]
/// either commits a complete assistant turn or restores the exact prior
/// state.
pub trait ChatModelEngine {
    fn identity(&self) -> &ModelIdentity;

    /// Absolute context position after everything committed so far.
    fn committed_len(&self) -> usize;

    /// Commit one rendered message into the context (prefill at the
    /// boundary). Returns the absolute span `[start, end)`.
    fn commit_message(&mut self, rendered: &str) -> Result<(usize, usize)>;

    /// Run one assistant generation as an atomic transaction:
    ///
    /// 1. prefill `prefix_rendered` speculatively at the boundary;
    /// 2. decode until eos / stop string / `max_new_tokens`;
    /// 3. on natural finish commit `suffix_rendered` per terminal policy;
    /// 4. on cancellation roll back to the boundary and return
    ///    `cancelled: true` with nothing committed.
    ///
    /// `on_token(id, piece)` observes each generated token (streaming);
    /// pieces may be empty for mid-codepoint fragments.
    fn generate_turn(
        &mut self,
        prefix_rendered: &str,
        suffix_rendered: &str,
        params: &GenerationParams,
        control: &CancelFlag,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GeneratedTurn>;

    /// Roll the context back to an earlier committed length.
    fn truncate_to(&mut self, len: usize) -> Result<()>;
}

// ---------------------------------------------------------------------------
// real implementation over Llama<CpuBackend>
// ---------------------------------------------------------------------------

/// v0.5-style seeded RNG: StdRng when a seed is given (deterministic),
/// the thread RNG otherwise.
enum AgentRng {
    Std(Box<rand::rngs::StdRng>),
    Thread(rand::rngs::ThreadRng),
}

impl rand::RngCore for AgentRng {
    fn next_u32(&mut self) -> u32 {
        match self {
            AgentRng::Std(rng) => rng.next_u32(),
            AgentRng::Thread(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            AgentRng::Std(rng) => rng.next_u64(),
            AgentRng::Thread(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            AgentRng::Std(rng) => rng.fill_bytes(dest),
            AgentRng::Thread(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            AgentRng::Std(rng) => rng.try_fill_bytes(dest),
            AgentRng::Thread(rng) => rng.try_fill_bytes(dest),
        }
    }
}

/// Real chat engine over a resident Llama-family model.
pub struct LlamaChatModel<'m> {
    model: &'m Llama<crate::backend::CpuBackend>,
    backend: &'m crate::backend::CpuBackend,
    tokenizer: &'m EmberTokenizer,
    cache: crate::kv_cache::KVCache,
    committed_len: usize,
    identity: ModelIdentity,
}

impl<'m> LlamaChatModel<'m> {
    /// Create the engine with a request-sized KV cache. `kv_capacity`
    /// bounds total committed context; exceeding it fails closed.
    pub fn new(
        model: &'m Llama<crate::backend::CpuBackend>,
        backend: &'m crate::backend::CpuBackend,
        tokenizer: &'m EmberTokenizer,
        kv_capacity: usize,
        mut identity: ModelIdentity,
    ) -> Self {
        let capacity = kv_capacity.min(model.config.max_seq_len);
        let cache = model.create_cache(backend, capacity);
        identity.context_len = capacity;
        Self {
            model,
            backend,
            tokenizer,
            cache,
            committed_len: 0,
            identity,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode_no_special(text)
            .context("failed to encode rendered message")
    }

    fn resolve_eos_ids(&self, extra: &[String]) -> Result<Vec<u32>> {
        let mut ids = self.tokenizer.eos_token_ids();
        for literal in extra {
            let id = self.tokenizer.token_to_id(literal).with_context(|| {
                format!("protocol eos literal `{literal}` missing from tokenizer")
            })?;
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    fn prefill_ids(&mut self, ids: &[u32]) -> Result<()> {
        ensure!(!ids.is_empty(), "cannot commit an empty rendered message");
        let start = self.committed_len;
        ensure!(
            start + ids.len() <= self.cache.max_seq_len(),
            "agent conversation exceeds KV capacity {}: committed {start} + new {}; \
             recreate the session with a larger capacity",
            self.cache.max_seq_len(),
            ids.len()
        );
        self.model
            .forward_last_logits_with_cache(self.backend, ids, &mut self.cache, start)?;
        self.committed_len += ids.len();
        Ok(())
    }
}

impl ChatModelEngine for LlamaChatModel<'_> {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn committed_len(&self) -> usize {
        self.committed_len
    }

    fn commit_message(&mut self, rendered: &str) -> Result<(usize, usize)> {
        let ids = self.encode(rendered)?;
        let start = self.committed_len;
        self.prefill_ids(&ids)?;
        debug_assert_eq!(self.cache.cursor(), self.committed_len);
        Ok((start, self.committed_len))
    }

    fn generate_turn(
        &mut self,
        prefix_rendered: &str,
        suffix_rendered: &str,
        params: &GenerationParams,
        control: &CancelFlag,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GeneratedTurn> {
        let reply_start = self.committed_len;

        // -- speculative scaffold --------------------------------------
        let prefix_ids = self.encode(prefix_rendered)?;
        let scaffold_rows = prefix_ids.len();
        let t_prefill = Instant::now();
        let logits = self.prefill_keep_logits(&prefix_ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        let vocab_size = self.backend.shape(&logits)[1];
        let mut logits = logits;

        let eos_ids = self.resolve_eos_ids(&params.extra_eos_tokens)?;
        let mut stream_decoder = self.tokenizer.incremental_decoder();
        let mut text_acc = String::new();
        let mut ids: Vec<u32> = Vec::new();
        let mut cancelled = false;
        let mut stop_hit: Option<TurnStop> = None;
        let mut bounded_text: Option<String> = None;
        let mut decode_ms = 0f64;
        let mut decode_evaluations = 0usize;
        let mut rng = match params.seed {
            Some(seed) => AgentRng::Std(Box::new(rand::rngs::StdRng::seed_from_u64(seed))),
            None => AgentRng::Thread(rand::thread_rng()),
        };

        loop {
            if control.is_cancelled() {
                cancelled = true;
                break;
            }
            if ids.len() >= params.max_new_tokens {
                stop_hit = Some(TurnStop::MaxTokens);
                break;
            }
            let logit_data = self.backend.data(&logits);
            let last_logits = &logit_data[..vocab_size];
            let next = if params.temperature == 0.0 {
                crate::sampler::argmax_token(last_logits)
            } else {
                crate::sampler::sample_token(
                    last_logits,
                    params.temperature,
                    params.top_k,
                    params.top_p,
                    &mut rng,
                )
            };
            let next = u32::try_from(next).context("token id exceeds u32")?;
            ids.push(next);

            let piece = stream_decoder.push(next)?;
            text_acc.push_str(&piece);
            on_token(next, &piece);

            if eos_ids.contains(&next) {
                stop_hit = Some(TurnStop::Eos);
                break;
            }

            // Stop strings bound the PARSED text, not the committed ids.
            let mut matched: Option<(&String, usize)> = None;
            for s in &params.stop_strings {
                if let Some(pos) = text_acc.find(s.as_str())
                    && matched.is_none_or(|(_, p)| pos < p)
                {
                    matched = Some((s, pos));
                }
            }
            if let Some((stop, byte_pos)) = matched {
                bounded_text = Some(text_acc[..byte_pos].to_string());
                stop_hit = Some(TurnStop::StopString(stop.clone()));
                break;
            }

            if ids.len() >= params.max_new_tokens {
                stop_hit = Some(TurnStop::MaxTokens);
                break;
            }
            if control.is_cancelled() {
                cancelled = true;
                break;
            }
            let t1 = Instant::now();
            logits = self.model.forward_last_logits_with_cache(
                self.backend,
                &[next],
                &mut self.cache,
                reply_start + scaffold_rows + ids.len() - 1,
            )?;
            decode_ms += t1.elapsed().as_secs_f64() * 1e3;
            decode_evaluations += 1;
        }

        if cancelled {
            self.cache.truncate_to(reply_start);
            self.committed_len = reply_start;
            debug_assert_eq!(self.cache.cursor(), reply_start);
            return Ok(GeneratedTurn {
                text: String::new(),
                committed_ids: Vec::new(),
                stop: None,
                cancelled: true,
                prompt_tokens_prefilled: scaffold_rows,
                decode_evaluations,
                prefill_ms,
                decode_ms,
            });
        }

        // flush held-back bytes (a split code point at the very end)
        let _tail = stream_decoder.finish()?;

        // -- terminal policy (mirrors VoiceSession) ---------------------
        self.resync_committed_to_cursor();
        let suffix_ids = self.encode(suffix_rendered)?;
        let terminal: Vec<u32> = if matches!(stop_hit, Some(TurnStop::Eos)) {
            vec![*ids.last().expect("EOS requires a generated token")]
        } else if ids.is_empty() {
            suffix_ids.clone()
        } else {
            let mut v = Vec::with_capacity(suffix_ids.len() + 1);
            v.push(*ids.last().unwrap());
            v.extend_from_slice(&suffix_ids);
            v
        };
        if !terminal.is_empty() {
            let _ = self.model.forward_last_logits_with_cache(
                self.backend,
                &terminal,
                &mut self.cache,
                self.committed_len,
            )?;
            self.committed_len += terminal.len();
        }
        debug_assert_eq!(self.cache.cursor(), self.committed_len);

        let mut full_ids = prefix_ids;
        full_ids.extend_from_slice(&ids[..ids.len().saturating_sub(1)]);
        full_ids.extend_from_slice(&terminal);
        debug_assert_eq!(full_ids.len(), self.committed_len - reply_start);
        let text = match bounded_text {
            Some(t) => t,
            None => self.tokenizer.decode(&ids)?,
        };

        Ok(GeneratedTurn {
            text,
            committed_ids: full_ids,
            stop: stop_hit,
            cancelled: false,
            prompt_tokens_prefilled: scaffold_rows,
            decode_evaluations,
            prefill_ms,
            decode_ms,
        })
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        ensure!(
            len <= self.committed_len,
            "truncate target {len} beyond committed {}",
            self.committed_len
        );
        self.cache.truncate_to(len);
        self.committed_len = len;
        debug_assert_eq!(self.cache.cursor(), len);
        Ok(())
    }
}

impl LlamaChatModel<'_> {
    /// Prefill that keeps the last-position logits (single forward pass;
    /// no double-counting of rows).
    fn prefill_keep_logits(&mut self, ids: &[u32]) -> Result<crate::tensor::CpuTensor> {
        ensure!(!ids.is_empty(), "cannot prefill zero tokens");
        let start = self.committed_len;
        ensure!(
            start + ids.len() <= self.cache.max_seq_len(),
            "agent conversation exceeds KV capacity {}: committed {start} + new {}; \
             recreate the session with a larger capacity",
            self.cache.max_seq_len(),
            ids.len()
        );
        let logits =
            self.model
                .forward_last_logits_with_cache(self.backend, ids, &mut self.cache, start)?;
        self.committed_len += ids.len();
        Ok(logits)
    }

    fn resync_committed_to_cursor(&mut self) {
        self.committed_len = self.cache.cursor();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::loader::{GgufLoader, GgufValue, LoadedTensor};
    use crate::tensor::CpuTensor;
    use std::collections::HashMap;

    // One real transformer block with an identity residual path. Its head
    // deterministically emits p -> a -> b -> EOS, while nonzero K/V rows
    // let the tests compare the live cache with independent full prefills.
    pub(crate) fn fixture() -> (Llama<CpuBackend>, EmberTokenizer, ModelIdentity) {
        let mut metadata = HashMap::from([(
            "general.architecture".into(),
            GgufValue::Str("llama".into()),
        )]);
        for (name, value) in [
            ("block_count", 1),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("embedding_length", 8),
            ("context_length", 64),
            ("vocab_size", 8),
        ] {
            metadata.insert(format!("llama.{name}"), GgufValue::U32(value));
        }
        let identity_matrix: Vec<f32> = (0..64)
            .map(|i| if i / 8 == i % 8 { 1.0 } else { 0.0 })
            .collect();
        let mut tensors = HashMap::new();
        for name in [
            "token_embd.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ] {
            tensors.insert(
                name.into(),
                LoadedTensor::F32(CpuTensor::from_data(vec![8, 8], identity_matrix.clone())),
            );
        }
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ] {
            tensors.insert(
                name.into(),
                LoadedTensor::F32(CpuTensor::from_data(vec![8], vec![1.0; 8])),
            );
        }
        for name in ["attn_q", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            tensors.insert(
                format!("blk.0.{name}.weight"),
                LoadedTensor::F32(CpuTensor::from_data(vec![8, 8], vec![0.0; 64])),
            );
        }
        let mut head = vec![0.0; 64];
        for (input, output) in [(0, 1), (1, 2), (2, 3)] {
            head[output * 8 + input] = 1.0;
        }
        tensors.insert(
            "output.weight".into(),
            LoadedTensor::F32(CpuTensor::from_data(vec![8, 8], head)),
        );
        let model = Llama::from_loader(GgufLoader {
            metadata,
            tensors,
            k_strategy: crate::quant_k::KStrategy::EagerF32,
            k_decisions: HashMap::new(),
            tensor_meta: HashMap::new(),
        })
        .unwrap();
        let tokenizer = EmberTokenizer::from_bytes(r#"{
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel","vocab":{"p":0,"a":1,"b":2,"<|eot_id|>":3,"s":4,"x":5,"y":6,"[UNK]":7},"unk_token":"[UNK]"}
        }"#).unwrap();
        let identity = ModelIdentity {
            model_path: "synthetic".into(),
            model_sha256: None,
            architecture: "llama".into(),
            quantization: None,
            n_layers: 1,
            embed_dim: 8,
            vocab_size: 8,
            tokenizer_sha256: None,
            context_len: 64,
        };
        (model, tokenizer, identity)
    }

    #[test]
    fn native_turn_ids_and_cache_cover_every_generated_token() {
        let (model, tokenizer, identity) = fixture();
        let backend = CpuBackend;
        for (prefix, budget, expected, stop) in [
            ("p", 8, vec![0, 1, 2, 3], TurnStop::Eos),
            ("b", 8, vec![2, 3], TurnStop::Eos),
            ("p", 2, vec![0, 1, 2, 4], TurnStop::MaxTokens),
            ("p", 1, vec![0, 1, 4], TurnStop::MaxTokens),
            ("p", 0, vec![0, 4], TurnStop::MaxTokens),
        ] {
            let mut engine =
                LlamaChatModel::new(&model, &backend, &tokenizer, 64, identity.clone());
            let mut emitted = Vec::new();
            let turn = engine
                .generate_turn(
                    prefix,
                    "s",
                    &GenerationParams {
                        max_new_tokens: budget,
                        ..Default::default()
                    },
                    &CancelFlag::new(),
                    &mut |id, _| emitted.push(id),
                )
                .unwrap();
            assert_eq!(turn.stop, Some(stop));
            assert_eq!(turn.committed_ids, expected);
            assert_eq!(turn.committed_ids.len(), engine.committed_len());
            assert_eq!(
                emitted.len(),
                turn.decode_evaluations + usize::from(budget > 0)
            );
            let mut reference = model.create_cache(&backend, expected.len());
            model
                .forward_last_logits_with_cache(&backend, &expected, &mut reference, 0)
                .unwrap();
            assert_eq!(
                engine.cache.export_compact_prefix(expected.len()).unwrap(),
                reference.export_compact_prefix(expected.len()).unwrap()
            );
        }
    }

    #[test]
    fn native_stop_string_keeps_generated_ids_and_truncation_restores_boundary() {
        let (model, tokenizer, identity) = fixture();
        let backend = CpuBackend;
        let mut engine = LlamaChatModel::new(&model, &backend, &tokenizer, 64, identity);
        engine.commit_message("x").unwrap();
        let boundary = engine.committed_len();
        let before = engine.cache.export_compact_prefix(boundary).unwrap();
        let turn = engine
            .generate_turn(
                "p",
                "s",
                &GenerationParams {
                    stop_strings: vec!["a".into()],
                    ..Default::default()
                },
                &CancelFlag::new(),
                &mut |_, _| {},
            )
            .unwrap();
        assert_eq!(turn.stop, Some(TurnStop::StopString("a".into())));
        assert_eq!(turn.committed_ids, vec![0, 1, 4]);
        assert!(turn.text.is_empty());
        assert_eq!(turn.committed_ids.len(), engine.committed_len() - boundary);
        engine.truncate_to(boundary).unwrap();
        assert_eq!(
            engine.cache.export_compact_prefix(boundary).unwrap(),
            before
        );
    }

    #[test]
    fn native_cancellation_discards_partial_token_bookkeeping() {
        let (model, tokenizer, identity) = fixture();
        let backend = CpuBackend;
        let mut engine = LlamaChatModel::new(&model, &backend, &tokenizer, 64, identity);
        engine.commit_message("x").unwrap();
        let before = engine.cache.export_compact_prefix(1).unwrap();
        let cancel = CancelFlag::new();
        let turn = engine
            .generate_turn(
                "p",
                "s",
                &GenerationParams::default(),
                &cancel,
                &mut |_, _| cancel.cancel(),
            )
            .unwrap();
        assert!(turn.cancelled);
        assert!(turn.committed_ids.is_empty());
        assert_eq!(engine.committed_len(), 1);
        assert_eq!(engine.cache.export_compact_prefix(1).unwrap(), before);
    }
}

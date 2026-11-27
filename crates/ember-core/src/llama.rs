use crate::backend::{AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, CpuError, Module};
use crate::experiments::{
    ActiveHooks, DisabledHooks, ExecutionContext, ExperimentRunner, ExperimentalForwardModel,
    LayerHooks, SliceActivation,
};
use crate::hook_types::DispatchPath;
use crate::model::{pool_layer_activation, ForwardModel, Linear};
use crate::plan::{ExecutionMode, ExecutionPlan, HookMode};
use crate::planned_decode::{
    forward_last_logits_planned, forward_last_logits_planned_into, PlannedDecodeState,
};
use crate::tensor::CpuTensor;
use crate::workspace::Workspace;
use alloc::vec::Vec;
use anyhow::Context;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

const INTERLEAVED_MIN_OUT_FEATURES: usize = 65_536;

#[path = "llama_batch.rs"]
mod batch;
pub use batch::{DecodeBatchExperiment, DecodeBatchSequence};

/// v0.4 execution-plan cache. Every input that changes serialized plan
/// content or scratch/KV capacity participates in the key; otherwise a
/// snapshot replay using a different cache capacity could reuse an undersized
/// score arena or stale provenance.
type PlanCacheKey = (
    ExecutionMode,
    HookMode,
    Vec<String>,
    usize,
    usize,
    String,
    String,
);
pub(crate) type PlanCache =
    std::sync::OnceLock<std::sync::Mutex<BTreeMap<PlanCacheKey, Arc<ExecutionPlan>>>>;

thread_local! {
    /// One decode workspace per calling thread. Dimensions are checked before
    /// every use so sequential inference with different models remains safe.
    static LLAMA_DECODE_WORKSPACE: RefCell<Option<Workspace>> = const { RefCell::new(None) };
    /// Reused `[vocab]` logits buffer for greedy decode (see
    /// `greedy_next_token_with_cache`).
    static GREEDY_LOGITS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Argmax of a full logits row with the sampler's tie rule (lowest index
/// wins), rejecting non-finite values like the materialized greedy route.
fn greedy_argmax(logits: &[f32]) -> Result<(u32, f32), CpuError> {
    if let Some((index, value)) = logits
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CpuError::ShapeMismatch(format!(
            "greedy logits contain non-finite value {value} at index {index}"
        )));
    }
    if logits.is_empty() {
        return Err(CpuError::ShapeMismatch("greedy logits are empty".into()));
    }
    let best = crate::sampler::argmax_token(logits);
    let best_u32 = u32::try_from(best).map_err(|_| {
        CpuError::ShapeMismatch("model vocabulary exceeds u32 token-ID space".into())
    })?;
    Ok((best_u32, logits[best]))
}

macro_rules! llama_trace_span {
    ($($argument:tt)*) => {
        if crate::trace::is_tracing() {
            crate::trace::span($($argument)*)
        } else {
            None
        }
    };
}

mod attention;
mod block;
mod config;
mod fast_decode;
mod load;
pub(crate) use attention::LlamaAttentionScratch;
#[cfg(test)]
use attention::{apply_rope_and_qk_norm, RopeQkNormSpec};
pub use attention::{LlamaAttention, DEFAULT_QK_NORM_EPS};
pub(crate) use block::LlamaLayerScratch;
pub use block::{LlamaBlock, LlamaMlp};
pub use config::{LlamaConfig, QkNormOrder, RopeLayout};
pub use load::llama_embed_tokens;
#[cfg(test)]
use load::take_llama_linear;

/// the full llama transformer model.
///
/// fields match the gguf tensor names in comments:
///   `token_embd.weight`      -> embed_tokens
///   `blk.{i}.*`              -> blocks
///   `output_norm.weight`     -> norm  (rms, no bias)
///   `output.weight`          -> head  (linear, no bias)
///
/// embedding lookup replaces gpt-2's `wte + wpe` with a single
/// token embedding (no learned position embeddings - rope handles
/// position). the `from_loader` builder reads llama-specific gguf
/// metadata keys.
pub enum LlamaEmbedding<B: Backend> {
    F32(B::Tensor),
    /// f16/bf16 table in its GGUF encoding (dims `[embed, vocab]`, one
    /// contiguous row per token), widened per looked-up row.
    Half(crate::half_weight::HalfWeight),
    Q8_0(crate::quant::QuantizedWeight),
    KQuant(crate::quant_k::KQuantWeight),
}

pub struct Llama<B: Backend> {
    /// token embedding table, shape [vocab_size, embed_dim]
    pub embed_tokens: LlamaEmbedding<B>,
    /// transformer decoder blocks
    pub blocks: Vec<LlamaBlock<B>>,
    /// final rms normalization weight
    pub norm: B::Tensor,
    /// lm head: projects hidden states to vocab logits (no bias)
    pub head: Linear<B>,
    /// model configuration
    pub config: LlamaConfig,
    /// Cached eligibility result for the allocation-free single-token path.
    fast_decode_inter_dim: Option<usize>,
    /// v0.4 execution-plan cache keyed by (mode, hook mode, active stages).
    pub(crate) plan_cache: PlanCache,
    /// Run-level hashes used by both the serialized and actually executed plan.
    plan_provenance: RefCell<(Option<String>, Option<String>, Option<usize>)>,
    /// Original per-tensor K-family decisions retained after the loader is
    /// consumed so execution plans preserve dtype and fallback provenance.
    pub(crate) k_decisions: BTreeMap<String, crate::quant_k::KTensorDecision>,
    /// Whether the LM head is tied to the embedding tensor (no output.weight).
    ///
    /// `pub(crate)` so `src/planned_decode.rs` (the plan interpreter) can
    /// resolve the final-op kernel without an accessor.
    pub(crate) head_tied: bool,
    /// Active v0.4 execution mode for decode (set by the CLI per run).
    execution_mode: RefCell<ExecutionMode>,
    /// v0.4 planned decode session (arena + resolved ops), keyed by plan
    /// hash; rebuilt when the plan changes.
    ///
    /// `pub(crate)` so `src/planned_decode.rs` (the plan interpreter) can
    /// own the session lifecycle.
    pub(crate) decode_state: RefCell<Option<PlannedDecodeState>>,
    /// Diagnostics: wall time spent building packed decode representations
    /// (Q8_0 VNNI tiles) during load.
    packing_ns: u64,
    /// Diagnostics: wall time spent building the interleaved lm-head layout.
    packing_interleaved_ns: u64,
    /// Per-frequency-pair `rope_freqs.weight` factors (llama3-style RoPE
    /// scaling) baked into the RoPE tables; `None` for uniform-theta RoPE.
    rope_frequency_factors: Option<Vec<f32>>,
}

/// Existing Llama projection groups that can receive the packed Q8_0 decode
/// representation. The tied embedding and LM head are deliberately excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LlamaPackedSelection {
    GateUp,
    Mlp,
    Attention,
    AttentionGateUp,
    All,
}

impl LlamaPackedSelection {
    #[inline]
    fn includes_attention(self) -> bool {
        matches!(self, Self::Attention | Self::AttentionGateUp | Self::All)
    }

    #[inline]
    fn includes_gate_up(self) -> bool {
        matches!(
            self,
            Self::GateUp | Self::Mlp | Self::AttentionGateUp | Self::All
        )
    }

    #[inline]
    fn includes_down(self) -> bool {
        matches!(self, Self::Mlp | Self::All)
    }
}

/// Work performed while constructing selected packed projections.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct LlamaPackingStats {
    pub weights_packed: usize,
    pub packed_bytes: usize,
    pub packing_ns: u64,
    pub eviction_attempts: usize,
    pub eviction_successes: usize,
    pub eviction_ns: u64,
}

/// Work performed by a later residency-only eviction pass.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct LlamaEvictionStats {
    pub eviction_attempts: usize,
    pub eviction_successes: usize,
    pub eviction_ns: u64,
}

impl ForwardModel<CpuBackend> for Llama<CpuBackend> {
    fn create_cache(&self, _backend: &CpuBackend, max_seq_len: usize) -> crate::kv_cache::KVCache {
        crate::kv_cache::KVCache::new(
            self.blocks.len(),
            self.config.n_kv_heads,
            self.config.head_dim,
            max_seq_len,
        )
    }
    fn max_seq_len(&self, _backend: &CpuBackend) -> usize {
        self.config.max_seq_len
    }
    fn forward_with_cache(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        Llama::forward_with_cache(self, backend, token_ids, cache, start_pos)
    }
    fn forward_last_logits_with_cache(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        self.forward_last_logits_with_cache_cpu(backend, token_ids, cache, start_pos)
    }
    fn forward_last_logits_with_cache_reusing(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        output: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        // Reusing the caller's buffer keeps the steady-state decode loop free
        // of logits allocations on the Q8_0 fast path and the planned
        // interpreter (the reference oracle still materializes internally and
        // copies).
        if output.shape() == [1, self.config.vocab_size] {
            return self.forward_last_logits_with_cache_cpu_into(
                backend,
                token_ids,
                cache,
                start_pos,
                output.data_mut(),
            );
        }
        *output = self.forward_last_logits_with_cache_cpu(backend, token_ids, cache, start_pos)?;
        Ok(())
    }

    fn forward_embeddings_with_cache(
        &self,
        backend: &CpuBackend,
        embeddings: &CpuTensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        Llama::forward_embeddings_with_cache(self, backend, embeddings, cache, start_pos)
    }
    fn greedy_next_token_with_cache(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<(u32, f32), CpuError> {
        // Greedy is defined as the argmax of exactly the logits the sampling
        // route would compute: same route order (fast, planned, reference),
        // same kernels and accumulation order on every architecture, so a
        // near-tie cannot resolve differently. The vocab-sized buffer is
        // thread-local and reused, so steady-state greedy decode performs no
        // heap allocation after the first token.
        let vocab = self.config.vocab_size;
        let mut logits = GREEDY_LOGITS.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        if logits.len() != vocab {
            logits.clear();
            logits.resize(vocab, 0.0);
        }
        let result = self
            .forward_last_logits_with_cache_cpu_into(
                backend,
                token_ids,
                cache,
                start_pos,
                &mut logits,
            )
            .and_then(|()| greedy_argmax(&logits));
        GREEDY_LOGITS.with(|cell| *cell.borrow_mut() = logits);
        result
    }
    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
    fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }
    fn vocab_size(&self, _backend: &CpuBackend) -> usize {
        self.config.vocab_size
    }
    fn forward_with_activations(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
        Llama::forward_with_activations(self, backend, token_ids)
    }

    fn forward_pooled_activations(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
        Llama::forward_pooled_activations(self, backend, token_ids, token_index_groups)
    }

    fn forward_pooled_hidden_states(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, CpuError> {
        Llama::forward_pooled_hidden_states(self, backend, token_ids, token_index_groups)
    }

    fn forward_pooled_with_blocks(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        block_boundaries: &[usize],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
        Llama::forward_pooled_with_blocks(
            self,
            backend,
            token_ids,
            block_boundaries,
            token_index_groups,
        )
    }
}

impl ExperimentalForwardModel for Llama<CpuBackend> {
    fn forward_last_logits_with_experiment(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        execution: ExecutionContext<'_>,
        runner: &mut ExperimentRunner,
    ) -> Result<CpuTensor, CpuError> {
        let hook_mode = runner.hook_mode();
        let active_site_keys = runner.active_plan_sites(self.blocks.len(), execution.phase);
        let active_sites: Vec<&str> = active_site_keys.iter().map(String::as_str).collect();
        let fast_eligible = token_ids.len() == 1
            && !crate::trace::is_tracing()
            && self.fast_decode_inter_dim.is_some();
        let mut hooks = ActiveHooks::new(runner, execution);

        // Q8_0's native hooked fast path retains precedence over v0.4 plans.
        // It records `Fast` only if the concrete route succeeds.
        if fast_eligible
            && let Some(result) = self
                .forward_decode_fast_hooked(backend, token_ids, cache, start_pos, &mut hooks, None)
        {
            // Experiment execution always materializes logits for the hooks.
            return result.map(|logits| {
                logits.expect("fast decode without a caller buffer materializes logits")
            });
        }

        // K-quant models are fast-path ineligible and reach the v0.4 plan.
        if self.planned_decode_eligible(token_ids) {
            hooks.note_dispatch_path(DispatchPath::Planned);
            let planned_hooks = if hook_mode == HookMode::Disabled {
                None
            } else {
                Some((&mut hooks, &execution))
            };
            return forward_last_logits_planned(
                self,
                token_ids,
                cache,
                start_pos,
                planned_hooks,
                hook_mode,
                &active_sites,
            );
        }

        // Record Generic only for the route that actually executes; do not
        // pre-record it for a planned evaluation.
        hooks.note_dispatch_path(DispatchPath::Generic);
        self.forward_last_logits_with_cache_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }
}

impl<B: Backend> Llama<B> {
    /// Final RMS norm over residual-stream rows (`[rows, embed_dim]`).
    ///
    /// This is the exact operation the generic (reference / prefill) forward
    /// path applies after the last block; the logit lens
    /// (`ember::v05::lens`) calls it so its last-layer projection is the
    /// model's own final-logits computation rather than a re-implementation.
    pub fn final_norm(&self, backend: &B, hidden: &B::Tensor) -> Result<B::Tensor, B::Error> {
        backend.rms_norm(hidden, &self.norm, self.config.norm_eps)
    }

    /// LM head (untied `output.weight` or tied token embeddings) over
    /// final-norm output rows. Shared with the generic forward path.
    pub fn lm_head(&self, backend: &B, normed: &B::Tensor) -> Result<B::Tensor, B::Error> {
        self.head.forward(backend, normed)
    }

    /// create a kv cache sized for this model's parameters.
    ///
    /// Whether the LM head reuses the token embedding (no `output.weight`).
    #[cfg(any(test, feature = "testkit"))]
    #[doc(hidden)]
    pub fn head_tied(&self) -> bool {
        self.head_tied
    }

    /// Whether a planned decode session has been built (the planned path ran).
    #[cfg(any(test, feature = "testkit"))]
    #[doc(hidden)]
    pub fn has_planned_session(&self) -> bool {
        self.decode_state.borrow().is_some()
    }

    /// The `rope_freqs.weight` factors applied to every RoPE frequency pair,
    /// or `None` when the model uses plain uniform-theta RoPE.
    pub fn rope_frequency_factors(&self) -> Option<&[f32]> {
        self.rope_frequency_factors.as_deref()
    }

    /// important difference from gpt-2: the cache allocates for
    /// `n_kv_heads` kv heads, not `n_heads` query heads.
    /// gqa repeats k/v during attention rather than storing duplicates.
    pub fn create_cache(&self, _backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache {
        crate::kv_cache::KVCache::new(
            self.blocks.len(),
            self.config.n_kv_heads,
            self.config.head_dim,
            max_seq_len,
        )
    }

    /// create a kv cache sized for ONE request instead of the model's full
    /// metadata context.
    ///
    /// KV-cache capacity never affects forward results (positions are
    /// explicit), but eager allocation at metadata context (e.g. 131072 for
    /// Llama-3.2-1B ≈ 8 GiB here) pushes interactive runs into swap on
    /// memory-constrained hosts and collapses decode throughput. Request-
    /// sized caches keep identical numerics with a footprint proportional
    /// to actual work.
    pub fn create_request_cache(
        &self,
        backend: &B,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> crate::kv_cache::KVCache {
        let needed = prompt_tokens
            .saturating_add(max_new_tokens)
            .min(self.config.max_seq_len);
        self.create_cache(backend, needed)
    }

    /// forward pass with incremental kv caching.
    ///
    /// mirrors `Gpt2::forward_with_cache` but:
    ///   - uses `LlamaBlock::forward_with_cache` which passes start_pos for rope
    ///   - normalizes with rms norm (via `backend.rms_norm`)
    ///   - no position embedding lookup (rope is in the attention layer)
    pub fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "Llama forward requires at least one token"
        );
        let embed_dim = self.config.embed_dim;
        let x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;
        self.forward_owned_embeddings_with_cache(backend, x, cache, start_pos)
    }

    /// forward pass with incremental kv caching over precomputed embeddings.
    ///
    /// `embeddings` is a `[seq_len, embed_dim]` tensor of *fully formed*
    /// model input embeddings (for a RoPE model these are the token-embedding
    /// rows — or, for multimodal input, the assembled text + visual embedding
    /// rows). This is the same internal path token prefill uses; the
    /// transformer never needs to know where the embeddings came from.
    ///
    /// positions are contiguous from `start_pos` (RoPE tables and the KV
    /// cache are addressed the same way as in `forward_with_cache`), and
    /// attention is causal over `cache.cursor() .. cache.cursor() + seq_len`.
    pub fn forward_embeddings_with_cache(
        &self,
        backend: &B,
        embeddings: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        self.forward_owned_embeddings_with_cache(backend, embeddings.clone(), cache, start_pos)
    }

    /// [`Llama::forward_embeddings_with_cache`] over an owned residual
    /// stream, which the layers update in place.
    fn forward_owned_embeddings_with_cache(
        &self,
        backend: &B,
        mut x: B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        cache.validate_start_pos(start_pos);
        let seq_len = backend.shape(&x)[0];
        assert!(seq_len > 0, "Llama forward requires at least one embedding");

        let mut hooks = DisabledHooks;
        let mut scratch = LlamaLayerScratch::new(backend)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            block.forward_with_cache_hooked(
                backend,
                &mut x,
                cache,
                layer,
                start_pos,
                &mut hooks,
                &mut scratch,
            )?;
        }
        // advance the cache cursor after all layers have stored k/v
        for _ in 0..seq_len {
            cache.advance_cursor();
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        self.head.forward(backend, &x)
    }

    pub fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut hooks = DisabledHooks;
        self.forward_last_logits_with_cache_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }

    /// last-logits forward over precomputed embeddings (no hooks).
    ///
    /// See [`Llama::forward_embeddings_with_cache`] for the embeddings
    /// contract; this variant only materializes the final token's logits.
    pub fn forward_last_logits_embeddings_with_cache(
        &self,
        backend: &B,
        embeddings: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut hooks = DisabledHooks;
        self.forward_last_logits_embeddings_with_cache_hooked(
            backend, embeddings, cache, start_pos, &mut hooks,
        )
    }

    fn forward_last_logits_with_cache_hooked<H>(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "Llama forward requires at least one token"
        );
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;

        // -- embedding lookup --
        let _span_emb = llama_trace_span!(
            "embedding",
            usize::MAX,
            OpKind::Embedding,
            vec![seq_len, embed_dim],
            trace::bytes_from_shape(&[seq_len, embed_dim]),
            trace::flops_embedding(),
        );
        let x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;
        if let Some(s) = _span_emb {
            s.end(
                vec![seq_len, embed_dim],
                trace::bytes_from_shape(&[seq_len, embed_dim]),
            );
        }

        self.forward_last_logits_owned_embeddings_with_cache_hooked(
            backend, x, cache, start_pos, hooks,
        )
    }

    /// last-logits forward over precomputed embeddings (hooked).
    ///
    /// See [`Llama::forward_embeddings_with_cache`] for the embeddings
    /// contract; this variant only materializes the final token's logits.
    pub(crate) fn forward_last_logits_embeddings_with_cache_hooked<H>(
        &self,
        backend: &B,
        embeddings: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        self.forward_last_logits_owned_embeddings_with_cache_hooked(
            backend,
            embeddings.clone(),
            cache,
            start_pos,
            hooks,
        )
    }

    /// Hooked last-logits forward over an owned residual stream, which the
    /// layers update in place through one shared [`LlamaLayerScratch`].
    fn forward_last_logits_owned_embeddings_with_cache_hooked<H>(
        &self,
        backend: &B,
        x: B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        self.forward_last_logits_from_layer_hooked(backend, x, 0, cache, start_pos, hooks)
    }

    /// Hooked last-logits forward that starts at block `first_layer`, with
    /// `x` as the residual stream entering that block.
    ///
    /// `first_layer == 0` is the ordinary embeddings forward above. A later
    /// start serves experiment prefix reuse (`crate::experiments::resume`):
    /// the caller supplies the exact hidden state and a cache whose layers
    /// `< first_layer` already hold this sequence's K/V, so every block that
    /// runs executes the same code on the same inputs as a full forward. The
    /// layer scratch is fresh either way: it is sized by the first block
    /// that runs and fully overwritten by every block.
    pub(crate) fn forward_last_logits_from_layer_hooked<H>(
        &self,
        backend: &B,
        mut x: B::Tensor,
        first_layer: usize,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        cache.validate_start_pos(start_pos);
        let seq_len = backend.shape(&x)[0];
        let embed_dim = self.config.embed_dim;
        assert!(seq_len > 0, "Llama forward requires at least one embedding");
        assert!(
            first_layer <= self.blocks.len(),
            "first layer {first_layer} exceeds the model's {} blocks",
            self.blocks.len()
        );

        let mut scratch = LlamaLayerScratch::new(backend)?;
        for (layer, block) in self.blocks.iter().enumerate().skip(first_layer) {
            hooks.before_layer(layer, &mut x)?;
            block.forward_with_cache_hooked(
                backend,
                &mut x,
                cache,
                layer,
                start_pos,
                hooks,
                &mut scratch,
            )?;
            hooks.after_layer(layer, &mut x)?;
        }
        for _ in 0..seq_len {
            cache.advance_cursor();
        }

        let last = backend.row_as_2d(&x, seq_len - 1)?;

        // -- final RMS norm --
        let _span_final_norm = llama_trace_span!(
            "final_norm",
            usize::MAX,
            OpKind::RmsNorm,
            vec![1, embed_dim],
            trace::bytes_from_shape(&[1, embed_dim]) + trace::bytes_from_shape(&[1, embed_dim]),
            trace::flops_rms_norm(1, embed_dim),
        );
        let mut last = self.final_norm(backend, &last)?;
        if let Some(s) = _span_final_norm {
            s.end(vec![1, embed_dim], trace::bytes_from_shape(&[1, embed_dim]));
        }
        hooks.before_logits(&mut last)?;

        // -- LM head --
        let _span_head = llama_trace_span!(
            "lm_head",
            usize::MAX,
            OpKind::MatMulQ8_0,
            vec![1, embed_dim],
            trace::bytes_matmul_input(1, embed_dim, self.head.weight_bytes(backend)),
            trace::flops_matmul(1, self.config.vocab_size, embed_dim),
        );
        let mut result = self.lm_head(backend, &last)?;
        let vocab_size = backend.shape(&result)[1];
        if let Some(s) = _span_head {
            s.end(
                vec![1, vocab_size],
                trace::bytes_matmul_output(1, vocab_size),
            );
        }
        hooks.after_logits(&mut result)?;

        Ok(result)
    }

    /// forward pass without caching (full sequence).
    pub fn forward(&self, backend: &B, token_ids: &[u32]) -> Result<B::Tensor, B::Error> {
        let embed_dim = self.config.embed_dim;
        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;

        for block in &self.blocks {
            x = block.forward(backend, &x)?;
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        self.head.forward(backend, &x)
    }

    /// forward pass with activation capture after each transformer block.
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let embed_dim = self.config.embed_dim;
        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;

        let mut activations = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            x = block.forward(backend, &x)?;
            let data = backend.data(&x);
            activations.push(data.to_vec());
        }
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((activations, logits))
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let (pooled, x) =
            self.forward_pooled_hidden_and_output(backend, token_ids, token_index_groups)?;
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((pooled, logits))
    }

    pub fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        self.forward_pooled_hidden_and_output(backend, token_ids, token_index_groups)
            .map(|(pooled, _)| pooled)
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_pooled_hidden_and_output(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let embed_dim = self.config.embed_dim;
        let mut pooled = token_index_groups
            .iter()
            .map(|_| vec![0.0f32; self.blocks.len() * embed_dim])
            .collect::<Vec<_>>();

        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;

        for (li, block) in self.blocks.iter().enumerate() {
            x = block.forward(backend, &x)?;
            let data = backend.data(&x);
            for (gi, token_indices) in token_index_groups.iter().enumerate() {
                let offset = li * embed_dim;
                pool_layer_activation(
                    data,
                    token_indices,
                    embed_dim,
                    &mut pooled[gi][offset..offset + embed_dim],
                );
            }
        }
        Ok((pooled, x))
    }

    /// batched forward pass with block-diagonal attention for independent sequences.
    ///
    /// `token_ids` is a concatenation of all stimuli tokens.
    /// `block_boundaries` marks the start position of each stimulus block
    /// (the first boundary is always 0, each subsequent boundary is the
    /// cumulative token count after the previous stimulus).
    /// `token_index_groups` maps each output group to the token position(s)
    /// to pool from within its stimulus block.
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_pooled_with_blocks(
        &self,
        backend: &B,
        token_ids: &[u32],
        block_boundaries: &[usize],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        crate::model::validate_block_partition(
            token_ids.len(),
            block_boundaries,
            token_index_groups,
        );
        let embed_dim = self.config.embed_dim;
        let mut pooled = token_index_groups
            .iter()
            .map(|_| vec![0.0f32; self.blocks.len() * embed_dim])
            .collect::<Vec<_>>();

        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;

        for (li, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_blocks(backend, &x, block_boundaries)?;
            let data = backend.data(&x);
            for (gi, token_indices) in token_index_groups.iter().enumerate() {
                let offset = li * embed_dim;
                pool_layer_activation(
                    data,
                    token_indices,
                    embed_dim,
                    &mut pooled[gi][offset..offset + embed_dim],
                );
            }
        }
        let mut last_rows = backend.zeroes(&[block_boundaries.len(), embed_dim])?;
        for (block_index, &start) in block_boundaries.iter().enumerate() {
            let end = block_boundaries
                .get(block_index + 1)
                .copied()
                .unwrap_or(token_ids.len());
            assert!(start < end && end <= token_ids.len());
            let row = backend.row_as_2d(&x, end - 1)?;
            backend.assign_row(&mut last_rows, block_index, &row);
        }
        let last_rows = backend.rms_norm(&last_rows, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &last_rows)?;
        Ok((pooled, logits))
    }
}

// ---------------------------------------------------------------------------
// v0.4 execution plan construction
// (contract: docs/v04-execution-contract.md, sections 3, 5, 10)
// ---------------------------------------------------------------------------

/// v0.4 planned decode is single-token (seq = 1); prefill stays on the
/// reference path. The scratch arena is sized for this capacity.
pub(crate) const PLAN_DECODE_SEQ: usize = 1;

/// Scratch alignment for all arena regions (AVX2-safe).
pub(crate) const PLAN_REGION_ALIGN: usize = 64;

/// v0.3 per-tensor execution decision name for provenance records.
/// `pub(crate)` for `src/planned_decode.rs` (kernel resolution).
pub(crate) fn k_execution_name(execution: crate::quant_k::KExecution) -> &'static str {
    match execution {
        crate::quant_k::KExecution::EagerF32 => "eager_f32",
        crate::quant_k::KExecution::CompressedScalar => "compressed_scalar",
        crate::quant_k::KExecution::CompressedX86 => "compressed_x86",
        crate::quant_k::KExecution::CompressedArm => "compressed_arm",
    }
}

impl Llama<CpuBackend> {
    /// Set the active v0.4 execution mode for decode (per run; default
    /// `Reference`).
    pub fn set_execution_mode(&self, mode: ExecutionMode) {
        *self.execution_mode.borrow_mut() = mode;
        log::info!("execution mode: {}", mode.name());
    }

    pub fn set_plan_provenance(&self, model: String, tokenizer: String, canonical_capacity: usize) {
        *self.plan_provenance.borrow_mut() =
            (Some(model), Some(tokenizer), Some(canonical_capacity));
    }

    /// Borrowed provenance: the planned decode validates a cached session
    /// against it per token, so this must not clone the hash strings.
    pub(crate) fn plan_provenance(
        &self,
    ) -> std::cell::Ref<'_, (Option<String>, Option<String>, Option<usize>)> {
        self.plan_provenance.borrow()
    }

    /// The active v0.4 execution mode.
    pub fn execution_mode(&self) -> ExecutionMode {
        *self.execution_mode.borrow()
    }

    /// Diagnostics: wall time spent packing decode representations at load.
    pub fn packing_ns(&self) -> u64 {
        self.packing_ns
    }

    /// Diagnostics: wall time spent building the interleaved lm-head layout.
    pub fn packing_interleaved_ns(&self) -> u64 {
        self.packing_interleaved_ns
    }

    /// Whether the plan-driven single-token decode path applies: mode is
    /// `Planned` or `PlannedFused`, exactly one token, and tracing is off.
    fn planned_decode_eligible(&self, token_ids: &[u32]) -> bool {
        matches!(
            *self.execution_mode.borrow(),
            ExecutionMode::Planned | ExecutionMode::PlannedFused
        ) && token_ids.len() == 1
            && !crate::trace::is_tracing()
    }

    /// Single-token logits with v0.4 dispatch: the Q8_0 native fast path
    /// keeps precedence (contract: Q8_0 is never rerouted), then the planned
    /// interpreter, then the generic reference path.
    pub(crate) fn forward_last_logits_with_cache_cpu(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        if let Some(result) = self.forward_decode_fast(backend, token_ids, cache, start_pos) {
            return result;
        }
        if self.planned_decode_eligible(token_ids) {
            return forward_last_logits_planned(
                self,
                token_ids,
                cache,
                start_pos,
                None,
                HookMode::Disabled,
                &[],
            );
        }
        Llama::forward_last_logits_with_cache(self, backend, token_ids, cache, start_pos)
    }

    /// Single-token logits written into `out` (the `[1, vocab]` payload).
    ///
    /// Mirrors [`Self::forward_last_logits_with_cache_cpu`]'s route order
    /// (fast path, planned interpreter, reference) but avoids materializing a
    /// `CpuTensor` for the two allocation-free routes. `out` must be exactly
    /// `vocab_size` long.
    pub(crate) fn forward_last_logits_with_cache_cpu_into(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        out: &mut [f32],
    ) -> Result<(), CpuError> {
        if out.len() != self.config.vocab_size {
            return Err(CpuError::ShapeMismatch(format!(
                "logits buffer has {} values, expected {}",
                out.len(),
                self.config.vocab_size
            )));
        }
        if let Some(result) =
            self.forward_decode_fast_into(backend, token_ids, cache, start_pos, out)
        {
            return result;
        }
        if self.planned_decode_eligible(token_ids) {
            return forward_last_logits_planned_into(
                self,
                token_ids,
                cache,
                start_pos,
                None,
                HookMode::Disabled,
                &[],
                out,
            );
        }
        let logits =
            Llama::forward_last_logits_with_cache(self, backend, token_ids, cache, start_pos)?;
        if logits.data().len() != out.len() {
            return Err(CpuError::ShapeMismatch(format!(
                "logits payload has {} values, expected {}",
                logits.data().len(),
                out.len()
            )));
        }
        out.copy_from_slice(logits.data());
        Ok(())
    }
}

/// Synthetic models shared by this crate's tests and the `ember` crate's
/// (which cannot reach `Llama`'s private fields). Test builds only: the
/// `testkit` feature, enabled by `ember`'s dev-dependency. Not API.
#[cfg(any(test, feature = "testkit"))]
#[doc(hidden)]
pub mod testkit;

#[cfg(test)]
mod tests;

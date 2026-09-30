//! Batched multi-sequence single-token decode for Q8_0 Llama models.
//!
//! # API contract
//!
//! [`Llama::forward_decode_batch`] advances `N` independent sequences by one
//! token each in a single forward pass. Each [`DecodeBatchSequence`] owns its
//! own KV cache, position, input token and logits buffer; nothing is shared
//! between sequences except the weights, whose every read is reused across the
//! `N` activation rows.
//!
//! * **Bit identity.** Sequence `i`'s logits, KV-cache contents and every hook
//!   activation are bit-identical to calling the single-sequence decode
//!   ([`crate::model::ForwardModel::forward_last_logits_with_cache`] on the
//!   Q8_0 fast path) with the same cache, token and position. Each output
//!   element of every projection is computed with exactly the single-row
//!   arithmetic in the same order; per-row operations (RMSNorm, RoPE,
//!   attention, SiLU, residual adds) run the same functions on each row.
//! * **Hooks.** [`Llama::forward_decode_batch_with_experiments`] gives each
//!   sequence its own [`ExperimentRunner`] and [`ExecutionContext`]. At every
//!   hook site the runners are called in sequence order with a `[1, width]`
//!   view of their own row — the view a single-sequence decode would pass —
//!   so interventions on one sequence never touch another. Runners observe
//!   the Q8_0 fast dispatch path exactly as single decode records it.
//! * **Positions.** Sequences may sit at different positions; `start_pos`
//!   must equal the cache cursor as for single decode.
//! * **Errors.** Shape errors are reported before any cache is touched. If a
//!   hook fails mid-pass the whole call returns that error; as with a failed
//!   single decode, no cache cursor is advanced (rows appended for the
//!   current position are overwritten by the next attempt).
//! * **Eligibility.** Only models on the allocation-free Q8_0 decode path
//!   (adjacent-pair RoPE, bias-free Q8_0 projections) are supported; see
//!   [`Llama::supports_batched_decode`]. Tracing must be off.
//!
//! The experiment runner, GUI and sweeps do not use this entry point yet.

use super::*;

thread_local! {
    static LLAMA_BATCH_WORKSPACE: RefCell<Option<BatchWorkspace>> = const { RefCell::new(None) };
}

/// One sequence advanced by [`Llama::forward_decode_batch`].
pub struct DecodeBatchSequence<'a> {
    /// Token fed at `start_pos`.
    pub token_id: u32,
    /// This sequence's KV cache; its cursor must equal `start_pos`.
    pub cache: &'a mut crate::kv_cache::KVCache,
    /// Absolute position of `token_id`.
    pub start_pos: usize,
    /// Receives the `[vocab]` next-token logits.
    pub logits: &'a mut [f32],
}

/// Per-sequence experiment hooks for
/// [`Llama::forward_decode_batch_with_experiments`].
pub struct DecodeBatchExperiment<'runner, 'model> {
    pub runner: &'runner mut ExperimentRunner,
    pub execution: ExecutionContext<'model>,
}

/// Row-major `[N, width]` buffers plus quantized activations for a batch.
struct BatchWorkspace {
    rows: usize,
    dims: [usize; 4],
    x: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gated: Vec<f32>,
    logits: Vec<f32>,
    quantized: Vec<u8>,
}

impl BatchWorkspace {
    fn new(rows: usize, embed: usize, q_dim: usize, kv_dim: usize, inter: usize) -> Self {
        let buffer = |width: usize| vec![0.0f32; rows * width];
        Self {
            rows,
            dims: [embed, q_dim, kv_dim, inter],
            x: buffer(embed),
            norm: buffer(embed),
            q: buffer(q_dim),
            k: buffer(kv_dim),
            v: buffer(kv_dim),
            attention: buffer(q_dim),
            projected: buffer(embed),
            gate: buffer(inter),
            up: buffer(inter),
            gated: buffer(inter),
            logits: Vec::new(),
            quantized: Vec::new(),
        }
    }
}

/// The decode layout the single-sequence path would read for a projection.
fn decode_weight<'a>(linear: &'a Linear<CpuBackend>) -> crate::simd::Q8DecodeWeight<'a> {
    match linear.packed_q8_weight_without_bias() {
        Some(packed) => crate::simd::Q8DecodeWeight::Packed16(packed),
        None => crate::simd::Q8DecodeWeight::Rows(
            linear
                .q8_weight_without_bias()
                .expect("batched decode eligibility checked"),
        ),
    }
}

fn row(buffer: &mut [f32], index: usize, width: usize) -> &mut [f32] {
    &mut buffer[index * width..(index + 1) * width]
}

impl Llama<CpuBackend> {
    /// Whether [`Self::forward_decode_batch`] can run this model.
    pub fn supports_batched_decode(&self) -> bool {
        self.fast_decode_inter_dim.is_some()
    }

    /// Advance every sequence by one token in one forward pass.
    ///
    /// See the [module documentation](self) for the contract: each
    /// sequence's logits and cache are bit-identical to a single-sequence
    /// decode of the same token.
    pub fn forward_decode_batch(
        &self,
        backend: &CpuBackend,
        sequences: &mut [DecodeBatchSequence<'_>],
    ) -> Result<(), CpuError> {
        let mut hooks: Vec<DisabledHooks> = (0..sequences.len()).map(|_| DisabledHooks).collect();
        self.forward_decode_batch_hooked(backend, sequences, &mut hooks)
    }

    /// [`Self::forward_decode_batch`] with one experiment runner per
    /// sequence; `experiments[i]` observes and may intervene on sequence `i`
    /// exactly as it would in a single-sequence decode.
    pub fn forward_decode_batch_with_experiments(
        &self,
        backend: &CpuBackend,
        sequences: &mut [DecodeBatchSequence<'_>],
        experiments: &mut [DecodeBatchExperiment<'_, '_>],
    ) -> Result<(), CpuError> {
        if experiments.len() != sequences.len() {
            return Err(CpuError::ShapeMismatch(format!(
                "batched decode has {} sequences but {} experiment runners",
                sequences.len(),
                experiments.len()
            )));
        }
        let mut hooks: Vec<ActiveHooks<'_, '_>> = experiments
            .iter_mut()
            .map(|experiment| ActiveHooks::new(&mut *experiment.runner, experiment.execution))
            .collect();
        self.forward_decode_batch_hooked(backend, sequences, &mut hooks)
    }

    pub(crate) fn forward_decode_batch_hooked<H>(
        &self,
        backend: &CpuBackend,
        sequences: &mut [DecodeBatchSequence<'_>],
        hooks: &mut [H],
    ) -> Result<(), CpuError>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        let n = sequences.len();
        if n == 0 {
            return Ok(());
        }
        if hooks.len() != n {
            return Err(CpuError::ShapeMismatch(format!(
                "batched decode has {n} sequences but {} hook sets",
                hooks.len()
            )));
        }
        let Some(inter_dim) = self.fast_decode_inter_dim else {
            return Err(CpuError::Kernel(
                "batched decode requires the Q8_0 fast decode path".into(),
            ));
        };
        if crate::trace::is_tracing() {
            return Err(CpuError::Kernel(
                "batched decode is unavailable while tracing".into(),
            ));
        }
        let vocab = self.config.vocab_size;
        for sequence in sequences.iter() {
            if sequence.logits.len() != vocab {
                return Err(CpuError::ShapeMismatch(format!(
                    "logits buffer has {} values, expected {vocab}",
                    sequence.logits.len()
                )));
            }
            if sequence.cache.n_layers() != self.blocks.len() {
                return Err(CpuError::ShapeMismatch(
                    "batched decode cache layer count mismatch".into(),
                ));
            }
            sequence.cache.validate_start_pos(sequence.start_pos);
        }
        let embed_dim = self.config.embed_dim;
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        LLAMA_BATCH_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let dims = [embed_dim, q_dim, kv_dim, inter_dim];
            if workspace
                .as_ref()
                .is_none_or(|current| current.rows < n || current.dims != dims)
            {
                *workspace = Some(BatchWorkspace::new(n, embed_dim, q_dim, kv_dim, inter_dim));
            }
            let workspace = workspace.as_mut().expect("batch workspace initialized");
            for hook in hooks.iter_mut() {
                hook.note_dispatch(DispatchPath::Fast);
            }
            self.forward_decode_batch_with_workspace(backend, sequences, hooks, workspace)
        })
    }

    fn forward_decode_batch_with_workspace<H>(
        &self,
        backend: &CpuBackend,
        sequences: &mut [DecodeBatchSequence<'_>],
        hooks: &mut [H],
        ws: &mut BatchWorkspace,
    ) -> Result<(), CpuError>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        use crate::simd::{matmul_q8_0_decode_tasks, Q8DecodeTask};
        let n = sequences.len();
        let [embed_dim, q_dim, kv_dim, inter_dim] = ws.dims;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let BatchWorkspace {
            x,
            norm,
            q,
            k,
            v,
            attention,
            projected,
            gate,
            up,
            gated,
            logits,
            quantized,
            ..
        } = ws;
        let x = &mut x[..n * embed_dim];
        let norm = &mut norm[..n * embed_dim];
        let q = &mut q[..n * q_dim];
        let k = &mut k[..n * kv_dim];
        let v = &mut v[..n * kv_dim];
        let attention = &mut attention[..n * q_dim];
        let projected = &mut projected[..n * embed_dim];
        let gate = &mut gate[..n * inter_dim];
        let up = &mut up[..n * inter_dim];
        let gated = &mut gated[..n * inter_dim];

        for (index, sequence) in sequences.iter().enumerate() {
            let x_row = row(x, index, embed_dim);
            match &self.embed_tokens {
                LlamaEmbedding::F32(table) => {
                    let token = sequence.token_id as usize;
                    if token >= table.shape()[0] {
                        return Err(CpuError::ShapeMismatch(format!(
                            "embedding token {token} out of bounds for vocabulary {}",
                            table.shape()[0]
                        )));
                    }
                    x_row
                        .copy_from_slice(&table.data()[token * embed_dim..(token + 1) * embed_dim]);
                }
                LlamaEmbedding::Q8_0(table) => {
                    if sequence.token_id as usize >= table.out_features() {
                        return Err(CpuError::ShapeMismatch(format!(
                            "embedding token {} out of bounds for vocabulary {}",
                            sequence.token_id,
                            table.out_features()
                        )));
                    }
                    table.dequantize_row(sequence.token_id as usize, x_row);
                }
                LlamaEmbedding::KQuant(_) => {
                    return Err(CpuError::ShapeMismatch(
                        "batched decode is ineligible for K-quant embeddings".into(),
                    ));
                }
            }
        }

        for (layer, block) in self.blocks.iter().enumerate() {
            for (index, hook) in hooks.iter_mut().enumerate() {
                let mut hidden = SliceActivation::new(1, embed_dim, row(x, index, embed_dim));
                hook.before_layer(layer, &mut hidden)?;
            }
            for index in 0..n {
                crate::simd::rms_norm_q8_decode_into(
                    &x[index * embed_dim..(index + 1) * embed_dim],
                    block.input_layernorm.data(),
                    block.norm_eps,
                    row(norm, index, embed_dim),
                );
            }
            crate::simd::quantize_q8_0_decode_into(norm, quantized);
            matmul_q8_0_decode_tasks(&mut [
                Q8DecodeTask::columns(quantized, decode_weight(&block.self_attn.q_proj), n, q),
                Q8DecodeTask::columns(quantized, decode_weight(&block.self_attn.k_proj), n, k),
                Q8DecodeTask::columns(quantized, decode_weight(&block.self_attn.v_proj), n, v),
            ]);

            for (index, sequence) in sequences.iter_mut().enumerate() {
                let q_row = row(q, index, q_dim);
                let k_row = row(k, index, kv_dim);
                block.self_attn.apply_decode_rope_and_qk_norm(
                    q_row,
                    n_heads,
                    sequence.start_pos,
                    block.self_attn.q_norm.as_ref(),
                );
                block.self_attn.apply_decode_rope_and_qk_norm(
                    k_row,
                    n_kv_heads,
                    sequence.start_pos,
                    block.self_attn.k_norm.as_ref(),
                );
                let cache = &mut *sequence.cache;
                let cursor = cache.cursor();
                cache.append(
                    layer,
                    cursor,
                    k_row,
                    &v[index * kv_dim..(index + 1) * kv_dim],
                );
                let spec = CachedAttentionSpec {
                    n_heads,
                    n_kv_heads,
                    head_dim: self.config.head_dim,
                    max_seq_len: cache.max_seq_len(),
                    total_seq_len: cursor + 1,
                };
                let (cached_k, cached_v, scratch) = cache.get_with_scratch(layer);
                backend.cached_causal_attention_into(
                    q_row,
                    cached_k,
                    cached_v,
                    spec,
                    scratch,
                    row(attention, index, q_dim),
                )?;
            }

            crate::simd::quantize_q8_0_decode_into(attention, quantized);
            matmul_q8_0_decode_tasks(&mut [Q8DecodeTask::columns(
                quantized,
                decode_weight(&block.self_attn.o_proj),
                n,
                projected,
            )]);
            for (index, hook) in hooks.iter_mut().enumerate() {
                let mut output =
                    SliceActivation::new(1, embed_dim, row(projected, index, embed_dim));
                hook.after_attention(layer, &mut output)?;
            }
            crate::simd::add_assign(x, projected);
            for index in 0..n {
                crate::simd::rms_norm_q8_decode_into(
                    &x[index * embed_dim..(index + 1) * embed_dim],
                    block.post_attention_layernorm.data(),
                    block.norm_eps,
                    row(norm, index, embed_dim),
                );
            }
            crate::simd::quantize_q8_0_decode_into(norm, quantized);
            matmul_q8_0_decode_tasks(&mut [
                Q8DecodeTask::columns(quantized, decode_weight(&block.mlp.gate_proj), n, gate),
                Q8DecodeTask::columns(quantized, decode_weight(&block.mlp.up_proj), n, up),
            ]);
            crate::simd::silu_mul_quantize_q8_0_decode_into(gate, up, gated, quantized);
            matmul_q8_0_decode_tasks(&mut [Q8DecodeTask::columns(
                quantized,
                decode_weight(&block.mlp.down_proj),
                n,
                projected,
            )]);
            for (index, hook) in hooks.iter_mut().enumerate() {
                let mut output =
                    SliceActivation::new(1, embed_dim, row(projected, index, embed_dim));
                hook.after_mlp(layer, &mut output)?;
            }
            crate::simd::add_assign(x, projected);
            for (index, hook) in hooks.iter_mut().enumerate() {
                let mut hidden = SliceActivation::new(1, embed_dim, row(x, index, embed_dim));
                hook.after_layer(layer, &mut hidden)?;
            }
        }
        for sequence in sequences.iter_mut() {
            sequence.cache.advance_cursor();
        }

        for index in 0..n {
            crate::simd::rms_norm_q8_decode_into(
                &x[index * embed_dim..(index + 1) * embed_dim],
                self.norm.data(),
                self.config.norm_eps,
                row(norm, index, embed_dim),
            );
        }
        for (index, hook) in hooks.iter_mut().enumerate() {
            let mut hidden = SliceActivation::new(1, embed_dim, row(norm, index, embed_dim));
            hook.before_logits(&mut hidden)?;
        }
        let head_weight = self
            .head
            .q8_weight_without_bias()
            .expect("batched decode eligibility checked");
        let vocab = head_weight.out_features();
        logits.resize(n * vocab, 0.0);
        let head = match self.head.interleaved.as_deref() {
            Some(interleaved) => crate::simd::Q8DecodeWeight::Interleaved(interleaved),
            None => crate::simd::Q8DecodeWeight::Rows(head_weight),
        };
        crate::simd::quantize_q8_0_decode_into(norm, quantized);
        matmul_q8_0_decode_tasks(&mut [Q8DecodeTask::columns(quantized, head, n, logits)]);
        for (index, (sequence, hook)) in sequences.iter_mut().zip(hooks.iter_mut()).enumerate() {
            sequence
                .logits
                .copy_from_slice(&logits[index * vocab..(index + 1) * vocab]);
            let mut output = SliceActivation::new(1, vocab, sequence.logits);
            hook.after_logits(&mut output)?;
        }
        Ok(())
    }
}

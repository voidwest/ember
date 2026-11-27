use super::*;

/// llama's multi-head self-attention with rotary position embeddings and gqa.
///
/// unlike gpt-2's combined qkv projection, llama uses three separate
/// linear layers (q_proj, k_proj, v_proj) with optional bias terms; qwen2/
/// qwen2.5 gguFs also carry `blk.{i}.attn_q.bias` (+ k/v) which are loaded
/// by `take_llama_linear_with_bias`.
/// rotary position embeddings are applied to q and k before attention.
/// grouped query attention (gqa) repeats k/v heads when `n_kv_heads < n_heads`.
///
/// gguf tensor names: `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`,
/// `blk.{i}.attn_v.weight`, `blk.{i}.attn_output.weight`.
///
/// reference material:
///   - llama paper (touvron et al. 2023)
///   - gqa paper (ainslie et al. 2023)
///   - rope paper (su et al. 2021)
///   - llama.cpp's attention in `llama-arch.cpp` - the gold standard
///     for a working reference that handles all the edge cases
///   - huggingface `LlamaAttention` for the pure-python reference
pub struct LlamaAttention<B: Backend> {
    /// query projection (no bias)
    pub(crate) q_proj: Linear<B>,
    /// key projection (no bias)
    pub(crate) k_proj: Linear<B>,
    /// value projection (no bias)
    pub(crate) v_proj: Linear<B>,
    /// attention output projection (no bias)
    pub(crate) o_proj: Linear<B>,
    /// number of query heads
    n_heads: usize,
    /// number of kv heads (< n_heads when using gqa)
    n_kv_heads: usize,
    /// dimension per head
    head_dim: usize,
    /// RoPE pairing convention used by this architecture.
    pub(super) rope_layout: RopeLayout,
    /// Q/K RMSNorm placement relative to RoPE.
    pub(super) qk_norm_order: QkNormOrder,
    /// precomputed rope cos table, shape [max_seq_len, head_dim]
    rope_cos: Arc<B::Tensor>,
    /// precomputed rope sin table, shape [max_seq_len, head_dim]
    rope_sin: Arc<B::Tensor>,
    /// optional qk normalization weight (qwen3): applied to q after rope, shape [head_dim]
    pub(crate) q_norm: Option<B::Tensor>,
    /// optional qk normalization weight (qwen3): applied to k after rope, shape [head_dim]
    pub(crate) k_norm: Option<B::Tensor>,
    /// RMSNorm epsilon for the Q/K norms: the model's parsed `norm_eps`
    /// when loaded from GGUF (see [`LlamaAttention::with_qk_norm_eps`]).
    pub(super) qk_norm_eps: f32,
}

impl<B: Backend> LlamaAttention<B> {
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per field of the record that it builds or compares"
    )]
    pub fn new(
        q_proj: Linear<B>,
        k_proj: Linear<B>,
        v_proj: Linear<B>,
        o_proj: Linear<B>,
        rope_cos: B::Tensor,
        rope_sin: B::Tensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_layout: RopeLayout,
        qk_norm_order: QkNormOrder,
        q_norm: Option<B::Tensor>,
        k_norm: Option<B::Tensor>,
    ) -> Self {
        Self::new_shared(
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            Arc::new(rope_cos),
            Arc::new(rope_sin),
            n_heads,
            n_kv_heads,
            head_dim,
            rope_layout,
            qk_norm_order,
            q_norm,
            k_norm,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per field of the record that it builds or compares"
    )]
    pub(super) fn new_shared(
        q_proj: Linear<B>,
        k_proj: Linear<B>,
        v_proj: Linear<B>,
        o_proj: Linear<B>,
        rope_cos: Arc<B::Tensor>,
        rope_sin: Arc<B::Tensor>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_layout: RopeLayout,
        qk_norm_order: QkNormOrder,
        q_norm: Option<B::Tensor>,
        k_norm: Option<B::Tensor>,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope_cos,
            rope_sin,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_layout,
            qk_norm_order,
            q_norm,
            k_norm,
            qk_norm_eps: DEFAULT_QK_NORM_EPS,
        }
    }

    /// Use `eps` for the Q/K RMSNorms. GGUF loading passes the model's
    /// parsed `attention.layer_norm_rms_epsilon`; direct constructors keep
    /// the historical [`DEFAULT_QK_NORM_EPS`] (Qwen3's value).
    pub fn with_qk_norm_eps(mut self, eps: f32) -> Self {
        self.qk_norm_eps = eps;
        self
    }
}

/// Q/K RMSNorm epsilon used when no model epsilon is supplied (Qwen3's
/// `attention.layer_norm_rms_epsilon`).
pub const DEFAULT_QK_NORM_EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
pub(super) struct RopeQkNormSpec {
    pub(super) qk_norm_eps: f32,
    pub(super) start_pos: usize,
    pub(super) n_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_layout: RopeLayout,
    pub(super) qk_norm_order: QkNormOrder,
}

fn apply_headwise_rms_norm(
    data: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    norm_data: &[f32],
    eps: f32,
) {
    let width = n_heads * head_dim;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * width + h * head_dim;
            let row = &mut data[base..base + head_dim];
            let sq_sum = crate::simd::sum_squares(row);
            let rstd = (sq_sum / head_dim as f32 + eps).sqrt().recip();
            for d in 0..head_dim {
                row[d] = row[d] * rstd * norm_data[d];
            }
        }
    }
}

pub(super) fn apply_rope_and_qk_norm<B: Backend>(
    backend: &B,
    mut x: B::Tensor,
    rope_cos: &B::Tensor,
    rope_sin: &B::Tensor,
    spec: RopeQkNormSpec,
    norm: Option<&B::Tensor>,
    block_boundaries: Option<&[usize]>,
) -> Result<B::Tensor, B::Error> {
    apply_rope_and_qk_norm_in_place(
        backend,
        &mut x,
        rope_cos,
        rope_sin,
        spec,
        norm,
        block_boundaries,
    )?;
    Ok(x)
}

/// RoPE plus optional Q/K RMSNorm over a `[seq_len, n_heads * head_dim]`
/// tensor, mutating its storage in place (no reallocation on the CPU
/// backend).
fn apply_rope_and_qk_norm_in_place<B: Backend>(
    backend: &B,
    x: &mut B::Tensor,
    rope_cos: &B::Tensor,
    rope_sin: &B::Tensor,
    spec: RopeQkNormSpec,
    norm: Option<&B::Tensor>,
    block_boundaries: Option<&[usize]>,
) -> Result<(), B::Error> {
    let seq_len = backend.shape(x)[0];
    let cos_data = backend.data(rope_cos);
    let sin_data = backend.data(rope_sin);
    let norm_data = norm.map(|norm| backend.data(norm));
    backend.update_in_place(x, |data| {
        rope_and_qk_norm_rows(
            data,
            seq_len,
            cos_data,
            sin_data,
            spec,
            norm_data,
            block_boundaries,
        );
    })
}

fn rope_and_qk_norm_rows(
    data: &mut [f32],
    seq_len: usize,
    cos_data: &[f32],
    sin_data: &[f32],
    spec: RopeQkNormSpec,
    norm_data: Option<&[f32]>,
    block_boundaries: Option<&[usize]>,
) {
    let width = spec.n_heads * spec.head_dim;
    let half = spec.head_dim / 2;

    if spec.qk_norm_order == QkNormOrder::BeforeRope
        && let Some(norm_data) = norm_data
    {
        apply_headwise_rms_norm(
            data,
            seq_len,
            spec.n_heads,
            spec.head_dim,
            norm_data,
            spec.qk_norm_eps,
        );
    }

    let mut block_start = 0;
    let mut boundary_index = 0;
    for s in 0..seq_len {
        if let Some(boundaries) = block_boundaries {
            while boundary_index < boundaries.len() && boundaries[boundary_index] <= s {
                block_start = boundaries[boundary_index];
                boundary_index += 1;
            }
        }
        let pos = spec.start_pos + s - block_start;
        let cos_row = &cos_data[pos * half..(pos + 1) * half];
        let sin_row = &sin_data[pos * half..(pos + 1) * half];
        let row = &mut data[s * width..(s + 1) * width];
        // Both kernels are bit-identical to the scalar
        // `x0.mul_add(c, -(x1 * s))` / `x0.mul_add(s, x1 * c)` rotation.
        match spec.rope_layout {
            RopeLayout::AdjacentPair => {
                crate::simd::rope_adjacent_pair(row, spec.n_heads, spec.head_dim, cos_row, sin_row)
            }
            RopeLayout::SplitHalf => {
                crate::simd::rope_split_half(row, spec.n_heads, spec.head_dim, cos_row, sin_row)
            }
        }
    }

    if spec.qk_norm_order == QkNormOrder::AfterRope
        && let Some(norm_data) = norm_data
    {
        apply_headwise_rms_norm(
            data,
            seq_len,
            spec.n_heads,
            spec.head_dim,
            norm_data,
            spec.qk_norm_eps,
        );
    }
}

impl LlamaAttention<CpuBackend> {
    /// Decode-step RoPE + optional QK norm, shared by the eager and planned
    /// decode paths. `pub(crate)` for `src/planned_decode.rs`.
    pub(crate) fn apply_decode_rope_and_qk_norm(
        &self,
        data: &mut [f32],
        n_heads: usize,
        position: usize,
        norm: Option<&CpuTensor>,
    ) {
        let width = n_heads * self.head_dim;
        debug_assert_eq!(data.len(), width);

        if self.qk_norm_order == QkNormOrder::BeforeRope
            && let Some(norm) = norm
        {
            apply_headwise_rms_norm(
                data,
                1,
                n_heads,
                self.head_dim,
                norm.data(),
                self.qk_norm_eps,
            );
        }

        let half = self.head_dim / 2;
        let table_start = position * half;
        let cos = &self.rope_cos.data()[table_start..table_start + half];
        let sin = &self.rope_sin.data()[table_start..table_start + half];
        match self.rope_layout {
            RopeLayout::SplitHalf => {
                crate::simd::rope_split_half(data, n_heads, self.head_dim, cos, sin);
            }
            RopeLayout::AdjacentPair => {
                crate::simd::rope_adjacent_pair(data, n_heads, self.head_dim, cos, sin);
            }
        }

        if self.qk_norm_order == QkNormOrder::AfterRope
            && let Some(norm) = norm
        {
            apply_headwise_rms_norm(
                data,
                1,
                n_heads,
                self.head_dim,
                norm.data(),
                self.qk_norm_eps,
            );
        }
    }
}

impl<B: Backend> LlamaAttention<B> {
    pub fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let q = self.q_proj.forward(backend, x)?;
        let k = self.k_proj.forward(backend, x)?;
        let v = self.v_proj.forward(backend, x)?;

        let head_dim = self.head_dim;

        let q = apply_rope_and_qk_norm(
            backend,
            q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos: 0,
                n_heads: self.n_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.q_norm.as_ref(),
            None,
        )?;

        let k = apply_rope_and_qk_norm(
            backend,
            k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos: 0,
                n_heads: self.n_kv_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.k_norm.as_ref(),
            None,
        )?;

        let result_tensor = backend.causal_attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim,
                block_boundaries: None,
            },
        )?;

        self.o_proj.forward(backend, &result_tensor)
    }

    /// forward with block-diagonal attention mask for batched independent sequences.
    /// `block_boundaries` marks the start position of each independent block.
    pub fn forward_with_blocks(
        &self,
        backend: &B,
        x: &B::Tensor,
        block_boundaries: &[usize],
    ) -> Result<B::Tensor, B::Error> {
        let q = self.q_proj.forward(backend, x)?;
        let k = self.k_proj.forward(backend, x)?;
        let v = self.v_proj.forward(backend, x)?;

        let head_dim = self.head_dim;

        let q = apply_rope_and_qk_norm(
            backend,
            q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos: 0,
                n_heads: self.n_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.q_norm.as_ref(),
            Some(block_boundaries),
        )?;

        let k = apply_rope_and_qk_norm(
            backend,
            k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos: 0,
                n_heads: self.n_kv_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.k_norm.as_ref(),
            Some(block_boundaries),
        )?;

        let result_tensor = backend.causal_attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim,
                block_boundaries: Some(block_boundaries),
            },
        )?;

        self.o_proj.forward(backend, &result_tensor)
    }

    /// forward with kv cache.
    ///
    /// the cache is allocated for `n_kv_heads` (not `n_heads`).
    /// during decode, cached k/v values are repeated via gqa to
    /// match the number of query heads before computing attention.
    pub fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut scratch = LlamaAttentionScratch::new(backend)?;
        let mut out = backend.zeroes(&[0, 0])?;
        self.forward_with_cache_reusing(
            backend,
            x,
            cache,
            layer,
            start_pos,
            &mut DisabledHooks,
            &mut scratch,
            &mut out,
        )?;
        Ok(out)
    }

    /// [`LlamaAttention::forward_with_cache`] writing the projected
    /// attention output into `out` and every intermediate into `scratch`,
    /// so a prefill reuses one set of buffers across all layers. `hooks`
    /// sees the per-head attention output before the output projection.
    #[allow(
        clippy::too_many_arguments,
        reason = "a forward pass takes its model state, input, cache and positions as separate arguments"
    )]
    pub(super) fn forward_with_cache_reusing<H>(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
        hooks: &mut H,
        scratch: &mut LlamaAttentionScratch<B::Tensor>,
        out: &mut B::Tensor,
    ) -> Result<(), B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        let LlamaAttentionScratch {
            q,
            k,
            v,
            attention: attention_out,
        } = scratch;
        let seq_len_in = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];

        // -- Q projection --
        let _span_q = llama_trace_span!(
            "q_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len_in, embed_dim],
            trace::bytes_matmul_input(seq_len_in, embed_dim, self.q_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len_in, self.n_heads * self.head_dim, embed_dim),
        );
        self.q_proj.forward_reusing(backend, x, q)?;
        let q_dim = backend.shape(q)[1];
        if let Some(s) = _span_q {
            s.end(
                vec![seq_len_in, q_dim],
                trace::bytes_matmul_output(seq_len_in, q_dim),
            );
        }

        // -- K projection --
        let kv_dim = self.n_kv_heads * self.head_dim;
        let _span_k = llama_trace_span!(
            "k_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len_in, embed_dim],
            trace::bytes_matmul_input(seq_len_in, embed_dim, self.k_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len_in, kv_dim, embed_dim),
        );
        self.k_proj.forward_reusing(backend, x, k)?;
        if let Some(s) = _span_k {
            s.end(
                vec![seq_len_in, kv_dim],
                trace::bytes_matmul_output(seq_len_in, kv_dim),
            );
        }

        // -- V projection --
        let _span_v = llama_trace_span!(
            "v_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len_in, embed_dim],
            trace::bytes_matmul_input(seq_len_in, embed_dim, self.v_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len_in, kv_dim, embed_dim),
        );
        self.v_proj.forward_reusing(backend, x, v)?;
        if let Some(s) = _span_v {
            s.end(
                vec![seq_len_in, kv_dim],
                trace::bytes_matmul_output(seq_len_in, kv_dim),
            );
        }

        let seq_len = backend.shape(q)[0];
        let head_dim = self.head_dim;

        // -- RoPE Q --
        let q_width = self.n_heads * head_dim;
        let _span_rope_q = llama_trace_span!(
            "rope_q",
            layer,
            OpKind::RoPE,
            vec![seq_len, q_width],
            trace::bytes_from_shape(&[seq_len, q_width]),
            trace::flops_rope(seq_len, q_width),
        );
        apply_rope_and_qk_norm_in_place(
            backend,
            q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos,
                n_heads: self.n_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.q_norm.as_ref(),
            None,
        )?;
        if let Some(s) = _span_rope_q {
            s.end(
                vec![seq_len, q_width],
                trace::bytes_from_shape(&[seq_len, q_width]),
            );
        }

        // -- RoPE K --
        let k_width = self.n_kv_heads * head_dim;
        let _span_rope_k = llama_trace_span!(
            "rope_k",
            layer,
            OpKind::RoPE,
            vec![seq_len, k_width],
            trace::bytes_from_shape(&[seq_len, k_width]),
            trace::flops_rope(seq_len, k_width),
        );
        apply_rope_and_qk_norm_in_place(
            backend,
            k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
                qk_norm_eps: self.qk_norm_eps,
                start_pos,
                n_heads: self.n_kv_heads,
                head_dim,
                rope_layout: self.rope_layout,
                qk_norm_order: self.qk_norm_order,
            },
            self.k_norm.as_ref(),
            None,
        )?;
        if let Some(s) = _span_rope_k {
            s.end(
                vec![seq_len, k_width],
                trace::bytes_from_shape(&[seq_len, k_width]),
            );
        }

        let k_data = backend.data(k);
        let v_data = backend.data(v);

        // -- KV cache store --
        let _span_kv_store = llama_trace_span!(
            "kv_cache_store",
            layer,
            OpKind::Other,
            vec![seq_len, kv_dim],
            (k_data.len() + v_data.len()) * 4,
            0,
        );
        let cursor = cache.cursor();
        for pos in 0..seq_len {
            let offset = pos * kv_dim;
            cache.append(
                layer,
                cursor + pos,
                &k_data[offset..offset + kv_dim],
                &v_data[offset..offset + kv_dim],
            );
        }
        if let Some(s) = _span_kv_store {
            s.end(vec![], 0);
        }

        // -- Attention score computation --
        let total_seq_len = cache.cursor() + seq_len;
        let max_seq_len = cache.max_seq_len();
        let _span_attn = llama_trace_span!(
            "attention_score",
            layer,
            OpKind::AttentionScore,
            vec![seq_len, q_width],
            trace::bytes_from_shape(&[seq_len, q_width])  // Q bytes
                + (total_seq_len * kv_dim * 4), // cached K bytes
            trace::flops_attention(seq_len, self.n_heads, head_dim, total_seq_len),
        );
        let (cached_k, cached_v, qk_scratch) = cache.get_with_scratch(layer);
        backend.cached_causal_attention_reusing(
            q,
            cached_k,
            cached_v,
            CachedAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim,
                max_seq_len,
                total_seq_len,
            },
            qk_scratch,
            attention_out,
        )?;
        let attn_out_dim = backend.shape(attention_out)[1];
        if let Some(s) = _span_attn {
            s.end(
                vec![seq_len, attn_out_dim],
                trace::bytes_from_shape(&[seq_len, attn_out_dim]),
            );
        }
        hooks.before_o_proj(layer, attention_out)?;

        // -- O projection --
        let _span_o = llama_trace_span!(
            "o_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len, attn_out_dim],
            trace::bytes_matmul_input(seq_len, attn_out_dim, self.o_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len, embed_dim, attn_out_dim),
        );
        self.o_proj.forward_reusing(backend, attention_out, out)?;
        if let Some(s) = _span_o {
            s.end(
                vec![seq_len, embed_dim],
                trace::bytes_matmul_output(seq_len, embed_dim),
            );
        }

        Ok(())
    }
}

/// Attention intermediates reused across the layers of one forward call.
pub(crate) struct LlamaAttentionScratch<T> {
    q: T,
    k: T,
    v: T,
    /// Attention context before the output projection.
    attention: T,
}

impl<T> LlamaAttentionScratch<T> {
    pub(super) fn new<B: Backend<Tensor = T>>(backend: &B) -> Result<Self, B::Error> {
        Ok(Self {
            q: backend.zeroes(&[0, 0])?,
            k: backend.zeroes(&[0, 0])?,
            v: backend.zeroes(&[0, 0])?,
            attention: backend.zeroes(&[0, 0])?,
        })
    }
}

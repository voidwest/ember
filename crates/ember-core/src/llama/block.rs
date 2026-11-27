use super::*;

/// llama's swiglu feed-forward network.
///
/// three linear projections (no bias):
///   `silu(gate_proj(x)) * up_proj(x) -> down_proj`
///
/// this replaces gpt-2's `Mlp` (which uses `c_fc` -> gelu -> `c_proj`).
/// gguf tensor names: `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`,
/// `blk.{i}.ffn_down.weight`.
///
/// reference: llama paper (touvron et al. 2023) section 3.3, the palm paper's
/// swiglu variant (shazeer 2020).
pub struct LlamaMlp<B: Backend> {
    /// gate projection (input -> 8/3 * input for standard llama)
    pub(crate) gate_proj: Linear<B>,
    /// up projection (input -> 8/3 * input, multiplied after gate)
    pub(crate) up_proj: Linear<B>,
    /// down projection (back to embed_dim)
    pub(crate) down_proj: Linear<B>,
}

impl<B: Backend> LlamaMlp<B> {
    pub fn new(gate_proj: Linear<B>, up_proj: Linear<B>, down_proj: Linear<B>) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

impl<B: Backend> Module<B> for LlamaMlp<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let mut gate = backend.zeroes(&[0, 0])?;
        let mut up = backend.zeroes(&[0, 0])?;
        let mut out = backend.zeroes(&[0, 0])?;
        self.forward_reusing(backend, x, &mut gate, &mut up, &mut out)?;
        Ok(out)
    }
}

impl<B: Backend> LlamaMlp<B> {
    /// SwiGLU forward writing into caller-owned buffers: `gate` and `up`
    /// hold the intermediates (SiLU and the gating multiply run in place on
    /// `gate`), `out` receives the down projection. Bit-identical to the
    /// allocating [`Module::forward`], which delegates here.
    fn forward_reusing(
        &self,
        backend: &B,
        x: &B::Tensor,
        gate: &mut B::Tensor,
        up: &mut B::Tensor,
        out: &mut B::Tensor,
    ) -> Result<(), B::Error> {
        use crate::trace::{self, OpKind};
        use std::time::Instant;

        let seq_len = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];
        let tracing = trace::is_tracing();

        // -- gate projection --
        let t0 = tracing.then(Instant::now);
        self.gate_proj.forward_reusing(backend, x, gate)?;
        let inter_dim = backend.shape(gate)[1];
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(gate)))
            } else {
                None
            };
            trace::record(
                "gate_proj",
                trace::current_layer(),
                OpKind::MatMulQ8_0,
                vec![seq_len, embed_dim],
                trace::bytes_matmul_input(seq_len, embed_dim, self.gate_proj.weight_bytes(backend)),
                vec![seq_len, inter_dim],
                trace::bytes_matmul_output(seq_len, inter_dim),
                trace::flops_matmul(seq_len, inter_dim, embed_dim),
                t0.elapsed().as_nanos() as u64,
                vals,
            );
        }

        // -- silu --
        let t0 = tracing.then(Instant::now);
        backend.silu_in_place(gate)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(gate)))
            } else {
                None
            };
            trace::record(
                "silu",
                trace::current_layer(),
                OpKind::Silu,
                vec![seq_len, inter_dim],
                trace::bytes_from_shape(&[seq_len, inter_dim]),
                vec![seq_len, inter_dim],
                trace::bytes_from_shape(&[seq_len, inter_dim]),
                trace::flops_silu(seq_len * inter_dim),
                t0.elapsed().as_nanos() as u64,
                vals,
            );
        }

        // -- up projection --
        let t0 = tracing.then(Instant::now);
        self.up_proj.forward_reusing(backend, x, up)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(up)))
            } else {
                None
            };
            trace::record(
                "up_proj",
                trace::current_layer(),
                OpKind::MatMulQ8_0,
                vec![seq_len, embed_dim],
                trace::bytes_matmul_input(seq_len, embed_dim, self.up_proj.weight_bytes(backend)),
                vec![seq_len, inter_dim],
                trace::bytes_matmul_output(seq_len, inter_dim),
                trace::flops_matmul(seq_len, inter_dim, embed_dim),
                t0.elapsed().as_nanos() as u64,
                vals,
            );
        }

        // -- elemul (gating) --
        let t0 = tracing.then(Instant::now);
        backend.elemul_in_place(gate, up)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(gate)))
            } else {
                None
            };
            trace::record(
                "elemul",
                trace::current_layer(),
                OpKind::Elemul,
                vec![seq_len, inter_dim],
                trace::bytes_from_shape(&[seq_len, inter_dim]) * 2,
                vec![seq_len, inter_dim],
                trace::bytes_from_shape(&[seq_len, inter_dim]),
                trace::flops_elemul(seq_len * inter_dim),
                t0.elapsed().as_nanos() as u64,
                vals,
            );
        }

        // -- down projection --
        let t0 = tracing.then(Instant::now);
        self.down_proj.forward_reusing(backend, gate, out)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(out)))
            } else {
                None
            };
            trace::record(
                "down_proj",
                trace::current_layer(),
                OpKind::MatMulQ8_0,
                vec![seq_len, inter_dim],
                trace::bytes_matmul_input(seq_len, inter_dim, self.down_proj.weight_bytes(backend)),
                vec![seq_len, embed_dim],
                trace::bytes_matmul_output(seq_len, embed_dim),
                trace::flops_matmul(seq_len, embed_dim, inter_dim),
                t0.elapsed().as_nanos() as u64,
                vals,
            );
        }

        Ok(())
    }
}

/// a single llama decoder block.
///
/// ```text
/// x -> rms_norm -> self_attention -> residual add
///   -> rms_norm -> swiglu_mlp -> residual add
/// ```
///
/// note the order: pre-norm (rms), then attention/mlp, then add.
/// this is the same pre-norm layout as gpt-2, but gpt-2 uses
/// layer norm (mean+var, bias) while llama uses rms norm
/// (no mean, no bias).
///
/// gguf tensor names:
///   `blk.{i}.attn_norm.weight` -> rms_norm weight for attention
///   `blk.{i}.ffn_norm.weight`  -> rms_norm weight for mlp
///   (no bias tensors - rms norm has no bias parameter)
pub struct LlamaBlock<B: Backend> {
    /// pre-attention rms normalization weight
    ///
    /// `pub(crate)`: read by `src/planned_decode.rs` when resolving planned
    /// ops (plan construction and the interpreter both walk the model).
    pub(crate) input_layernorm: B::Tensor,
    /// multi-head self-attention
    pub(crate) self_attn: LlamaAttention<B>,
    /// pre-mlp rms normalization weight
    pub(crate) post_attention_layernorm: B::Tensor,
    /// swiglu feed-forward network
    pub(crate) mlp: LlamaMlp<B>,
    /// epsilon for rms normalization (from model config)
    pub(super) norm_eps: f32,
}

impl<B: Backend> LlamaBlock<B> {
    pub fn new(
        input_layernorm: B::Tensor,
        self_attn: LlamaAttention<B>,
        post_attention_layernorm: B::Tensor,
        mlp: LlamaMlp<B>,
        norm_eps: f32,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            norm_eps,
        }
    }
}

impl<B: Backend> LlamaBlock<B> {
    pub fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let mut hooks = DisabledHooks;
        let mut scratch = LlamaLayerScratch::new(backend)?;
        let mut x = x.clone();
        self.forward_with_cache_hooked(
            backend,
            &mut x,
            cache,
            layer,
            start_pos,
            &mut hooks,
            &mut scratch,
        )?;
        Ok(x)
    }

    /// One cached decoder block over the residual stream `x`, updated in
    /// place. Every intermediate lives in `scratch`, which the caller keeps
    /// across layers, so a prefill allocates its buffers once instead of
    /// once per layer and operation. Bit-identical to the former
    /// allocate-per-op form: same kernels in the same order, and the
    /// residual adds are the same single IEEE add per element.
    #[allow(
        clippy::too_many_arguments,
        reason = "a forward pass takes its model state, input, cache and positions as separate arguments"
    )]
    pub(super) fn forward_with_cache_hooked<H>(
        &self,
        backend: &B,
        x: &mut B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
        hooks: &mut H,
        scratch: &mut LlamaLayerScratch<B::Tensor>,
    ) -> Result<(), B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        trace::set_current_layer(layer);

        let LlamaLayerScratch {
            normed,
            attention,
            attention_out,
            gate,
            up,
            mlp_out,
        } = scratch;
        let embed_dim = backend.shape(x)[1];
        let seq_len = backend.shape(x)[0];
        let x_bytes = trace::bytes_from_shape(backend.shape(x));
        let norm_bytes = trace::bytes_from_shape(backend.shape(&self.input_layernorm));

        // -- attn RMS norm --
        let _span_attn_norm = llama_trace_span!(
            "attn_rms_norm",
            layer,
            OpKind::RmsNorm,
            vec![seq_len, embed_dim],
            x_bytes + norm_bytes,
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        backend.rms_norm_reusing(x, &self.input_layernorm, self.norm_eps, normed)?;
        if let Some(s) = _span_attn_norm {
            s.end(
                backend.shape(normed).to_vec(),
                trace::bytes_from_shape(backend.shape(normed)),
            );
        }

        // attention (cached) — sub-spans are inside LlamaAttention::forward_with_cache
        self.self_attn.forward_with_cache_reusing(
            backend,
            normed,
            cache,
            layer,
            start_pos,
            hooks,
            attention,
            attention_out,
        )?;
        hooks.after_attention(layer, attention_out)?;

        // -- attn residual add --
        let attn_out_bytes = trace::bytes_from_shape(backend.shape(attention_out));
        let _span_attn_add = llama_trace_span!(
            "attn_residual_add",
            layer,
            OpKind::ResidualAdd,
            vec![seq_len, embed_dim],
            x_bytes + attn_out_bytes,
            trace::flops_residual_add(backend.data(x).len()),
        );
        backend.add_in_place(x, attention_out)?;
        if let Some(s) = _span_attn_add {
            s.end(
                vec![backend.shape(x)[0], backend.shape(x)[1]],
                trace::bytes_from_shape(backend.shape(x)),
            );
        }

        // -- mlp RMS norm --
        let mlp_norm_bytes = trace::bytes_from_shape(backend.shape(&self.post_attention_layernorm));
        let _span_mlp_norm = llama_trace_span!(
            "mlp_rms_norm",
            layer,
            OpKind::RmsNorm,
            vec![backend.shape(x)[0], backend.shape(x)[1]],
            trace::bytes_from_shape(backend.shape(x)) + mlp_norm_bytes,
            trace::flops_rms_norm(backend.shape(x)[0], backend.shape(x)[1]),
        );
        backend.rms_norm_reusing(x, &self.post_attention_layernorm, self.norm_eps, normed)?;
        if let Some(s) = _span_mlp_norm {
            s.end(
                backend.shape(normed).to_vec(),
                trace::bytes_from_shape(backend.shape(normed)),
            );
        }

        // swiglu mlp — sub-spans are inside LlamaMlp::forward_reusing
        self.mlp
            .forward_reusing(backend, normed, gate, up, mlp_out)?;
        hooks.after_mlp(layer, mlp_out)?;

        // -- mlp residual add --
        let mlp_out_bytes = trace::bytes_from_shape(backend.shape(mlp_out));
        let _span_mlp_add = llama_trace_span!(
            "mlp_residual_add",
            layer,
            OpKind::ResidualAdd,
            vec![backend.shape(x)[0], backend.shape(x)[1]],
            trace::bytes_from_shape(backend.shape(x)) + mlp_out_bytes,
            trace::flops_residual_add(backend.data(x).len()),
        );
        backend.add_in_place(x, mlp_out)?;
        if let Some(s) = _span_mlp_add {
            s.end(
                vec![backend.shape(x)[0], backend.shape(x)[1]],
                trace::bytes_from_shape(backend.shape(x)),
            );
        }

        Ok(())
    }
}

/// Decoder-block intermediates reused across the layers of one eager
/// (generic/hooked) forward call. Each buffer grows to the prompt's size on
/// the first layer and is overwritten, never reallocated, by later layers.
pub(crate) struct LlamaLayerScratch<T> {
    normed: T,
    attention: LlamaAttentionScratch<T>,
    /// Attention block output (after `o_proj`), exposed to hooks.
    attention_out: T,
    gate: T,
    up: T,
    /// MLP block output (after `down_proj`), exposed to hooks.
    mlp_out: T,
}

impl<T> LlamaLayerScratch<T> {
    pub(crate) fn new<B: Backend<Tensor = T>>(backend: &B) -> Result<Self, B::Error> {
        Ok(Self {
            normed: backend.zeroes(&[0, 0])?,
            attention: LlamaAttentionScratch::new(backend)?,
            attention_out: backend.zeroes(&[0, 0])?,
            gate: backend.zeroes(&[0, 0])?,
            up: backend.zeroes(&[0, 0])?,
            mlp_out: backend.zeroes(&[0, 0])?,
        })
    }
}

impl<B: Backend> LlamaBlock<B> {
    /// forward with block-diagonal attention mask.
    pub fn forward_with_blocks(
        &self,
        backend: &B,
        x: &B::Tensor,
        block_boundaries: &[usize],
    ) -> Result<B::Tensor, B::Error> {
        let normed = backend.rms_norm(x, &self.input_layernorm, self.norm_eps)?;
        let attn_out = self
            .self_attn
            .forward_with_blocks(backend, &normed, block_boundaries)?;
        let x = backend.add(x, &attn_out)?;
        let normed = backend.rms_norm(&x, &self.post_attention_layernorm, self.norm_eps)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        backend.add(&x, &mlp_out)
    }
}

impl<B: Backend> Module<B> for LlamaBlock<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        // rms_norm -> attention -> residual add
        let normed = backend.rms_norm(x, &self.input_layernorm, self.norm_eps)?;
        let attn_out = self.self_attn.forward(backend, &normed)?;
        let x = backend.add(x, &attn_out)?;

        // rms_norm -> swiglu mlp -> residual add
        let normed = backend.rms_norm(&x, &self.post_attention_layernorm, self.norm_eps)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        backend.add(&x, &mlp_out)
    }
}

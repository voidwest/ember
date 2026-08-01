use crate::artifact::DispatchPath;
use crate::backend::{AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, CpuError, Module};
use crate::experiments::{
    ActiveHooks, DisabledHooks, ExecutionContext, ExperimentRunner, ExperimentalForwardModel,
    LayerHooks, SliceActivation,
};
use crate::model::{pool_layer_activation, ForwardModel, Linear};
use crate::tensor::CpuTensor;
use crate::workspace::Workspace;
use alloc::vec::Vec;
use std::cell::RefCell;
use std::sync::Arc;

const INTERLEAVED_MIN_OUT_FEATURES: usize = 65_536;

thread_local! {
    /// One decode workspace per calling thread. Dimensions are checked before
    /// every use so sequential inference with different models remains safe.
    static LLAMA_DECODE_WORKSPACE: RefCell<Option<Workspace>> = const { RefCell::new(None) };
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    AdjacentPair,
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormOrder {
    BeforeRope,
    AfterRope,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// number of transformer layers
    pub n_layers: usize,
    /// number of query heads
    pub n_heads: usize,
    /// number of key/value heads (gqa: may be < n_heads)
    pub n_kv_heads: usize,
    /// hidden dimension per token
    pub embed_dim: usize,
    /// dimension per attention head (embed_dim / n_heads, often 128)
    pub head_dim: usize,
    /// maximum sequence length the model was trained for
    pub max_seq_len: usize,
    /// base frequency for rotary position embeddings
    /// (10000.0 for llama-2, 500000.0 for llama-3)
    pub rope_theta: f32,
    /// epsilon for rms normalization (typically 1e-5)
    pub norm_eps: f32,
    /// RoPE pairing convention for this architecture.
    pub rope_layout: RopeLayout,
    /// Q/K RMSNorm placement relative to RoPE for architectures that use it.
    pub qk_norm_order: QkNormOrder,
    /// token vocabulary size
    pub vocab_size: usize,
}

impl LlamaConfig {
    /// read config from gguf metadata, supporting multiple architectures.
    ///
    /// detects the architecture from `general.architecture` and uses the
    /// appropriate prefix (`llama.*`, `qwen2.*`).  falls back to `llama.*`
    /// when the architecture key is missing for backward compatibility.
    ///
    /// mapped metadata keys (per-architecture prefix):
    ///
    ///   `{prefix}.block_count`                       -> n_layers (default 32)
    ///   `{prefix}.attention.head_count`              -> n_heads (default 32)
    ///   `{prefix}.attention.head_count_kv`           -> n_kv_heads (default n_heads)
    ///   `{prefix}.embedding_length`                  -> embed_dim (default 4096)
    ///   `{prefix}.context_length`                    -> max_seq_len (default 2048)
    ///   `{prefix}.rope.freq_base`                    -> rope_theta (default 10000.0)
    ///   `{prefix}.attention.layer_norm_rms_epsilon`  -> norm_eps (default 1e-5)
    ///   `{prefix}.vocab_size`                        -> vocab_size (default 32000)
    ///
    /// supported architectures: llama, qwen2 (including qwen2.5)
    pub fn from_gguf_metadata(loader: &crate::loader::GgufLoader) -> Self {
        use crate::loader::GgufValue;

        // detect architecture prefix from gguf metadata.
        // llama models use "llama.*", qwen2.5 uses "qwen2.*", etc.
        // fall back to "llama" for backward compatibility.
        let arch_prefix = match loader.metadata.get("general.architecture") {
            Some(GgufValue::Str(s)) => s.as_str(),
            _ => "llama",
        };
        // normalize: qwen2 covers qwen2.5 and qwen3 (same arch family)
        let prefix = match arch_prefix {
            "qwen2" => "qwen2",
            "qwen3" => "qwen3",
            _ => "llama",
        };

        let (rope_layout, qk_norm_order) = match prefix {
            // Qwen-family GGUFs use the split-half RoPE convention and apply
            // Q/K RMSNorm before RoPE. This was validated against llama.cpp
            // with golden-logit prompt ladders.
            "qwen2" | "qwen3" => (RopeLayout::SplitHalf, QkNormOrder::BeforeRope),
            _ => (RopeLayout::AdjacentPair, QkNormOrder::AfterRope),
        };

        let get_u32 = |key: &str, default: u32| -> u32 {
            // try architecture-specific key first, then fall back to llama
            let arch_key = format!("{}.{}", prefix, key);
            let llama_key = format!("llama.{}", key);
            match (
                loader.metadata.get(&arch_key),
                loader.metadata.get(&llama_key),
            ) {
                (Some(GgufValue::U32(v)), _) => *v,
                (_, Some(GgufValue::U32(v))) => *v,
                _ => default,
            }
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            let arch_key = format!("{}.{}", prefix, key);
            let llama_key = format!("llama.{}", key);
            match (
                loader.metadata.get(&arch_key),
                loader.metadata.get(&llama_key),
            ) {
                (Some(GgufValue::F32(v)), _) => *v,
                (_, Some(GgufValue::F32(v))) => *v,
                _ => default,
            }
        };

        let n_layers = get_u32("block_count", 32) as usize;
        let n_heads = get_u32("attention.head_count", 32) as usize;
        let n_kv_heads = get_u32("attention.head_count_kv", n_heads as u32) as usize;
        let embed_dim = get_u32("embedding_length", 4096) as usize;
        // some architectures (qwen3, deepseek, etc.) specify head_dim explicitly
        // in the gguf metadata. fall back to embed_dim / n_heads when absent.
        let head_dim = get_u32("attention.key_length", (embed_dim / n_heads) as u32) as usize;
        let max_seq_len = get_u32("context_length", 2048) as usize;
        let rope_theta = get_f32("rope.freq_base", 10000.0);
        let norm_eps = get_f32("attention.layer_norm_rms_epsilon", 1e-5);
        let vocab_size = get_u32("vocab_size", 32000) as usize;

        Self {
            n_layers,
            n_heads,
            n_kv_heads,
            embed_dim,
            head_dim,
            max_seq_len,
            rope_theta,
            norm_eps,
            rope_layout,
            qk_norm_order,
            vocab_size,
        }
    }
}

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
#[allow(dead_code)]
pub struct LlamaMlp<B: Backend> {
    /// gate projection (input -> 8/3 * input for standard llama)
    gate_proj: Linear<B>,
    /// up projection (input -> 8/3 * input, multiplied after gate)
    up_proj: Linear<B>,
    /// down projection (back to embed_dim)
    down_proj: Linear<B>,
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
        use crate::trace::{self, OpKind};
        use std::time::Instant;

        let seq_len = backend.shape(x)[0];
        let embed_dim = backend.shape(x)[1];
        let tracing = trace::is_tracing();

        // -- gate projection --
        let t0 = tracing.then(Instant::now);
        let gate = self.gate_proj.forward(backend, x)?;
        let inter_dim = backend.shape(&gate)[1];
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(&gate)))
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
        let gate = backend.silu(&gate)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(&gate)))
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
        let up = self.up_proj.forward(backend, x)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(&up)))
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
        let gated = backend.elemul(&gate, &up)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(&gated)))
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
        let result = self.down_proj.forward(backend, &gated)?;
        if let Some(t0) = t0 {
            let vals = if trace::values_enabled() {
                Some(trace::compute_tensor_values(backend.data(&result)))
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

        Ok(result)
    }
}

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
#[allow(dead_code)]
pub struct LlamaAttention<B: Backend> {
    /// query projection (no bias)
    q_proj: Linear<B>,
    /// key projection (no bias)
    k_proj: Linear<B>,
    /// value projection (no bias)
    v_proj: Linear<B>,
    /// attention output projection (no bias)
    o_proj: Linear<B>,
    /// number of query heads
    n_heads: usize,
    /// number of kv heads (< n_heads when using gqa)
    n_kv_heads: usize,
    /// dimension per head
    head_dim: usize,
    /// RoPE pairing convention used by this architecture.
    rope_layout: RopeLayout,
    /// Q/K RMSNorm placement relative to RoPE.
    qk_norm_order: QkNormOrder,
    /// precomputed rope cos table, shape [max_seq_len, head_dim]
    rope_cos: Arc<B::Tensor>,
    /// precomputed rope sin table, shape [max_seq_len, head_dim]
    rope_sin: Arc<B::Tensor>,
    /// optional qk normalization weight (qwen3): applied to q after rope, shape [head_dim]
    q_norm: Option<B::Tensor>,
    /// optional qk normalization weight (qwen3): applied to k after rope, shape [head_dim]
    k_norm: Option<B::Tensor>,
}

impl<B: Backend> LlamaAttention<B> {
    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    fn new_shared(
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
        }
    }
}

#[derive(Clone, Copy)]
struct RopeQkNormSpec {
    start_pos: usize,
    n_heads: usize,
    head_dim: usize,
    rope_layout: RopeLayout,
    qk_norm_order: QkNormOrder,
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

fn apply_rope_and_qk_norm<B: Backend>(
    backend: &B,
    x: &B::Tensor,
    rope_cos: &B::Tensor,
    rope_sin: &B::Tensor,
    spec: RopeQkNormSpec,
    norm: Option<&B::Tensor>,
    block_boundaries: Option<&[usize]>,
) -> Result<B::Tensor, B::Error> {
    let seq_len = backend.shape(x)[0];
    let width = spec.n_heads * spec.head_dim;
    let half = spec.head_dim / 2;
    let cos_data = backend.data(rope_cos);
    let sin_data = backend.data(rope_sin);
    let mut data = backend.data(x).to_vec();

    if spec.qk_norm_order == QkNormOrder::BeforeRope {
        if let Some(norm) = norm {
            apply_headwise_rms_norm(
                &mut data,
                seq_len,
                spec.n_heads,
                spec.head_dim,
                backend.data(norm),
                1e-6,
            );
        }
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

        for h in 0..spec.n_heads {
            let base = s * width + h * spec.head_dim;

            for d in 0..half {
                let (i0, i1) = match spec.rope_layout {
                    RopeLayout::AdjacentPair => (base + 2 * d, base + 2 * d + 1),
                    RopeLayout::SplitHalf => (base + d, base + d + half),
                };

                let x0 = data[i0];
                let x1 = data[i1];
                let c = cos_row[d];
                let si = sin_row[d];

                data[i0] = x0 * c - x1 * si;
                data[i1] = x0 * si + x1 * c;
            }
        }
    }

    if spec.qk_norm_order == QkNormOrder::AfterRope {
        if let Some(norm) = norm {
            apply_headwise_rms_norm(
                &mut data,
                seq_len,
                spec.n_heads,
                spec.head_dim,
                backend.data(norm),
                1e-6,
            );
        }
    }

    backend.load_from_cpu(data, &[seq_len, width])
}

impl LlamaAttention<CpuBackend> {
    fn apply_decode_rope_and_qk_norm(
        &self,
        data: &mut [f32],
        n_heads: usize,
        position: usize,
        norm: Option<&CpuTensor>,
    ) {
        let width = n_heads * self.head_dim;
        debug_assert_eq!(data.len(), width);

        if self.qk_norm_order == QkNormOrder::BeforeRope {
            if let Some(norm) = norm {
                apply_headwise_rms_norm(data, 1, n_heads, self.head_dim, norm.data(), 1e-6);
            }
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
                for head in 0..n_heads {
                    let head_start = head * self.head_dim;
                    for d in 0..half {
                        let i0 = head_start + 2 * d;
                        let i1 = i0 + 1;
                        let x0 = data[i0];
                        let x1 = data[i1];
                        data[i0] = x0 * cos[d] - x1 * sin[d];
                        data[i1] = x0 * sin[d] + x1 * cos[d];
                    }
                }
            }
        }

        if self.qk_norm_order == QkNormOrder::AfterRope {
            if let Some(norm) = norm {
                apply_headwise_rms_norm(data, 1, n_heads, self.head_dim, norm.data(), 1e-6);
            }
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
            &q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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
            &k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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
            &q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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
            &k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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
        use crate::trace::{self, OpKind};

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
        let q = self.q_proj.forward(backend, x)?;
        let q_dim = backend.shape(&q)[1];
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
        let k = self.k_proj.forward(backend, x)?;
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
        let v = self.v_proj.forward(backend, x)?;
        if let Some(s) = _span_v {
            s.end(
                vec![seq_len_in, kv_dim],
                trace::bytes_matmul_output(seq_len_in, kv_dim),
            );
        }

        let seq_len = backend.shape(&q)[0];
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
        let q = apply_rope_and_qk_norm(
            backend,
            &q,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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
        let k = apply_rope_and_qk_norm(
            backend,
            &k,
            &self.rope_cos,
            &self.rope_sin,
            RopeQkNormSpec {
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

        let k_data = backend.data(&k);
        let v_data = backend.data(&v);

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
        let result = backend.cached_causal_attention_with_scratch(
            &q,
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
        )?;
        let attn_out_dim = backend.shape(&result)[1];
        if let Some(s) = _span_attn {
            s.end(
                vec![seq_len, attn_out_dim],
                trace::bytes_from_shape(&[seq_len, attn_out_dim]),
            );
        }

        // -- O projection --
        let _span_o = llama_trace_span!(
            "o_proj",
            layer,
            OpKind::MatMulQ8_0,
            vec![seq_len, attn_out_dim],
            trace::bytes_matmul_input(seq_len, attn_out_dim, self.o_proj.weight_bytes(backend)),
            trace::flops_matmul(seq_len, embed_dim, attn_out_dim),
        );
        let result = self.o_proj.forward(backend, &result)?;
        if let Some(s) = _span_o {
            s.end(
                vec![seq_len, embed_dim],
                trace::bytes_matmul_output(seq_len, embed_dim),
            );
        }

        Ok(result)
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
#[allow(dead_code)]
pub struct LlamaBlock<B: Backend> {
    /// pre-attention rms normalization weight
    input_layernorm: B::Tensor,
    /// multi-head self-attention
    self_attn: LlamaAttention<B>,
    /// pre-mlp rms normalization weight
    post_attention_layernorm: B::Tensor,
    /// swiglu feed-forward network
    mlp: LlamaMlp<B>,
    /// epsilon for rms normalization (from model config)
    norm_eps: f32,
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
        self.forward_with_cache_hooked(backend, x, cache, layer, start_pos, &mut hooks)
    }

    fn forward_with_cache_hooked<H>(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
        hooks: &mut H,
    ) -> Result<B::Tensor, B::Error>
    where
        H: LayerHooks<B::Tensor, B::Error>,
    {
        use crate::trace::{self, OpKind};

        trace::set_current_layer(layer);

        let embed_dim = backend.shape(x)[1];
        let seq_len = backend.shape(x)[0];
        let x_shape = vec![seq_len, embed_dim];
        let x_bytes = trace::bytes_from_shape(backend.shape(x));
        let norm_bytes = trace::bytes_from_shape(backend.shape(&self.input_layernorm));

        // -- attn RMS norm --
        let _span_attn_norm = llama_trace_span!(
            "attn_rms_norm",
            layer,
            OpKind::RmsNorm,
            x_shape.clone(),
            x_bytes + norm_bytes,
            trace::flops_rms_norm(seq_len, embed_dim),
        );
        let normed = backend.rms_norm(x, &self.input_layernorm, self.norm_eps)?;
        let normed_shape = backend.shape(&normed).to_vec();
        let normed_bytes = trace::bytes_from_shape(backend.shape(&normed));
        if let Some(s) = _span_attn_norm {
            s.end(normed_shape, normed_bytes);
        }

        // attention (cached) — sub-spans are inside LlamaAttention::forward_with_cache
        let mut attn_out = self
            .self_attn
            .forward_with_cache(backend, &normed, cache, layer, start_pos)?;
        hooks.after_attention(layer, &mut attn_out)?;

        // -- attn residual add --
        let attn_out_bytes = trace::bytes_from_shape(backend.shape(&attn_out));
        #[allow(clippy::needless_borrow)]
        let _span_attn_add = llama_trace_span!(
            "attn_residual_add",
            layer,
            OpKind::ResidualAdd,
            x_shape.clone(),
            x_bytes + attn_out_bytes,
            trace::flops_residual_add(backend.data(&x).len()),
        );
        let x = backend.add(x, &attn_out)?;
        if let Some(s) = _span_attn_add {
            s.end(
                vec![backend.shape(&x)[0], backend.shape(&x)[1]],
                trace::bytes_from_shape(backend.shape(&x)),
            );
        }

        // -- mlp RMS norm --
        let mlp_norm_bytes = trace::bytes_from_shape(backend.shape(&self.post_attention_layernorm));
        let _span_mlp_norm = llama_trace_span!(
            "mlp_rms_norm",
            layer,
            OpKind::RmsNorm,
            vec![backend.shape(&x)[0], backend.shape(&x)[1]],
            trace::bytes_from_shape(backend.shape(&x)) + mlp_norm_bytes,
            trace::flops_rms_norm(backend.shape(&x)[0], backend.shape(&x)[1]),
        );
        let normed = backend.rms_norm(&x, &self.post_attention_layernorm, self.norm_eps)?;
        let normed_shape = backend.shape(&normed).to_vec();
        let normed_bytes = trace::bytes_from_shape(backend.shape(&normed));
        if let Some(s) = _span_mlp_norm {
            s.end(normed_shape, normed_bytes);
        }

        // swiglu mlp — sub-spans are inside LlamaMlp::forward
        let mut mlp_out = self.mlp.forward(backend, &normed)?;
        hooks.after_mlp(layer, &mut mlp_out)?;

        // -- mlp residual add --
        let mlp_out_bytes = trace::bytes_from_shape(backend.shape(&mlp_out));
        #[allow(clippy::needless_borrow)]
        let _span_mlp_add = llama_trace_span!(
            "mlp_residual_add",
            layer,
            OpKind::ResidualAdd,
            vec![backend.shape(&x)[0], backend.shape(&x)[1]],
            trace::bytes_from_shape(backend.shape(&x)) + mlp_out_bytes,
            trace::flops_residual_add(backend.data(&x).len()),
        );
        let result = backend.add(&x, &mlp_out)?;
        if let Some(s) = _span_mlp_add {
            s.end(
                vec![backend.shape(&result)[0], backend.shape(&result)[1]],
                trace::bytes_from_shape(backend.shape(&result)),
            );
        }

        Ok(result)
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
    Q8_0(crate::quant::QuantizedWeight),
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
        if let Some(result) = self.forward_decode_fast(backend, token_ids, cache, start_pos) {
            return result;
        }
        Llama::forward_last_logits_with_cache(self, backend, token_ids, cache, start_pos)
    }
    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
    fn embed_dim(&self) -> usize {
        self.config.embed_dim
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
        let fast_eligible = token_ids.len() == 1
            && !crate::trace::is_tracing()
            && self.fast_decode_inter_dim.is_some();
        if !fast_eligible {
            runner.note_dispatch(execution.phase, DispatchPath::Generic);
        }
        let mut hooks = ActiveHooks::new(runner, execution);
        if fast_eligible {
            if let Some(result) =
                self.forward_decode_fast_hooked(backend, token_ids, cache, start_pos, &mut hooks)
            {
                return result;
            }
        }
        self.forward_last_logits_with_cache_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }
}

impl Llama<CpuBackend> {
    fn eligible_fast_decode_inter_dim(&self) -> Option<usize> {
        // The allocation-free path is validated for Llama's adjacent-pair
        // RoPE. Real Qwen3 end-to-end coverage showed decode divergence with
        // split-half RoPE, despite synthetic single-layer parity, so Qwen stays
        // on the generic implementation until exact real-model parity exists.
        if self.config.rope_layout != RopeLayout::AdjacentPair {
            return None;
        }

        let q_dim = self.config.n_heads.checked_mul(self.config.head_dim)?;
        let kv_dim = self.config.n_kv_heads.checked_mul(self.config.head_dim)?;
        let embed_dim = self.config.embed_dim;
        let mut inter_dim = None;

        for block in &self.blocks {
            let q = block.self_attn.q_proj.q8_weight_without_bias()?;
            let k = block.self_attn.k_proj.q8_weight_without_bias()?;
            let v = block.self_attn.v_proj.q8_weight_without_bias()?;
            let o = block.self_attn.o_proj.q8_weight_without_bias()?;
            let gate = block.mlp.gate_proj.q8_weight_without_bias()?;
            let up = block.mlp.up_proj.q8_weight_without_bias()?;
            let down = block.mlp.down_proj.q8_weight_without_bias()?;

            if q.in_features() != embed_dim
                || q.out_features() != q_dim
                || k.in_features() != embed_dim
                || k.out_features() != kv_dim
                || v.in_features() != embed_dim
                || v.out_features() != kv_dim
                || o.in_features() != q_dim
                || o.out_features() != embed_dim
                || gate.in_features() != embed_dim
                || up.in_features() != embed_dim
                || gate.out_features() != up.out_features()
                || down.in_features() != gate.out_features()
                || down.out_features() != embed_dim
            {
                return None;
            }

            match inter_dim {
                Some(expected) if expected != gate.out_features() => return None,
                None => inter_dim = Some(gate.out_features()),
                _ => {}
            }
        }

        let head = self.head.q8_weight_without_bias()?;
        if head.in_features() != embed_dim {
            return None;
        }
        inter_dim
    }

    /// Run the allocation-free Q8_0 path for a single decode token.
    ///
    /// `None` means the model uses a mixed/F32 weight layout or tracing is
    /// active, in which case the generic implementation remains authoritative.
    fn forward_decode_fast(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Option<Result<CpuTensor, CpuError>> {
        let mut hooks = DisabledHooks;
        self.forward_decode_fast_hooked(backend, token_ids, cache, start_pos, &mut hooks)
    }

    fn forward_decode_fast_hooked<H>(
        &self,
        backend: &CpuBackend,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        hooks: &mut H,
    ) -> Option<Result<CpuTensor, CpuError>>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        if token_ids.len() != 1 || crate::trace::is_tracing() {
            return None;
        }
        let inter_dim = self.fast_decode_inter_dim?;
        hooks.note_dispatch(DispatchPath::Fast);
        let embed_dim = self.config.embed_dim;
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        Some(LLAMA_DECODE_WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let needs_resize = workspace.as_ref().is_none_or(|current| {
                current.max_rows() != 1
                    || current.embed_dim() != embed_dim
                    || current.inter_dim() != inter_dim
                    || current.q_dim() != q_dim
                    || current.kv_dim() != kv_dim
            });
            if needs_resize {
                *workspace = Some(Workspace::new(
                    1,
                    embed_dim,
                    inter_dim,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                ));
            }
            self.forward_decode_with_workspace(
                backend,
                token_ids[0],
                cache,
                start_pos,
                workspace.as_mut().expect("decode workspace initialized"),
                hooks,
            )
        }))
    }

    fn forward_decode_with_workspace<H>(
        &self,
        backend: &CpuBackend,
        token_id: u32,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
        workspace: &mut Workspace,
        hooks: &mut H,
    ) -> Result<CpuTensor, CpuError>
    where
        H: for<'a> LayerHooks<SliceActivation<'a>, CpuError>,
    {
        let embed_dim = self.config.embed_dim;
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        let inter_dim = workspace.inter_dim();
        let profile_operators = crate::decode_profile::is_enabled();

        let Workspace {
            norm_out,
            residual_out,
            q_out,
            k_out,
            v_out,
            attn_out,
            gate_out,
            up_out,
            gated_out,
            mlp_out,
            ..
        } = workspace;
        let x = &mut residual_out[..embed_dim];
        match &self.embed_tokens {
            LlamaEmbedding::F32(table) => {
                if token_id as usize >= table.shape()[0] {
                    return Err(CpuError::ShapeMismatch(format!(
                        "embedding token {} out of bounds for vocabulary {}",
                        token_id,
                        table.shape()[0]
                    )));
                }
                let embedding_start = token_id as usize * embed_dim;
                x.copy_from_slice(&table.data()[embedding_start..embedding_start + embed_dim]);
            }
            LlamaEmbedding::Q8_0(table) => {
                if token_id as usize >= table.out_features() {
                    return Err(CpuError::ShapeMismatch(format!(
                        "embedding token {} out of bounds for vocabulary {}",
                        token_id,
                        table.out_features()
                    )));
                }
                table.dequantize_row(token_id as usize, x);
            }
        }

        let norm = &mut norm_out[..embed_dim];
        let q = &mut q_out[..q_dim];
        let k = &mut k_out[..kv_dim];
        let v = &mut v_out[..kv_dim];
        let attention = &mut attn_out[..q_dim];
        let gate = &mut gate_out[..inter_dim];
        let up = &mut up_out[..inter_dim];
        let gated = &mut gated_out[..inter_dim];
        let projected = &mut mlp_out[..embed_dim];

        for (layer, block) in self.blocks.iter().enumerate() {
            {
                let mut hidden = SliceActivation::new(1, embed_dim, x);
                hooks.before_layer(layer, &mut hidden)?;
            }
            crate::simd::rms_norm_into(x, block.input_layernorm.data(), block.norm_eps, norm);

            let q_weight = block
                .self_attn
                .q_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let k_weight = block
                .self_attn
                .k_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let v_weight = block
                .self_attn
                .v_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_q = block.self_attn.q_proj.packed_q8_weight_without_bias();
            let packed_k = block.self_attn.k_proj.packed_q8_weight_without_bias();
            let packed_v = block.self_attn.v_proj.packed_q8_weight_without_bias();
            if let (Some(packed_q), Some(packed_k), Some(packed_v)) = (packed_q, packed_k, packed_v)
            {
                if profile_operators {
                    let elapsed = backend.matmul_q8_0_packed_triple_into_timed(
                        norm, packed_q, packed_k, packed_v, q, k, v,
                    );
                    record_profiled_packed(layer, "q", packed_q, elapsed[0]);
                    record_profiled_packed(layer, "k", packed_k, elapsed[1]);
                    record_profiled_packed(layer, "v", packed_v, elapsed[2]);
                } else {
                    backend.matmul_q8_0_packed_triple_into(
                        norm, packed_q, packed_k, packed_v, q, k, v,
                    );
                }
            } else if profile_operators {
                let elapsed = backend
                    .matmul_q8_0_triple_into_timed(norm, q_weight, k_weight, v_weight, q, k, v);
                record_profiled_q8(layer, "q", q_weight, elapsed[0]);
                record_profiled_q8(layer, "k", k_weight, elapsed[1]);
                record_profiled_q8(layer, "v", v_weight, elapsed[2]);
            } else {
                backend.matmul_q8_0_triple_into(norm, 1, q_weight, k_weight, v_weight, q, k, v);
            }

            block.self_attn.apply_decode_rope_and_qk_norm(
                q,
                self.config.n_heads,
                start_pos,
                block.self_attn.q_norm.as_ref(),
            );
            block.self_attn.apply_decode_rope_and_qk_norm(
                k,
                self.config.n_kv_heads,
                start_pos,
                block.self_attn.k_norm.as_ref(),
            );

            let cursor = cache.cursor();
            cache.append(layer, cursor, k, v);
            let attention_spec = CachedAttentionSpec {
                n_heads: self.config.n_heads,
                n_kv_heads: self.config.n_kv_heads,
                head_dim: self.config.head_dim,
                max_seq_len: cache.max_seq_len(),
                total_seq_len: cursor + 1,
            };
            let (cached_k, cached_v, qk_scratch) = cache.get_with_scratch(layer);
            backend.cached_causal_attention_into(
                q,
                cached_k,
                cached_v,
                attention_spec,
                qk_scratch,
                attention,
            )?;

            let o_weight = block
                .self_attn
                .o_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_o = block.self_attn.o_proj.packed_q8_weight_without_bias();
            if let Some(packed_o) = packed_o {
                if profile_operators {
                    let elapsed =
                        backend.matmul_q8_0_packed_into_timed(attention, packed_o, projected);
                    record_profiled_packed(layer, "o", packed_o, elapsed);
                } else {
                    backend.matmul_q8_0_packed_into(attention, packed_o, projected);
                }
            } else if profile_operators {
                let elapsed = backend.matmul_q8_0_into_timed(attention, o_weight, projected);
                record_profiled_q8(layer, "o", o_weight, elapsed);
            } else {
                backend.matmul_q8_0_into(attention, 1, o_weight, projected);
            }
            {
                let mut attention_output = SliceActivation::new(1, embed_dim, projected);
                hooks.after_attention(layer, &mut attention_output)?;
            }
            crate::simd::add_assign(x, projected);

            crate::simd::rms_norm_into(
                x,
                block.post_attention_layernorm.data(),
                block.norm_eps,
                norm,
            );
            let gate_weight = block
                .mlp
                .gate_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let up_weight = block
                .mlp
                .up_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_gate = block.mlp.gate_proj.packed_q8_weight_without_bias();
            let packed_up = block.mlp.up_proj.packed_q8_weight_without_bias();
            if let (Some(packed_gate), Some(packed_up)) = (packed_gate, packed_up) {
                if profile_operators {
                    let elapsed = backend.matmul_q8_0_packed_pair_into_timed(
                        norm,
                        packed_gate,
                        packed_up,
                        gate,
                        up,
                    );
                    record_profiled_packed(layer, "gate", packed_gate, elapsed[0]);
                    record_profiled_packed(layer, "up", packed_up, elapsed[1]);
                } else {
                    backend.matmul_q8_0_packed_pair_into(norm, packed_gate, packed_up, gate, up);
                }
            } else if profile_operators {
                let elapsed =
                    backend.matmul_q8_0_pair_into_timed(norm, gate_weight, up_weight, gate, up);
                record_profiled_q8(layer, "gate", gate_weight, elapsed[0]);
                record_profiled_q8(layer, "up", up_weight, elapsed[1]);
            } else {
                backend.matmul_q8_0_pair_into(norm, 1, gate_weight, up_weight, gate, up);
            }
            crate::simd::silu_mul_into(gate, up, gated);

            let down_weight = block
                .mlp
                .down_proj
                .q8_weight_without_bias()
                .expect("fast path eligibility checked");
            let packed_down = block.mlp.down_proj.packed_q8_weight_without_bias();
            if let Some(packed_down) = packed_down {
                if profile_operators {
                    let elapsed =
                        backend.matmul_q8_0_packed_into_timed(gated, packed_down, projected);
                    record_profiled_packed(layer, "down", packed_down, elapsed);
                } else {
                    backend.matmul_q8_0_packed_into(gated, packed_down, projected);
                }
            } else if profile_operators {
                let elapsed = backend.matmul_q8_0_into_timed(gated, down_weight, projected);
                record_profiled_q8(layer, "down", down_weight, elapsed);
            } else {
                backend.matmul_q8_0_into(gated, 1, down_weight, projected);
            }
            {
                let mut mlp_output = SliceActivation::new(1, embed_dim, projected);
                hooks.after_mlp(layer, &mut mlp_output)?;
            }
            crate::simd::add_assign(x, projected);
            {
                let mut hidden = SliceActivation::new(1, embed_dim, x);
                hooks.after_layer(layer, &mut hidden)?;
            }
        }
        cache.advance_cursor();

        crate::simd::rms_norm_into(x, self.norm.data(), self.config.norm_eps, norm);
        {
            let mut hidden = SliceActivation::new(1, embed_dim, norm);
            hooks.before_logits(&mut hidden)?;
        }
        let head_weight = self
            .head
            .q8_weight_without_bias()
            .expect("fast path eligibility checked");
        let mut logits = vec![0.0; head_weight.out_features()];
        if let Some(interleaved) = self.head.interleaved.as_ref() {
            if profile_operators {
                let elapsed =
                    backend.matmul_q8_0_interleaved_into_timed(norm, interleaved, &mut logits);
                crate::decode_profile::record(
                    usize::MAX,
                    "lm_head",
                    interleaved.in_features(),
                    interleaved.out_features(),
                    if rayon::current_num_threads() > 1 {
                        crate::decode_profile::DecodeExecutionMode::InterleavedRowParallelRayon
                    } else {
                        crate::decode_profile::DecodeExecutionMode::InterleavedSerial
                    },
                    elapsed,
                );
            } else {
                backend.matmul_q8_0_interleaved_into(norm, interleaved, &mut logits);
            }
        } else {
            if profile_operators {
                let elapsed = backend.matmul_q8_0_into_timed(norm, head_weight, &mut logits);
                record_profiled_q8(usize::MAX, "lm_head", head_weight, elapsed);
            } else {
                backend.matmul_q8_0_into(norm, 1, head_weight, &mut logits);
            }
        }
        {
            let mut output = SliceActivation::new(1, head_weight.out_features(), &mut logits);
            hooks.after_logits(&mut output)?;
        }
        Ok(CpuTensor::from_data(
            vec![1, head_weight.out_features()],
            logits,
        ))
    }

    /// build a llama model from a gguf loader.
    ///
    /// reads metadata keys under the `llama.*` namespace (as written
    /// by llama.cpp's `llama-arch.cpp`) and maps gguf tensor names
    /// from the llama naming convention.
    ///
    /// expected gguf tensor names per layer:
    ///   `blk.{i}.attn_q.weight`       -> q_proj
    ///   `blk.{i}.attn_k.weight`       -> k_proj
    ///   `blk.{i}.attn_v.weight`       -> v_proj
    ///   `blk.{i}.attn_output.weight`  -> o_proj
    ///   `blk.{i}.ffn_gate.weight`     -> gate_proj
    ///   `blk.{i}.ffn_up.weight`       -> up_proj
    ///   `blk.{i}.ffn_down.weight`     -> down_proj
    ///   `blk.{i}.attn_norm.weight`    -> input_layernorm (rms, no bias)
    ///   `blk.{i}.ffn_norm.weight`     -> post_attention_layernorm (rms, no bias)
    ///
    /// global tensors:
    ///   `token_embd.weight`           -> embed_tokens
    ///   `output_norm.weight`          -> final rms norm (no bias)
    ///   `output.weight`               -> lm_head (linear, no bias)
    ///
    /// design note: f32/f16 linear weights are loaded with their gguf logical
    /// shape and transposed when building `Linear`, matching `Gpt2::from_loader`.
    /// q8_0 weights are loaded into `QuantizedWeight` with the reversed
    /// `[out_features, in_features]` shape expected by the quantized matmul path.
    pub fn from_loader(loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        Self::from_loader_with_max_seq_len(loader, None)
    }

    /// build a llama model from a gguf loader, optionally capping runtime
    /// context length and rope table allocation below the GGUF metadata value.
    pub fn from_loader_with_max_seq_len(
        loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, max_seq_len, true)
    }

    /// Build a Llama-family model without consulting the automatic packed
    /// decode environment switch. Lifecycle experiments use this constructor
    /// so packing can occur at an explicit phase boundary in the same binary.
    pub fn from_loader_without_packed_decode(
        loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::from_loader_impl(loader, max_seq_len, false)
    }

    fn from_loader_impl(
        mut loader: crate::loader::GgufLoader,
        max_seq_len: Option<usize>,
        allow_automatic_packing: bool,
    ) -> anyhow::Result<Self> {
        use crate::loader::LoadedTensor;
        use crate::tensor::compute_rope_freqs;

        let mut config = LlamaConfig::from_gguf_metadata(&loader);
        if let Some(max_seq_len) = max_seq_len {
            config.max_seq_len = config.max_seq_len.min(max_seq_len);
        }
        log::debug!("llama config: {:?}", config);
        let n_layers = config.n_layers;
        let packed_decode_enabled = allow_automatic_packing
            && config.rope_layout == RopeLayout::AdjacentPair
            && std::env::var_os("EMBER_LLAMA_PACKED_Q8").is_none_or(|value| value != "0");

        // precompute rope tables once, shared across all attention layers
        let (rope_cos, rope_sin) =
            compute_rope_freqs(config.max_seq_len, config.head_dim, config.rope_theta, None);
        log::debug!(
            "rope_cos shape: {:?}, rope_sin shape: {:?}",
            rope_cos.shape(),
            rope_sin.shape()
        );

        let rope_cos = Arc::new(rope_cos);
        let rope_sin = Arc::new(rope_sin);

        let embed_tokens: LlamaEmbedding<CpuBackend> =
            match loader.take_tensor("token_embd.weight")? {
                LoadedTensor::F32(tensor) => {
                    // GGUF stores the embedding as [embed, vocab] with vocab
                    // rows contiguous, i.e. already row-major [vocab, embed];
                    // only the dims need swapping for the row lookup.
                    let shape = tensor.shape();
                    debug_assert_eq!(shape.len(), 2);
                    LlamaEmbedding::F32(crate::tensor::CpuTensor::from_data(
                        vec![shape[1], shape[0]],
                        tensor.data().to_vec(),
                    ))
                }
                LoadedTensor::Q8_0(weight) => LlamaEmbedding::Q8_0(weight),
            };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // optionally load qk norm weights (qwen3, etc.)
            let qk_q_norm =
                take_optional_llama_norm(&mut loader, &format!("blk.{}.attn_q_norm.weight", i));
            let qk_k_norm =
                take_optional_llama_norm(&mut loader, &format!("blk.{}.attn_k_norm.weight", i));

            let attn = LlamaAttention::new_shared(
                take_llama_linear_with_bias(
                    &mut loader,
                    &format!("blk.{}.attn_q.weight", i),
                    &format!("blk.{}.attn_q.bias", i),
                    packed_decode_enabled,
                )?,
                take_llama_linear_with_bias(
                    &mut loader,
                    &format!("blk.{}.attn_k.weight", i),
                    &format!("blk.{}.attn_k.bias", i),
                    packed_decode_enabled,
                )?,
                take_llama_linear_with_bias(
                    &mut loader,
                    &format!("blk.{}.attn_v.weight", i),
                    &format!("blk.{}.attn_v.bias", i),
                    packed_decode_enabled,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.attn_output.weight", i),
                    packed_decode_enabled,
                )?,
                Arc::clone(&rope_cos),
                Arc::clone(&rope_sin),
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.rope_layout,
                config.qk_norm_order,
                qk_q_norm,
                qk_k_norm,
            );

            let mlp = LlamaMlp::new(
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_gate.weight", i),
                    packed_decode_enabled,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_up.weight", i),
                    packed_decode_enabled,
                )?,
                take_llama_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_down.weight", i),
                    packed_decode_enabled,
                )?,
            );

            blocks.push(LlamaBlock::new(
                loader.take_f32(&format!("blk.{}.attn_norm.weight", i))?,
                attn,
                loader.take_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                mlp,
                config.norm_eps,
            ));
        }

        // lm_head: use output.weight if present, otherwise tie with embed_tokens
        let mut head = match loader.tensors.remove("output.weight") {
            Some(LoadedTensor::F32(tensor)) => Linear::new(gguf_to_row_major_f32(tensor), None),
            Some(LoadedTensor::Q8_0(weight)) => Linear::new_q8_0(weight, None),
            None => match &embed_tokens {
                // Tied embeddings are already laid out as [vocab, embed] in
                // QuantizedWeight, exactly the [out, in] layout the Q8 matmul
                // expects. Reusing the mapping avoids a second ~1 GiB F32
                // embedding copy and keeps decode on the packed integer path.
                LlamaEmbedding::Q8_0(weight) => {
                    Linear::<CpuBackend>::new_q8_0(weight.clone(), None)
                }
                LlamaEmbedding::F32(tensor) => {
                    // The embedding is already reinterpreted as [vocab, embed]
                    // row-major; the linear needs [embed, vocab], so a real
                    // transpose (data reorder) is required — not the raw-GGUF
                    // helper, which would double-transpose.
                    Linear::<CpuBackend>::new(tensor.clone().transpose(), None)
                }
            },
        };
        head.prepare_interleaved(INTERLEAVED_MIN_OUT_FEATURES);

        let mut model = Self {
            embed_tokens,
            blocks,
            norm: loader.take_f32("output_norm.weight")?,
            head,
            config,
            fast_decode_inter_dim: None,
        };
        model.fast_decode_inter_dim = model.eligible_fast_decode_inter_dim();
        log::debug!(
            "llama q8 decode workspace: {}",
            if model.fast_decode_inter_dim.is_some() {
                "enabled"
            } else {
                "unavailable (mixed or unsupported weight shapes)"
            }
        );
        Ok(model)
    }

    /// Construct packed representations for the selected existing projection
    /// group. Packing and eviction are timed separately; when eviction is
    /// requested it happens immediately after each weight is packed, matching
    /// the production path's bounded temporary residency.
    pub fn prepare_packed_decode_selected(
        &mut self,
        selection: LlamaPackedSelection,
        evict_source_pages: bool,
    ) -> anyhow::Result<LlamaPackingStats> {
        if self.config.rope_layout != RopeLayout::AdjacentPair {
            anyhow::bail!(
                "packed lifecycle experiments require adjacent-pair Llama RoPE; \
                 this architecture remains on the generic path"
            );
        }
        if !crate::simd::packed_q8_0_vnni_supported() {
            anyhow::bail!("packed Q8_0 AVX-512 VNNI kernel is unavailable on this CPU");
        }

        let mut stats = LlamaPackingStats::default();
        for block in &mut self.blocks {
            if selection.includes_attention() {
                prepare_experimental_linear(
                    &mut block.self_attn.q_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.k_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.v_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.self_attn.o_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
            if selection.includes_gate_up() {
                prepare_experimental_linear(
                    &mut block.mlp.gate_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
                prepare_experimental_linear(
                    &mut block.mlp.up_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
            if selection.includes_down() {
                prepare_experimental_linear(
                    &mut block.mlp.down_proj,
                    evict_source_pages,
                    &mut stats,
                )?;
            }
        }
        Ok(stats)
    }

    /// Re-issue `MADV_DONTNEED` for selected packed projection sources after a
    /// generic prefill may have faulted their row-contiguous pages back in.
    pub fn reevict_packed_decode_sources(
        &self,
        selection: LlamaPackedSelection,
    ) -> anyhow::Result<LlamaEvictionStats> {
        let mut stats = LlamaEvictionStats::default();
        for block in &self.blocks {
            if selection.includes_attention() {
                reevict_experimental_linear(&block.self_attn.q_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.k_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.v_proj, &mut stats)?;
                reevict_experimental_linear(&block.self_attn.o_proj, &mut stats)?;
            }
            if selection.includes_gate_up() {
                reevict_experimental_linear(&block.mlp.gate_proj, &mut stats)?;
                reevict_experimental_linear(&block.mlp.up_proj, &mut stats)?;
            }
            if selection.includes_down() {
                reevict_experimental_linear(&block.mlp.down_proj, &mut stats)?;
            }
        }
        Ok(stats)
    }

    /// Return the number and byte size of selected packed projections now
    /// owned by the model.
    pub fn packed_decode_summary(&self, selection: LlamaPackedSelection) -> (usize, usize) {
        let mut weights = 0;
        let mut bytes = 0;
        let mut account = |linear: &Linear<CpuBackend>| {
            if linear.has_packed_decode() {
                weights += 1;
                bytes += linear.packed_decode_bytes();
            }
        };
        for block in &self.blocks {
            if selection.includes_attention() {
                account(&block.self_attn.q_proj);
                account(&block.self_attn.k_proj);
                account(&block.self_attn.v_proj);
                account(&block.self_attn.o_proj);
            }
            if selection.includes_gate_up() {
                account(&block.mlp.gate_proj);
                account(&block.mlp.up_proj);
            }
            if selection.includes_down() {
                account(&block.mlp.down_proj);
            }
        }
        (weights, bytes)
    }
}

/// GGUF stores 2D tensors with the first dim contiguous, i.e. the data is
/// row-major over `[out, in]` for a logical `[in, out]` tensor. The f32
/// matmul expects row-major `[in, out]`, so reinterpret and transpose once.
fn gguf_to_row_major_f32(tensor: crate::tensor::CpuTensor) -> crate::tensor::CpuTensor {
    let shape = tensor.shape();
    debug_assert_eq!(shape.len(), 2);
    let reordered =
        crate::tensor::CpuTensor::from_data(vec![shape[1], shape[0]], tensor.data().to_vec());
    reordered.transpose()
}

fn take_llama_linear(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
    prepare_packed: bool,
) -> anyhow::Result<Linear<CpuBackend>> {
    use crate::loader::LoadedTensor;

    let mut linear = match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => Linear::new(gguf_to_row_major_f32(tensor), None),
        LoadedTensor::Q8_0(weight) => Linear::new_q8_0(weight, None),
    };
    if prepare_packed {
        linear.prepare_packed_decode();
    }
    Ok(linear)
}

/// like `take_llama_linear`, but also loads an optional f32 bias tensor.
///
/// qwen2/qwen2.5 attention projections carry `blk.{i}.attn_q.bias` /
/// `attn_k.bias` / `attn_v.bias`; llama and qwen3 GGUFs do not, and pass
/// through with no bias.
fn take_llama_linear_with_bias(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
    bias_name: &str,
    prepare_packed: bool,
) -> anyhow::Result<Linear<CpuBackend>> {
    use crate::loader::LoadedTensor;

    let bias = loader.take_optional_f32(&[bias_name.to_string()]);
    let mut linear = match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => Linear::new(gguf_to_row_major_f32(tensor), bias),
        LoadedTensor::Q8_0(weight) => Linear::new_q8_0(weight, bias),
    };
    if prepare_packed {
        linear.prepare_packed_decode();
    }
    Ok(linear)
}

fn take_optional_llama_norm(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
) -> Option<CpuTensor> {
    match loader.tensors.remove(name) {
        Some(crate::loader::LoadedTensor::F32(tensor)) => Some(tensor),
        _ => None,
    }
}

fn prepare_experimental_linear(
    linear: &mut Linear<CpuBackend>,
    evict_source_pages: bool,
    stats: &mut LlamaPackingStats,
) -> anyhow::Result<()> {
    let packing_start = std::time::Instant::now();
    let packed_bytes = linear.prepare_packed_decode_without_eviction();
    let packing_elapsed = packing_start.elapsed();
    let Some(packed_bytes) = packed_bytes else {
        return Ok(());
    };

    stats.weights_packed += 1;
    stats.packed_bytes += packed_bytes;
    stats.packing_ns = stats
        .packing_ns
        .saturating_add(packing_elapsed.as_nanos() as u64);

    if evict_source_pages && linear.has_mapped_q8_source() {
        stats.eviction_attempts += 1;
        let eviction_start = std::time::Instant::now();
        let evicted = linear.evict_packed_source_pages()?;
        stats.eviction_ns = stats
            .eviction_ns
            .saturating_add(eviction_start.elapsed().as_nanos() as u64);
        stats.eviction_successes += usize::from(evicted);
    }
    Ok(())
}

fn reevict_experimental_linear(
    linear: &Linear<CpuBackend>,
    stats: &mut LlamaEvictionStats,
) -> anyhow::Result<()> {
    if !linear.has_packed_decode() || !linear.has_mapped_q8_source() {
        return Ok(());
    }
    stats.eviction_attempts += 1;
    let eviction_start = std::time::Instant::now();
    let evicted = linear.evict_packed_source_pages()?;
    stats.eviction_ns = stats
        .eviction_ns
        .saturating_add(eviction_start.elapsed().as_nanos() as u64);
    stats.eviction_successes += usize::from(evicted);
    Ok(())
}

#[inline]
fn record_profiled_q8(
    layer: usize,
    operator: &'static str,
    weight: &crate::quant::QuantizedWeight,
    elapsed: std::time::Duration,
) {
    let execution_mode =
        if crate::simd::q8_decode_uses_row_parallel(weight.out_features(), weight.in_features()) {
            crate::decode_profile::DecodeExecutionMode::RowParallelRayon
        } else {
            crate::decode_profile::DecodeExecutionMode::Serial
        };
    crate::decode_profile::record(
        layer,
        operator,
        weight.in_features(),
        weight.out_features(),
        execution_mode,
        elapsed,
    );
}

#[inline]
fn record_profiled_packed(
    layer: usize,
    operator: &'static str,
    weight: &crate::quant::QuantizedWeightVnni,
    elapsed: std::time::Duration,
) {
    crate::decode_profile::record(
        layer,
        operator,
        weight.in_features(),
        weight.out_features(),
        crate::decode_profile::DecodeExecutionMode::PackedRowParallelRayon,
        elapsed,
    );
}

fn llama_embed_tokens<B: Backend>(
    backend: &B,
    table: &LlamaEmbedding<B>,
    token_ids: &[u32],
    embed_dim: usize,
) -> Result<B::Tensor, B::Error> {
    let mut output = backend.zeroes(&[token_ids.len(), embed_dim])?;
    for (row, &token) in token_ids.iter().enumerate() {
        match table {
            LlamaEmbedding::F32(table) => {
                backend.assign_row_from_table(&mut output, row, table, token as usize)?;
            }
            LlamaEmbedding::Q8_0(table) => {
                backend.assign_row_from_q8_0(&mut output, row, table, token as usize)?;
            }
        }
    }
    Ok(output)
}

impl<B: Backend> Llama<B> {
    /// create a kv cache sized for this model's parameters.
    ///
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
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;

        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(backend, &x, cache, layer, start_pos)?;
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
        let mut x = llama_embed_tokens(backend, &self.embed_tokens, token_ids, embed_dim)?;
        if let Some(s) = _span_emb {
            s.end(
                vec![seq_len, embed_dim],
                trace::bytes_from_shape(&[seq_len, embed_dim]),
            );
        }

        for (layer, block) in self.blocks.iter().enumerate() {
            hooks.before_layer(layer, &mut x)?;
            x = block.forward_with_cache_hooked(backend, &x, cache, layer, start_pos, hooks)?;
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
        let mut last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
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
        let mut result = self.head.forward(backend, &last)?;
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
    #[allow(clippy::type_complexity)]
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

    #[allow(clippy::type_complexity)]
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

    #[allow(clippy::type_complexity)]
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
    #[allow(clippy::type_complexity)]
    pub fn forward_pooled_with_blocks(
        &self,
        backend: &B,
        token_ids: &[u32],
        block_boundaries: &[usize],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
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
            debug_assert!(start < end && end <= token_ids.len());
            let row = backend.row_as_2d(&x, end - 1)?;
            backend.assign_row(&mut last_rows, block_index, &row);
        }
        let last_rows = backend.rms_norm(&last_rows, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &last_rows)?;
        Ok((pooled, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::test_support::RecordingExperiment;
    use crate::experiments::{
        ExecutionPhase, ExperimentHook, ModelContext, ModelFamily, TracingState, ZeroLayerOutput,
        ZeroLayerOutputSpec, ZeroLayerOutputStage,
    };
    use crate::loader::{GgufLoader, GgufValue};
    use crate::quant::{QuantizedWeight, Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::collections::HashMap;

    struct ThreadCountingAllocator;

    thread_local! {
        static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for ThreadCountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            TRACK_ALLOCATIONS
                .try_with(|tracking| {
                    if tracking.get() {
                        ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                    }
                })
                .ok();
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            TRACK_ALLOCATIONS
                .try_with(|tracking| {
                    if tracking.get() {
                        ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                    }
                })
                .ok();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            TRACK_ALLOCATIONS
                .try_with(|tracking| {
                    if tracking.get() {
                        ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                    }
                })
                .ok();
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

    fn count_current_thread_allocations<T>(run: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let result = run();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let allocations = ALLOCATION_COUNT.with(Cell::get);
        (result, allocations)
    }

    #[test]
    fn llama_config_honors_full_context_length_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("llama.context_length".to_string(), GgufValue::U32(131_072));
        let loader = GgufLoader {
            metadata,
            tensors: HashMap::new(),
        };

        let config = LlamaConfig::from_gguf_metadata(&loader);

        assert_eq!(config.max_seq_len, 131_072);
    }

    fn test_q8_linear(out_features: usize, in_features: usize, seed: usize) -> Linear<CpuBackend> {
        assert!(in_features.is_multiple_of(Q8_0_BLOCK_SIZE));
        let blocks = out_features * in_features / Q8_0_BLOCK_SIZE;
        let mut data = Vec::with_capacity(blocks * Q8_0_TYPE_SIZE);
        for block in 0..blocks {
            let scale = half::f16::from_f32(0.005 + (block % 7) as f32 * 0.001);
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            for index in 0..Q8_0_BLOCK_SIZE {
                let quant = ((block * 17 + index * 13 + seed) % 31) as i8 - 15;
                data.push(quant as u8);
            }
        }
        Linear::new_q8_0(
            QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap(),
            None,
        )
    }

    fn test_attention_with_rope(
        rope_cos: Arc<CpuTensor>,
        rope_sin: Arc<CpuTensor>,
        seed: usize,
    ) -> LlamaAttention<CpuBackend> {
        LlamaAttention::new_shared(
            test_q8_linear(32, 32, seed),
            test_q8_linear(16, 32, seed + 1),
            test_q8_linear(16, 32, seed + 2),
            test_q8_linear(32, 32, seed + 3),
            rope_cos,
            rope_sin,
            2,
            1,
            16,
            RopeLayout::AdjacentPair,
            QkNormOrder::AfterRope,
            None,
            None,
        )
    }

    #[test]
    fn attention_layers_share_rope_tables() {
        let (rope_cos, rope_sin) = crate::tensor::compute_rope_freqs(8, 16, 10_000.0, None);
        let rope_cos = Arc::new(rope_cos);
        let rope_sin = Arc::new(rope_sin);
        let first = test_attention_with_rope(Arc::clone(&rope_cos), Arc::clone(&rope_sin), 1);
        let second = test_attention_with_rope(Arc::clone(&rope_cos), Arc::clone(&rope_sin), 5);

        assert!(Arc::ptr_eq(&first.rope_cos, &second.rope_cos));
        assert!(Arc::ptr_eq(&first.rope_sin, &second.rope_sin));
    }

    fn test_llama_model() -> Llama<CpuBackend> {
        test_llama_model_with_layers(1)
    }

    fn test_llama_model_with_layers(n_layers: usize) -> Llama<CpuBackend> {
        let embed_dim = 32;
        let head_dim = 16;
        let n_heads = 2;
        let n_kv_heads = 1;
        let inter_dim = 64;
        let vocab_size = 32;
        let max_seq_len = 8;
        let blocks = (0..n_layers)
            .map(|layer| {
                let seed = layer * 8 + 1;
                let (rope_cos, rope_sin) =
                    crate::tensor::compute_rope_freqs(max_seq_len, head_dim, 10_000.0, None);
                let attention = LlamaAttention::new(
                    test_q8_linear(embed_dim, embed_dim, seed),
                    test_q8_linear(head_dim, embed_dim, seed + 1),
                    test_q8_linear(head_dim, embed_dim, seed + 2),
                    test_q8_linear(embed_dim, embed_dim, seed + 3),
                    rope_cos,
                    rope_sin,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    RopeLayout::AdjacentPair,
                    QkNormOrder::AfterRope,
                    None,
                    None,
                );
                let mlp = LlamaMlp::new(
                    test_q8_linear(inter_dim, embed_dim, seed + 4),
                    test_q8_linear(inter_dim, embed_dim, seed + 5),
                    test_q8_linear(embed_dim, inter_dim, seed + 6),
                );
                LlamaBlock::new(
                    CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
                    attention,
                    CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
                    mlp,
                    1e-5,
                )
            })
            .collect();
        let embedding = (0..vocab_size * embed_dim)
            .map(|index| ((index * 19 % 101) as f32 - 50.0) * 0.002)
            .collect();
        Llama {
            embed_tokens: LlamaEmbedding::F32(CpuTensor::from_data(
                vec![vocab_size, embed_dim],
                embedding,
            )),
            blocks,
            norm: CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
            head: test_q8_linear(vocab_size, embed_dim, 8),
            config: LlamaConfig {
                n_layers,
                n_heads,
                n_kv_heads,
                embed_dim,
                head_dim,
                max_seq_len,
                rope_theta: 10_000.0,
                norm_eps: 1e-5,
                rope_layout: RopeLayout::AdjacentPair,
                qk_norm_order: QkNormOrder::AfterRope,
                vocab_size,
            },
            fast_decode_inter_dim: Some(inter_dim),
        }
    }

    fn configure_as_test_qwen(model: &mut Llama<CpuBackend>) {
        model.config.rope_layout = RopeLayout::SplitHalf;
        model.config.qk_norm_order = QkNormOrder::BeforeRope;
        for block in &mut model.blocks {
            block.self_attn.rope_layout = RopeLayout::SplitHalf;
            block.self_attn.qk_norm_order = QkNormOrder::BeforeRope;
            block.self_attn.q_norm = Some(CpuTensor::from_data(
                vec![model.config.head_dim],
                vec![1.0; model.config.head_dim],
            ));
            block.self_attn.k_norm = Some(CpuTensor::from_data(
                vec![model.config.head_dim],
                vec![1.0; model.config.head_dim],
            ));
        }
        model.fast_decode_inter_dim = model.eligible_fast_decode_inter_dim();
    }

    #[test]
    fn fast_q8_decode_matches_generic_path() {
        let model = test_llama_model();
        let backend = CpuBackend;
        let mut generic_cache = model.create_cache(&backend, model.config.max_seq_len);
        let generic =
            Llama::forward_last_logits_with_cache(&model, &backend, &[3], &mut generic_cache, 0)
                .unwrap();
        let mut fast_cache = model.create_cache(&backend, model.config.max_seq_len);
        let fast = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[3],
            &mut fast_cache,
            0,
        )
        .unwrap();

        for (index, (expected, actual)) in generic.data().iter().zip(fast.data()).enumerate() {
            let tolerance = 1e-4 * expected.abs().max(actual.abs()).max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "logit {index}: generic={expected} fast={actual}"
            );
        }
    }

    fn warmed_disabled_decode_allocation_count(n_layers: usize) -> usize {
        let model = test_llama_model_with_layers(n_layers);
        let backend = CpuBackend;
        let mut cache = model.create_cache(&backend, model.config.max_seq_len);
        ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut cache, 0)
            .unwrap();
        let (result, allocations) = count_current_thread_allocations(|| {
            ForwardModel::forward_last_logits_with_cache(&model, &backend, &[5], &mut cache, 1)
        });
        result.unwrap();
        allocations
    }

    #[test]
    fn disabled_hooks_add_no_per_layer_allocations() {
        let one_layer = warmed_disabled_decode_allocation_count(1);
        let four_layers = warmed_disabled_decode_allocation_count(4);

        assert_eq!(
            four_layers, one_layer,
            "disabled hook dispatch must not allocate once per layer"
        );
    }

    #[test]
    fn split_half_rope_remains_on_generic_decode_path() {
        let mut model = test_llama_model();
        configure_as_test_qwen(&mut model);
        assert!(model.fast_decode_inter_dim.is_none());
    }

    #[test]
    fn experiment_hook_order_covers_prefill_and_fast_decode() {
        let model = test_llama_model();
        let backend = CpuBackend;
        let model_context =
            ModelContext::new(ModelFamily::Llama, None, "llama", 1, model.config.embed_dim);
        let (experiment, records) = RecordingExperiment::new();
        let mut runner = ExperimentRunner::new(experiment);
        let mut cache = model.create_cache(&backend, model.config.max_seq_len);

        let prefill = ExecutionContext::new(
            model_context,
            ExecutionPhase::Prefill,
            0,
            2,
            TracingState::Disabled,
        );
        ExperimentalForwardModel::forward_last_logits_with_experiment(
            &model,
            &backend,
            &[3, 5],
            &mut cache,
            0,
            prefill,
            &mut runner,
        )
        .unwrap();

        let decode = ExecutionContext::new(
            model_context,
            ExecutionPhase::Decode,
            2,
            1,
            TracingState::Disabled,
        );
        ExperimentalForwardModel::forward_last_logits_with_experiment(
            &model,
            &backend,
            &[7],
            &mut cache,
            2,
            decode,
            &mut runner,
        )
        .unwrap();

        let records = records.lock().unwrap();
        let expected_per_evaluation = [
            ExperimentHook::BeforeLayer,
            ExperimentHook::AfterAttention,
            ExperimentHook::AfterMlp,
            ExperimentHook::AfterLayer,
            ExperimentHook::BeforeLogits,
            ExperimentHook::AfterLogits,
        ];
        assert_eq!(records.len(), expected_per_evaluation.len() * 2);
        assert_eq!(
            records.iter().map(|record| record.hook).collect::<Vec<_>>(),
            expected_per_evaluation
                .into_iter()
                .chain(expected_per_evaluation)
                .collect::<Vec<_>>()
        );
        for record in &records[..expected_per_evaluation.len()] {
            assert_eq!(record.phase, Some(ExecutionPhase::Prefill));
            assert_eq!(record.sequence_length, Some(2));
        }
        for record in &records[expected_per_evaluation.len()..] {
            assert_eq!(record.phase, Some(ExecutionPhase::Decode));
            assert_eq!(record.sequence_length, Some(3));
        }
        assert_eq!(records[0].shape, Some([2, model.config.embed_dim]));
        assert_eq!(records[4].shape, Some([1, model.config.embed_dim]));
        assert_eq!(records[5].shape, Some([1, model.config.vocab_size]));
        assert_eq!(records[6].shape, Some([1, model.config.embed_dim]));
        for record in records.iter().filter(|record| record.layer_index.is_some()) {
            assert_eq!(record.layer_index, Some(0));
        }
    }

    #[test]
    fn qwen_observation_hooks_preserve_generic_logits_exactly() {
        let mut model = test_llama_model();
        configure_as_test_qwen(&mut model);
        let backend = CpuBackend;
        let mut normal_cache = model.create_cache(&backend, model.config.max_seq_len);
        let normal = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[3],
            &mut normal_cache,
            0,
        )
        .unwrap();

        let model_context =
            ModelContext::new(ModelFamily::Qwen3, None, "qwen3", 1, model.config.embed_dim);
        let execution = ExecutionContext::new(
            model_context,
            ExecutionPhase::Decode,
            0,
            1,
            TracingState::Disabled,
        );
        let (experiment, records) = RecordingExperiment::new();
        let mut runner = ExperimentRunner::new(experiment);
        let mut observed_cache = model.create_cache(&backend, model.config.max_seq_len);
        let observed = ExperimentalForwardModel::forward_last_logits_with_experiment(
            &model,
            &backend,
            &[3],
            &mut observed_cache,
            0,
            execution,
            &mut runner,
        )
        .unwrap();

        assert_eq!(observed, normal);
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 6);
        assert!(records
            .iter()
            .all(|record| record.phase == Some(ExecutionPhase::Decode)));
    }

    #[test]
    fn zero_attention_intervention_changes_fast_decode_logits() {
        let model = test_llama_model();
        let backend = CpuBackend;

        let mut normal_cache = model.create_cache(&backend, model.config.max_seq_len);
        ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[3, 5],
            &mut normal_cache,
            0,
        )
        .unwrap();
        let normal = ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[7],
            &mut normal_cache,
            2,
        )
        .unwrap();

        let mut experiment_cache = model.create_cache(&backend, model.config.max_seq_len);
        ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[3, 5],
            &mut experiment_cache,
            0,
        )
        .unwrap();
        let model_context =
            ModelContext::new(ModelFamily::Llama, None, "llama", 1, model.config.embed_dim);
        let mut runner = ExperimentRunner::new(ZeroLayerOutput::new(ZeroLayerOutputSpec::new(
            0,
            ZeroLayerOutputStage::Attention,
        )));
        runner.on_model_loaded(&model_context).unwrap();
        let execution = ExecutionContext::new(
            model_context,
            ExecutionPhase::Decode,
            2,
            1,
            TracingState::Disabled,
        );
        let intervened = ExperimentalForwardModel::forward_last_logits_with_experiment(
            &model,
            &backend,
            &[7],
            &mut experiment_cache,
            2,
            execution,
            &mut runner,
        )
        .unwrap();

        assert_ne!(intervened, normal);
    }

    #[test]
    fn trace_guard_preserves_semantic_events() {
        let model = test_llama_model();
        let backend = CpuBackend;
        let mut cache = model.create_cache(&backend, model.config.max_seq_len);

        assert!(crate::trace::enable_tracing("prefill", 0));
        let result =
            ForwardModel::forward_last_logits_with_cache(&model, &backend, &[3], &mut cache, 0);
        let report = crate::trace::disable_tracing().expect("trace report");
        result.unwrap();

        let names = report
            .events
            .iter()
            .map(|event| event.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "embedding",
                "attn_rms_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope_q",
                "rope_k",
                "kv_cache_store",
                "attention_score",
                "o_proj",
                "attn_residual_add",
                "mlp_rms_norm",
                "gate_proj",
                "silu",
                "up_proj",
                "elemul",
                "down_proj",
                "mlp_residual_add",
                "final_norm",
                "lm_head",
            ]
        );
    }

    #[test]
    fn hidden_only_pooling_matches_logits_path() {
        let model = test_llama_model();
        let backend = CpuBackend;
        let groups = vec![vec![0, 1]];
        let hidden_only = model
            .forward_pooled_hidden_states(&backend, &[3, 5], &groups)
            .unwrap();
        let (with_logits, _) = model
            .forward_pooled_activations(&backend, &[3, 5], &groups)
            .unwrap();

        assert_eq!(hidden_only, with_logits);
    }

    #[test]
    fn block_pooled_forward_matches_independent_sequences() {
        let model = test_llama_model();
        let backend = CpuBackend;
        let first_tokens = [3, 5];
        let second_tokens = [7, 11, 13];
        let token_ids = [first_tokens.as_slice(), second_tokens.as_slice()].concat();
        let boundaries = [0, first_tokens.len()];
        let groups = vec![vec![0, 1], vec![2, 3, 4]];

        let (batched_pooled, batched_logits) = ForwardModel::forward_pooled_with_blocks(
            &model,
            &backend,
            &token_ids,
            &boundaries,
            &groups,
        )
        .unwrap();

        for (block_index, tokens) in [first_tokens.as_slice(), second_tokens.as_slice()]
            .into_iter()
            .enumerate()
        {
            let local_group = vec![(0..tokens.len()).collect()];
            let (independent_pooled, independent_logits) = model
                .forward_pooled_activations(&backend, tokens, &local_group)
                .unwrap();
            assert_eq!(
                batched_pooled[block_index], independent_pooled[0],
                "pooled activations differ for block {block_index}"
            );

            let vocab_size = model.config.vocab_size;
            let batched_row =
                &batched_logits.data()[block_index * vocab_size..(block_index + 1) * vocab_size];
            let independent_data = independent_logits.data();
            let independent_row =
                &independent_data[independent_data.len() - vocab_size..independent_data.len()];
            assert_eq!(batched_row, independent_row);
        }
    }
}

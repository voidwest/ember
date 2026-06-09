use crate::backend::{AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, Module};
use crate::model::{pool_layer_activation, ForwardModel, Linear};
use crate::tensor::CpuTensor;
use alloc::vec::Vec;

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
        let gate = self.gate_proj.forward(backend, x)?;
        let gate = backend.silu(&gate)?;
        let up = self.up_proj.forward(backend, x)?;
        let gated = backend.elemul(&gate, &up)?;
        self.down_proj.forward(backend, &gated)
    }
}

/// llama's multi-head self-attention with rotary position embeddings and gqa.
///
/// unlike gpt-2's combined qkv projection, llama uses three separate
/// linear layers (q_proj, k_proj, v_proj) with **no bias terms**.
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
    rope_cos: B::Tensor,
    /// precomputed rope sin table, shape [max_seq_len, head_dim]
    rope_sin: B::Tensor,
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

    for s in 0..seq_len {
        let pos = spec.start_pos + s;
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
        )?;

        let result_tensor = backend.causal_attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim,
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
        let q = self.q_proj.forward(backend, x)?;
        let k = self.k_proj.forward(backend, x)?;
        let v = self.v_proj.forward(backend, x)?;

        let seq_len = backend.shape(&q)[0];
        let kv_dim = self.n_kv_heads * self.head_dim;
        let head_dim = self.head_dim;

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
        )?;
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
        )?;

        let k_data = backend.data(&k);
        let v_data = backend.data(&v);

        // store k/v in cache (n_kv_heads per layer)
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

        // compute attention against full cached k/v
        let total_seq_len = cache.cursor() + seq_len;
        let max_seq_len = cache.max_seq_len();
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
        self.o_proj.forward(backend, &result)
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
        // rms_norm -> attention (cached) -> residual add
        let normed = backend.rms_norm(x, &self.input_layernorm, self.norm_eps)?;
        let attn_out = self
            .self_attn
            .forward_with_cache(backend, &normed, cache, layer, start_pos)?;
        let x = backend.add(x, &attn_out)?;

        // rms_norm -> swiglu mlp -> residual add
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
pub struct Llama<B: Backend> {
    /// token embedding table, shape [vocab_size, embed_dim]
    pub embed_tokens: B::Tensor,
    /// transformer decoder blocks
    pub blocks: Vec<LlamaBlock<B>>,
    /// final rms normalization weight
    pub norm: B::Tensor,
    /// lm head: projects hidden states to vocab logits (no bias)
    pub head: Linear<B>,
    /// model configuration
    pub config: LlamaConfig,
}

impl<B: Backend> ForwardModel<B> for Llama<B> {
    fn create_cache(&self, _backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache {
        crate::kv_cache::KVCache::new(
            self.blocks.len(),
            self.config.n_kv_heads,
            self.config.head_dim,
            max_seq_len,
        )
    }
    fn max_seq_len(&self, _backend: &B) -> usize {
        self.config.max_seq_len
    }
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Llama::forward_with_cache(self, backend, token_ids, cache, start_pos)
    }
    fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
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
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Llama::forward_with_activations(self, backend, token_ids)
    }

    fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Llama::forward_pooled_activations(self, backend, token_ids, token_index_groups)
    }
}

impl Llama<CpuBackend> {
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
        use crate::loader::LoadedTensor;
        use crate::tensor::compute_rope_freqs;

        let mut config = LlamaConfig::from_gguf_metadata(&loader);
        if let Some(max_seq_len) = max_seq_len {
            config.max_seq_len = config.max_seq_len.min(max_seq_len);
        }
        log::debug!("llama config: {:?}", config);
        let n_layers = config.n_layers;

        // precompute rope tables once, shared across all attention layers
        let (rope_cos, rope_sin) =
            compute_rope_freqs(config.max_seq_len, config.head_dim, config.rope_theta, None);
        log::debug!(
            "rope_cos shape: {:?}, rope_sin shape: {:?}",
            rope_cos.shape(),
            rope_sin.shape()
        );

        // helper: get a tensor as f32 (for embeddings, norms, etc.)
        let get_f32 = |name: &str| -> anyhow::Result<CpuTensor> {
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(t)) => Ok(t.clone()),
                Some(LoadedTensor::Q8_0(qw)) => Ok(qw.dequantize_all()),
                None => anyhow::bail!("Missing tensor: {}", name),
            }
        };

        // helper: build a linear from a weight tensor (may be f32 or q8_0).
        // llama weights have no bias, so this takes only the weight name.
        let get_linear = |name: &str| -> anyhow::Result<Linear<CpuBackend>> {
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(t)) => Ok(Linear::new(t.clone().transpose(), None)),
                Some(LoadedTensor::Q8_0(qw)) => Ok(Linear::new_q8_0(qw.clone(), None)),
                None => anyhow::bail!("Missing tensor: {}", name),
            }
        };

        let embed_tokens = get_f32("token_embd.weight")?;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // optionally load qk norm weights (qwen3, etc.)
            let qk_q_norm = loader
                .tensors
                .get(&format!("blk.{}.attn_q_norm.weight", i))
                .and_then(|t| match t {
                    LoadedTensor::F32(t) => Some(t.clone()),
                    _ => None,
                });
            let qk_k_norm = loader
                .tensors
                .get(&format!("blk.{}.attn_k_norm.weight", i))
                .and_then(|t| match t {
                    LoadedTensor::F32(t) => Some(t.clone()),
                    _ => None,
                });

            let attn = LlamaAttention::new(
                get_linear(&format!("blk.{}.attn_q.weight", i))?,
                get_linear(&format!("blk.{}.attn_k.weight", i))?,
                get_linear(&format!("blk.{}.attn_v.weight", i))?,
                get_linear(&format!("blk.{}.attn_output.weight", i))?,
                rope_cos.clone(),
                rope_sin.clone(),
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.rope_layout,
                config.qk_norm_order,
                qk_q_norm,
                qk_k_norm,
            );

            let mlp = LlamaMlp::new(
                get_linear(&format!("blk.{}.ffn_gate.weight", i))?,
                get_linear(&format!("blk.{}.ffn_up.weight", i))?,
                get_linear(&format!("blk.{}.ffn_down.weight", i))?,
            );

            blocks.push(LlamaBlock::new(
                get_f32(&format!("blk.{}.attn_norm.weight", i))?,
                attn,
                get_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                mlp,
                config.norm_eps,
            ));
        }

        // lm_head: use output.weight if present, otherwise tie with embed_tokens
        let head = match loader.tensors.get("output.weight") {
            Some(LoadedTensor::F32(t)) => Linear::new(t.clone().transpose(), None),
            Some(LoadedTensor::Q8_0(qw)) => Linear::new_q8_0(qw.clone(), None),
            None => Linear::new(embed_tokens.clone().transpose(), None),
        };

        Ok(Self {
            embed_tokens,
            blocks,
            norm: get_f32("output_norm.weight")?,
            head,
            config,
        })
    }
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
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            backend.assign_row_from_table(&mut x, i, &self.embed_tokens, tok as usize)?;
        }

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
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            backend.assign_row_from_table(&mut x, i, &self.embed_tokens, tok as usize)?;
        }

        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(backend, &x, cache, layer, start_pos)?;
        }
        for _ in 0..seq_len {
            cache.advance_cursor();
        }

        let last = backend.row_as_2d(&x, seq_len - 1)?;
        let last = backend.rms_norm(&last, &self.norm, self.config.norm_eps)?;
        self.head.forward(backend, &last)
    }

    /// forward pass without caching (full sequence).
    pub fn forward(&self, backend: &B, token_ids: &[u32]) -> Result<B::Tensor, B::Error> {
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            backend.assign_row_from_table(&mut x, i, &self.embed_tokens, tok as usize)?;
        }

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
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            backend.assign_row_from_table(&mut x, i, &self.embed_tokens, tok as usize)?;
        }

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
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut pooled = token_index_groups
            .iter()
            .map(|_| vec![0.0f32; self.blocks.len() * embed_dim])
            .collect::<Vec<_>>();

        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            backend.assign_row_from_table(&mut x, i, &self.embed_tokens, tok as usize)?;
        }

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
        let x = backend.rms_norm(&x, &self.norm, self.config.norm_eps)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((pooled, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{GgufLoader, GgufValue};
    use std::collections::HashMap;

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
}

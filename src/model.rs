use crate::backend::{AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, Module};
use crate::quant::QuantizedWeight;
use crate::tensor::CpuTensor;
use alloc::vec::Vec;

/// a model that can run inference with a kv cache.
/// both `Gpt2` and `Llama` implement this trait so the
/// `generate` / `demo_mode` / `interactive_mode` functions
/// in `main.rs` are generic over architecture.
pub trait ForwardModel<B: Backend> {
    fn create_cache(&self, backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache;
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error>;
    /// number of transformer layers (for display/debug).
    fn n_layers(&self) -> usize;
    /// hidden dimension (for display/debug).
    fn embed_dim(&self) -> usize;

    /// run forward pass and collect hidden states after each transformer block.
    ///
    /// returns `(per_layer_activations, final_logits)` where
    /// `per_layer_activations[layer]` is the flattened hidden state after
    /// block `layer`, with shape `[seq_len * embed_dim]`.
    ///
    /// this is the probing entry point - the same model code, but collecting
    /// intermediate representations instead of discarding them.
    #[allow(clippy::type_complexity)]
    fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(alloc::vec::Vec<alloc::vec::Vec<f32>>, B::Tensor), B::Error>;
}

/// the kind of weight backing a `Linear` layer.
///
/// `F32` is the standard path - f32/f16 tensors loaded from gguf and
/// stored as the backend's native tensor type.  `Q8_0` keeps weights in
/// their raw block-compressed form and dequantizes on the fly during
/// matmul, saving ~4x memory.
pub enum WeightKind<B: Backend> {
    /// f32 weight tensor, shape [in_features, out_features]
    F32(B::Tensor),
    /// q8_0 block-compressed weight, never stored as f32.
    /// dequantized column-by-column during `matmul_q8_0`.
    Q8_0(QuantizedWeight),
}

/// a linear (fully-connected) layer: `y = xW + b`.
/// weight must be `[in_features, out_features]`.
pub struct Linear<B: Backend> {
    /// weight matrix, shape [in_features, out_features]
    weight: WeightKind<B>,
    /// optional bias vector, shape [out_features]
    bias: Option<B::Tensor>,
}

impl<B: Backend> Linear<B> {
    /// create a linear layer with an f32 weight tensor.
    pub fn new(weight: B::Tensor, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::F32(weight),
            bias,
        }
    }

    /// create a linear layer with a q8_0 quantized weight.
    /// the weight stays in block-compressed form; `forward()` calls `matmul_q8_0`.
    pub fn new_q8_0(qw: QuantizedWeight, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::Q8_0(qw),
            bias,
        }
    }

    pub fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let mut out = match &self.weight {
            WeightKind::F32(w) => backend.matmul(x, w)?,
            WeightKind::Q8_0(qw) => backend.matmul_q8_0(x, qw)?,
        };
        if let Some(ref b) = self.bias {
            out = backend.add_broadcast(&out, b)?;
        }
        Ok(out)
    }
}

/// gpt-2's two-layer feed-forward network: `c_fc` -> gelu -> `c_proj`.
pub struct Mlp<B: Backend> {
    /// hidden layer (in_features -> 4*in_features in gpt-2)
    c_fc: Linear<B>,
    /// projection layer (4*in_features -> in_features)
    c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(c_fc: Linear<B>, c_proj: Linear<B>) -> Self {
        Self { c_fc, c_proj }
    }
}

impl<B: Backend> Module<B> for Mlp<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let x = self.c_fc.forward(backend, x)?;
        let x = backend.gelu(&x)?;
        self.c_proj.forward(backend, &x)
    }
}

/// causal multi-head self-attention.
///
/// splits a combined qkv projection into query, key, and value,
/// applies scaled dot-product attention with a causal mask
/// (token `i` can only attend to tokens `0..=i`), then projects
/// the output through `c_proj`.
pub struct Attention<B: Backend> {
    /// combined q, k, v projection
    c_attn: Linear<B>,
    /// attention output projection
    c_proj: Linear<B>,
    /// number of attention heads
    n_heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(c_attn: Linear<B>, c_proj: Linear<B>, n_heads: usize) -> Self {
        Self {
            c_attn,
            c_proj,
            n_heads,
        }
    }

    /// forward with kv cache.
    ///
    /// during the **prefill** pass (`seq_len > 1`) the full attention is
    /// computed and k/v projections for every position are stored in the
    /// cache. during **decode** (`seq_len == 1`) only the new token's q
    /// is computed; cached k/v from all prior positions are reused, turning
    /// the O(n^2*d) full-sequence attention into O(n*d) per step.
    ///
    /// the cache uses a `[layer][head][seq_position][head_dim]` layout.
    /// this method appends the current step's k/v to the cache before
    /// computing attention so the causal masking is always correct.
    pub fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
    ) -> Result<B::Tensor, B::Error> {
        let qkv = self.c_attn.forward(backend, x)?;
        let embed_dim = backend.shape(&qkv)[1] / 3;
        let seq_len = backend.shape(x)[0];
        let head_dim = embed_dim / self.n_heads;

        let q = backend.slice_cols(&qkv, 0, embed_dim);
        let k = backend.slice_cols(&qkv, embed_dim, 2 * embed_dim);
        let v = backend.slice_cols(&qkv, 2 * embed_dim, 3 * embed_dim);

        let k_data = backend.data(&k);
        let v_data = backend.data(&v);

        // -- 1. store k/v for the current step(s) into the cache ------
        //      (cursor advances after all layers have stored, in gpt2::forward_with_cache)
        let cursor = cache.cursor();
        for pos in 0..seq_len {
            let offset = pos * embed_dim;
            cache.append(
                layer,
                cursor + pos,
                &k_data[offset..offset + embed_dim],
                &v_data[offset..offset + embed_dim],
            );
        }

        // -- 2. compute attention against the *full* cached k/v -------
        //      (cursor hasn't advanced yet - it advances after all layers
        //      finish, in gpt2::forward_with_cache)
        let total_seq_len = cache.cursor() + seq_len;
        let (cached_k, cached_v) = cache.get(layer);

        let result = backend.cached_causal_attention(
            &q,
            cached_k,
            cached_v,
            CachedAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_heads,
                head_dim,
                max_seq_len: cache.max_seq_len(),
                total_seq_len,
            },
        )?;
        self.c_proj.forward(backend, &result)
    }
}
impl<B: Backend> Module<B> for Attention<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let qkv = self.c_attn.forward(backend, x)?;
        let embed_dim = backend.shape(&qkv)[1] / 3;

        let q = backend.slice_cols(&qkv, 0, embed_dim);
        let k = backend.slice_cols(&qkv, embed_dim, 2 * embed_dim);
        let v = backend.slice_cols(&qkv, 2 * embed_dim, 3 * embed_dim);

        let head_dim = embed_dim / self.n_heads;
        let result_tensor = backend.causal_attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_heads,
                head_dim,
            },
        )?;
        self.c_proj.forward(backend, &result_tensor)
    }
}
/// a single transformer block: layer_norm -> attention -> residual add
/// -> layer_norm -> mlp -> residual add.
pub struct Block<B: Backend> {
    /// pre-attention layer norm
    ln_1: LayerNorm<B>,
    /// multi-head self-attention
    attn: Attention<B>,
    /// pre-mlp layer norm
    ln_2: LayerNorm<B>,
    /// feed-forward network (c_fc -> gelu -> c_proj)
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(ln_1: LayerNorm<B>, attn: Attention<B>, ln_2: LayerNorm<B>, mlp: Mlp<B>) -> Self {
        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }
}

impl<B: Backend> Block<B> {
    /// forward with kv cache. layer index is required for cache lookups.
    pub fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
    ) -> Result<B::Tensor, B::Error> {
        let normed = self.ln_1.forward(backend, x)?;
        let attn_out = self
            .attn
            .forward_with_cache(backend, &normed, cache, layer)?;
        let x = backend.add(x, &attn_out)?;

        let normed = self.ln_2.forward(backend, &x)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        backend.add(&x, &mlp_out)
    }
}

impl<B: Backend> Module<B> for Block<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let normed = self.ln_1.forward(backend, x)?;
        let attn_out = self.attn.forward(backend, &normed)?;
        let x = backend.add(x, &attn_out)?;

        let normed = self.ln_2.forward(backend, &x)?;
        let mlp_out = self.mlp.forward(backend, &normed)?;
        backend.add(&x, &mlp_out)
    }
}

/// gpt-2's pre-norm layer normalization with learned scale and bias.
pub struct LayerNorm<B: Backend> {
    /// learned scale parameter
    weight: B::Tensor,
    /// learned bias parameter
    bias: B::Tensor,
    /// epsilon for numerical stability in division
    eps: f32,
}
impl<B: Backend> LayerNorm<B> {
    pub fn new(weight: B::Tensor, bias: B::Tensor, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
}

impl<B: Backend> Module<B> for LayerNorm<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        backend.layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}

/// the full gpt-2 transformer model.
///
/// `wte` and `wpe` are transposed on load so that `index_select`
/// picks a row directly (original layout is `[vocab, embed]`; we store
/// `[vocab, embed]` and use `transpose()` for matmuls where needed).
pub struct Gpt2<B: Backend> {
    /// word token embeddings, transposed so index_select picks a row directly
    pub wte: B::Tensor,
    /// word position embeddings, transposed
    pub wpe: B::Tensor,
    /// transformer decoder blocks
    pub blocks: Vec<Block<B>>,
    /// final layer norm
    pub ln_f: LayerNorm<B>,
    /// lm head: projects hidden states to vocab logits
    pub head: Linear<B>,
    /// number of attention heads (used to construct the kv cache)
    pub n_heads: usize,
    /// hidden dimension (cached for display access)
    pub embed_dim: usize,
}

impl Gpt2<CpuBackend> {
    /// build a gpt-2 model from a gguf loader.
    ///
    /// expects the following gguf tensor names:
    /// - `token_embd.weight`, `position_embd.weight`
    /// - `blk.{i}.attn_qkv.weight`, `blk.{i}.attn_qkv.bias`
    /// - `blk.{i}.attn_output.weight`, `blk.{i}.attn_output.bias`
    /// - `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_up.bias`
    /// - `blk.{i}.ffn_down.weight`, `blk.{i}.ffn_down.bias`
    /// - `blk.{i}.attn_norm.weight`, `blk.{i}.attn_norm.bias`
    /// - `blk.{i}.ffn_norm.weight`, `blk.{i}.ffn_norm.bias`
    /// - `output_norm.weight`, `output_norm.bias`
    /// - `output.weight`
    ///
    /// metadata keys `gpt2.block_count` and `gpt2.attention.head_count` control
    /// the number of layers and heads (default 12 each if missing).
    pub fn from_loader(loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        use crate::loader::LoadedTensor;

        // helper: get a tensor that must be available as f32.
        // embeddings, norms, and biases go through this path.
        // if a tensor happens to be q8_0 (e.g. a quantized embedding),
        // it is fully dequantized here so index_select still works.
        let get_f32 = |name: &str| -> anyhow::Result<CpuTensor> {
            match loader.tensors.get(name) {
                Some(LoadedTensor::F32(t)) => Ok(t.clone()),
                Some(LoadedTensor::Q8_0(qw)) => Ok(qw.dequantize_all()),
                None => anyhow::bail!("Missing tensor: {}", name),
            }
        };

        // helper: build a Linear from a weight tensor (may be f32 or q8_0)
        // and an optional f32 bias tensor.
        let get_linear =
            |name: &str, bias_name: Option<&str>| -> anyhow::Result<Linear<CpuBackend>> {
                let bias = match bias_name {
                    Some(bname) => Some(get_f32(bname)?),
                    None => None,
                };
                match loader.tensors.get(name) {
                    Some(LoadedTensor::F32(t)) => Ok(Linear::new(t.clone().transpose(), bias)),
                    Some(LoadedTensor::Q8_0(qw)) => Ok(Linear::new_q8_0(qw.clone(), bias)),
                    None => anyhow::bail!("Missing tensor: {}", name),
                }
            };

        // metadata
        let n_layers = match loader.metadata.get("gpt2.block_count") {
            Some(crate::loader::GgufValue::U32(n)) => *n as usize,
            _ => 12,
        };
        let n_heads = match loader.metadata.get("gpt2.attention.head_count") {
            Some(crate::loader::GgufValue::U32(n)) => *n as usize,
            _ => 12,
        };

        // blocks
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn = Attention::new(
                get_linear(
                    &format!("blk.{}.attn_qkv.weight", i),
                    Some(&format!("blk.{}.attn_qkv.bias", i)),
                )?,
                get_linear(
                    &format!("blk.{}.attn_output.weight", i),
                    Some(&format!("blk.{}.attn_output.bias", i)),
                )?,
                n_heads,
            );

            let mlp = Mlp::new(
                get_linear(
                    &format!("blk.{}.ffn_up.weight", i),
                    Some(&format!("blk.{}.ffn_up.bias", i)),
                )?,
                get_linear(
                    &format!("blk.{}.ffn_down.weight", i),
                    Some(&format!("blk.{}.ffn_down.bias", i)),
                )?,
            );

            blocks.push(Block::new(
                LayerNorm::new(
                    get_f32(&format!("blk.{}.attn_norm.weight", i))?,
                    get_f32(&format!("blk.{}.attn_norm.bias", i))?,
                    1e-5,
                ),
                attn,
                LayerNorm::new(
                    get_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                    get_f32(&format!("blk.{}.ffn_norm.bias", i))?,
                    1e-5,
                ),
                mlp,
            ));
        }

        let wte = get_f32("token_embd.weight")?;
        let embed_dim = wte.shape[1];

        Ok(Self {
            wte,
            wpe: get_f32("position_embd.weight")?,
            blocks,
            ln_f: LayerNorm::new(
                get_f32("output_norm.weight")?,
                get_f32("output_norm.bias")?,
                1e-5,
            ),
            head: get_linear("output.weight", None)?,
            n_heads,
            embed_dim,
        })
    }
}

impl<B: Backend> ForwardModel<B> for Gpt2<B> {
    fn create_cache(&self, backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache {
        Gpt2::create_cache(self, backend, max_seq_len)
    }
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Gpt2::forward_with_cache(self, backend, token_ids, cache, start_pos)
    }
    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
    fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Gpt2::forward_with_activations(self, backend, token_ids)
    }
}

impl<B: Backend> Gpt2<B> {
    pub fn new(
        wte: B::Tensor,
        wpe: B::Tensor,
        blocks: Vec<Block<B>>,
        ln_f: LayerNorm<B>,
        head: Linear<B>,
        n_heads: usize,
        embed_dim: usize,
    ) -> Self {
        Self {
            wte,
            wpe,
            blocks,
            ln_f,
            head,
            n_heads,
            embed_dim,
        }
    }

    /// look up token + position embeddings for a batch of token ids.
    ///
    /// `start_pos` is the absolute position offset for the position embeddings
    /// (0 for a new sequence, `prompt_len + step` during incremental decode).
    fn embed_with_offset(
        &self,
        backend: &B,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let seq_len = tokens.len();
        let embed_dim = backend.shape(&self.wte)[1];
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &token_id) in tokens.iter().enumerate() {
            let word_vec = backend.index_select(&self.wte, token_id as usize)?;
            let pos_vec = backend.index_select(&self.wpe, start_pos + i)?;
            let combined = backend.add(&word_vec, &pos_vec)?;
            backend.assign_row(&mut x, i, &combined);
        }
        Ok(x)
    }

    fn embed(&self, backend: &B, tokens: &[u32]) -> Result<B::Tensor, B::Error> {
        self.embed_with_offset(backend, tokens, 0)
    }

    /// create a kv cache sized for this model's parameters.
    pub fn create_cache(&self, backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache {
        let embed_dim = backend.shape(&self.wte)[1];
        let head_dim = embed_dim / self.n_heads;
        crate::kv_cache::KVCache::new(self.blocks.len(), self.n_heads, head_dim, max_seq_len)
    }

    /// forward pass with incremental kv caching.
    ///
    /// `start_pos` is the absolute position offset for position embeddings
    /// (0 during prefill, `prompt_len + step` for decode steps).
    pub fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let seq_len = token_ids.len();
        let mut x = self.embed_with_offset(backend, token_ids, start_pos)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(backend, &x, cache, layer)?;
        }
        // advance the cache cursor by seq_len after all layers have
        // stored their k/v for these positions.
        for _ in 0..seq_len {
            cache.advance_cursor();
        }
        let x = self.ln_f.forward(backend, &x)?;
        self.head.forward(backend, &x)
    }

    pub fn forward(&self, backend: &B, token_ids: &[u32]) -> Result<B::Tensor, B::Error> {
        let mut x = self.embed(backend, token_ids)?;

        for block in &self.blocks {
            x = block.forward(backend, &x)?;
        }
        let x = self.ln_f.forward(backend, &x)?;

        let logits = self.head.forward(backend, &x)?;

        Ok(logits)
    }

    /// forward pass with activation capture after each transformer block.
    ///
    /// returns `(per_layer_hidden_states, final_logits)`.
    /// each hidden state is the flattened sequence state after the block's
    /// residual add, shape `[seq_len * embed_dim]`.
    #[allow(clippy::type_complexity)]
    pub fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let mut x = self.embed(backend, token_ids)?;
        let mut activations = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            x = block.forward(backend, &x)?;
            let data = backend.data(&x);
            activations.push(data.to_vec());
        }
        let x = self.ln_f.forward(backend, &x)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((activations, logits))
    }
}

// -- llama support --------------------------------------------
// Llama-family architectures share the `ForwardModel` interface with GPT-2.
// Demo and interactive CLI modes are still GPT-2-only.
// -------------------------------------------------------------

/// architectural parameters for a llama-family model.
///
/// read from gguf metadata during `Llama::from_loader`.
/// the key names here mirror meta's gguf convention
/// (`llama.block_count`, `llama.attention.head_count`, etc.).
#[derive(Debug)]
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
        // normalize: qwen2 covers qwen2.5 (same arch family)
        let prefix = match arch_prefix {
            "qwen2" => "qwen2",
            _ => "llama",
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
        let head_dim = embed_dim / n_heads;
        let max_seq_len = get_u32("context_length", 2048).min(4096) as usize;
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
/// reference: llama paper (touvron et al. 2023) section 3.3, the PaLM paper's
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
    /// precomputed rope cos table, shape [max_seq_len, head_dim]
    rope_cos: B::Tensor,
    /// precomputed rope sin table, shape [max_seq_len, head_dim]
    rope_sin: B::Tensor,
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
        }
    }
}

impl<B: Backend> Module<B> for LlamaAttention<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        // project q, k, v separately (no bias)
        let q = self.q_proj.forward(backend, x)?;
        let k = self.k_proj.forward(backend, x)?;
        let v = self.v_proj.forward(backend, x)?;

        let seq_len = backend.shape(&q)[0];
        let embed_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let head_dim = self.head_dim;

        // apply RoPE inline (llama.cpp style): rotate pairs (d, d+half) within
        // each head using precomputed cos/sin tables.
        let half = head_dim / 2;
        let cos_data = backend.data(&self.rope_cos);
        let sin_data = backend.data(&self.rope_sin);

        let q_raw = backend.data(&q);
        let mut q_rope = q_raw.to_vec();
        for s in 0..seq_len {
            let cos_row = &cos_data[s * half..(s + 1) * half];
            let sin_row = &sin_data[s * half..(s + 1) * half];
            for h in 0..self.n_heads {
                let base = s * embed_dim + h * head_dim;
                for d in 0..half {
                    let x = q_rope[base + d];
                    let y = q_rope[base + d + half];
                    q_rope[base + d] = x * cos_row[d] - y * sin_row[d];
                    q_rope[base + d + half] = x * sin_row[d] + y * cos_row[d];
                }
            }
        }
        let q = backend.load_from_cpu(q_rope, &[seq_len, embed_dim])?;

        let k_raw = backend.data(&k);
        let mut k_rope = k_raw.to_vec();
        for s in 0..seq_len {
            let cos_row = &cos_data[s * half..(s + 1) * half];
            let sin_row = &sin_data[s * half..(s + 1) * half];
            for h in 0..self.n_kv_heads {
                let base = s * kv_dim + h * head_dim;
                for d in 0..half {
                    let x = k_rope[base + d];
                    let y = k_rope[base + d + half];
                    k_rope[base + d] = x * cos_row[d] - y * sin_row[d];
                    k_rope[base + d + half] = x * sin_row[d] + y * cos_row[d];
                }
            }
        }
        let k = backend.load_from_cpu(k_rope, &[seq_len, kv_dim])?;

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
}

impl<B: Backend> LlamaAttention<B> {
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
        let embed_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let head_dim = self.head_dim;

        // apply RoPE inline (llama.cpp style): rotate pairs (d, d+half) within each
        // head using the precomputed cos/sin tables. no intermediate tensors.
        let half = head_dim / 2;
        let cos_data = backend.data(&self.rope_cos);
        let sin_data = backend.data(&self.rope_sin);

        let q_raw = backend.data(&q);
        let mut q_rope = q_raw.to_vec();
        for s in 0..seq_len {
            let pos = start_pos + s;
            let cos_row = &cos_data[pos * half..(pos + 1) * half];
            let sin_row = &sin_data[pos * half..(pos + 1) * half];
            for h in 0..self.n_heads {
                let base = s * embed_dim + h * head_dim;
                for d in 0..half {
                    let x = q_rope[base + d];
                    let y = q_rope[base + d + half];
                    q_rope[base + d] = x * cos_row[d] - y * sin_row[d];
                    q_rope[base + d + half] = x * sin_row[d] + y * cos_row[d];
                }
            }
        }
        let q = backend.load_from_cpu(q_rope, &[seq_len, embed_dim])?;

        let k_raw = backend.data(&k);
        let mut k_rope = k_raw.to_vec();
        for s in 0..seq_len {
            let pos = start_pos + s;
            let cos_row = &cos_data[pos * half..(pos + 1) * half];
            let sin_row = &sin_data[pos * half..(pos + 1) * half];
            for h in 0..self.n_kv_heads {
                let base = s * kv_dim + h * head_dim;
                for d in 0..half {
                    let x = k_rope[base + d];
                    let y = k_rope[base + d + half];
                    k_rope[base + d] = x * cos_row[d] - y * sin_row[d];
                    k_rope[base + d + half] = x * sin_row[d] + y * cos_row[d];
                }
            }
        }
        let k = backend.load_from_cpu(k_rope, &[seq_len, kv_dim])?;

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
        let (cached_k, cached_v) = cache.get(layer);
        let result = backend.cached_causal_attention(
            &q,
            cached_k,
            cached_v,
            CachedAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                head_dim,
                max_seq_len: cache.max_seq_len(),
                total_seq_len,
            },
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
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Llama::forward_with_cache(self, backend, token_ids, cache, start_pos)
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
    /// design note: quantized llama models from llama.cpp use the same
    /// column-major storage as quantized gpt-2 models. as with `Gpt2::from_loader`,
    /// quantized linear weights need `.transpose()` after loading.
    /// f32 weights should not be transposed (the loader leaves them
    /// in natural row-major order).
    pub fn from_loader(loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        use crate::loader::LoadedTensor;
        use crate::tensor::compute_rope_freqs;

        let config = LlamaConfig::from_gguf_metadata(&loader);
        log::debug!("llama config: {:?}", config);
        let n_layers = config.n_layers;

        // precompute rope tables once, shared across all attention layers
        let (rope_cos, rope_sin) =
            compute_rope_freqs(config.max_seq_len, config.head_dim, config.rope_theta);
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

        // helper: build a Linear from a weight tensor (may be f32 or q8_0).
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
            let word_vec = backend.index_select(&self.embed_tokens, tok as usize)?;
            backend.assign_row(&mut x, i, &word_vec);
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

    /// forward pass without caching (full sequence).
    pub fn forward(&self, backend: &B, token_ids: &[u32]) -> Result<B::Tensor, B::Error> {
        let seq_len = token_ids.len();
        let embed_dim = self.config.embed_dim;
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &tok) in token_ids.iter().enumerate() {
            let word_vec = backend.index_select(&self.embed_tokens, tok as usize)?;
            backend.assign_row(&mut x, i, &word_vec);
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
            let word_vec = backend.index_select(&self.embed_tokens, tok as usize)?;
            backend.assign_row(&mut x, i, &word_vec);
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
}

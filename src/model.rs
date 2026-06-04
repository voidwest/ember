use crate::backend::{AttentionSpec, Backend, CachedAttentionSpec, CpuBackend, Module};
use crate::quant::QuantizedWeight;
use crate::tensor::CpuTensor;
use alloc::vec::Vec;

pub use crate::llama::{Llama, LlamaConfig};

/// a model that can run inference with a kv cache.
/// gpt-2 and llama-family models implement this trait so the
/// `generate` / `demo_mode` / `interactive_mode` functions
/// in `main.rs` are generic over architecture.
pub trait ForwardModel<B: Backend> {
    fn create_cache(&self, backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache;
    /// maximum sequence length supported by this loaded model.
    fn max_seq_len(&self, backend: &B) -> usize;
    fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error>;
    fn forward_last_logits_with_cache(
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

    /// run forward pass and collect pooled hidden states after each block.
    ///
    /// `token_index_groups` contains one or more token-index sets to pool from
    /// each layer. The returned vector has one flat `[n_layers * embed_dim]`
    /// buffer per group. Implementations can override this to avoid storing
    /// full sequence activations.
    #[allow(clippy::type_complexity)]
    fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[alloc::vec::Vec<usize>],
    ) -> Result<(alloc::vec::Vec<alloc::vec::Vec<f32>>, B::Tensor), B::Error> {
        let (layer_states, logits) = self.forward_with_activations(backend, token_ids)?;
        let embed_dim = self.embed_dim();
        let n_layers = self.n_layers();
        let mut pooled = token_index_groups
            .iter()
            .map(|_| alloc::vec![0.0f32; n_layers * embed_dim])
            .collect::<alloc::vec::Vec<_>>();
        for (li, state) in layer_states.iter().enumerate() {
            for (gi, token_indices) in token_index_groups.iter().enumerate() {
                let offset = li * embed_dim;
                pool_layer_activation(
                    state,
                    token_indices,
                    embed_dim,
                    &mut pooled[gi][offset..offset + embed_dim],
                );
            }
        }
        Ok((pooled, logits))
    }
}

pub fn pool_layer_activation(
    layer_state: &[f32],
    token_indices: &[usize],
    embed_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), embed_dim);
    out.fill(0.0);
    for &token_index in token_indices {
        let row_start = token_index * embed_dim;
        for (j, value) in out.iter_mut().enumerate() {
            *value += layer_state[row_start + j];
        }
    }
    let scale = 1.0 / token_indices.len() as f32;
    for value in out {
        *value *= scale;
    }
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
        let max_seq_len = cache.max_seq_len();
        let (cached_k, cached_v, qk_scratch) = cache.get_with_scratch(layer);

        let result = backend.cached_causal_attention_with_scratch(
            &q,
            cached_k,
            cached_v,
            CachedAttentionSpec {
                n_heads: self.n_heads,
                n_kv_heads: self.n_heads,
                head_dim,
                max_seq_len,
                total_seq_len,
            },
            qk_scratch,
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

        // helper: build a linear from a weight tensor (may be f32 or q8_0)
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
    fn max_seq_len(&self, backend: &B) -> usize {
        backend.shape(&self.wpe)[0]
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
    fn forward_last_logits_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Gpt2::forward_last_logits_with_cache(self, backend, token_ids, cache, start_pos)
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

    fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        Gpt2::forward_pooled_activations(self, backend, token_ids, token_index_groups)
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
            backend.assign_row_sum_from_tables(
                &mut x,
                i,
                &self.wte,
                token_id as usize,
                &self.wpe,
                start_pos + i,
            )?;
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

    pub fn forward_last_logits_with_cache(
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
        for _ in 0..seq_len {
            cache.advance_cursor();
        }

        let last = backend.row_as_2d(&x, seq_len - 1)?;
        let last = self.ln_f.forward(backend, &last)?;
        self.head.forward(backend, &last)
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

    #[allow(clippy::type_complexity)]
    pub fn forward_pooled_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        let n_layers = self.blocks.len();
        let embed_dim = self.embed_dim;
        let mut pooled = token_index_groups
            .iter()
            .map(|_| vec![0.0f32; n_layers * embed_dim])
            .collect::<Vec<_>>();

        let mut x = self.embed(backend, token_ids)?;
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
        let x = self.ln_f.forward(backend, &x)?;
        let logits = self.head.forward(backend, &x)?;
        Ok((pooled, logits))
    }
}

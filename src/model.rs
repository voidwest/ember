use crate::backend::{Backend, CpuBackend, Module};
use alloc::vec::Vec;

/// a linear (fully-connected) layer: `y = xW + b`.
/// weight must be `[in_features, out_features]`.
pub struct Linear<B: Backend> {
    /// weight matrix, shape [in_features, out_features]
    weight: B::Tensor,
    /// optional bias vector, shape [out_features]
    bias: Option<B::Tensor>,
}

impl<B: Backend> Linear<B> {
    pub fn new(weight: B::Tensor, bias: Option<B::Tensor>) -> Self {
        Self { weight, bias }
    }
    pub fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let mut out = backend.matmul(x, &self.weight)?;
        if let Some(ref b) = self.bias {
            out = backend.add_broadcast(&out, b)?;
        }
        Ok(out)
    }
}

/// gpt-2's two-layer feed-forward network: `c_fc` → gelu → `c_proj`.
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
    /// the O(n²·d) full-sequence attention into O(n·d) per step.
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
        let scale = (head_dim as f32).sqrt().recip();

        let q = backend.slice_cols(&qkv, 0, embed_dim);
        let k = backend.slice_cols(&qkv, embed_dim, 2 * embed_dim);
        let v = backend.slice_cols(&qkv, 2 * embed_dim, 3 * embed_dim);

        let q_data = backend.data(&q);
        let k_data = backend.data(&k);
        let v_data = backend.data(&v);

        // ── 1. store k/v for the current step(s) into the cache ──────
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

        // ── 2. compute attention against the *full* cached k/v ───────
        //      (cursor hasn't advanced yet — it advances after all layers
        //      finish, in gpt2::forward_with_cache)
        let total_seq_len = cache.cursor() + seq_len;
        let (cached_k, cached_v) = cache.get(layer);
        let cache_head_stride = cache.max_seq_len() * head_dim;

        let mut attn_buf = vec![0.0f32; seq_len * embed_dim];

        for h in 0..self.n_heads {
            let q_head_offset = h * head_dim;

            for i in 0..seq_len {
                // causal mask: position i (in the current batch) attends to
                // positions 0..=total_seq_len - seq_len + i in the cache.
                let max_j = total_seq_len - seq_len + i;

                let mut qk_row = vec![f32::NEG_INFINITY; total_seq_len];
                let q_idx_abs = i * embed_dim + q_head_offset;

                for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                    let k_cache_abs = h * cache_head_stride + j * head_dim;
                    let dot: f32 = q_data[q_idx_abs..q_idx_abs + head_dim]
                        .iter()
                        .zip(cached_k[k_cache_abs..k_cache_abs + head_dim].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    *slot = dot * scale;
                }

                // softmax
                let max_val = qk_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if max_val == f32::NEG_INFINITY {
                    let uniform = 1.0 / (total_seq_len as f32);
                    for slot in qk_row.iter_mut().take(total_seq_len) {
                        *slot = uniform;
                    }
                } else {
                    let mut sum = 0.0;
                    for s in qk_row.iter_mut() {
                        *s = (*s - max_val).exp();
                        sum += *s;
                    }
                    let inv_sum = sum.recip();
                    for s in qk_row.iter_mut() {
                        *s *= inv_sum;
                    }
                }

                // weighted sum of values
                for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                    if weight == 0.0 {
                        continue;
                    }
                    let v_cache_abs = h * cache_head_stride + j * head_dim;
                    let out_offset = i * embed_dim + q_head_offset;
                    for d in 0..head_dim {
                        attn_buf[out_offset + d] += weight * cached_v[v_cache_abs + d];
                    }
                }
            }
        }

        let result = backend.from_cpu(attn_buf, &[seq_len, embed_dim])?;
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
        let scale = (head_dim as f32).sqrt().recip();
        let x_shape = backend.shape(x);
        let seq_len = x_shape[0];

        let q_data = backend.data(&q);
        let k_data = backend.data(&k);
        let v_data = backend.data(&v);

        let mut attn_buf = vec![0.0; seq_len * embed_dim];

        for h in 0..self.n_heads {
            let q_head_offset = h * head_dim;
            let k_head_offset = h * head_dim;
            let v_head_offset = h * head_dim;

            let mut qk = vec![f32::NEG_INFINITY; seq_len * seq_len];

            for i in 0..seq_len {
                // causal mask: token i can only attend to tokens 0..i (including itself)
                for j in 0..=i {
                    let q_idx = i * embed_dim + q_head_offset;
                    let k_idx = j * embed_dim + k_head_offset;
                    let dot: f32 = q_data[q_idx..q_idx + head_dim]
                        .iter()
                        .zip(k_data[k_idx..k_idx + head_dim].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    qk[i * seq_len + j] = dot * scale;
                }
            }

            let max_per_row: Vec<f32> = qk
                .chunks(seq_len)
                .map(|row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
                .collect();

            let mut row_sums = vec![0.0; seq_len];
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let row = &mut qk[row_start..row_start + seq_len];
                let max = max_per_row[i];
                if max == f32::NEG_INFINITY {
                    let uniform = 1.0 / (seq_len as f32);
                    for s in row.iter_mut() {
                        *s = uniform;
                    }
                    row_sums[i] = 1.0;
                    continue;
                }
                let mut sum = 0.0;
                for s in row.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                row_sums[i] = sum;
            }

            for (row, sum) in qk.chunks_mut(seq_len).zip(row_sums.iter()) {
                let inv_sum = sum.recip();
                for s in row.iter_mut() {
                    *s *= inv_sum;
                }
            }

            for i in 0..seq_len {
                for j in 0..=i {
                    let weight = qk[i * seq_len + j];
                    if weight == 0.0 {
                        continue;
                    }

                    let v_offset = j * embed_dim + v_head_offset;
                    let out_offset = i * embed_dim + q_head_offset;

                    let dst = &mut attn_buf[out_offset..out_offset + head_dim];
                    for d in 0..head_dim {
                        dst[d] += weight * v_data[v_offset + d];
                    }
                }
            }
        }

        let result_tensor = backend.from_cpu(attn_buf, &[seq_len, embed_dim])?;
        self.c_proj.forward(backend, &result_tensor)
    }
}
/// a single transformer block: layer_norm → attention → residual add
/// → layer_norm → mlp → residual add.
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
        let get_t = |name: &str| {
            loader
                .tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))
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
            // attention mapping — quantized linear weights need transpose
            // because the loader reverses dims for column-major q8_0 storage.
            let attn = Attention::new(
                Linear::new(
                    get_t(&format!("blk.{}.attn_qkv.weight", i))?.transpose(),
                    Some(get_t(&format!("blk.{}.attn_qkv.bias", i))?),
                ),
                Linear::new(
                    get_t(&format!("blk.{}.attn_output.weight", i))?.transpose(),
                    Some(get_t(&format!("blk.{}.attn_output.bias", i))?),
                ),
                n_heads,
            );

            let mlp = Mlp::new(
                Linear::new(
                    get_t(&format!("blk.{}.ffn_up.weight", i))?.transpose(),
                    Some(get_t(&format!("blk.{}.ffn_up.bias", i))?),
                ),
                Linear::new(
                    get_t(&format!("blk.{}.ffn_down.weight", i))?.transpose(),
                    Some(get_t(&format!("blk.{}.ffn_down.bias", i))?),
                ),
            );

            blocks.push(Block::new(
                LayerNorm::new(
                    get_t(&format!("blk.{}.attn_norm.weight", i))?,
                    get_t(&format!("blk.{}.attn_norm.bias", i))?,
                    1e-5,
                ),
                attn,
                LayerNorm::new(
                    get_t(&format!("blk.{}.ffn_norm.weight", i))?,
                    get_t(&format!("blk.{}.ffn_norm.bias", i))?,
                    1e-5,
                ),
                mlp,
            ));
        }

        Ok(Self {
            // embeddings are loaded with dims reversed by the loader
            // (column-major q8_0 → [vocab, embed]), so no manual
            // transpose needed — index_select already picks rows directly.
            wte: get_t("token_embd.weight")?,
            wpe: get_t("position_embd.weight")?,
            blocks,
            ln_f: LayerNorm::new(
                get_t("output_norm.weight")?,
                get_t("output_norm.bias")?,
                1e-5,
            ),
            head: Linear::new(get_t("output.weight")?.transpose(), None),
            n_heads,
        })
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
    ) -> Self {
        Self {
            wte,
            wpe,
            blocks,
            ln_f,
            head,
            n_heads,
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
}

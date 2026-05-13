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

        // pre-allocate scratch buffer once per call (not per head, not per token).
        // resizing never re-allocates because capacity == cache.max_seq_len() ≥ total_seq_len.
        let mut qk_scratch = Vec::with_capacity(cache.max_seq_len());

        for h in 0..self.n_heads {
            let q_head_offset = h * head_dim;

            for i in 0..seq_len {
                // causal mask: position i (in the current batch) attends to
                // positions 0..=total_seq_len - seq_len + i in the cache.
                let max_j = total_seq_len - seq_len + i;

                qk_scratch.clear();
                qk_scratch.resize(total_seq_len, f32::NEG_INFINITY);
                let qk_row = qk_scratch.as_mut_slice();
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

        let result = backend.load_from_cpu(attn_buf, &[seq_len, embed_dim])?;
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

        let result_tensor = backend.load_from_cpu(attn_buf, &[seq_len, embed_dim])?;
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

// ── llama support ────────────────────────────────────────────
// (work in progress; see LLAMA.md for the full plan)
// ─────────────────────────────────────────────────────────────

/// architectural parameters for a llama-family model.
///
/// read from gguf metadata during `Llama::from_loader`.
/// the key names here mirror meta's gguf convention
/// (`llama.block_count`, `llama.attention.head_count`, etc.).
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
    /// read llama config from gguf metadata.
    ///
    /// ## todo
    /// map these gguf metadata keys:
    ///   `llama.block_count`              → n_layers
    ///   `llama.attention.head_count`     → n_heads
    ///   `llama.attention.head_count_kv`  → n_kv_heads
    ///   `llama.rope.freq_base`           → rope_theta (default 10000.0)
    ///   `llama.context_length`           → max_seq_len
    ///   `general.architecture`           → "llama" sanity check
    ///
    /// fall back to sensible defaults when metadata is missing
    /// (e.g. n_kv_heads defaults to n_heads for non-gqa models).
    ///
    /// reference: llama.cpp reads the same keys in `llama-arch.cpp`.
    pub fn from_gguf_metadata(metadata: &crate::loader::GgufLoader) -> Self {
        let _ = metadata;
        todo!("LlamaConfig::from_gguf_metadata: read llama.* metadata keys")
    }
}

/// llama's swiglu feed-forward network.
///
/// three linear projections (no bias):
///   `silu(gate_proj(x)) * up_proj(x) → down_proj`
///
/// this replaces gpt-2's `Mlp` (which uses `c_fc` → gelu → `c_proj`).
/// gguf tensor names: `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`,
/// `blk.{i}.ffn_down.weight`.
///
/// reference: llama paper (touvron et al. 2023) §3.3, the PaLM paper's
/// swiglu variant (shazeer 2020).
pub struct LlamaMlp<B: Backend> {
    /// gate projection (input → 8/3 * input for standard llama)
    gate_proj: Linear<B>,
    /// up projection (input → 8/3 * input, multiplied after gate)
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
        // swiglu: silu(gate(x)) * up(x), then down-project
        //
        // ## todo
        // 1. gate = self.gate_proj.forward(backend, x)?
        // 2. gate = backend.silu(&gate)?
        // 3. up   = self.up_proj.forward(backend, x)?
        // 4. gated = element-wise multiply: backend.add(?) — need a mul op
        //    or do it manually via data().
        //    *actually*: the Backend trait doesn't have a mul() yet.
        //    options: (a) add mul to the trait, (b) do the multiply
        //    via data()/load_from_cpu like gpt-2's attention does,
        //    (c) add a higher-level fused swiglu to the backend.
        //    option (b) matches the gpt-2 pattern and keeps the trait
        //    surface small. for now this is todo.
        // 5. out = self.down_proj.forward(backend, &gated)?
        let _ = (backend, x);
        todo!("LlamaMlp::forward: implement swiglu")
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
///   • llama paper (touvron et al. 2023)
///   • gqa paper (ainslie et al. 2023)
///   • rope paper (su et al. 2021)
///   • llama.cpp's attention in `llama-arch.cpp` — the gold standard
///     for a working reference that handles all the edge cases
///   • huggingface `LlamaAttention` for the pure-python reference
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
    /// rope frequency base
    rope_theta: f32,
}

impl<B: Backend> LlamaAttention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: Linear<B>,
        k_proj: Linear<B>,
        v_proj: Linear<B>,
        o_proj: Linear<B>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
        }
    }
}

impl<B: Backend> Module<B> for LlamaAttention<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let _ = (backend, x);
        todo!("LlamaAttention::forward: separate q/k/v proj, rope, gqa, causal attention")
    }
}

impl<B: Backend> LlamaAttention<B> {
    /// forward with kv cache.
    ///
    /// the cache is allocated for `n_kv_heads` (not `n_heads`).
    /// during decode, cached k/v values are repeated via gqa to
    /// match the number of query heads before computing attention.
    ///
    /// ## todo
    /// follow the same prefill→cache→decode pattern as
    /// `Attention::forward_with_cache`, but:
    ///   • project q, k, v separately (no bias)
    ///   • apply rotary embeddings to q and k via `CpuTensor::apply_rotary_emb`
    ///     (or the backend equivalent once it's plumbed through)
    ///   • repeat k/v heads for gqa: `n_repeat = n_heads / n_kv_heads`
    ///     the standard llama.cpp approach repeats interleaved:
    ///       for h in 0..n_heads:
    ///         kv_h = h / n_repeat
    ///         k[.., h, ..] = cached_k[.., kv_h, ..]
    ///   • scaled dot-product attention with causal mask
    ///   • project through o_proj
    pub fn forward_with_cache(
        &self,
        backend: &B,
        x: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        layer: usize,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let _ = (backend, x, cache, layer, start_pos);
        todo!("LlamaAttention::forward_with_cache: rope + gqa + causal + cache append")
    }
}

/// a single llama decoder block.
///
/// ```text
/// x → rms_norm → self_attention → residual add
///   → rms_norm → swiglu_mlp → residual add
/// ```
///
/// note the order: pre-norm (rms), then attention/mlp, then add.
/// this is the same pre-norm layout as gpt-2, but gpt-2 uses
/// layer norm (mean+var, bias) while llama uses rms norm
/// (no mean, no bias).
///
/// gguf tensor names:
///   `blk.{i}.attn_norm.weight` → rms_norm weight for attention
///   `blk.{i}.ffn_norm.weight`  → rms_norm weight for mlp
///   (no bias tensors — rms norm has no bias parameter)
pub struct LlamaBlock<B: Backend> {
    /// pre-attention rms normalization weight
    input_layernorm: B::Tensor,
    /// multi-head self-attention
    self_attn: LlamaAttention<B>,
    /// pre-mlp rms normalization weight
    post_attention_layernorm: B::Tensor,
    /// swiglu feed-forward network
    mlp: LlamaMlp<B>,
}

impl<B: Backend> LlamaBlock<B> {
    pub fn new(
        input_layernorm: B::Tensor,
        self_attn: LlamaAttention<B>,
        post_attention_layernorm: B::Tensor,
        mlp: LlamaMlp<B>,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
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
        let _ = (backend, x, cache, layer, start_pos);
        todo!("LlamaBlock::forward_with_cache: rms_norm → attn (cached) → add → rms_norm → mlp → add")
    }
}

impl<B: Backend> Module<B> for LlamaBlock<B> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let _ = (backend, x);
        todo!("LlamaBlock::forward: rms_norm → attn → add → rms_norm → mlp → add")
    }
}

/// the full llama transformer model.
///
/// fields match the gguf tensor names in comments:
///   `token_embd.weight`      → embed_tokens
///   `blk.{i}.*`              → blocks
///   `output_norm.weight`     → norm  (rms, no bias)
///   `output.weight`          → head  (linear, no bias)
///
/// embedding lookup replaces gpt-2's `wte + wpe` with a single
/// token embedding (no learned position embeddings — rope handles
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

impl Llama<CpuBackend> {
    /// build a llama model from a gguf loader.
    ///
    /// reads metadata keys under the `llama.*` namespace (as written
    /// by llama.cpp's `llama-arch.cpp`) and maps gguf tensor names
    /// from the llama naming convention.
    ///
    /// expected gguf tensor names per layer:
    ///   `blk.{i}.attn_q.weight`       → q_proj
    ///   `blk.{i}.attn_k.weight`       → k_proj
    ///   `blk.{i}.attn_v.weight`       → v_proj
    ///   `blk.{i}.attn_output.weight`  → o_proj
    ///   `blk.{i}.ffn_gate.weight`     → gate_proj
    ///   `blk.{i}.ffn_up.weight`       → up_proj
    ///   `blk.{i}.ffn_down.weight`     → down_proj
    ///   `blk.{i}.attn_norm.weight`    → input_layernorm (rms, no bias)
    ///   `blk.{i}.ffn_norm.weight`     → post_attention_layernorm (rms, no bias)
    ///
    /// global tensors:
    ///   `token_embd.weight`           → embed_tokens
    ///   `output_norm.weight`          → final rms norm (no bias)
    ///   `output.weight`               → lm_head (linear, no bias)
    ///
    /// design note: quantized llama models from llama.cpp use the same
    /// column-major storage as quantized gpt-2 models. as with `Gpt2::from_loader`,
    /// quantized linear weights need `.transpose()` after loading.
    /// f32 weights should not be transposed (the loader leaves them
    /// in natural row-major order).
    ///
    /// ## todo
    ///   • read config via `LlamaConfig::from_gguf_metadata`
    ///   • allocate kv cache sizes based on n_kv_heads instead of n_heads
    ///   • precompute rope frequency tables
    ///   • build all layers in a loop
    ///   • test with an actual llama gguf file from llama.cpp
    pub fn from_loader(loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        let _ = loader;
        todo!("Llama::from_loader: read metadata, build blocks, return model")
    }
}

impl<B: Backend> Llama<B> {
    /// create a kv cache sized for this model's parameters.
    ///
    /// important difference from gpt-2: the cache allocates for
    /// `n_kv_heads` kv heads, not `n_heads` query heads.
    /// gqa repeats k/v during attention rather than storing duplicates.
    pub fn create_cache(&self, _backend: &B, max_seq_len: usize) -> crate::kv_cache::KVCache {
        let _ = max_seq_len;
        todo!("Llama::create_cache: allocate cache for n_kv_heads * max_seq_len * head_dim")
    }

    /// forward pass with incremental kv caching.
    ///
    /// mirrors `Gpt2::forward_with_cache` but:
    ///   • uses `LlamaBlock::forward_with_cache` which passes start_pos for rope
    ///   • normalizes with rms norm (via `backend.rms_norm`)
    ///   • no position embedding lookup (rope is in the attention layer)
    pub fn forward_with_cache(
        &self,
        backend: &B,
        token_ids: &[u32],
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        let _ = (backend, token_ids, cache, start_pos);
        todo!("Llama::forward_with_cache: embed → blocks → norm → head")
    }

    /// forward pass without caching (full sequence).
    pub fn forward(&self, backend: &B, token_ids: &[u32]) -> Result<B::Tensor, B::Error> {
        let _ = (backend, token_ids);
        todo!("Llama::forward: embed → blocks → norm → head")
    }
}

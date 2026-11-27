use super::*;

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
                block_boundaries: None,
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
    pub(super) weight: B::Tensor,
    /// learned bias parameter
    pub(super) bias: B::Tensor,
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
    pub fn from_loader(mut loader: crate::loader::GgufLoader) -> anyhow::Result<Self> {
        // GPT-2 consumes f32 tensors only; widen f16/bf16 ones exactly as
        // the loader used to.
        loader.materialize_half_tensors();
        if let Some(architecture) = loader.metadata.get("general.architecture") {
            match architecture {
                crate::loader::GgufValue::Str(architecture) => anyhow::ensure!(
                    architecture == "gpt2",
                    "unsupported GPT-2 GGUF architecture '{architecture}'"
                ),
                _ => anyhow::bail!("GGUF general.architecture must be a string"),
            }
        }
        // metadata
        let n_layers = match loader.metadata.get("gpt2.block_count") {
            Some(crate::loader::GgufValue::U32(n)) => *n as usize,
            _ => 12,
        };
        let n_heads = match loader.metadata.get("gpt2.attention.head_count") {
            Some(crate::loader::GgufValue::U32(n)) => *n as usize,
            _ => 12,
        };
        anyhow::ensure!(n_layers > 0, "GPT-2 model must contain at least one layer");
        anyhow::ensure!(n_heads > 0, "GPT-2 attention head count must be non-zero");
        // Metadata values are attacker-controlled; enforce construction caps
        // before sizing the block vector or loading per-block projections.
        anyhow::ensure!(
            n_layers <= crate::loader::limits::MAX_LAYERS,
            "gpt2.block_count {n_layers} exceeds the {} layer limit",
            crate::loader::limits::MAX_LAYERS
        );
        anyhow::ensure!(
            n_heads <= crate::loader::limits::MAX_HEADS,
            "gpt2.attention.head_count {n_heads} exceeds the {} head limit",
            crate::loader::limits::MAX_HEADS
        );

        // Verify the complete tensor inventory and cheap geometry before
        // dequantizing/allocating any block. A hostile but syntactically valid
        // block weight must not consume the construction budget before a
        // malformed embedding or output shape is discovered.
        let mut required = vec![
            "token_embd.weight".to_string(),
            "position_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "output_norm.bias".to_string(),
            "output.weight".to_string(),
        ];
        required.reserve(n_layers * 12);
        for i in 0..n_layers {
            for suffix in [
                "attn_qkv.weight",
                "attn_qkv.bias",
                "attn_output.weight",
                "attn_output.bias",
                "ffn_up.weight",
                "ffn_up.bias",
                "ffn_down.weight",
                "ffn_down.bias",
                "attn_norm.weight",
                "attn_norm.bias",
                "ffn_norm.weight",
                "ffn_norm.bias",
            ] {
                required.push(format!("blk.{i}.{suffix}"));
            }
        }
        crate::loader::require_tensors(&loader, &required)?;
        loader.check_model_dequantization_budget()?;

        let token_shape = gpt2_embedding_shape(&loader, "token_embd.weight")?;
        let position_shape = gpt2_embedding_shape(&loader, "position_embd.weight")?;
        let (vocab_size, embed_dim) = token_shape;
        anyhow::ensure!(
            vocab_size > 0 && vocab_size <= crate::loader::limits::MAX_VOCAB_SIZE,
            "GPT-2 vocabulary size {vocab_size} is outside the supported range"
        );
        anyhow::ensure!(
            embed_dim > 0 && embed_dim <= crate::loader::limits::MAX_EMBED_DIM,
            "GPT-2 embedding width {embed_dim} is outside the supported range"
        );
        anyhow::ensure!(
            position_shape.0 > 0 && position_shape.0 <= crate::loader::limits::MAX_CONTEXT_LEN,
            "GPT-2 context length {} is outside the supported range",
            position_shape.0
        );
        anyhow::ensure!(
            position_shape.1 == embed_dim,
            "GPT-2 token/position embedding widths differ: {embed_dim} vs {}",
            position_shape.1
        );
        anyhow::ensure!(
            embed_dim.is_multiple_of(n_heads),
            "GPT-2 embedding width {embed_dim} is not divisible by {n_heads} heads"
        );
        let head_dim = embed_dim / n_heads;
        anyhow::ensure!(
            head_dim <= crate::loader::limits::MAX_HEAD_DIM,
            "GPT-2 attention head dimension {head_dim} exceeds the {} limit",
            crate::loader::limits::MAX_HEAD_DIM
        );
        gpt2_check_linear(&loader, "output.weight", embed_dim, vocab_size)?;
        gpt2_check_vector(&loader, "output_norm.weight", embed_dim)?;
        gpt2_check_vector(&loader, "output_norm.bias", embed_dim)?;
        for i in 0..n_layers {
            let prefix = format!("blk.{i}.");
            gpt2_check_linear(
                &loader,
                &format!("{prefix}attn_qkv.weight"),
                embed_dim,
                embed_dim * 3,
            )?;
            gpt2_check_vector(&loader, &format!("{prefix}attn_qkv.bias"), embed_dim * 3)?;
            gpt2_check_linear(
                &loader,
                &format!("{prefix}attn_output.weight"),
                embed_dim,
                embed_dim,
            )?;
            gpt2_check_vector(&loader, &format!("{prefix}attn_output.bias"), embed_dim)?;
            let (mlp_input, intermediate) =
                gpt2_linear_shape(&loader, &format!("{prefix}ffn_up.weight"))?;
            anyhow::ensure!(
                mlp_input == embed_dim && intermediate > 0 && intermediate <= crate::loader::limits::MAX_INTERMEDIATE_DIM,
                "GPT-2 block {i} MLP up projection must be {embed_dim} -> an intermediate width <= {}, got {mlp_input} -> {intermediate}",
                crate::loader::limits::MAX_INTERMEDIATE_DIM
            );
            gpt2_check_vector(&loader, &format!("{prefix}ffn_up.bias"), intermediate)?;
            gpt2_check_linear(
                &loader,
                &format!("{prefix}ffn_down.weight"),
                intermediate,
                embed_dim,
            )?;
            gpt2_check_vector(&loader, &format!("{prefix}ffn_down.bias"), embed_dim)?;
            gpt2_check_vector(&loader, &format!("{prefix}attn_norm.weight"), embed_dim)?;
            gpt2_check_vector(&loader, &format!("{prefix}attn_norm.bias"), embed_dim)?;
            gpt2_check_vector(&loader, &format!("{prefix}ffn_norm.weight"), embed_dim)?;
            gpt2_check_vector(&loader, &format!("{prefix}ffn_norm.bias"), embed_dim)?;
        }

        // blocks
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn = Attention::new(
                take_gpt2_linear(
                    &mut loader,
                    &format!("blk.{}.attn_qkv.weight", i),
                    Some(&format!("blk.{}.attn_qkv.bias", i)),
                )?,
                take_gpt2_linear(
                    &mut loader,
                    &format!("blk.{}.attn_output.weight", i),
                    Some(&format!("blk.{}.attn_output.bias", i)),
                )?,
                n_heads,
            );

            let mlp = Mlp::new(
                take_gpt2_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_up.weight", i),
                    Some(&format!("blk.{}.ffn_up.bias", i)),
                )?,
                take_gpt2_linear(
                    &mut loader,
                    &format!("blk.{}.ffn_down.weight", i),
                    Some(&format!("blk.{}.ffn_down.bias", i)),
                )?,
            );

            blocks.push(Block::new(
                LayerNorm::new(
                    loader.take_f32(&format!("blk.{}.attn_norm.weight", i))?,
                    loader.take_f32(&format!("blk.{}.attn_norm.bias", i))?,
                    1e-5,
                ),
                attn,
                LayerNorm::new(
                    loader.take_f32(&format!("blk.{}.ffn_norm.weight", i))?,
                    loader.take_f32(&format!("blk.{}.ffn_norm.bias", i))?,
                    1e-5,
                ),
                mlp,
            ));
        }

        let wte = take_gpt2_embedding(&mut loader, "token_embd.weight")?;
        let wpe = take_gpt2_embedding(&mut loader, "position_embd.weight")?;
        let ln_f = LayerNorm::new(
            loader.take_f32("output_norm.weight")?,
            loader.take_f32("output_norm.bias")?,
            1e-5,
        );
        let head = take_gpt2_linear(&mut loader, "output.weight", None)?;
        validate_gpt2_norm(&ln_f, embed_dim, "output_norm")?;
        anyhow::ensure!(
            head.in_features(&CpuBackend) == embed_dim
                && head.out_features(&CpuBackend) == wte.shape()[0],
            "GPT-2 output head must be {} -> {}, got {} -> {}",
            embed_dim,
            wte.shape()[0],
            head.in_features(&CpuBackend),
            head.out_features(&CpuBackend)
        );
        for (index, block) in blocks.iter().enumerate() {
            validate_gpt2_block(block, embed_dim, index)?;
        }

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f,
            head,
            n_heads,
            embed_dim,
        })
    }
}

fn gpt2_linear_shape(
    loader: &crate::loader::GgufLoader,
    name: &str,
) -> anyhow::Result<(usize, usize)> {
    use crate::loader::LoadedTensor;
    let tensor = loader
        .tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))?;
    match tensor {
        LoadedTensor::F32(tensor) => match tensor.shape() {
            [output, input] => Ok((*input, *output)),
            shape => anyhow::bail!("{name} must be a 2D linear weight, got {shape:?}"),
        },
        LoadedTensor::Q8_0(weight) => Ok((weight.in_features(), weight.out_features())),
        LoadedTensor::KQuant(_) => {
            anyhow::bail!("{name} uses unsupported compressed K-quant tensors")
        }
        LoadedTensor::Half(_) => {
            anyhow::bail!("{name}: f16/bf16 tensors must be widened before GPT-2 construction")
        }
    }
}

fn gpt2_embedding_shape(
    loader: &crate::loader::GgufLoader,
    name: &str,
) -> anyhow::Result<(usize, usize)> {
    use crate::loader::LoadedTensor;
    let tensor = loader
        .tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))?;
    match tensor {
        LoadedTensor::F32(tensor) => match tensor.shape() {
            [embedding, vocab] => Ok((*vocab, *embedding)),
            shape => anyhow::bail!("{name} must be a 2D embedding, got {shape:?}"),
        },
        LoadedTensor::Q8_0(weight) => Ok((weight.out_features(), weight.in_features())),
        LoadedTensor::KQuant(_) => {
            anyhow::bail!("{name} uses unsupported compressed K-quant tensors")
        }
        LoadedTensor::Half(_) => {
            anyhow::bail!("{name}: f16/bf16 tensors must be widened before GPT-2 construction")
        }
    }
}

fn gpt2_check_linear(
    loader: &crate::loader::GgufLoader,
    name: &str,
    input: usize,
    output: usize,
) -> anyhow::Result<()> {
    let actual = gpt2_linear_shape(loader, name)?;
    anyhow::ensure!(
        actual == (input, output),
        "GPT-2 {name} must be {input} -> {output}, got {} -> {}",
        actual.0,
        actual.1
    );
    Ok(())
}

fn gpt2_check_vector(
    loader: &crate::loader::GgufLoader,
    name: &str,
    width: usize,
) -> anyhow::Result<()> {
    use crate::loader::LoadedTensor;
    let tensor = loader
        .tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor: {name}"))?;
    match tensor {
        LoadedTensor::F32(tensor) => anyhow::ensure!(
            tensor.shape() == [width],
            "GPT-2 {name} must have shape [{width}], got {:?}",
            tensor.shape()
        ),
        _ => anyhow::bail!("GPT-2 {name} must be an f32 vector of width {width}"),
    }
    Ok(())
}

fn take_gpt2_embedding(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
) -> anyhow::Result<CpuTensor> {
    use crate::loader::LoadedTensor;
    match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => {
            anyhow::ensure!(tensor.ndim() == 2, "{name} must be 2D");
            let shape = tensor.shape();
            let embedding_shape = [shape[1], shape[0]];
            Ok(tensor.into_reshape(&embedding_shape))
        }
        LoadedTensor::Q8_0(weight) => {
            crate::loader::check_f32_dequantization_size(
                name,
                weight.out_features(),
                weight.in_features(),
            )?;
            Ok(weight.dequantize_all())
        }
        LoadedTensor::KQuant(_) => {
            anyhow::bail!("gpt2 does not support compressed K-quant tensors in v0.3")
        }
        LoadedTensor::Half(_) => {
            anyhow::bail!("{name}: f16/bf16 tensors must be widened before GPT-2 construction")
        }
    }
}

fn validate_gpt2_linear_bias(
    linear: &Linear<CpuBackend>,
    expected: usize,
    name: &str,
) -> anyhow::Result<()> {
    if let Some(bias) = linear.bias() {
        anyhow::ensure!(
            bias.shape() == [expected],
            "GPT-2 {name} bias must have shape [{expected}], got {:?}",
            bias.shape()
        );
    }
    Ok(())
}

fn validate_gpt2_norm(
    norm: &LayerNorm<CpuBackend>,
    embed_dim: usize,
    name: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        norm.weight.shape() == [embed_dim] && norm.bias.shape() == [embed_dim],
        "GPT-2 {name} weights must both have shape [{embed_dim}], got {:?} and {:?}",
        norm.weight.shape(),
        norm.bias.shape()
    );
    Ok(())
}

fn validate_gpt2_block(
    block: &Block<CpuBackend>,
    embed_dim: usize,
    index: usize,
) -> anyhow::Result<()> {
    let backend = CpuBackend;
    let qkv_width = embed_dim
        .checked_mul(3)
        .context("GPT-2 QKV width overflow")?;
    anyhow::ensure!(
        block.attn.c_attn.in_features(&backend) == embed_dim
            && block.attn.c_attn.out_features(&backend) == qkv_width,
        "GPT-2 block {index} QKV projection must be {embed_dim} -> {qkv_width}"
    );
    anyhow::ensure!(
        block.attn.c_proj.in_features(&backend) == embed_dim
            && block.attn.c_proj.out_features(&backend) == embed_dim,
        "GPT-2 block {index} attention output projection must be {embed_dim} -> {embed_dim}"
    );
    validate_gpt2_linear_bias(&block.attn.c_attn, qkv_width, &format!("block {index} QKV"))?;
    validate_gpt2_linear_bias(
        &block.attn.c_proj,
        embed_dim,
        &format!("block {index} attention output"),
    )?;
    let intermediate = block.mlp.c_fc.out_features(&backend);
    anyhow::ensure!(
        block.mlp.c_fc.in_features(&backend) == embed_dim
            && intermediate > 0
            && block.mlp.c_proj.in_features(&backend) == intermediate
            && block.mlp.c_proj.out_features(&backend) == embed_dim,
        "GPT-2 block {index} MLP projection shapes are inconsistent"
    );
    validate_gpt2_linear_bias(
        &block.mlp.c_fc,
        intermediate,
        &format!("block {index} MLP up"),
    )?;
    validate_gpt2_linear_bias(
        &block.mlp.c_proj,
        embed_dim,
        &format!("block {index} MLP down"),
    )?;
    validate_gpt2_norm(
        &block.ln_1,
        embed_dim,
        &format!("block {index} attention norm"),
    )?;
    validate_gpt2_norm(&block.ln_2, embed_dim, &format!("block {index} MLP norm"))?;
    Ok(())
}

pub(super) fn take_gpt2_linear(
    loader: &mut crate::loader::GgufLoader,
    name: &str,
    bias_name: Option<&str>,
) -> anyhow::Result<Linear<CpuBackend>> {
    use crate::loader::LoadedTensor;

    let bias = bias_name.map(|name| loader.take_f32(name)).transpose()?;
    match loader.take_tensor(name)? {
        LoadedTensor::F32(tensor) => {
            anyhow::ensure!(
                tensor.ndim() == 2,
                "{name} must be a 2D linear weight, got {:?}",
                tensor.shape()
            );
            Ok(Linear::new(tensor.transpose(), bias))
        }
        LoadedTensor::Q8_0(weight) => Ok(Linear::new_q8_0(weight, bias)),
        LoadedTensor::KQuant(_) => {
            anyhow::bail!("gpt2 does not support compressed K-quant tensors in v0.3")
        }
        LoadedTensor::Half(_) => {
            anyhow::bail!("{name}: f16/bf16 tensors must be widened before GPT-2 construction")
        }
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
    fn forward_embeddings_with_cache(
        &self,
        backend: &B,
        embeddings: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        Gpt2::forward_embeddings_with_cache(self, backend, embeddings, cache, start_pos)
    }
    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
    fn vocab_size(&self, backend: &B) -> usize {
        backend.shape(&self.wte)[0]
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

    fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        Gpt2::forward_pooled_hidden_states(self, backend, token_ids, token_index_groups)
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
        assert!(
            self.n_heads > 0,
            "GPT-2 attention head count must be non-zero"
        );
        assert!(
            embed_dim.is_multiple_of(self.n_heads),
            "GPT-2 embedding width must be divisible by its head count"
        );
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
        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "GPT-2 forward requires at least one token"
        );
        let x = self.embed_with_offset(backend, token_ids, start_pos)?;
        self.forward_embeddings_with_cache(backend, &x, cache, start_pos)
    }

    /// forward the transformer body on precomputed embeddings.
    ///
    /// contract: `embeddings` must be *fully formed* model input embeddings —
    /// for GPT-2 that means the caller has already added the learned position
    /// embeddings (`wte + wpe` rows at the correct absolute positions).
    pub fn forward_embeddings_with_cache(
        &self,
        backend: &B,
        embeddings: &B::Tensor,
        cache: &mut crate::kv_cache::KVCache,
        start_pos: usize,
    ) -> Result<B::Tensor, B::Error> {
        cache.validate_start_pos(start_pos);
        let seq_len = backend.shape(embeddings)[0];
        assert!(seq_len > 0, "GPT-2 forward requires at least one embedding");
        let mut x = embeddings.clone();
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
        cache.validate_start_pos(start_pos);
        assert!(
            !token_ids.is_empty(),
            "GPT-2 forward requires at least one token"
        );
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
        assert!(
            !token_ids.is_empty(),
            "GPT-2 forward requires at least one token"
        );
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
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    pub fn forward_with_activations(
        &self,
        backend: &B,
        token_ids: &[u32],
    ) -> Result<(Vec<Vec<f32>>, B::Tensor), B::Error> {
        assert!(
            !token_ids.is_empty(),
            "GPT-2 forward requires at least one token"
        );
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
        let (pooled, last) =
            self.forward_pooled_hidden_and_last(backend, token_ids, token_index_groups)?;
        let last = self.ln_f.forward(backend, &last)?;
        let logits = self.head.forward(backend, &last)?;
        Ok((pooled, logits))
    }

    pub fn forward_pooled_hidden_states(
        &self,
        backend: &B,
        token_ids: &[u32],
        token_index_groups: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, B::Error> {
        self.forward_pooled_hidden_and_last(backend, token_ids, token_index_groups)
            .map(|(pooled, _)| pooled)
    }

    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
    fn forward_pooled_hidden_and_last(
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
        let last = backend.row_as_2d(&x, token_ids.len() - 1)?;
        Ok((pooled, last))
    }
}

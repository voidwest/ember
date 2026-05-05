use crate::backend::{Backend, CpuBackend, Module};
use alloc::vec::Vec;

pub struct Linear<B: Backend> {
    weight: B::Tensor,
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

pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
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

pub struct Attention<B: Backend> {
    c_attn: Linear<B>,
    c_proj: Linear<B>,
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

            for i in 0..seq_len {
                let row_start = i * seq_len;
                let inv_sum = row_sums[i].recip();
                let row = &mut qk[row_start..row_start + seq_len];
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
pub struct Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: Attention<B>,
    ln_2: LayerNorm<B>,
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

pub struct LayerNorm<B: Backend> {
    weight: B::Tensor,
    bias: B::Tensor,
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

pub struct Gpt2<B: Backend> {
    pub wte: B::Tensor,
    pub wpe: B::Tensor,
    pub blocks: Vec<Block<B>>,
    pub ln_f: LayerNorm<B>,
    pub head: Linear<B>,
}

impl Gpt2<CpuBackend> {
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
        }; // default val

        // blocks
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // attention mapping
            let attn = Attention::new(
                Linear::new(
                    get_t(&format!("blk.{}.attn_qkv.weight", i))?,
                    Some(get_t(&format!("blk.{}.attn_qkv.bias", i))?),
                ),
                Linear::new(
                    get_t(&format!("blk.{}.attn_output.weight", i))?,
                    Some(get_t(&format!("blk.{}.attn_output.bias", i))?),
                ),
                n_heads,
            );

            let mlp = Mlp::new(
                Linear::new(
                    get_t(&format!("blk.{}.ffn_up.weight", i))?,
                    Some(get_t(&format!("blk.{}.ffn_up.bias", i))?),
                ),
                Linear::new(
                    get_t(&format!("blk.{}.ffn_down.weight", i))?,
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
            wte: get_t("token_embd.weight")?.transpose(),
            wpe: get_t("position_embd.weight")?.transpose(),
            blocks,
            ln_f: LayerNorm::new(
                get_t("output_norm.weight")?,
                get_t("output_norm.bias")?,
                1e-5,
            ),
            head: Linear::new(get_t("output.weight")?, None),
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
    ) -> Self {
        Self {
            wte,
            wpe,
            blocks,
            ln_f,
            head,
        }
    }
    fn embed(&self, backend: &B, tokens: &[u32]) -> Result<B::Tensor, B::Error> {
        let seq_len = tokens.len();
        let embed_dim = backend.shape(&self.wte)[1];
        let mut x = backend.zeroes(&[seq_len, embed_dim])?;
        for (i, &token_id) in tokens.iter().enumerate() {
            let word_vec = backend.index_select(&self.wte, token_id as usize)?;

            let pos_vec = backend.index_select(&self.wpe, i)?;

            let combined = backend.add(&word_vec, &pos_vec)?;

            backend.assign_row(&mut x, i, &combined);
        }
        Ok(x)
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

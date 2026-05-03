use crate::backend::{Backend, CpuBackend, Module};
use alloc::vec::Vec;
use reqwest::get;

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
            out = backend.add(&out, b)?;
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
        let embed_dim = qkv.shape()[1] / 3;
        let q = backend.slice_cols(&qkv, 0, embed_dim)?;
        let k = backend.slice_cols(&qkv, embed_dim, 2 * embed_dim)?;
        let v = backend.slice_cols(&qkv, 2 * embed_dim, 3 * embed_dim)?;
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
        let n_heads = 12; // default val

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
            wte: get_t("token_embd.weight")?,
            wpe: get_t("position_embd.weight")?,
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
        let mut x = backend.zeroes(&[seq_len, 768])?;
        for (i, &token_id) in tokens.iter().enumerate() {
            let word_vec = backend.index_select(&self.wte, token_id as usize)?;

            let pos_vec = backend.index_select(&self.wpe, i)?;

            let combined = backend.add(&word_vec, &pos_vec)?;

            backend.assign_row(&mut x, i, &combined);
        }
        Ok(x)
    }

    pub fn forward(&self, backend: &B, token_ids: &[usize]) -> Result<B::Tensor, B::Error> {
        let vocab_size = 50257;
        backend.zeroes(&[token_ids.len(), vocab_size])
    }
}

use crate::backend::{Backend, Module};
use alloc::vec::Vec;

pub struct Linear<B: Backend> {
    weight: B::Tensor,
    bias: Option<B::Tensor>,
}

pub struct Attention<B: Backend> {
    pub w_qkve: Linear<B>,
    pub w_out: Linear<B>,
    pub n_heads: usize,
}

pub struct Gpt2<B: Backend> {
    pub wte: B::Tensor,
    pub wpe: B::Tensor,
    pub blocks: Vec<Block<B>>,
    pub ln_f: LayerNorm<B>,
    pub head: Linear<B>,
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
        self.c_proj.forward(backend, x) // simple attention
                                        // 1. project x to q, k, v
                                        // 2. split to heads
                                        // 3. scaled dot products (q@ k^t / sqrt(head_dim))
                                        // apply casual mask
                                        // softmax
                                        // attention @ v
                                        // concat heads
                                        // output projection
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
    wte: B::Tensor,
    wpe: B::Tensor,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
}

impl<B: Backend> Gpt2<B> {
    pub fn new(wte: B::Tensor, wpe: B::Tensor, blocks: Vec<Block<B>>, ln_f: LayerNorm<B>) -> Self {
        Self {
            wte,
            wpe,
            blocks,
            ln_f,
        }
    }

    pub fn forward(&self, backend: &B, token_ids: &[usize]) -> Result<B::Tensor, B::Error> {
        let vocab_size = 50257;
        backend.zeroes(&[token_ids.len(), vocab_size])
    }
}

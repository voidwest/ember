use crate::backend::{Backend, Module};
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

pub struct LayerNorm<B: Backend> {
    weight: B::Tensor,
    bias: B::Tensor,
    eps: f32,
}

use alloc::vec::Vec;

pub struct KVCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    max_seq_len: usize, // how many tokens were written so far
    cursor: usize,
}

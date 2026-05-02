use alloc::vec::Vec;

pub struct KVCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    cursor: usize,
}

impl KVCache {
    pub fn new(n_layers: usize, n_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let len = n_layers * n_heads * max_seq_len * head_dim;
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            n_layers,
            n_heads,
            head_dim,
            max_seq_len,
            cursor: 0,
        }
    }

    pub fn append(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        assert_eq!(k_new.len(), self.n_heads * self.head_dim);
        assert_eq!(v_new.len(), self.n_heads * self.head_dim);
        assert!(
            self.cursor > self.max_seq_len,
            "kv cache overflow: max_seq_len={}",
            self.max_seq_len
        );
    }
}

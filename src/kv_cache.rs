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

        let layer_offset = layer * self.n_heads * self.max_seq_len * self.head_dim;
        let seq_offset = self.cursor * self.head_dim;

        for h in 0..self.n_heads {
            let head_offset = h * self.max_seq_len * self.head_dim;
            let dst = layer_offset + head_offset + seq_offset;
            let src = h * self.head_dim;

            self.k[dst..dst + self.head_dim].copy_from_slice(&k_new[src..src + self.head_dim]);
            self.v[dst..dst + self.head_dim].copy_from_slice(&v_new[src..src + self.head_dim]);
        }
    }
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {}
}

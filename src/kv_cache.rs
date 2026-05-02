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
            self.cursor < self.max_seq_len,
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
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        let layer_offset = layer * self.n_heads * self.max_seq_len * self.head_dim;
        let len = self.n_heads * self.cursor * self.head_dim;
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
        )
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn advance_cursor(&mut self) {
        self.cursor += 1;
    }
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(2, 4, 8, 128);
        let k = vec![1.0; 4 * 8];
        let v = vec![2.0; 4 * 8];

        cache.append(0, &k, &v);
        cache.advance_cursor();
        assert_eq!(cache.cursor(), 1);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), 4 * 1 * 8);
        assert_eq!(v_out.len(), 4 * 1 * 8);
    }
}

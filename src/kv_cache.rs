use alloc::vec::Vec;

/// a flat, pre-allocated key/value cache for transformer attention.
///
/// memory layout: `[layer][head][seq_position][head_dim]`.
/// wired into `Attention::forward_with_cache` - during prefill the full
/// k/v projection is cached; subsequent decode steps read from the cache
/// instead of recomputing against the full sequence each pass.
pub struct KVCache {
    /// key cache, flat layout: [layer][head][pos][head_dim]
    k: Vec<f32>,
    /// value cache, flat layout: [layer][head][pos][head_dim]
    v: Vec<f32>,
    /// stored for allocation size, not read back
    #[allow(dead_code)]
    n_layers: usize,
    /// pre-allocated scratch buffer for attention score rows.
    /// reused across all heads and tokens during a decode step
    /// so the hot path never allocates.
    qk_scratch: Vec<f32>,
    /// number of kv heads stored in the cache.
    /// for gpt-2 this equals n_heads; for llama with gqa it may be less.
    n_kv_heads: usize,
    /// size per head
    head_dim: usize,
    /// maximum sequence length the cache was allocated for
    max_seq_len: usize,
    /// write position in the sequence dimension
    cursor: usize,
}

impl KVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let len = n_layers * n_kv_heads * max_seq_len * head_dim;
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            n_layers,
            n_kv_heads,
            qk_scratch: vec![0.0; max_seq_len],
            head_dim,
            max_seq_len,
            cursor: 0,
        }
    }

    pub fn append(&mut self, layer: usize, pos: usize, k_new: &[f32], v_new: &[f32]) {
        assert_eq!(k_new.len(), self.n_kv_heads * self.head_dim);
        assert_eq!(v_new.len(), self.n_kv_heads * self.head_dim);
        self.append_with_head_dim(layer, pos, k_new, v_new, self.head_dim);
    }

    pub fn append_with_head_dim(
        &mut self,
        layer: usize,
        pos: usize,
        k_new: &[f32],
        v_new: &[f32],
        active_head_dim: usize,
    ) {
        assert!(active_head_dim <= self.head_dim);
        assert_eq!(k_new.len(), self.n_kv_heads * active_head_dim);
        assert_eq!(v_new.len(), self.n_kv_heads * active_head_dim);
        assert!(
            pos < self.max_seq_len,
            "kv cache overflow: pos={}, max_seq_len={}",
            pos,
            self.max_seq_len
        );

        let layer_offset = layer * self.n_kv_heads * self.max_seq_len * self.head_dim;
        let seq_offset = pos * self.head_dim;

        for h in 0..self.n_kv_heads {
            let head_offset = h * self.max_seq_len * self.head_dim;
            let dst = layer_offset + head_offset + seq_offset;
            let src = h * active_head_dim;

            self.k[dst..dst + active_head_dim].copy_from_slice(&k_new[src..src + active_head_dim]);
            self.v[dst..dst + active_head_dim].copy_from_slice(&v_new[src..src + active_head_dim]);
        }
    }
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        let layer_offset = layer * self.n_kv_heads * self.max_seq_len * self.head_dim;
        let len = self.n_kv_heads * self.max_seq_len * self.head_dim;
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
        )
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// return a mutable reference to the pre-allocated `qk_scratch` buffer.
    ///
    /// the caller should `clear()` and then `resize(total_seq_len, f32::NEG_INFINITY)`
    /// before use. because the buffer was allocated to `max_seq_len`,
    /// `resize` will never reallocate as long as `total_seq_len <= max_seq_len`.
    #[inline]
    pub fn qk_scratch_mut(&mut self) -> &mut Vec<f32> {
        &mut self.qk_scratch
    }

    /// maximum sequence length the cache was allocated for
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn advance_cursor(&mut self) {
        self.cursor += 1;
    }
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// number of kv heads stored in the cache.
    /// for gpt-2 this equals n_heads; for llama with gqa it may be less.
    #[inline]
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
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

        cache.append(0, 0, &k, &v);
        cache.advance_cursor();
        assert_eq!(cache.cursor(), 1);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), 4 * 128 * 8);
        assert_eq!(v_out.len(), 4 * 128 * 8);
    }
}

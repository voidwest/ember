use alloc::vec::Vec;
use half::{f16, slice::HalfFloatSliceExt};

/// a flat, pre-allocated key/value cache for transformer attention.
///
/// memory layout: `[layer][head][seq_position][head_dim]`.
/// wired into `Attention::forward_with_cache` - during prefill the full
/// k/v projection is cached; subsequent decode steps read from the cache
/// instead of recomputing against the full sequence each pass.
pub struct KVCache {
    /// key cache, flat layout: [layer][head][pos][head_dim]
    k: Vec<f16>,
    /// value cache, flat layout: [layer][head][pos][head_dim]
    v: Vec<f16>,
    /// number of cache layers
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
        Self::try_new(n_layers, n_kv_heads, head_dim, max_seq_len)
            .expect("invalid or unallocatable KV cache geometry")
    }

    /// Fallible cache allocation for metadata-driven import paths.
    ///
    /// Ordinary model construction continues to use [`KVCache::new`], whose
    /// assertion-level contract is unchanged. Snapshot import uses this
    /// method so malformed or excessive dimensions fail before decode rather
    /// than overflowing shape arithmetic or panicking during allocation.
    pub fn try_new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if n_layers == 0 {
            return Err("kv cache requires at least one layer".into());
        }
        if n_kv_heads == 0 {
            return Err("kv cache requires at least one KV head".into());
        }
        if head_dim == 0 {
            return Err("kv cache requires a non-zero head dimension".into());
        }
        if max_seq_len == 0 {
            return Err("kv cache requires a positive sequence length".into());
        }
        let len = [n_layers, n_kv_heads, max_seq_len, head_dim]
            .into_iter()
            .try_fold(1usize, |count, dim| count.checked_mul(dim))
            .ok_or_else(|| "kv cache shape product overflow".to_string())?;

        let allocate_f16 = |name: &str| -> Result<Vec<f16>, String> {
            let mut values = Vec::new();
            values.try_reserve_exact(len).map_err(|error| {
                format!("cannot allocate {name} KV payload ({len} f16): {error}")
            })?;
            values.resize(len, f16::ZERO);
            Ok(values)
        };
        let mut qk_scratch = Vec::new();
        qk_scratch.try_reserve_exact(max_seq_len).map_err(|error| {
            format!("cannot allocate KV attention scratch ({max_seq_len} f32): {error}")
        })?;
        qk_scratch.resize(max_seq_len, 0.0);

        Ok(Self {
            k: allocate_f16("key")?,
            v: allocate_f16("value")?,
            n_layers,
            n_kv_heads,
            qk_scratch,
            head_dim,
            max_seq_len,
            cursor: 0,
        })
    }

    pub fn append(&mut self, layer: usize, pos: usize, k_new: &[f32], v_new: &[f32]) {
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
        self.append_with_layout(layer, pos, k_new, v_new, self.n_kv_heads, active_head_dim);
    }

    /// Append K/V values when a layer uses fewer heads and/or a narrower head
    /// dimension than the cache's maximum allocation.
    pub fn append_with_layout(
        &mut self,
        layer: usize,
        pos: usize,
        k_new: &[f32],
        v_new: &[f32],
        active_kv_heads: usize,
        active_head_dim: usize,
    ) {
        assert!(layer < self.n_layers, "kv cache layer out of bounds");
        assert!(
            active_kv_heads > 0,
            "kv cache requires at least one active head"
        );
        assert!(active_kv_heads <= self.n_kv_heads);
        assert!(
            active_head_dim > 0,
            "kv cache requires a non-zero head dimension"
        );
        assert!(active_head_dim <= self.head_dim);
        let source_len = active_kv_heads
            .checked_mul(active_head_dim)
            .expect("kv cache append shape product overflow");
        assert_eq!(k_new.len(), source_len);
        assert_eq!(v_new.len(), source_len);
        assert!(
            pos < self.max_seq_len,
            "kv cache overflow: pos={}, max_seq_len={}",
            pos,
            self.max_seq_len
        );

        let layer_offset = self.layer_offset(layer);
        let seq_offset = pos
            .checked_mul(self.head_dim)
            .expect("kv cache sequence offset overflow");

        for h in 0..active_kv_heads {
            let head_offset = h
                .checked_mul(self.max_seq_len)
                .and_then(|offset| offset.checked_mul(self.head_dim))
                .expect("kv cache head offset overflow");
            let dst = layer_offset + head_offset + seq_offset;
            let src = h * active_head_dim;

            self.k[dst..dst + active_head_dim]
                .convert_from_f32_slice(&k_new[src..src + active_head_dim]);
            self.v[dst..dst + active_head_dim]
                .convert_from_f32_slice(&v_new[src..src + active_head_dim]);
        }
    }
    /// Export the initialized prefix into compact
    /// `[layer][head][position][dimension]` payloads.
    ///
    /// Unlike the live allocation, the returned head stride is
    /// `sequence_length * head_dim`; unused capacity is not serialized.
    /// This is a read-only copy and never mutates or aliases the cache.
    pub(crate) fn export_compact_prefix(
        &self,
        sequence_length: usize,
    ) -> Result<(Vec<f16>, Vec<f16>), String> {
        if sequence_length != self.cursor {
            return Err(format!(
                "snapshot sequence length {sequence_length} does not match cache cursor {}",
                self.cursor
            ));
        }
        if sequence_length > self.max_seq_len {
            return Err(format!(
                "snapshot sequence length {sequence_length} exceeds cache capacity {}",
                self.max_seq_len
            ));
        }
        let compact_head = sequence_length
            .checked_mul(self.head_dim)
            .ok_or_else(|| "compact KV head stride overflow".to_string())?;
        let compact_len = self
            .n_layers
            .checked_mul(self.n_kv_heads)
            .and_then(|count| count.checked_mul(compact_head))
            .ok_or_else(|| "compact KV payload length overflow".to_string())?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(compact_len)
            .map_err(|error| format!("cannot allocate compact key payload: {error}"))?;
        values
            .try_reserve_exact(compact_len)
            .map_err(|error| format!("cannot allocate compact value payload: {error}"))?;
        for layer in 0..self.n_layers {
            let layer_offset = self.layer_offset(layer);
            for head in 0..self.n_kv_heads {
                let start = layer_offset + head * self.max_seq_len * self.head_dim;
                let end = start + compact_head;
                keys.extend_from_slice(&self.k[start..end]);
                values.extend_from_slice(&self.v[start..end]);
            }
        }
        Ok((keys, values))
    }

    /// Restore compact prefix payloads without f16 -> f32 -> f16 conversion.
    ///
    /// This is restricted to the snapshot layer so external callers cannot
    /// bypass compatibility validation. The copy owns its destination and
    /// therefore never aliases snapshot memory.
    pub(crate) fn import_compact_prefix(
        &mut self,
        sequence_length: usize,
        keys: &[f16],
        values: &[f16],
    ) -> Result<(), String> {
        if sequence_length > self.max_seq_len {
            return Err(format!(
                "snapshot sequence length {sequence_length} exceeds cache capacity {}",
                self.max_seq_len
            ));
        }
        let compact_head = sequence_length
            .checked_mul(self.head_dim)
            .ok_or_else(|| "compact KV head stride overflow".to_string())?;
        let expected = self
            .n_layers
            .checked_mul(self.n_kv_heads)
            .and_then(|count| count.checked_mul(compact_head))
            .ok_or_else(|| "compact KV payload length overflow".to_string())?;
        if keys.len() != expected || values.len() != expected {
            return Err(format!(
                "compact KV payload length mismatch: expected {expected} elements each, got keys={} values={}",
                keys.len(),
                values.len()
            ));
        }
        let mut source = 0usize;
        for layer in 0..self.n_layers {
            let layer_offset = self.layer_offset(layer);
            for head in 0..self.n_kv_heads {
                let destination = layer_offset + head * self.max_seq_len * self.head_dim;
                self.k[destination..destination + compact_head]
                    .copy_from_slice(&keys[source..source + compact_head]);
                self.v[destination..destination + compact_head]
                    .copy_from_slice(&values[source..source + compact_head]);
                source += compact_head;
            }
        }
        self.cursor = sequence_length;
        Ok(())
    }

    pub fn get(&self, layer: usize) -> (&[f16], &[f16]) {
        let layer_offset = self.layer_offset(layer);
        let len = self.layer_stride();
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
        )
    }

    pub fn get_with_scratch(&mut self, layer: usize) -> (&[f16], &[f16], &mut Vec<f32>) {
        let layer_offset = self.layer_offset(layer);
        let len = self.layer_stride();
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
            &mut self.qk_scratch,
        )
    }

    /// Number of layer slabs in the cache.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Assert that a caller's absolute position agrees with cache state.
    /// Keeping two independent cursors without this check can silently
    /// overwrite or skip K/V positions.
    pub fn validate_start_pos(&self, start_pos: usize) {
        assert_eq!(
            start_pos, self.cursor,
            "kv cache start_pos {start_pos} does not match cursor {}",
            self.cursor
        );
    }

    /// maximum sequence length the cache was allocated for
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// bytes reserved for K and V storage, excluding the small score scratch.
    #[cfg(test)]
    fn storage_bytes(&self) -> usize {
        self.k
            .capacity()
            .saturating_add(self.v.capacity())
            .saturating_mul(core::mem::size_of::<f16>())
    }
    pub fn advance_cursor(&mut self) {
        assert!(
            self.cursor < self.max_seq_len,
            "kv cache cursor overflow: cursor={}, max_seq_len={}",
            self.cursor,
            self.max_seq_len
        );
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

    fn layer_stride(&self) -> usize {
        self.n_kv_heads
            .checked_mul(self.max_seq_len)
            .and_then(|stride| stride.checked_mul(self.head_dim))
            .expect("kv cache layer stride overflow")
    }

    fn layer_offset(&self, layer: usize) -> usize {
        assert!(layer < self.n_layers, "kv cache layer out of bounds");
        layer
            .checked_mul(self.layer_stride())
            .expect("kv cache layer offset overflow")
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
        assert_eq!(
            cache.storage_bytes(),
            2 * 2 * 4 * 128 * 8 * core::mem::size_of::<f16>()
        );
        assert_eq!(k_out[0].to_f32(), 1.0);
        assert_eq!(v_out[0].to_f32(), 2.0);
    }

    #[test]
    fn append_with_layout_supports_layers_with_fewer_heads() {
        let mut cache = KVCache::new(1, 4, 8, 2);
        cache.append_with_layout(0, 0, &[1.0; 8], &[2.0; 8], 2, 4);
        let (k, v) = cache.get(0);
        assert_eq!(k[0].to_f32(), 1.0);
        assert_eq!(k[4].to_f32(), 0.0);
        let second_head = 2 * 8;
        assert_eq!(k[second_head].to_f32(), 1.0);
        assert_eq!(v[second_head].to_f32(), 2.0);
        let inactive_head = 2 * 2 * 8;
        assert_eq!(k[inactive_head].to_f32(), 0.0);
    }

    #[test]
    #[should_panic(expected = "does not match cursor")]
    fn start_position_must_match_cursor() {
        KVCache::new(1, 1, 1, 2).validate_start_pos(1);
    }
}

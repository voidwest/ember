use alloc::vec::Vec;
use half::{f16, slice::HalfFloatSliceExt};

/// Physical storage requested by one model layer. Shared layers read an
/// earlier layer's owner slab and never write a second copy.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvLayerLayout {
    Owned { n_kv_heads: usize, head_dim: usize },
    Shared { source_layer: usize },
}

#[derive(Clone, Copy)]
struct LayerAllocation {
    offset: usize,
    owner: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

/// a flat, pre-allocated key/value cache for transformer attention.
///
/// Each owner slab is `[head][seq_position][head_dim]`. Ordinary models
/// use uniform `[layer][head][seq_position][head_dim]` storage; heterogeneous
/// models can allocate different slab shapes and alias earlier owners.
/// wired into `Attention::forward_with_cache` - during prefill the full
/// k/v projection is cached; subsequent decode steps read from the cache
/// instead of recomputing against the full sequence each pass.
///
/// `Clone` exists for session-level provisional inference: a scratch copy
/// lets speculative prefill/decode run without touching the committed
/// cache. Clones are independent; bytes beyond a rolled-back cursor are
/// never read (see [`KVCache::truncate_to`]).
#[derive(Clone)]
pub struct KVCache {
    /// Key owner slabs, each laid out as [head][pos][head_dim].
    k: Vec<f16>,
    /// Value owner slabs with the same offsets and shapes as the keys.
    v: Vec<f16>,
    /// number of cache layers
    n_layers: usize,
    /// Absent for the original uniform layout. Heterogeneous models retain
    /// one small descriptor per logical layer, including shared aliases.
    layers: Option<Vec<LayerAllocation>>,
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

        Self::allocate(n_layers, n_kv_heads, head_dim, max_seq_len, len, None)
    }

    pub(crate) fn try_new_per_layer(
        layouts: &[KvLayerLayout],
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if layouts.is_empty() || max_seq_len == 0 {
            return Err("KV cache requires layers and a positive sequence length".into());
        }
        let mut layers: Vec<LayerAllocation> = Vec::new();
        layers
            .try_reserve_exact(layouts.len())
            .map_err(|error| format!("cannot allocate KV layer metadata: {error}"))?;
        let mut len = 0usize;
        let mut max_heads = 0;
        let mut max_dim = 0;
        for (layer, layout) in layouts.iter().enumerate() {
            let allocation = match *layout {
                KvLayerLayout::Owned {
                    n_kv_heads,
                    head_dim,
                } => {
                    if n_kv_heads == 0 || head_dim == 0 {
                        return Err(format!("KV layer {layer} requires positive head geometry"));
                    }
                    let offset = len;
                    len = n_kv_heads
                        .checked_mul(max_seq_len)
                        .and_then(|count| count.checked_mul(head_dim))
                        .and_then(|count| len.checked_add(count))
                        .ok_or_else(|| "KV layer allocation size overflow".to_string())?;
                    LayerAllocation {
                        offset,
                        owner: layer,
                        n_kv_heads,
                        head_dim,
                    }
                }
                KvLayerLayout::Shared { source_layer } => {
                    if source_layer >= layer {
                        return Err(format!("KV layer {layer} must share an earlier layer"));
                    }
                    layers[source_layer]
                }
            };
            max_heads = max_heads.max(allocation.n_kv_heads);
            max_dim = max_dim.max(allocation.head_dim);
            layers.push(allocation);
        }
        if layers.iter().enumerate().all(|(layer, allocation)| {
            allocation.owner == layer
                && allocation.n_kv_heads == max_heads
                && allocation.head_dim == max_dim
        }) {
            return Self::try_new(layouts.len(), max_heads, max_dim, max_seq_len);
        }
        Self::allocate(
            layouts.len(),
            max_heads,
            max_dim,
            max_seq_len,
            len,
            Some(layers),
        )
    }

    fn allocate(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        len: usize,
        layers: Option<Vec<LayerAllocation>>,
    ) -> Result<Self, String> {
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
            layers,
            n_kv_heads,
            qk_scratch,
            head_dim,
            max_seq_len,
            cursor: 0,
        })
    }

    pub fn append(&mut self, layer: usize, pos: usize, k_new: &[f32], v_new: &[f32]) {
        self.append_with_head_dim(layer, pos, k_new, v_new, self.layer_head_dim(layer));
    }

    pub fn append_with_head_dim(
        &mut self,
        layer: usize,
        pos: usize,
        k_new: &[f32],
        v_new: &[f32],
        active_head_dim: usize,
    ) {
        self.append_with_layout(
            layer,
            pos,
            k_new,
            v_new,
            self.layer_n_kv_heads(layer),
            active_head_dim,
        );
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
        let allocation = self.layer_allocation(layer);
        assert_eq!(
            allocation.owner, layer,
            "shared KV layers cannot be written"
        );
        assert!(
            active_kv_heads > 0,
            "kv cache requires at least one active head"
        );
        assert!(active_kv_heads <= allocation.n_kv_heads);
        assert!(
            active_head_dim > 0,
            "kv cache requires a non-zero head dimension"
        );
        assert!(active_head_dim <= allocation.head_dim);
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

        let layer_offset = allocation.offset;
        let seq_offset = pos
            .checked_mul(allocation.head_dim)
            .expect("kv cache sequence offset overflow");

        for h in 0..active_kv_heads {
            let head_offset = h
                .checked_mul(self.max_seq_len)
                .and_then(|offset| offset.checked_mul(allocation.head_dim))
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
        self.require_uniform_snapshot()?;
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
        self.require_uniform_snapshot()?;
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
        let allocation = self.layer_allocation(layer);
        let layer_offset = allocation.offset;
        let len = allocation.n_kv_heads * self.max_seq_len * allocation.head_dim;
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
        )
    }

    pub fn get_with_scratch(&mut self, layer: usize) -> (&[f16], &[f16], &mut Vec<f32>) {
        let allocation = self.layer_allocation(layer);
        let layer_offset = allocation.offset;
        let len = allocation.n_kv_heads * self.max_seq_len * allocation.head_dim;
        (
            &self.k[layer_offset..layer_offset + len],
            &self.v[layer_offset..layer_offset + len],
            &mut self.qk_scratch,
        )
    }

    /// Number of logical model layers, including layers sharing storage.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Uniform head dimension, or the maximum for a heterogeneous cache.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Physical head width used by this layer's cache slab.
    pub fn layer_head_dim(&self, layer: usize) -> usize {
        self.layer_allocation(layer).head_dim
    }

    pub fn layer_n_kv_heads(&self, layer: usize) -> usize {
        self.layer_allocation(layer).n_kv_heads
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
    pub fn storage_bytes(&self) -> usize {
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

    /// Roll the write/read cursor back to `pos`, logically discarding every
    /// position >= `pos`.
    ///
    /// Safety contract (session cancellation / provisional rollback):
    /// attention reads exactly `[0, cursor + seq_len)` (see the
    /// `total_seq_len` computation in `LlamaBlock::forward_with_cache`), so
    /// stale bytes beyond the rolled-back cursor are never read and are
    /// overwritten by subsequent appends. Positions are explicit, so no
    /// other state needs clearing. Debug-asserted monotonicity keeps the
    /// API from silently "rewinding" into a shorter-than-requested prefix.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(
            pos <= self.cursor,
            "kv cache truncate_to({pos}) cannot exceed cursor {}",
            self.cursor
        );
        debug_assert!(pos <= self.max_seq_len);
        self.cursor = pos;
    }

    /// Uniform KV head count, or the maximum for a heterogeneous cache.
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
        self.layer_allocation(layer).offset
    }

    fn layer_allocation(&self, layer: usize) -> LayerAllocation {
        assert!(layer < self.n_layers, "kv cache layer out of bounds");
        if let Some(layers) = &self.layers {
            return layers[layer];
        }
        LayerAllocation {
            offset: layer
                .checked_mul(self.layer_stride())
                .expect("kv cache layer offset overflow"),
            owner: layer,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }

    fn require_uniform_snapshot(&self) -> Result<(), String> {
        if self.layers.is_some() {
            return Err("KV snapshot v1 requires uniform non-shared geometry; per-layer caches need a future schema".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heterogeneous_cache(capacity: usize) -> KVCache {
        KVCache::try_new_per_layer(
            &[
                KvLayerLayout::Owned {
                    n_kv_heads: 2,
                    head_dim: 4,
                },
                KvLayerLayout::Owned {
                    n_kv_heads: 1,
                    head_dim: 8,
                },
                KvLayerLayout::Shared { source_layer: 0 },
            ],
            capacity,
        )
        .unwrap()
    }

    #[test]
    fn heterogeneous_storage_aliases_owners_and_preserves_rollback_and_clone() {
        let mut cache = heterogeneous_cache(5);
        assert_eq!(cache.storage_bytes(), (2 * 4 + 8) * 5 * 4);
        assert_eq!(cache.n_layers(), 3);
        assert_eq!(cache.layer_head_dim(0), 4);
        assert_eq!(cache.layer_head_dim(1), 8);
        assert_eq!(cache.layer_n_kv_heads(1), 1);
        for pos in 0..3 {
            cache.append(0, pos, &[pos as f32 + 1.0; 8], &[2.0; 8]);
            cache.append(1, pos, &[pos as f32 + 10.0; 8], &[20.0; 8]);
            cache.advance_cursor();
        }
        assert_eq!(cache.get(0).0.as_ptr(), cache.get(2).0.as_ptr());
        assert_eq!(cache.get(0).1.as_ptr(), cache.get(2).1.as_ptr());
        let original = cache.clone();
        assert_ne!(cache.get(0).0.as_ptr(), original.get(0).0.as_ptr());
        cache.truncate_to(1);
        cache.append(0, 1, &[9.0; 8], &[8.0; 8]);
        cache.append(1, 1, &[7.0; 8], &[6.0; 8]);
        cache.advance_cursor();
        assert_eq!(cache.get(2).0[4].to_f32(), 9.0);
        assert_eq!(cache.get(1).0[8].to_f32(), 7.0);
        assert_eq!(original.get(0).0[4].to_f32(), 2.0);
        assert_eq!(original.cursor(), 3);
        assert_eq!(cache.cursor(), 2);
        cache.reset();
        assert_eq!(cache.cursor(), 0);
    }

    #[test]
    fn heterogeneous_geometry_rejects_malformed_owners_and_uniform_snapshots() {
        for layouts in [
            vec![],
            vec![KvLayerLayout::Shared { source_layer: 0 }],
            vec![KvLayerLayout::Owned {
                n_kv_heads: 0,
                head_dim: 4,
            }],
            vec![KvLayerLayout::Owned {
                n_kv_heads: 1,
                head_dim: 0,
            }],
            vec![KvLayerLayout::Owned {
                n_kv_heads: usize::MAX,
                head_dim: 4,
            }],
        ] {
            assert!(KVCache::try_new_per_layer(&layouts, 2).is_err());
        }
        let mut cache = heterogeneous_cache(2);
        cache.append(0, 0, &[1.0; 8], &[2.0; 8]);
        cache.append(1, 0, &[3.0; 8], &[4.0; 8]);
        cache.advance_cursor();
        let before = cache.get(0).0.to_vec();
        assert!(cache
            .export_compact_prefix(1)
            .unwrap_err()
            .contains("uniform non-shared"));
        assert!(cache
            .import_compact_prefix(1, &[], &[])
            .unwrap_err()
            .contains("uniform non-shared"));
        assert_eq!(cache.get(0).0, before);
        assert_eq!(cache.cursor(), 1);
    }

    #[test]
    fn uniform_layer_layouts_retain_snapshot_roundtrip() {
        let layouts = [KvLayerLayout::Owned {
            n_kv_heads: 2,
            head_dim: 4,
        }; 2];
        let mut cache = KVCache::try_new_per_layer(&layouts, 3).unwrap();
        assert!(cache.layers.is_none());
        for layer in 0..2 {
            cache.append(layer, 0, &[layer as f32 + 1.0; 8], &[3.0; 8]);
        }
        cache.advance_cursor();
        let (keys, values) = cache.export_compact_prefix(1).unwrap();
        let mut imported = KVCache::new(2, 2, 4, 5);
        imported.import_compact_prefix(1, &keys, &values).unwrap();
        assert_eq!(imported.export_compact_prefix(1).unwrap(), (keys, values));
    }

    #[test]
    #[should_panic(expected = "shared KV layers cannot be written")]
    fn shared_layers_reject_duplicate_writes() {
        heterogeneous_cache(2).append(2, 0, &[0.0; 8], &[0.0; 8]);
    }

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

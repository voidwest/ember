# Bugs

## Bug 1 — `KVCache::get()` returns too short a slice

### Symptom

```
thread 'main' panicked at src/model.rs:127:38:
range start index 4992 out of range for slice of length 4608
```

### Cause

`KVCache::append()` stores K/V in a `[layer][head][seq_position][head_dim]` layout where each head occupies `max_seq_len * head_dim` contiguous elements — the stride between heads within a layer is `max_seq_len * head_dim`. `KVCache::get()` was written as if the layout were dense (stride `cursor * head_dim`), returning only `n_heads * cursor * head_dim` elements starting from the layer offset.

For GPT-2 small (12 heads, head_dim=64, max_seq_len=26, cursor=6):

```
Cache buffer per-layer layout (append):
  [Head 0: 26×64 = 1664 elems] [Head 1: 1664 elems] [Head 2: 1664 elems] ...
  ↑ 6 positions valid    ↑ written at cursor=0    ↑ written at cursor=0
```

`get()` returned `self.k[layer_offset..layer_offset + 12×6×64]` = `[0..4608)`. This covers Head 0 (1664 elems), Head 1 (1664 elems), and 1280 elems of Head 2 — only ~2.77 heads of data. When the attention code accessed head 3 at offset `3 × 1664 = 4992`, it sliced past the returned buffer.

### Before

```rust
pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
    let layer_offset = layer * self.n_heads * self.max_seq_len * self.head_dim;
    let len = self.n_heads * self.cursor * self.head_dim;  // ← dense layout
    (&self.k[layer_offset..layer_offset + len], ...)
}
```

### After

```rust
pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
    let layer_offset = layer * self.n_heads * self.max_seq_len * self.head_dim;
    let len = self.n_heads * self.max_seq_len * self.head_dim;  // ← matches append stride
    (&self.k[layer_offset..layer_offset + len], ...)
}
```

### How the fix works

`get()` now returns the full per-layer allocation (`n_heads × max_seq_len × head_dim` elements). The attention code already computes head offsets with `cache_head_stride = max_seq_len * head_dim`, so the returned slice and the access pattern are now consistent. Uninitialised padding positions beyond `cursor` are never read — the attention loop bounds `j` to `max_j ≤ total_seq_len - 1`.

---

## Bug 2 — Cache cursor advanced per-layer instead of per-forward-pass

### Symptom

```
thread 'main' panicked at src/kv_cache.rs:37:9:
kv cache overflow: max_seq_len=26
```

### Cause

`Attention::forward_with_cache()` called `cache.advance_cursor()` inside its per-position loop. But this method is invoked once per layer (12 times for GPT-2 small). Each forward pass of a 6-token prompt advanced the cursor 12 × 6 = 72 times instead of 6.

The cursor is global across all layers — it tracks how many token positions have been stored, not how many K/V writes have happened. All layers should store their K/V at the same absolute sequence position for a given token, then the cursor advances once per token.

### Before

In `Attention::forward_with_cache` (`src/model.rs`):

```rust
for pos in 0..seq_len {
    let offset = pos * embed_dim;
    cache.append(layer, &k_data[offset..offset + embed_dim], &v_data[offset..offset + embed_dim]);
    cache.advance_cursor();  // ← advanced 12× per token (once per layer)
}
```

### After

`Attention::forward_with_cache` — removed the advancements:

```rust
// (cursor advances after all layers have stored, in Gpt2::forward_with_cache)
for pos in 0..seq_len {
    let offset = pos * embed_dim;
    cache.append(layer, &k_data[offset..offset + embed_dim], &v_data[offset..offset + embed_dim]);
}
```

`Gpt2::forward_with_cache` — advancements added here, after the per-layer loop:

```rust
let seq_len = token_ids.len();
let mut x = self.embed_with_offset(backend, token_ids, start_pos)?;
for (layer, block) in self.blocks.iter().enumerate() {
    x = block.forward_with_cache(backend, &x, cache, layer)?;
}
// Advance the cache cursor by seq_len after all layers have
// stored their K/V for these positions.
for _ in 0..seq_len {
    cache.advance_cursor();
}
let x = self.ln_f.forward(backend, &x)?;
self.head.forward(backend, &x)
```

### Cascading adjustment

Since cursor advancement moved out of the attention layer, `total_seq_len` needed to account for the not-yet-advanced cursor:

```diff
- let total_seq_len = cache.cursor();
+ let total_seq_len = cache.cursor() + seq_len;
```

### How the fix works

Cursor advancement moved from `Attention::forward_with_cache` (called per-layer) to `Gpt2::forward_with_cache` (called once per forward pass). All 12 layers append their K/V at the same absolute sequence positions (starting at `cursor`), then the cursor advances by `seq_len` once. This keeps the cursor in sync with the actual number of stored token positions.

---

## Bug 3 — `test_kv_cache` assertion used wrong length after Bug 1 fix

### Symptom

```
---- kv_cache::tests::test_kv_cache stdout ----
thread 'kv_cache::tests::test_kv_cache' panicked at src/kv_cache.rs:94:9:
assertion `left == right` failed
  left: 4096
 right: 32
```

### Cause

When Bug 1 was fixed — changing `KVCache::get()` to return the full per-layer allocation
(`n_heads × max_seq_len × head_dim` elements) — the unit test was not updated to match.
The test created a cache with `KVCache::new(2, 4, 8, 128)` (4 heads, head_dim=8, max_seq_len=128)
and expected `get()` to return only the cursor-occupied portion: `4 × 1 × 8 = 32` elements.
After the fix, `get()` returns the full pre-allocated buffer: `4 × 128 × 8 = 4096` elements.

The test assertion was wrong — it encoded the old Bug 1 behaviour as if it were correct.

### Before

```rust
let (k_out, v_out) = cache.get(0);
assert_eq!(k_out.len(), 4 * 1 * 8);  // 32 — old dense-layout expectation
assert_eq!(v_out.len(), 4 * 1 * 8);
```

### After

```rust
let (k_out, v_out) = cache.get(0);
assert_eq!(k_out.len(), 4 * 128 * 8);  // 4096 — matches full per-layer buffer
assert_eq!(v_out.len(), 4 * 128 * 8);
```

### How the fix works

The test now asserts against `n_heads × max_seq_len × head_dim`, which matches the
actual return size of `get()` after the Bug 1 fix. The attention code bounds reads to
`cursor`-relevant positions via the `j` loop bounds, so the larger slice is safe.

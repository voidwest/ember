# Bugs

## Bug 1 — `KVCache::get()` returns too short a slice

**Symptom**: panic — range out of bounds when accessing cached K/V during attention.

**Cause**: `get()` computed the slice length as `n_heads * cursor * head_dim` (dense layout), but the cache uses a fixed pre-allocation with stride `max_seq_len * head_dim` between heads. After enough positions were written, head offsets past the first ~2.77 heads fell outside the returned slice.

**Fix**: changed `get()` to return the full per-layer allocation (`n_heads * max_seq_len * head_dim`). The attention code bounds its reads by `max_j ≤ total_seq_len - 1`, so padding positions beyond `cursor` are never accessed.

---

## Bug 2 — Cache cursor advanced per-layer instead of per-forward-pass

**Symptom**: cache overflow panic (`cursor > max_seq_len`) after a few generated tokens.

**Cause**: `Attention::forward_with_cache` (called once per layer, 12× for GPT-2 small) advanced `cache.cursor` inside its per-position loop. Each token advanced the cursor 12 times instead of once.

**Fix**: moved cursor advancement out of `Attention::forward_with_cache` into `Gpt2::forward_with_cache`, after the per-layer loop. Cursor now advances by exactly `seq_len` per forward pass. Adjusted `total_seq_len` to `cache.cursor() + seq_len` since cursor hasn't yet been advanced when attention runs.

---

## Bug 3 — `test_kv_cache` assertion encoded Bug 1's wrong behavior

**Symptom**: unit test failure after Bug 1 was fixed — assertion expected `get()` to return 32 elements but it returned 4096.

**Cause**: the test was written against the old (broken) dense-layout length. It expected `n_heads * cursor * head_dim` elements.

**Fix**: updated the assertion to `n_heads * max_seq_len * head_dim`, matching the corrected `get()`.

---

## Bug 4 — Q8_0 and F16 tensors loaded in wrong memory layout

**Symptom**: every prompt produced complete gibberish even at temperature 0. llama.cpp with the same GGUF file produced coherent text.

**Cause**: GGUF stores Q8_0 (and by convention F16) tensors in column-major order so the innermost dimension is a multiple of 32 (Q8_0 block requirement). The tensor info header reports the logical shape (e.g. `[768, 50257]`), but the dequantized flat buffer is laid out column-major. ember reshaped with logical dims in row-major order, scrambling every weight matrix. Verified: row-major reshape → correlation 0.004 with PyTorch; column-major → 0.999975.

**Fix**: after dequantizing Q8_0 / converting F16, reverse the `dims` vec in the loader so the reshape matches the column-major storage layout. Then adjusted transposes in `Gpt2::from_loader` — embeddings get no manual transpose (loader already reversed dims → index_select picks rows directly), while linear weights get a `.transpose()` to restore the expected `[in_features, out_features]` for matmul.

---

## Bug 5 — KV cache prefill overwrite (2026-05-08, commit `1c420e1`)

### Before

```
$ cargo run -- --prompt "1, 2, 3 ,4 5" -t 0.3
/5/5/5/5/5/5/5/5/5/5

$ cargo run -- --prompt "1, 2, 3 ,4 5" -t 1.0
 Yes Yes Age 18-40-50 10-50 age 10 21-50 age 21-50
```

### After

```
$ cargo run -- --prompt "1, 2, 3 ,4 5" -t 0.0
, 6, 7, 8, 9, 10, 11, 12,

$ cargo run -- --prompt "1, 2, 3 ,4 5" -t 0.3
, 6, 7, 8, 9, 10, 11, 12,
```

### What went wrong

`KVCache::append` wrote every entry at `self.cursor`, but `cursor` is deliberately not advanced until `Gpt2::forward_with_cache` finishes all layers (see Bug 2). During prefill, `Attention::forward_with_cache` called `append` once per prompt token in a loop — every call landed at `cursor = 0`, each overwriting the last. Only the final prompt token's K/V survived at cache slot 0; all other slots remained zero. The model could attend to at most one prompt token, producing nonsense.

### How it was fixed

`KVCache::append` now takes an explicit `pos: usize` parameter instead of reading `self.cursor`. `Attention::forward_with_cache` snapshots `cache.cursor()` before the loop and passes `cursor + pos`, so each token in the batch lands at its correct absolute cache slot. The cursor is still batch-advanced afterward in `Gpt2::forward_with_cache` — that part was already correct from Bug 2's fix.

---

## Bug 6 — `cargo fmt` CI failure (2026-05-08, commit `6bbad15`)

**Symptom**: `cargo fmt -- --check` failed on GitHub CI.

**Cause**: Two formatting violations slipped through:

1. `src/backend.rs:44` — `load_from_cpu` trait method signature was split across 4 lines; `rustfmt` wanted it on a single line.
2. `src/kv_cache.rs:119` — missing trailing newline at end of file.

**Fix**:

- Collapsed `load_from_cpu(&self, data: Vec<f32>, shape: &[usize])` onto one line.
- Added trailing `\n` to `src/kv_cache.rs`.

**Prevention**: Run `cargo fmt` before pushing. CI already checks this; a pre-push hook would catch it earlier.

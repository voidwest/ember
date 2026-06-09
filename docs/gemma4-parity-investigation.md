# Gemma 4 GGUF Parity Investigation

## Summary

- LLaMA and Qwen3 golden-logit parity are at cosine 0.9995+.
- Gemma 4 after structural fixes produces cosine ~0.87 against llama.cpp reference, coherent
  output, 26 tests pass.
- The remaining cosine gap (~0.13) is not a known structural mismatch.
- Current evidence points to small numerical drift introduced per-layer being amplified by
  RMSNorm weights, especially at global attention layers.
- L0 `attn_norm` is bit-identical between Ember and llama.cpp (cosine 1.000, identical
  floating-point values). The pipeline starts correctly and diverges gradually.

## Baseline Results

| Model   | Token IDs consistent<br>with reference | Top-1 match | Cosine   | Notes                            |
|---------|---------------------------------------|-------------|----------|----------------------------------|
| LLaMA   | yes (copied from llama.cpp metadata)  | yes         | ~0.9995  | Golden-logit parity achieved     |
| Qwen3   | yes (copied from llama.cpp metadata)  | yes         | ~0.9998  | Golden-logit parity achieved     |
| Gemma 4 (initial) | yes (same reference tokens)  | no          | 0.18     | Flat logits; Arabic prompt produced multilingual gibberish |
| Gemma 4 (after fixes) | yes (same reference tokens) | no     | 0.87     | Coherent English output; Arabic still diverges |

## Fixed Issues

### PLE projection shape/orientation

Gemma per-layer embedding (PLE) pathway uses two linear projections per block:

- `blk.{i}.inp_gate.weight` — shape `[1536, 256]`, projects hidden `1536 -> 256` (no transpose).
- `blk.{i}.proj.weight` — shape `[256, 1536]`, projects gated PLE `256 -> 1536` (no transpose).

Previously `blk.{i}.proj.weight` was loaded with `get_linear` (transposed), which would cause a
dimension mismatch at runtime if the PLE path were exercised. Changed to `get_linear_no_transpose`.

### Final softcap metadata parsing

- GGUF key: `gemma4.final_logit_softcapping`
- Value: `30.0`
- Previously hardcoded to `Some(15.0)`, halving the effective softcap range.

### Tied logits

Gemma GGUF contains `token_embd.weight` with no separate `output.weight`. The LM head uses tied
embeddings:

```
hidden -> output_norm -> token_embd tied logits -> final softcap(30.0)
```

### Block layout / structural alignment

Ember's `forward_with_cache` now matches llama.cpp's graph build order:

1. `attn_norm` → self-attention → `post_attn_norm` → residual add
2. `ffn_norm` → FFN (gate/gelu/up/down) → `post_ffn_norm` → residual add
3. PLE (inp_gate/gelu/multiply/proj/post_ple_norm) → residual add
4. Multiply by `layer_output_scale`

Additional structural fixes:

- Token embeddings scaled by `sqrt(1536)` (matching llama.cpp).
- Global PLE projection (`per_layer_model_proj` [1536, 8960] + `per_layer_proj_norm` [256])
  combined with raw PLE lookup, scaled by `1/sqrt(2)`.
- `layer_output_scale` (geometric mean ~0.42) applied per block.
- GELU in both MLP and PLE uses tanh approximation (matching `ggml_gelu`).
- RoPE `freq_factors` from `rope_freqs.weight` (64 × 1.0, 192 × 1e30) applied to gate
  frequency pairs.

### BF16 loading

BF16 tensor support added in `src/loader.rs` for tensor type 30:

```rust
30 => {
    let mut buf = vec![0u8; element_count * 2];
    reader.read_exact(&mut buf)?;
    let mut data = vec![0.0f32; element_count];
    for (i, dst) in data.iter_mut().enumerate().take(element_count) {
        let start = i * 2;
        let bits = u16::from_le_bytes(buf[start..start + 2].try_into()?);
        *dst = f32::from_bits((bits as u32) << 16);
    }
    LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
}
```

Required for `per_layer_model_proj.weight` which is stored as BF16 in the GGUF.

### RoPE freq_factors

`compute_rope_freqs` in `src/tensor.rs` now accepts optional `freq_factors`:

```rust
pub fn compute_rope_freqs(
    max_seq_len: usize,
    head_dim: usize,
    theta_base: f32,
    freq_factors: Option<&[f32]>,
) -> (CpuTensor, CpuTensor)
```

- LLaMA passes `None`.
- Gemma local and global RoPE both pass `rope_freqs.as_deref()`.
- Frequency pairs where factor > 1e10 get `freq = 0` (identity rotation).

## Rejected / Non-Final Hypotheses

| Hypothesis | Change tested | Result | Status |
|------------|--------------|--------|--------|
| PLE disabled | Remove PLE from block | cosine 0.08 (fail) | Rejected |
| Softcap disabled | Remove final softcap | Minimal change | Rejected |
| PLE at end of block | Move PLE after FFN residual | cosine 0.10 (fail) | Rejected — correct placement confirmed |
| PLE at start of block | Move PLE before attention | cosine 0.72 | Accepted — matches llama.cpp |
| Embedding scaling disabled | Remove `sqrt(1536)` | cosine 0.86 | Minor — kept for completeness |
| Layer output scale disabled | Remove per-layer scalar | cosine -0.54 (fail) | Rejected — essential |
| V unweighted RMS norm | Normalize V before cache | cosine 0.70 (drop) | Rejected — degrades results |
| Wrong RMS norm formula | Compared SIMD vs scalar | Identical | Rejected — correct |
| Wrong RMS norm weights | Dumped weights vs GGUF | Cosine 1.0, diff 0.0 | Rejected — correct |
| Q8_0 dequantization | F32 MLP vs Q8_0 MLP | Identical cosine | Rejected — correct |
| `sum_squares` AVX2 bug | Forced scalar path | Identical output | Rejected — correct |
| FP non-associativity (root cause) | L0 attn_norm comparison | Bit-identical | Rejected — not the root cause |
| Global layer RoPE | Added freq_factors | Minimal cosine change | Minor improvement |

## Layerwise Comparison Pipeline

### Setup

- llama.cpp binary at `/tmp/dump_llama2` evaluates BOS token and writes per-layer hidden states
  to `/tmp/llama_35layers.bin`.
- Ember `forward_last_logits_with_cache` writes per-layer states to
  `/tmp/ember_35layers.bin`.
- Python comparison script computes layer-by-layer cosine and L2 norms.

### Key Findings

- Layers 0–3 match at cosine 0.99+.
- Divergence is gradual, not a single catastrophic mismatch.
- Global attention layers (every 5th layer: L5, L10, L15, L20, L25, L30) show larger drops.
  Worst: L15 cosine 0.096, L23 cosine 0.031.
- Final L34 cosine: ~0.51 between Ember and llama.cpp hidden states.
- Final logit cosine: ~0.87.

## RMSNorm Verification

### Methodology

1. Dumped Ember's L4 input (block input tensor) and attn_norm weight.
2. Dumped llama.cpp's L4 input and attn_norm output.
3. Applied Python RMSNorm formula `x * 1/sqrt(mean(x^2) + eps) * weight` to both inputs.
4. Compared Python results against Ember and llama.cpp outputs.

### Results

| Comparison | Cosine | L2 (Python) | L2 (Actual) |
|------------|--------|-------------|-------------|
| Python(llama_input) vs llama.cpp L4 attn_norm | 1.000000 | 44.21 | 44.21 |
| Python(ember_input) vs Ember L4 attn_norm | 1.000000 | 28.57 | 28.57 |
| Python(llama_input) vs Python(ember_input) | 0.457198 | 44.21 | 28.57 |
| Ember L0 attn_norm vs llama.cpp L0 attn_norm | 1.000000 | 452.85 | 452.85 |

### Conclusion

- L0 `attn_norm` is bit-identical (same floating-point values).
- RMSNorm formula verified correct in both implementations.
- RMSNorm weights verified identical to GGUF (cosine 1.0, L2 diff 0.0).
- SIMD `sum_squares` verified matching scalar implementation with model data.
- **Small upstream angular differences are amplified by large RMSNorm weights.** Example:
  raw inputs at cosine 0.996 produce RMSNorm outputs at cosine 0.457 because the weight
  vector (RMS 40, max 236) strongly weights certain dimensions.

## Current Interpretation

The remaining gap is **not** attributed to:

- Incorrect RMSNorm formula
- Incorrect RMSNorm weight loading
- Final softcap
- Tied output head
- Obvious tensor orientation bugs
- Rust vs C is not a sufficient explanation by itself: L0 bit-identical output
  shows the implementations can match exactly when the operation boundary is
  aligned. It does not rule out later low-level accumulation differences, but it
  rules out "Rust is inherently wrong" as a blanket explanation.

The current best explanation:

1. The pipeline starts perfectly (L0 attn_norm bit-identical).
2. Small upstream numerical differences — potentially from GELU, matmul accumulation,
   attention scoring, or quantization order — are introduced at each layer.
3. The next layer's RMSNorm weight (with values up to 236) amplifies these micro-differences:
   a 0.4% angular difference in the input becomes a 54% difference in the RMSNorm output.
4. This amplification compounds over 35 layers, producing the 13% final cosine gap.

The specific source of the per-layer differences has not been isolated to a single
operation. Candidates include:

- GELU tanh implementation details (e.g., `f32::tanh` vs `tanhf`, FMA vs separate mul+add)
- Q8_0 matmul block ordering differences
- Attention scoring accumulation order
- Softmax numerical path
- Global attention layer numerical sensitivity (head_dim=512, larger RoPE theta)

## Current Test Status

- `cargo test --lib`: 26 passed, 0 failed, 2 ignored (benchmarks)
- Build: 0 warnings (after removing debug code)

## Files Changed

| File | Change |
|------|--------|
| `src/gemma4.rs` | Block layout, PLE pathway (inp_gate/gelu/proj/norm), global PLE projection, embedding scaling `sqrt(1536)`, per-layer `layer_output_scale`, GELU tanh, RoPE `freq_factors` for local and global layers, BF16 `per_layer_model_proj` loading |
| `src/tensor.rs` | `compute_rope_freqs` accepts optional `freq_factors: Option<&[f32]>` |
| `src/llama.rs` | Passes `None` for `freq_factors` |
| `src/loader.rs` | BF16 (type 30) tensor loading |
| `src/simd.rs` | `sum_squares_simd_matches_scalar` test |

## Next Steps

1. Dump L4 intermediate tensors from both implementations: attn_norm input, gate_proj output,
   GELU output, up_proj output, element-wise product, down_proj output.
2. Compare the FFN intermediate at the point where cosine first drops below 0.99.
3. Identify whether the drop comes from GELU precision, matmul precision, or accumulation
   order.
4. Add regression tests for the fixed structural issues (PLE projection shape, softcap
   value, block layout, embedding scaling).

## Appendix: Key Commands

### Run Gemma comparison

```sh
# Ember
target/release/ember --model gemma-4-E2B-it-Q8_0.gguf --arch gemma4 \
  --prompt "Hello world" --max-seq-len 128 --temperature 0 --dump-logits /tmp/t.npy

# Compare with live llama.cpp
python3 -c "
import numpy as np
e = np.load('/tmp/t.npy')[0]
r = np.load('/tmp/llamacpp_live.npz')['logits'][0]
cos = np.dot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r))
print(f'cosine={cos:.6f}')
"
```

### Dump per-layer states from llama.cpp

```sh
cd /tmp/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama -j4
g++ -std=c++17 -I./include -I./ggml/include -I./src \
  /tmp/dump_llama2.cpp \
  ./build/src/libllama.a ./build/ggml/src/libggml.a \
  ./build/ggml/src/libggml-base.a ./build/ggml/src/libggml-cpu.a \
  -lpthread -ldl -lm -o /tmp/dump_llama2

LLAMA_LOG=-1 /tmp/dump_llama2 gemma-4-E2B-it-Q8_0.gguf
# Produces /tmp/llama_l4_dump.bin with per-layer states (requires source patches)
```

### Dump per-layer states from Ember

Requires adding per-layer collection code to `forward_last_logits_with_cache` and writing to
`/tmp/ember_35layers.bin`. See commit history for the debug patch.

### Run tests

```sh
cargo test --lib
```

### GGML standalone RMSNorm test

```sh
cd /tmp/llama.cpp
g++ -std=c++17 -I./ggml/include -I./ggml/src -I./ggml/src/ggml-cpu \
  /tmp/test_ggml.cpp \
  ./build/ggml/src/libggml.a ./build/ggml/src/libggml-base.a \
  ./build/ggml/src/libggml-cpu.a \
  -lpthread -ldl -lm -o /tmp/test_ggml
/tmp/test_ggml
```

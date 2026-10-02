# Ember architecture and design

Internals, design notes, and optimization context.
Moved from the top-level README.

## what this demonstrates

- **systems programming in rust**: manual memory layout for the kv cache
  (`[layer][head][pos][head_dim]`), explicit stride math for tensor indexing,
  and scoped allocations that can be profiled and optimized directly.
- **generic backend architecture**: the transformer is written against a
  `Backend` trait - the same model code works on cpu today and could run
  on gpu tomorrow without modification.
- **ml fundamentals**: causal multi-head attention with kv caching,
  numerically stable softmax (handles all-masked rows), layer norm,
  gelu activation, top-k/top-p sampling.
- **file format parsing**: gguf v3 loader for f32, f16, bf16, q8_0 and the
  k-family quantizations; the [support matrix](support.md) says which of
  them run compressed and which fall back to f32.
- **memory-conscious inference**: q8_0 weights stay mapped in their compressed
  representation, tied q8_0 embeddings can be reused directly as the LM head,
  and K/V state is stored as f16.
- **edge case handling**: uniform fallback when every logit is -inf,
  categorical sampling with inverse cdf, nucleus cutoff logic.


## architecture

the crate is one library (`src/lib.rs`) plus the `ember` binary
(`src/main.rs`). `main.rs` parses the command line and dispatches to the
`cli_*.rs` modules. Those and the two consoles belong to the binary; the
engine, kernels and research layer live in the library.

text generation goes through `cli_generation.rs` -> `generate()`, a generic
`ForwardModel` path used by gpt-2, llama/qwen, and gemma 4. it runs a
two-phase loop:

1. **prefill** - forward pass on the full prompt, populating the kv cache.
2. **decode** - one token at a time, reading from the cache.

shared model primitives live in `src/model.rs` (`ForwardModel`, `Linear`, and
the gpt-2 blocks). llama/qwen lives in `src/llama.rs`, gemma 4 lives in
`src/gemma4.rs`, tensors are `CpuTensor` in `src/tensor.rs`, and the gguf
parser is `src/loader.rs`.

the modules group into layers. the list names the owning files, not every
file; each module's own `//!` header is the authoritative description.

```text
inference engine
  loader.rs            gguf v3 parser, mmap-backed weights, hostile-input limits
  model.rs             shared model primitives + gpt-2 transformer
  llama.rs             llama/qwen transformer (reference and fast decode paths)
  gemma4.rs            dense text-only gemma 4 transformer
  backend.rs           backend trait + cpu backend impl
  tensor.rs            row-major f32 tensor, rope, silu, elemul
  kv_cache.rs          flat f16 k/v cache, gqa-aware (n_kv_heads)
  sampler.rs           temperature, top-k, top-p sampling
  tokenizer.rs         huggingface tokenizer wrapper

quantized kernels
  quant.rs             q8_0 packing + owned/mmap-backed QuantizedWeight
  simd.rs              q8_0 matmul and f32/f16 kernels (avx2, avx-512, neon)
  q8_gemm.rs           tiled multi-row q8_0 gemm for prefill (aarch64)
  attention_kernels.rs multi-row cached-attention kernels for prefill (aarch64)
  quant_k.rs           k-family super-block dequantization
  k_quant_matmul.rs    q4_k/q6_k x q8_k compressed matmul (scalar, avx2, neon)
  packed_cache.rs      on-disk cache of packed q8_0 decode layouts

planned execution (docs/v04-execution-contract.md)
  plan.rs, plan_build.rs   execution-plan types and construction
  planned_decode.rs        the planned-decode interpreter
  runtime_schedule.rs      runtime-only schedule derived from a plan

research layer (docs/v05-research-contract.md)
  v05/                 experiment specs, hook sites, captures, interventions,
                       the runner, bundle writer and offline verifier, plus
                       sweeps, logit lens, steering, attribution and the
                       probe bridge
  experiments/         execution hooks and the v0.2 capture/patch experiments
  artifact/, compare.rs, extraction.rs   v0.2 activation artifacts
  kv_snapshot.rs, kv_compare.rs, kv_diagnostics.rs   kv-prefix snapshots

consoles
  gui_native.rs, gui_native/   native console (`ember gui`, `gui` feature)
  gui.rs, gui_page.html        browser console (`ember web-gui`)
  app_store.rs                 persisted console state

separate tracks (see the README's "related tools and research")
  agent/               tool protocols, approval gating and traces
  multimodal/, smolvlm.rs, smolvlm_video.rs, ultravox.rs   image/video/audio input
  tts/, duplex.rs      speech output and duplex audio (device i/o needs `audio`)
  diff_corpus.rs, diff_outcome.rs, subprocess.rs, quant_fault.rs   EmberSEC harnesses

outside the crate
  probes/              python probe scripts (linear, cca, rsa, divergence)
  bindings/python/     optional PyO3 binding over the same core
```


## design notes

- **backend trait**: the transformer is generic - `CpuBackend` is the default,
  but any type implementing `Backend` works. the trait abstracts linear ops,
  element-wise math, layer norm, attention, and tensor lifecycle. the current
  CPU attention backend uses runtime-dispatched SIMD helpers and Rayon, but the
  model no longer owns those kernels directly.
- **q8_0 quantization**: 8-bit block quantization (fp16 scale + 32 int8
  values per block). File-loaded weights remain in mmap-backed q8_0 storage.
  The CPU backend quantizes each activation row to the same block format and
  accumulates packed integer dots, with scalar fallback, x86 AVX2/AVX-512
  VNNI dispatch, and NEON dot-product kernels on aarch64
  ([ARM kernels](arm-kernels.md)).
- **kv cache**: flat `[layer][head][seq_position][head_dim]` layout. prefill
  stores k/v for all prompt tokens; decode reads from cache and appends one
  token at a time. K/V values are stored as f16 and converted inside the
  attention kernels. The cache uses `n_kv_heads` (not `n_heads`), supporting
  grouped-query attention without storing repeated query-head copies.


## design justifications

these are the non-obvious trade-offs made in this codebase.

**embedding storage and tied heads.** model builders expose embedding tables as
row-addressable token data; q8_0 dimensions are normalized to contiguous
`[vocab, embed]` rows. Token lookup copies or dequantizes only the requested
row. When `output.weight` is absent, Llama/Qwen and Gemma can reuse the same
embedding table as the language-model head without expanding a q8_0 table.

**`load_from_cpu` on the backend trait.** the method loads host-side f32
data into a backend tensor. for `CpuBackend` this is a thin wrapper around
`CpuTensor::from_data`; a future gpu backend would copy the data to device
memory here. the name was chosen over `from_cpu` to avoid tripping
`clippy::wrong_self_convention` (which expects `from_*` to be a constructor
without `&self`).

**mixed CPU matmul paths.** f32 tensors use `matrixmultiply::sgemm`. q8_0
weights use Ember's packed Q8×Q8 kernels instead. Q/K/V and gate/up grouped
methods share activation quantization when possible, while the `Backend` trait
keeps those choices out of model code.

**f16 kv cache.** attention projections are computed in f32, converted to f16
when appended to the cache, and consumed through f32×f16 dot/accumulate
helpers. This halves K/V storage relative to an f32 cache while keeping
attention scores and outputs in f32.

**softmax returns uniform for all-masked input.** when every logit is -inf
(fully masked row), softmax normally produces NaN. this code detects that
case and returns `1/n` per position. it costs one extra branch per row and
prevents the generation loop from producing NaNs on degenerate input.


## prerequisites

- rust 1.98.1, pinned by `rust-toolchain.toml`, for the default build with
  the native console; headless builds (`--no-default-features`) keep the
  1.92 MSRV
- a gguf model file (e.g. gpt2 in q8_0)
- a tokenizer file for the model (`tokenizer-gpt2.json` for gpt-2, `tokenizer.json` for llama/qwen, `tokenizer-qwen3.json` for qwen3, `tokenizer-gemma4.json` for gemma 4; all four are included in the repo)


## current limitations

- attention math is abstracted behind the `Backend` trait, but the only
  implementation today is the cpu backend. it uses SIMD helpers for inner
  dot/accumulate work and Rayon for larger per-head workloads; there is no gpu
  backend yet.
- q8_0 matmul quantizes activations to q8_0 for both prefill and decode. The
  packed kernels need AVX2 or AVX-512 on x86 and NEON with the dot-product
  and FP16 extensions on aarch64; other CPUs use the scalar Q8×Q8
  implementation.
- model loader supports gpt-2, llama/qwen, and dense text-only gemma 4 ggufs
  through architecture-specific tensor names. Demo, single-prompt generation,
  and probe mode use the shared model interface; interactive mode remains
  GPT-2-only.
- not fully no_std - file i/o and mmap require std.


## optimization notes

the probe pipeline and CPU backend include these CPU-friendly optimizations:

- grouped extraction avoids redundant forwards across positions for the same
  template.
- pooled activation extraction writes only selected hidden-state spans instead
  of storing full per-layer sequence activations.
- `run_probe_matrix.py --jobs` parallelizes independent downstream analysis
  bundles after extraction.
- full and cached attention paths use the shared SIMD dot-product and
  weighted-accumulate helpers where their head dimensions are contiguous.
- K/V caches use f16 storage with direct f32×f16 attention helpers.
- prompt-only inference sizes its cache to the actual token count instead of a
  model's potentially very large metadata context.
- file-loaded q8_0 tensors retain mmap ranges, avoiding a second full model
  copy and allowing the operating system to page data lazily.
- q8_0 prefill uses tiled multi-row kernels; Q/K/V and gate/up projections
  reuse packed activation rows.
- q8_0 single-row decode matmuls split output rows across Rayon workers once
  their measured work exceeds the decode crossover.
- shared CPU attention parallelizes prefill by output row and decode by head,
  with worker-local score scratch to avoid scatter buffers.

the next useful optimization targets are:

1. **lm-head specialization**: the tied Gemma head now remains Q8_0, but a
   fused top-k-aware path could avoid materializing all vocabulary logits.
2. **richer thread-count benchmarks**: run `scripts/benchmark_threads.py` across
   Qwen3 0.6B, LLaMA 1B, Gemma 4, and selected 3B slices, then use the results
   to tune the parallelism thresholds.

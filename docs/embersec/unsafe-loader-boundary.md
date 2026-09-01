# EmberSEC: unsafe loader boundary audit

> **Phase I provenance:** frozen audit documentation from branch snapshot
> `e1fe6269`; the measured hardened Ember target is `3ceb7039`. Current main
> retains the applicable hardening, but implementation names and dataflow may
> have evolved. Read this as the published Phase I evidence record.

Every `unsafe` item in the crate, classified by whether it can receive
values derived from model files and whether those values are validated
before the unsafe operation. Line numbers refer to this branch (they
drift; the classification is the durable content).

Classification legend:

- [directly model-controlled]  the unsafe operation consumes bytes or
  counts taken straight from the GGUF without a prior validated gate.
- [validated before unsafe]    an explicit check (loader validation,
  constructor validation, or shape assert) runs before the unsafe op.
- [not reachable from loader]  the path cannot be driven by GGUF bytes.
- [needs follow-up]            the gate is procedural and should become
  structural (type-level) in a later EmberSEC slice.

## 1. GGUF file mapping

| site | what | classification |
|---|---|---|
| `loader.rs` `load_gguf_with_k_strategy`: `unsafe { memmap2::Mmap::map(&f) }` | maps the attacker-controlled file read-only | [validated before unsafe] — the mapping itself is length-checked by the OS; the safety comment documents the caller contract (no truncation/concurrent mutation while loaded). The *contents* are treated as untrusted thereafter. |

## 2. Mapped-weight storage

| site | what | classification |
|---|---|---|
| `quant.rs` `QuantizedWeight::try_from_mmap` (caller of `try_new_storage`) | stores a `Range<usize>` into the mapping | [validated before unsafe] — range start/end and `end <= mmap.len()` are checked here, and the loader's range gate ran first; `try_new_storage` additionally verifies byte-length vs. shape. |
| `quant.rs` `QuantizedWeight::evict_mapped_pages` (`unsafe fn` + inner `unsafe` block, `MADV_DONTNEED` via `unchecked_advise_range`) | drops resident pages of a mapped weight | [validated before unsafe] — operates on an already-validated `QuantizedWeight`; the caller (`model.rs` repack) guarantees no live borrows. Caller-ordering contract, not model-data contract. |
| `quant_k.rs` `KQuantWeight::try_from_mmap` / `try_new_storage` | same pattern for K-family weights | [validated before unsafe] — same checks as Q8_0 plus dtype/execution consistency. |

## 3. Numerical kernels (the bulk of the crate's `unsafe`)

All SIMD kernels in `simd.rs` (`dequantize_row_avx2`, `matmul_q8_0_decode_*`,
`rms_norm_into_avx2`, `rope_split_half_avx2`, the NEON family, and the
dispatch wrappers that call them) and in `k_quant_matmul.rs`
(`quantize_q8_k_into`, `q4_k_dot_q8_k`, `q6_k_dot_q8_k`, `*_x4`, their
dispatch) are `unsafe fn` whose contract is **CPU-feature availability**,
not data validity.

- Input slices come from `CpuTensor`, `QuantizedWeight`, or
  `KQuantWeight` objects, whose shapes/lengths were established by the
  loader gate + constructor validation.
- The safe dispatch wrappers assert length relationships
  (e.g. `dequantize_row`: `required_bytes <= data.len()` and
  `required_values <= dst.len()`; `rms_norm_into`:
  `dst.len() == x.len() == weight.len()`; `rope_split_half`:
  `x.len() == n_heads * head_dim`).
- The *values* in the buffers are model-controlled (weights) or
  activation-derived; that is inherent to inference, not a memory-safety
  boundary. The shape/layout side is what matters for memory safety.

Classification: [validated before unsafe] for shape/length; the
unsafe-ness is ISA dispatch. Status update (2026-08, EmberSEC slice 3):
the contract is now structural for the Q8_0 paths —

- every matmul entry point (`simd::matmul_q8_0_decode`,
  `matmul_q8_0_batch`, interleaved/packed variants,
  `k_quant_matmul::matmul_k_q8_into`) already accepted
  `&QuantizedWeight` / `&KQuantWeight` / the repacked layout types, all
  of which are constructible only from validated weights
  (`QuantizedWeightInterleaved`/`Vnni` have no raw-data constructor);
- the one raw-slice pub primitive (`dequantize_q8_0_row`) was replaced
  by `simd::dequantize_row(&Q8WeightView, ...)`; `Q8WeightView` is
  constructible only via `QuantizedWeight::row_view()` and re-asserts
  the storage invariant, so raw `(data, blocks_per_row)` pairs cannot
  reach the dequant kernels from outside `simd.rs`;
- the AVX2/AVX-512/NEON arch kernels live in private modules
  (`mod x86_64`, `mod aarch64`) — `pub` there is crate-internal only.

Remaining follow-up: tie the `plan.rs` scratch-arena sizes to validated
tensors explicitly (they are currently checked at plan build, not at
the type level).

| site | what | classification |
|---|---|---|
| `tensor.rs` `CpuTensor::matmul` `unsafe { matrixmultiply::sgemm(...) }` | BLAS call with `m/k/n` from shapes | [validated before unsafe] — 2-D shape asserts + inner-dim equality + checked output length precede the call. |
| `plan.rs` `DecodeArena::region_f32` / `region_array` (`from_raw_parts_mut`) | aligned scratch-arena views sized by the execution plan | [validated before unsafe] — sizes come from the post-load execution plan (checked at plan build); not directly model-file bytes. [not reachable from loader] in the raw sense; [needs follow-up] to tie plan sizes to validated tensors explicitly. |

## 4. Not reachable from the GGUF loader

| site | what |
|---|---|
| `alloc_counter.rs` `unsafe impl GlobalAlloc` | process-global allocation counting; driven by any allocation, not by model structure. |
| `v05/bundle.rs`, `v05/spec.rs` | the word "unsafe" appears only in path-validation error strings and tests (no `unsafe` blocks); the v0.5 experiment-spec boundary is a separate input surface (see fuzzing-plan.md). |
| `k_quant_matmul.rs` `unsafe impl Send for DstColumns` | marker impl for a rayon task payload; no data dereference. |

## 5. Summary

- The GGUF parsing path itself contains **no unsafe parsing** — all
  scalar reads are `read_exact` into fixed buffers; all index
  arithmetic is checked (see loader-threat-surface.md).
- Every unsafe operation reachable from model loading currently sits
  behind a procedural validation gate (loader range gate, constructor
  checks, or dispatch-level asserts).
- The structural weakness is that these gates are procedural: a future
  refactor could construct a view or call a kernel with raw parsed
  values. The first EmberSEC slice replaces the loader's procedural
  gate with a distinct `ValidatedTensorInfo` type that raw descriptors
  must pass through, so view construction consumes only validated
  descriptors. Kernel-side enforcement (kernels accepting only
  validated objects) is a follow-up.

# K-quant batch-1 GEMV redesign — design notes (pre-implementation)

Status: design + pre-coding analysis for the K-quant decode optimization phase.

## 1. What the current code does (traced end-to-end)

Q4_K projection during batch-1 decode (`planned_decode.rs` Matvec op →
`planned_linear_into` → `matmul_k_into[_parallel]` → `matmul_q4_k_avx2_row1_into`):

```
for each output column j in 0..out_features:        # 2048 for q/o, 8192 for gate
    acc_j = 0
    for b in 0..blocks_per_row:                      # 8 for in=2048
        block = data[j * 8*144 + b*144 ..]           # 144-byte Q4_K super-block
        d = f16(block[0..2]); min = f16(block[2..4])
        (ds[8], ms[8]) = unpack_k4_scales(block[4..16])   # get_scale_min_k4 × 8
        for g in 0..4:                               # 64 values: 32 low + 32 high nibble
            d1 = d*ds[2g]; m1 = min*ms[2g]; d2 = d*ds[2g+1]; m2 = min*ms[2g+1]
            for c in 0..4:                           # 8 values (AVX2 ymm)
                q8 = load(qs + g*32 + c*8)
                ql = cvtepu8_epi32(q8 & 0x0F); qh = cvtepu8_epi32((q8>>4) & 0x0F)
                vl = fma(cvt(ql), d1, -m1); vh = fma(cvt(qh), d2, -m2)
                acc = 0 (RESET every c!)
                acc = fma(x_low, vl); acc = fma(x_high, vh)
                store lanes; acc_j += sum8(lanes)    # horizontal reduce every 8 values
    dst[j] += acc_j
```

## 2. Physical layout (GGUF / llama.cpp `block_q4_K`, 144 B per 256 values)

- `[0..2]` f16 `d`  (super-block scale), `[2..4]` f16 `min`
- `[4..16]` 12 B of six-bit (scale, min) pairs for 8 sub-blocks of 32 values
  (K4-style 12-byte bit-reshuffle, `get_scale_min_k4`)
- `[16..144]` 128 B nibble-packed quants; sub-block 2g = low nibbles of
  `qs[32g..32g+32]` (values 64g..64g+32), sub-block 2g+1 = high nibbles
  (values 64g+32..64g+64)
- value = `d * ds * q - min * ms`, q ∈ 0..15

Q6_K (210 B, used for ~18 of 114 tensors in Llama-1B Q4_K_M incl. half the
`down` and `v` projections): per-16 int8 scales, 6-bit quants from ql/qh.

## 3. Where the redundant work is (measured, not guessed)

Each super-block is decoded exactly once per matmul (once for its own
column) — the traversal does not re-read weight bytes. The inefficiency is
**per-value decode/reduction overhead** that keeps the kernel ~8× below the
machine's memory bandwidth (4.3 GB/s effective vs ~35 GB/s):

1. **Horizontal reduction every 8 values**: `store lanes; acc_j += sum8`
   per ymm chunk → ~10 uops/8 values of pure reduction overhead; the vector
   accumulator is never carried across the block.
2. **Scale/min unpacking re-broadcast per 8-value chunk** (`get_scale_min_k4`
   scalar bit-ops + f16→f32 per column/block), instead of once per block.
3. **The min term is computed per value** (`fma(q, d1, -m1)` then
   `fma(x, v, acc)`) — two chained FMAs per value in a dependency chain.
4. **f32 dequantization per value** (cvtepu8_epi32 → cvtps) with no
   integer MACs; Q8_0 uses AVX-512 VNNI (vpdpbusd) but K-quant is AVX2-only
   float, so the machine's 512-bit + VNNI units sit idle for K-quant.
5. Scalar path round-trips every block through a 1 KiB `BLOCK_SCRATCH`
   (write + read) and dots in k-order with per-block scalar acc.

Because each weight byte is already read exactly once, the *weight traffic*
is already optimal; the fix is to make the per-value work cheap enough that
DRAM bandwidth becomes the limit (Q8_0's 35.7 GB/s shows that is achievable
on this host).

## 4. llama.cpp / ggml comparison (MIT source in repo, adapted ideas only)

llama.cpp's batch-1 K-quant path (`ggml_vec_dot_q4_K_q8_K`, AVX2):
- quantizes the activation vector once per matmul to int8 (q8_K, with
  per-16 `bsums`), shared across all output columns;
- unpacks the 12 scale bytes once per block into a vector
  (`mins_and_scales`, the same `get_scale_min_k4` reshuffle);
- does the dot with `maddubs_epi16` (8-bit×8-bit) + `madd_epi16` scale
  multiply, and the min correction from integer sums (`dmin * Σ m_s·bsum`);
- keeps the accumulator in vectors across the whole block (one horizontal
  sum per dot).

Idea we adopt: **unpack scales once per block, keep vector accumulators
across the whole column, one horizontal reduction per output element**.
Idea we do NOT adopt for this phase: **int8 activation quantization
(q8_K/q8_1)** — it changes Ember's numerical semantics (activation
quantization error) and would push the eager-vs-compressed logit delta past
the frozen Gate-B bound (1e-2). The Q8_0 path may use VNNI on quantized
activations because its own gates allow it; the K-quant gates compare
against an exact-f32 oracle. Exact-f32 activations also have the same
bandwidth ceiling, so nothing is lost.

## 5. Chosen strategy (smallest change with the most measured impact)

New module `src/k_gemv.rs`: a batch-1 (`rows == 1`) Q4_K/Q6_K GEMV that
- keeps **exact f32 activations** (no quantization);
- traverses output columns, unpacking each block's header **once** into
  registers and consuming its 256 values with 16-lane (AVX-512) or 8-lane
  (AVX2) vector accumulators carried across the whole block;
- applies scale/min via broadcast FMAs directly in the accumulator (two
  independent FMA chains per sub-block pair to hide latency);
- does **one horizontal reduction per output element**;
- parallelizes with one coarse static rayon split over output chunks
  (bit-identical per-column body), with a **shape-dependent threshold
  measured experimentally** instead of the inherited 8M-MAC rule;
- ships a portable scalar fallback and dispatches AVX-512 > AVX2 > scalar
  by runtime feature detection (same pattern as the Q8_0 path).

Wiring: `matmul_k_into` / `matmul_k_into_parallel` route `rows == 1`
K-quant through the new path; `rows > 1` (prefill) keeps the existing
kernels unchanged. Planned decode, generic decode, hooks, and the planner
are untouched (the kernel is below the observable tensor boundary).

Validation: unit tests vs the scalar reference on seeded data; Gate A/B/C
via `tests/k_parity.rs` on the real model; golden logits; generation
parity; Q8 regression re-run.

## 6. Measurement limits

No perf (hardware counters unavailable, paranoid=2); powersave governor
noise handled with medians over ≥5 reps; AVX-512 presence checked at
runtime; llama.cpp comparison via the pinned `llama-bench` binary on the
same files (isolated per-projection llama.cpp benchmarks are not exposed by
that build — full-model tg is used).

## 7. Post-implementation verification and the column-blocking experiment (2026-08-10)

The final session edit (an AVX-512 column-pair body, CB=2) shipped with a
dispatch bug: the fallback `while i < dst_chunk.len()` loop had lost its
`i += 1`, so every non-col2 column (all Q6_K, and Q4_K without AVX-512)
looped forever on column 0. The handoff-required `cargo test --release
--lib k_gemv` hung; the fix is a one-line `i += 1` and the suite now
passes in ~0.3 s. Lesson: the col2 edit restructured the loop body and the
increment was dropped — kernel dispatch loops need a regression test that
terminates.

### Column-blocking A/B (CB = columns per activation L1 load)

Three AVX-512 bodies were measured: single-column (CB=1, the shipped
pre-edit path), column-pair (CB=2), and 4-column (CB=4), on the real
Q4_K_M/Q6_K Llama-3.2-1B projections and end-to-end, interleaved to cancel
thermal drift (the host idles at ~84 °C with the GUI up; long sequential
matrices are unusable — the same binary measures 6.1 tps hot vs 13.6 tps
cool).

- Microbenchmark (isolated kernels, medians of 7): CB=2/CB=4 are within
  ±10 % of CB=1 on every projection — no systematic win at 1t or 4t.
- End-to-end `bench-decode --execution planned` Q4_K_M 8t: **CB=1 13.55 /
  CB=2 11.63 / CB=4 7.51 tps** (round 1; round 2 10.33 / 9.94 / 6.77).
- Conclusion: **CB=1 is the default**. The kernels are FMA/issue-bound,
  not activation-load-bound; multi-column bodies interleave 2/4 weight-row
  streams and defeat the sequential DRAM prefetcher that the single-column
  traversal feeds perfectly. The earlier "activation L1 re-reads are ~half
  the time" estimate did not hold on this host.
- The CB=2/CB=4 bodies remain in `src/k_gemv.rs`, selectable with
  `EMBER_KGEMV_CB=2|4` for hosts with different load-port ratios, and are
  covered by `all_column_blocking_widths_match_oracle` (every CB verified
  against the eager-f32 oracle, serial ≡ parallel bit-identical).

### Verification state of the shipped path (CB=1 default)

- `k_gemv` unit suite: 3/3 (scale unpacking, seeded oracle incl.
  serial≡parallel, all-CB oracle sweep).
- Full lib suite: 296/296.
- `tests/k_parity.rs` (real-model Gates B/C/E): Q4_K_M 5/5 and Q6_K 5/5.
- Golden logits vs llama.cpp (recreated reference, llama-cpp-python
  `logits_all=True`): Q8_0 top-1 match, top-10 overlap 1.0, cosine 0.9998;
  Q4_K_M top-1 match, top-10 overlap 1.0, cosine 0.9995. (The stored
  `data/golden/llama32_1b_ember_logits.npy` is a June-3 ember-vs-ember
  artifact of a different build era and legitimately differs from the
  current build; the llama.cpp cross-check is the live external gate.)
- End-to-end decode (cool, interleaved): Q4_K_M ~13.5 tps, Q6_K ~11 tps at
  8t planned (vs 6.25/6.04 before this phase; ~2×).

## 8. K-quant prefill: register-blocked GEMM (2026-08-10, second phase)

The v0.3 prefill path (`rows > 1`) was catastrophic: 4.7 tok/s vs llama.cpp
121-132 tok/s (a 28x gap). The old kernels had the same anti-patterns the
decode phase fixed, worse: for each output column they re-read every
activation from L2, horizontal-reduced every 8 values, and
read-modify-wrote `dst` per 8-value chunk — and prefill was never
parallelized (the generic path called the serial `matmul_k_into`).

### Implementation (`src/k_prefill.rs`)

A register-blocked exact-f32 GEMM (AVX-512):
- fixed (RT x CT) tiles with **compile-time loop bounds** — the first
  attempt used runtime `acc[r][col]` indexing and the compiler spilled every
  accumulator to the stack (`vmovaps (%rsp)... / vfmadd / vmovaps ...(%rsp)`
  per FMA — ~3 memory ops per FMA, ~10-15 GFLOPS). The macro-generated
  constant tiles keep all 16-32 accumulators in zmm registers (zero spills).
- Q4_K main tile 4x4 (measured best of 4x2/2x4/4x4 via `EMBER_KPREFILL_TILE`),
  Q6_K tile 2x1, small fixed remainder tiles.
- dequantized weight chunk shared across the RT rows; activation chunk
  shared across the CT columns.
- parallel split over 256-column tiles (bit-identical to serial; each task
  reads only its own weight columns; x re-reads hit L2).
- `CpuBackend::matmul_k` now routes through the parallel entry for both
  decode and prefill (matches the Q8_0 batch pattern); non-AVX-512 builds
  fall back to the v0.3 kernels via `matmul_k_legacy_prefill_into`.

### Measured results (cool host, best of 3 rounds, 26-token Arabic prompt)

| model | before | after | llama.cpp | gap |
|---|---|---|---|---|
| llama-1B Q4_K_M | 4.7 tok/s | **33.0** (7.0x) | 121 | 3.7x |
| llama-1B Q6_K | ~5.4 | **38.1** (~7x) | ~121 | ~3.2x |
| qwen-1.5B Q4_K_M | — | **22.1** | 96.4 | 4.4x |

Isolated kernel (26 rows, 8 threads, aggregate): q 82 GFLOP/s, o 87,
gate/up 71-74, down (Q6K) 100; single-thread ~20-28 GFLOPS (the
single-thread kernel is the main remaining inefficiency, ~13% of the
Tiger-Lake FMA ceiling; parallel scaling caps at ~4x on 8 threads,
memory-bound).

### Verification

- `k_prefill` unit suite 4/4: Q4_K/Q6_K vs the eager-f32 oracle (Gate A,
  1e-4 relative) across shapes incl. rows=26/33; serial == parallel
  bit-identical; zero-scale blocks contribute exactly zero; length
  mismatches rejected.
- Full lib suite 301, k_parity (real-model Gates B/C/E, which **includes
  multi-token prefill logits**): Q4_K_M 5/5 and Q6_K 5/5.
- Decode regression check: Q4_K_M 15.4 -> 19.0 tps (cooler run; no
  regression from the routing change).
- clippy -D warnings + fmt clean.

### Remaining (ranked)

1. Single-thread tile kernel efficiency (~20 GFLOPS; dequant port
   contention + x L1 re-reads); deeper unrolling or a hybrid
   dequant-once-two-level cache scheme.
2. Parallel scaling caps at ~4x (8 threads) — memory contention on x
   re-reads/weights; larger per-task column tiles may help.
3. Generic-path per-op Vec materialization for the non-matmul ops (silu,
   norms, residual adds) — the Q8 path has the same overhead.

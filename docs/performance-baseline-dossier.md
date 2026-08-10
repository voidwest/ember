# Ember performance baseline dossier — batch-1 decode, prefill, startup

Status: baseline only. **No optimization was performed.** All numbers below
were produced with the codebase at `f8e97a7` plus isolated profiling
instrumentation (see "Files changed" at the end). Numerics unchanged: the
full test suite passes (294 lib + 46 integration + 38 + 5 + 12) including
the v0.4 Gate E zero-allocation parity tests.

## 0. Methodology

| field | value |
|---|---|
| commit | `f8e97a7` + uncommitted profiling instrumentation (all tests green) |
| CPU | Intel Core i5-1135G7 (Tiger Lake), 4 physical / 8 logical, up to 4.2 GHz |
| ISA | x86_64, AVX2+FMA+F16C, AVX-512 VNNI/VL/BW (used by Q8 kernels) |
| OS / kernel | Arch Linux, 7.1.5-arch1-2, glibc 2.44 |
| compiler | rustc 1.92.0 (pinned toolchain), release profile, LTO off |
| governor | powersave; CPU scaling active → repeated trials, medians reported |
| perf | NOT available (`perf_event_open` → EACCES, paranoid=2, no package, no root). No hardware counters (cycles/IPC/cache misses) in this dossier; wall-clock per-op timing + procfs faults + allocator counts used instead |
| DRAM ceiling (measured) | copy 14–37 GiB/s (1–8 threads), multithreaded ceiling ≈ 28 GiB/s ≈ 30 GB/s |
| models | Llama-3.2-1B-Instruct Q8_0 (sha256 `432f310a…`), Q4_K_M (807,694,080 B), Q6_K (1,021,800,192 B); Qwen2.5-1.5B-Instruct Q8_0 (1,894,532,128 B); llama.cpp comparison used the identical `models/v03-ladder/` files |
| prompts | bench-decode: deterministic token-id 1, context grows 1→N; generate: 26-token Arabic morphology prompt, 64 generated tokens, greedy |
| trials | bench-decode: 5 reps (128 tokens, Q8) / 3 reps (64 tokens, K-quant), 1–2 warmups; generate: 3+ runs; llama.cpp: `-r 3` |
| warm/cold | "cold" = model+tokenizer evicted from page cache via `posix_fadvise(DONTNEED)`; "warm" = page-cache resident |

Raw data: `artifacts/performance-baseline/2026-08-10/raw/` (73 files) and
`artifacts/performance-baseline/2026-08-10/llamacpp/`, `startup.json`.
Reproduce with `scripts/performance/profile_baseline.py`,
`scripts/performance/startup_cold_warm.py` (see section 11).

## 1. EXECUTIVE SUMMARY

### Top 5 observed bottlenecks (batch-1 decode, representative model Llama-3.2-1B Q8_0, 8 threads, 36.8 ms/token)

1. **The whole decode is DRAM-bandwidth-bound.** Every token streams ~1.31 GB
   of quantized weights (MLP 856 MB + LM head 279 MB + QKV/O 178 MB) at a
   measured 35.7 GB/s — at the machine's measured DRAM ceiling (28–40 GB/s).
   Kernel math is not the limit for Q8; bytes-per-token is. *(confidence:
   high — per-op timing + per-token weight bytes + independent bandwidth probe)*
2. **MLP gate/up/down projections: 18.2 ms/token (49.6%)** — three 16.8M-MAC
   GEMVs per layer × 16 layers. Bandwidth-bound like everything else, but the
   largest single consumer. *(high)*
3. **LM head: 6.24 ms/token (17.0%)** — one 262.7M-MAC GEMV (2048×128256).
   A single operator. *(high)*
4. **K-quant decode is ~8× below memory bandwidth (4.3 GB/s vs 35.7 GB/s).**
   The Q4_K_M/Q6_K kernels dequantize every weight block once per output
   column (no block reuse across columns) and only parallelize projections
   ≥ 8M MACs; q/k/v/o run serial and get *slower* with more threads. End
   result: planned K-quant decode 6.25 tps vs llama.cpp 42.2 tps on the same
   file. *(high — direct comparison + per-op scaling)*
5. **Inter-op overhead grows with thread count: 1.8–6.3 ms/token (5–17%) on
   the fast path at 8 threads** — ~80 rayon join/wake barriers per token,
   plus a per-token 513 KB logits allocation/copy. *(medium — residual
   measurement; see section 2)*

Prefill-specific: the generic prefill path materializes every intermediate
(~1,700 allocations/token at decode; prefill similar per-row) and the K-quant
prefill batch kernel is catastrophic (4.7 tok/s vs llama.cpp 131.9 tok/s —
28× behind).

### Top 5 suspected optimization opportunities

1. **Bandwidth-competitive K-quant GEMV** (block reuse across output columns;
   extend column-parallel to all projections). Expected: planned K-quant
   decode 6.25 → ~30–40 tps (5–6×), K-quant prefill 4.7 → >50 tok/s.
2. **K-quant prefill batched kernel** (parallel rows × column tiles with
   shared dequant). Expected: prefill 4.7 → 50–130 tok/s (10–28×).
3. **Per-operator thread thresholds + fewer rayon joins** (serialize tiny
   ops, fuse gate/up/down dispatch, avoid negative q/o K-quant scaling).
   Expected: 5–15% on fast path, more on K-quant.
4. **Allocation-free logits output** (caller-owned/reused buffer): removes the
   only remaining steady-state heap allocation (513 KB/token) on the
   fast/planned paths; small wall-clock win (<2%) but removes the generic
   path's 1,695 allocs/6.5 MB per token if it can be made allocation-free.
5. **Tokenize model load** (tokenizer.json init 396 ms > model mmap 181 ms):
   tokenizer deserialization is the largest startup cost; cache or lazy-load.

Confidence: 1–2 high (measured vs llama.cpp), 3 medium (residual gap), 4 high
(allocator counts), 5 high (phase timings).

### Amdahl ceilings (Llama-1B Q8 fast path, 8 threads, 36.8 ms/token)

| subsystem | share | ceiling if infinitely fast |
|---|---|---|
| MLP (gate+up+down) | 49.6% | 1.98× |
| LM head | 17.0% | 1.20× |
| Q+O projections | 10.2% | 1.11× |
| inter-op residual (mean) | 7.5% | 1.08× |
| attention | 1.6% | 1.02× |
| norms/rope/silu | 1.6% | 1.02× |

But Q8 decode already runs at ~90–100% of DRAM bandwidth, so even "infinitely
fast kernels" buy only ~1.1× wall-clock. The only large lever for Q8 decode is
fewer bytes per token (lower-bit quant), and the only large lever overall is
fixing K-quant, which currently wastes 8× of its bandwidth budget.

## 2. DECODE LATENCY BREAKDOWN

### Llama-3.2-1B Q8_0, fast (workspace) path, 8 threads — 36.8 ms/token (27.2 tps, ctx 1→129)

From `bench-decode --profile-operators` (per-op medians × 16 layers):

| subsystem | ms/token | % | notes |
|---|---|---|---|
| MLP gate/up/down | 18.23 | 49.6 | 3 × 2048×8192 GEMV/layer, packed VNNI, row-parallel |
| LM head | 6.24 | 17.0 | 2048×128256 interleaved VNNI, row-parallel |
| Q + O projections | 3.74 | 10.2 | 2048×2048 each |
| K + V projections | 1.21 | 3.3 | 2048×512 |
| attention (cached) | 0.58 | 1.6 | 36 µs/layer @ avg ctx 33; grows ~linearly with ctx |
| norms (2/layer) | 0.05 | 0.1 | 1.6 µs each |
| RoPE (q+k) | 0.10 | 0.3 | 5–8 µs |
| KV store | 0.02 | 0.1 | f32→f16 convert |
| SiLU×mul, residual adds | 0.15 | 0.4 | |
| embedding | 0.001 | 0.0 | row copy |
| **op-sum** | **30.5** | **82.9** | |
| inter-op residual (sync/loop/alloc) | 2.8–6.3 | 7.5–17.1 | mean-based 2.8 ms, median-based 6.3 ms (see note) |
| **total** | **36.8** | 100 | |

Matmuls are **80%** of token time; everything non-matmul is ≤ 4%.
The inter-op residual grows with threads (1.9% at 1 t → 17.1% at 8 t) and is
the sum of ~80 rayon join/wake barriers, the per-token logits allocation
(513 KB), and loop bookkeeping. Median-vs-mean bias bounds it between 2.8 and
6.3 ms; a dedicated loop-instrumentation pass is the honest next step.

### Llama-3.2-1B Q4_K_M, planned path, 8 threads — 160 ms/token (6.25 tps)

| subsystem | ms/token | % |
|---|---|---|
| gate + up + down | 77.3 | 48.1 |
| LM head | 22.8 | 14.2 |
| Q + O (serial kernels!) | 50.6 | 31.5 |
| K + V | 12.4 | 7.7 |
| attention | 0.7 | 0.4 |
| norms/rope/silu | 0.4 | 0.3 |
| inter-op residual | 5.9 | 3.5 |

Q and O (4.2M MACs each) fall under the 8M-MAC column-parallel threshold and
run serial; at 8 threads they cost as much as the parallelized gate
(1.58 ms vs 1.65 ms per layer) and *regress* with thread count (1.18 ms @1 t
→ 1.58 ms @8 t).

### Generic (hooked) path, Llama-3.2-1B Q8_0, 8 threads — 57.6 ms/token (trace-instrumented)

Shares are stable across baseline/capture/intervene (99.2% covered by spans):
up 21.5%, gate 21.2%, lm_head 19.0%, down 15.8%, q 6.9%, o 6.4%, k 2.7%,
v 2.6%, silu 1.6%, attention 1.4%, everything else < 1%. The generic path is
**1.85× slower than the fast path** on the same model (57.6 vs 31 ms/token)
— allocations + per-op dispatch + no workspace.

### vs llama.cpp (identical files, 8 threads)

| model | ember decode (ctx 1→65) | llama.cpp tg (ctx 26→90) | ratio |
|---|---|---|---|
| llama-3.2-1b Q8_0 fast | 31.25 tps | 30.65 tps | **1.02×** |
| llama-3.2-1b Q4_K_M planned | 6.26 tps | 42.21 tps | **0.15×** |
| llama-3.2-1b Q6_K planned | 6.04 tps | 38.56 tps | 0.16× |
| qwen2.5-1.5b Q8_0 generic | 11.47 tps | 24.42 tps | 0.47× |
| qwen2.5-1.5b Q4_K_M planned | ~3.0 tps (old artifact) | 34.80 tps | ~0.09× |

llama.cpp is itself bandwidth-bound (Q8 40.2 GB/s, Q4 30.6 GB/s effective):
Q4 is faster than Q8 there *because it reads fewer bytes*.

## 3. PREFILL BREAKDOWN

26-token Arabic prompt, 8 threads (generate `--benchmark`, plus trace spans):

| model/path | prefill tok/s | vs llama.cpp |
|---|---|---|
| llama-1B Q8_0 generic | 47.5 (trace: 37.2) | 0.44× (llama.cpp 107.5) |
| llama-1B Q4_K_M generic | 4.7 | 0.036× (llama.cpp 131.9) |
| qwen-1.5B Q8_0 generic | 33.3 | 0.50× (llama.cpp 66.1) |

Op breakdown (llama Q8, 26 rows, trace): down 26.6%, up 24.4%, gate 23.6%,
o 7.1%, q 6.9%, silu 2.8%, k/v 2.7% each, lm_head 1.6%, attention 0.5% —
matmuls ≈ 94%. Prefill uses the generic tensor path (per-op Vec
materialization); the batch Q8 matmul tiles rows 4-per-task. K-quant prefill
uses the *serial* `matmul_k_into` batch loop (no parallelization at all for
rows>1) with per-column dequant → 28× behind llama.cpp.

## 4. ALLOCATION REPORT

Per-token (counting allocator; `bench-decode --allocations`):

| path | alloc events/token (caller) | bytes/token | hot sites |
|---|---|---|---|
| llama Q8 fast | 3–5 | ~513 KB | `logits` Vec (128256×f32) in `forward_decode_with_workspace`; workspace is thread-local and reused (0 steady-state beyond logits) |
| llama Q4 planned | 4 | ~514 KB | `dst.to_vec()` in the plan's Logits op (arena→owned tensor; arena itself is allocation-free — Gate E holds for the compute loop) |
| qwen Q8 generic | **1,695** | **6.5 MB** | every op materializes new Vecs (rms_norm, q/k/v/o, gate/up/down, rope, attention, logits) across 28 layers |

Worker-thread allocations (global delta − caller): ~10 events and ~24 KB per
token (attention qk-scratch `resize` growth). Global net-outstanding delta is
tiny (allocs balanced by frees).

- Scratch reuse: fast path reuses a thread-local `Workspace`; planned path
  reuses a 20 MB arena (212 regions); generic path reuses nothing.
- The one per-token allocation on fast/planned paths is the returned logits
  tensor (~513 KB) — it is the API contract, not waste, but it is the only
  remaining steady-state heap traffic and it is trivially removable with a
  caller-provided buffer.
- Per-token alloc accounting adds no measurable timing perturbation
  (27.20 vs 27.17 tps with/without `--allocations`).

## 5. THREADING REPORT

End-to-end decode scaling (median tps; RAYON_NUM_THREADS):

| model/path | 1 t | 2 t | 3 t | 4 t | 6 t | 8 t | 8/1 |
|---|---|---|---|---|---|---|---|
| llama Q8 fast | 11.86 | 19.79 | 24.95 | 26.68 | 27.10 | 27.17 | 2.29× |
| qwen Q8 generic | 5.20 | 8.16 | 10.20 | 10.57 | 11.09 | 11.47 | 2.21× |
| llama Q4 planned | 2.94 | 4.68 | — | 5.82 | — | 6.25 | 2.13× |
| llama Q4 reference | 2.92 | 2.86 | — | 2.94 | — | 2.95 | 1.01× |
| llama Q6 planned | 3.00 | 4.43 | — | 5.38 | — | 6.04 | 2.01× |

- **4 physical threads ≈ saturation** on this host (Q8: +1.8% from 4→8 t).
- **Reference (generic) K-quant path does not scale at all** — its matvec
  dispatch is serial per tensor; the 2.1× planned-vs-reference gap at 8 t is
  entirely the plan's column-parallel matvec (at 1 t they are identical:
  2.94 vs 2.92 tps → arena/plan overhead ≈ 0).
- Per-operator scaling (Q8 fast, medians): gate/up/down 1.15 ms→0.38 ms
  (3.0×); lm_head 23.2→6.24 ms (3.7×); q/o 0.29→0.12 ms (2.4×); attention
  40→36 µs (≈1×, already tiny). Q4 planned: q/o **regress** 1.18→1.58 ms.
- Synchronization: ~80 rayon joins/token on the fast path at 8 t; inter-op
  residual grows from 1.9% (1 t) to 17.1% (8 t) of token time (fast path),
  1.1%→3.5% (planned). Small operators pay sync cost disproportionate to
  their work (e.g., a 1.6 µs RMSNorm between two 0.4 ms parallel matmuls).
- The q8_matmul microbench (gate+up 2048×8192, hot cache): 2.14 ms @1 t,
  1.15 @2 t, 0.69 @4 t, 0.63 @8 t — saturates at 4 threads; packed-paired
  layout is 1.4–2.0× faster than the generic path and bit-identical.

## 6. KERNEL REPORT

Hottest kernels (Llama-3.2-1B, decode):

| kernel | shape (out×in) | MACs | quant | exec mode | per-token ms @8 t | share |
|---|---|---|---|---|---|---|
| lm_head | 128256×2048 | 262.7M | Q8_0 | interleaved VNNI, row-parallel | 6.24 | 17.0% |
| gate/up ×16 | 8192×2048 | 16.8M each | Q8_0 | packed VNNI, row-parallel | 5.9–6.1 each | ~20% each |
| down ×16 | 2048×8192 | 16.8M | Q8_0 | packed VNNI, row-parallel | 6.2 | 20.4% |
| q/o ×16 | 2048×2048 | 4.2M | Q8_0 | packed VNNI | 1.9/1.8 | 6/6% |
| k/v ×16 | 512×2048 | 1.05M | Q8_0 | packed VNNI | 0.6/0.6 | 2/2% |
| K-quant variants | same shapes | | Q4_K/Q6_K | avx2-q4k/q6k, column-parallel ≥8M MACs else serial | see §2 | |

Facts:
- Decode is pure **GEMV/narrow-matrix** (rows=1): no GEMM path is exercised
  at batch 1; prefill (rows=26) uses the tiled Q8 batch kernel.
- Weights are **packed**: Q8_0 in VNNI-interleaved 16-row tiles (persistent,
  mmap-backed source, packed copy resident); K-quant stays **compressed on
  mmap** and is dequantized per block *inside* the matvec — with no reuse
  across output columns (each 256-element super-block is dequantized once per
  output column; scaling metadata is read repeatedly per column, adding
  traffic). Q4 planned achieves only 4.3 GB/s effective vs 35.7 GB/s Q8.
- Dispatch is dynamic per call: `is_x86_feature_detected!` checks, parallel
  thresholds (`out×in ≥ 1,048,576` Q8; `≥ 8,000,000` K-quant decode) and
  chunk-row computation repeat every invocation even though shapes are fixed
  after load.

## 7. MEMORY / KV REPORT

- KV cache: flat f16 `[layer][head][pos][head_dim]`, allocated once at
  `required_context`; head stride = `max_seq×head_dim` (e.g., 8.4 MB for
  llama ctx 131072), pos stride = head_dim (64). Full-context allocation for
  llama-1B would be 4.3 GB (K+V) — generation caps the cache at
  prompt+generated length, keeping it small (≈3 MB for these runs).
- Decode access pattern: one new (k,v) written per layer (`f32→f16`
  convert+copy, ~2.4 µs); attention reads the whole K/V prefix per layer with
  head-stride jumps of `max_seq×head_dim` — strided, cache-hostile at large
  ctx, but the volume is small (KV ≈ 2 bytes/elem vs 1.06 bytes/elem × 16
  projections of weights).
- Attention time scales linearly with ctx (40 µs/layer @ ctx 26 → 59 µs/layer
  @ ctx 88) and is ~1.6% of decode at these contexts; at 4k–8k ctx it would
  be ~1.5–2.5 ms/layer×16 → dominant. Not measured here (out of scope for
  this baseline; see §10).
- qk_scratch `[max_seq]` f32 preallocated and reused; worker threads grow
  their own score scratch (~24 KB/token until capacity).
- Startup memory: peak RSS 1.64 GB (llama Q8, warm), 1.89 GB (qwen Q8);
  steady-state decode adds only workspace (~2 MB) + KV.

### Startup (page-cache cold vs warm; `/usr/bin/time -v`)

| model | cold wall | cold major faults | warm wall | warm major faults | peak RSS |
|---|---|---|---|---|---|
| llama-1B Q8 | 2.70 s | 741 | 1.66 s | 0 | 1.64 GB |
| qwen-1.5B Q8 | 5.38 s | 5,600 | 2.66 s | 0 | 1.89 GB |

Warm phase split (llama Q8, bench-lifecycle): model mmap+construct 181 ms,
tokenizer init + prompt encode 396 ms, prefill 150 ms (6 tok), decode 258 ms
(7 evals). **Tokenizer deserialization (396 ms) exceeds model load (181 ms).**

## 8. EXECUTION-PLAN REPORT

- Resolved once per `(model, execution, hook-mode)` (cached `Arc<ExecutionPlan>`):
  op sequence, tensor table (147 tensors for llama Q4_K_M: 96 avx2-q4k +
  18 avx2-q6k + 33 eager-f32), per-tensor KernelId, scratch arena (19.99 MB,
  212 regions), KV layout, thread strategy (`column-parallel-rayon`), hook
  sites. Session state (`ResolvedOps`: region indices + roles) built once per
  plan hash.
- Repeated every token: plan-hash check; the interpreter's `match` over ~17
  resolved ops per layer; per-call kernel dispatch (`is_x86_feature_detected!`,
  parallel thresholds, chunk sizes); attention dispatch re-decided per layer
  (`should_parallel_attention`); K-quant parallel/serial decision per
  projection (`in×out ≥ 8M`).
- Shapes are fixed after load and the plan already encodes them; the residual
  dynamic work is cheap but nonzero, and the *parallel-vs-serial* decisions
  are provably wrong for q/o K-quant at ≥4 threads (see §5).
- Evidence that the plan concept pays: planned vs reference K-quant decode at
  8 t = 6.25 vs 2.95 tps (2.1×, entirely parallelism); at 1 t identical
  (plan overhead ≈ 0). An immutable plan is effectively already present and
  working; no redesign is justified by this data.

## 9. OBSERVABILITY COST REPORT

Same model/prompt/tokens, generate path, decode evals/s:

| mode | llama Q8 fast, 1 t | llama Q8 fast, 8 t | qwen Q8 generic, 8 t |
|---|---|---|---|
| A. normal (hooks disabled) | 12.20 | 32.20 | 14.06 |
| B. capture (after-attn+after-mlp, all 16 layers, every token — 32 records, 258 KB/token, verified in manifest) | 11.62–12.18 (noise-bound) | 32.16 | 14.38 |
| C. intervention (zero layer-8 attention output) + capture | 11.01–11.62 | 31.58 | 14.13 |
| noop experiment attached (hooks_overhead bench, planned path Q6_K) | — | +4.0% | — |

- **Capture and intervention are effectively free at 8 threads (≤2%)** and at
  worst ~5% at 1 thread (noise-limited). The decode is bandwidth-bound; hook
  dispatch (66 calls/token), 258 KB/token of copies, and per-record Vec
  allocs are hidden under the 1.31 GB weight stream. Cost sources, in order:
  hook dispatch/selection checks → capture `to_vec()` copies + allocs →
  planned-path de-fusion when a hook is active → record buffering
  (serialized only at generation end).
- Trace mode forces the generic path: op-level spans on the generic path are
  1.85× slower than the fast path — the path switch, not the spans, is the
  cost (spans cover 99.2% of generic-path time).
- `hooks_overhead` on Q8 actually exercises the *planned* interpreter (Q8
  serial matvecs), not the fast path; use the generate-path numbers above for
  fast-path hook cost.

## 10. PRIORITIZED NEXT EXPERIMENTS

Ranked by (expected impact, cost, risk, evidence quality):

1. **K-quant GEMV with block reuse + full column-parallel coverage.** Impact
   up to 5–6× on K-quant decode (6.25 → ~35 tps) and most of the 28× prefill
   gap; cost: high (new kernel core); risk: medium (must keep Gate A/B
   parity; a "dequant once per block, accumulate across a column tile"
   restructure changes accumulation order — needs tolerance testing or
   bit-preserving formulation); observability: low; evidence: high (4.3 vs
   30+ GB/s, llama.cpp 6.7× faster, q/o serial regression).
2. **K-quant prefill batched kernel** (parallel row×column tiles, shared
   dequant scratch). Impact: 10–28× on K-quant prefill; cost: medium; risk:
   medium (same parity concerns); evidence: high (4.7 vs 131.9 tok/s).
3. **Per-operator thread thresholds + join reduction.** Serialize tiny ops at
   high thread counts, raise the K-quant parallel threshold downward for
   q/o, batch gate/up/down into fewer rayon calls, hoist
   `should_parallel_*`/feature checks out of the per-token loop. Impact:
   5–15% fast path, 10–30% K-quant; cost: low–medium; risk: low (bit-identical
   kernels already exist per mode); evidence: medium (inter-op residual +
   q/o regression).
4. **Allocation-free logits output** (caller-provided buffer / arena-pinned
   return). Impact: <2% fast/planned wall, removes the last steady-state heap
   traffic; enables a truly allocation-free steady state; cost: low; risk:
   low; evidence: high (allocator counts).
5. **Verify at longer context (2k–8k).** Attention is 1.6% at ctx ≤ 130 but
   grows linearly; KV-stride layout may dominate at 4k+. Cost: low (existing
   tooling); needed before any KV work.
6. **Tokenizer init caching** for startup (396 ms > 181 ms model load).
   Impact: ~0.4 s of a 1.7–2.7 s startup; cost: low; risk: low.

**Evaluation of the five named hypotheses:** batch-1 quantized GEMV
specialization — *justified for K-quant* (the current kernel is the measured
bottleneck; Q8 GEMV is already specialized and at bandwidth); per-operator
thread thresholds — *justified* (q/o regression, 17% sync residual);
allocation-free steady-state decode — *already ~true* on fast/planned paths
(one 513 KB logits alloc remains); persistent packed weights — *already done*
for Q8; immutable model-specific execution plans — *already done and
working*. **No IR/JIT/compiler-framework/architectural redesign is justified
by this data.** The Q8 fast path should be left alone; the K-quant path and
threading policy are where the measured problems are.

## 11. Files changed, commands, raw data

Changed (all additive/isolated; full test suite green):
- `src/alloc_counter.rs` — thread-local byte tracking
  (`count_allocations_with_bytes`); existing `count_allocations` unchanged.
- `src/main.rs` — `bench-decode --allocations` flag.
- `src/cli_commands.rs` — allocation report in bench-decode JSON.
- `src/llama.rs` — `--profile-operators` now also times RMSNorm/RoPE/KV
  store/attention/SiLU/residual-adds/embedding on the fast path (guarded by
  the same flag; normal path unchanged).
- `scripts/performance/{common,profile_baseline,startup_cold_warm,analyze}.py`
- `docs/performance-baseline-dossier.md` (this file); `.gitignore` whitelist
  entry for it.

Exact commands:
```bash
# matrix (models × executions × thread counts, resumable)
python3 scripts/performance/profile_baseline.py            # or --only llama-1b-q8 --quick
# cold/warm startup (posix_fadvise-based page-cache eviction)
python3 scripts/performance/startup_cold_warm.py llama-1b-q8 qwen-1.5b-q8
# single-run examples
./target/release/ember bench-decode -m Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokens 128 --repetitions 5 --execution reference --profile-operators --allocations
./target/release/ember bench-decode -m Llama-3.2-1B-Instruct.Q4_K_M.gguf --arch llama \
  --tokens 64 --repetitions 3 --execution planned --profile-operators
./target/release/ember --model Llama-3.2-1B-Instruct-Q8_0.gguf --arch llama \
  --tokenizer tokenizer.json -p "في الجملة التالية، الكلمة المميزة هي: كِتَاب. اشرح معناها." \
  -n 64 --temperature 0 --benchmark --capture-activations <capture.toml>
./target/release/ember --model ... -n 64 --temperature 0 --benchmark --zero-layer-output 8:attention
./target/release/ember ... --trace ops --trace-out trace.json --trace-run-metadata
cargo bench --bench q8_matmul -- --model Llama-3.2-1B-Instruct-Q8_0.gguf --rows 1 --threads 1,2,4,8 --cache hot,cold
cargo bench --bench hooks_overhead -- --model Llama-3.2-1B-Instruct.Q6_K.gguf
~/.cache/ember/llama.cpp/build/bin/llama-bench -m models/v03-ladder/llama-3.2-1b-q8_0.gguf -p 26 -n 64 -t 8 -r 3 -o json
# aggregation
python3 scripts/performance/analyze.py artifacts/performance-baseline/2026-08-10/raw
```

Raw data: `artifacts/performance-baseline/2026-08-10/raw/` (73 JSON files:
bench_*, profile_*, alloc_*, gen_*, trace_*), `.../llamacpp/*.json`,
`.../startup.json`. Re-run everything with the two scripts above (resumable,
skips completed runs).

Measurement limitations: no hardware counters (perf unavailable); powersave
governor noise (medians over ≥3–5 reps; 1-thread generate runs vary ±5%);
"cold" approximates a cold page cache (SSD-backed); bench-decode context grows
1→N (attention heavier in the middle of the run); trace mode measures the
generic path only; the inter-op residual is bounded but not fully decomposed.

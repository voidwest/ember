# Ember Phase 3 runtime report — decode hot path, load instrumentation, packed cache

Date: 2026-09-11
Baseline revision: `ae550f0f` (working tree with uncommitted `residual_patch` /
`cli_intervene` / `cli_handoff` user work left untouched)
Commits: `b008525a` (planned session fast path), `d3c99548` (logits into-buffer),
`4b4e6b04` (runtime schedule), `7cd2dcb9` (load instrumentation),
`0872c215` (packed cache)

## Host and environment

| item | value |
|---|---|
| CPU | Intel Core i5-1135G7 (Tiger Lake), 4 physical cores / 8 logical |
| OS / kernel | Arch Linux, kernel 7.1.5-2 |
| compiler | rustc 1.92.0, release profile `lto = "thin"` |
| governor | `powersave` (`intel_pstate`) |
| perf | available (paranoid=2, own-process user-space counters) |
| thermal state | 60–95 °C; package trip ≈ 87 °C; throttle counters already in the 10^5–10^6 range; a user browser (helium.AppImage) held ~100% of one core during the final matrix |
| measurement | `RAYON_NUM_THREADS=4`, `taskset -c 0-3`, warmups + medians, ABBA-interleaved A/B against the preserved baseline binary, bounded cool-downs |

Absolute tok/s on this box is throttled and drifts up to ~30% between
consecutive runs (observed in the Stage-0 matrix). Every before/after claim
below is either a deterministic counter (allocations, cache hits, phase
timings) or an interleaved median pair; sub-3% throughput differences are
treated as noise, matching the phase-2 convention.

## Baseline profile (Stage 0, revision `ae550f0f`)

Decode matrix, llama-3.2-1B ladder, 4 threads (two rounds; thermal drift visible):

| config | t/s r1 | t/s r2 | alloc events/token (median) | alloc bytes/token |
|---|---:|---:|---:|---:|
| q4_K_M `--execution planned` | 36.9 | 26.1 | 5 | 516 KB |
| q6_K `--execution planned` | 30.5 | 23.7 | 5 | 516 KB |
| q8_0 `--execution reference` (fast path) | 28.8 | 26.6 | 5 | 516 KB |
| q4_K_M `--execution reference` (CLI default) | 38.7 | 38.0 | **833** | 3.93 MB |

Paired planned-vs-reference on q4_K_M (3 rounds each, interleaved):
planned 37.5 t/s vs reference 37.9 t/s — the default reference path allocates
166× more per token and runs the same speed; allocation churn is not a
throughput bottleneck on this host.

`perf record` on planned q4 decode (16 tokens): `q4_k_dot_q8_k` 52.7%,
`q6_k_dot_q8_k` 34.6%, `dot_column` 2.3%, `compute_rope_freqs` 1.4% — ≈87% of
cycles in the two K-quant dots, reproducing the phase-2 profile.

Per-operator profile (planned q4, per token summed over layers): lm_head
5.80 ms, down 5.60 ms, up 4.83 ms, gate 4.78 ms, q 1.39 ms, o 1.37 ms.

Startup (llama-1B, `/usr/bin/time -v`, tiny decode so load dominates):

| case | wall | RSS | major faults |
|---|---:|---:|---:|
| q8_0 warm | 0.77 s | 1.57 GiB | 0 |
| q8_0 warm + `EMBER_PARALLEL_REPACK=1` | 0.44 s | 1.57 GiB | 0 |
| q8_0 cold | 1.70 s | 1.57 GiB | 632 |
| q8_0 cold + parallel repack | 1.97 s | 1.57 GiB | 3,859 |
| q4_K_M warm | 0.21 s | 0.81 GiB | 0 |
| q4_K_M cold | 1.23 s | 0.81 GiB | 2,784 |

Note on the allocation report: `caller_thread_alloc_events_per_token` divides
the total by the per-repetition token count, so multi-repetition runs
overstate it by the repetition factor. The `per_token_alloc_events` array is
the authoritative per-token figure and is what this report uses.

## Prioritized bottleneck list (as found)

1. Planned-path per-token setup: plan-cache mutex + BTreeMap probe + key
   construction + provenance clone per token; scheduler resolution evaluated
   even with profiling off; strategy/hash string compares.
2. Logits materialization: one vocab-sized allocation + copy per token in
   every path (the dominant steady-state allocation).
3. Load/first-token invisibility: no load-phase timers; `bench-decode` hid
   load and prefill from its JSON.
4. Per-process repacking: the Q8_0 VNNI repack (~0.5 s on llama-1B) paid on
   every process start; `EMBER_PARALLEL_REPACK` was the only mitigation.
5. Per-op thread selection was global, not inspectable.
6. Inspectable-plan gaps: lifetimes/ownership existed in JSON but not in the
   text view; no host-aware runtime schedule.

Not bottlenecks (evidence): allocator churn (planned == reference speed),
K-quant dot kernels (bandwidth-bound; phase-2 E1–E9 were all noise), the
Q8 fast path's DRAM throughput.

## Implemented changes

### 1. Planned decode session fast path (`b008525a`)

`forward_last_logits_planned` re-entered `execution_plan()` every token
(mutex, BTreeMap probe over a 7-tuple key, key construction, `Arc` clone,
provenance hash clone) and compared plan-hash/thread-strategy strings per
token. It also evaluated `planned_scheduler()` eagerly as an `OpTimer`
argument even when operator profiling was off.

Now the decode session stores the plan inputs it was built from (mode, hook
mode, active stages, capacity, thread count, provenance) plus the precomputed
`parallel_matvec` flag; a token reuses the arena when they all match, and the
plan cache is touched only when the configuration actually changes. Scheduler
labels are computed lazily inside the profiling guard.

Paired result (baseline vs change, ABBA): q4_K_M planned −2.4%, q8_0 fast
+3.0%, q4 reference +1.5% — all inside the noise band; the change is kept for
correctness/hygiene, not throughput.

### 2. Allocation-free logits (`d3c99548`)

`ForwardModel::forward_last_logits_with_cache_into` writes the `[1, vocab]`
logits into a caller buffer (materialize-and-copy default; Llama overrides
with a copy-free route: fast path → planned → reference, same order as the
allocating dispatch). The planned interpreter runs the final LM-head matvec
straight into the caller buffer when hooks are inactive; the Q8 fast path's
workspace accepts an optional output slice. `bench-decode` and the standard
generation loop keep one persistent logits buffer.

Measured (paired ABBA, llama-1B):

| config | alloc events/token | alloc bytes/token | t/s delta |
|---|---:|---:|---:|
| q4_K_M planned | 5 → **2** | 515,716 → **3,040** | +0.2% |
| q8_0 fast path | 5 → **2** | 515,716 → **3,040** | +0.5% |
| q4_K_M reference (control) | 833 | 3.93 MB | +2.8% (noise) |

The remaining 2 events / 3,040 bytes per token are rayon job structures under
pool contention (the documented Gate-E allowance). New k_parity test asserts
bit-identical logits and ≤2 allocations on the into-route.

### 3. Runtime schedule and inspectable plan detail (`4b4e6b04`)

`runtime_schedule::RuntimeSchedule::from_plan` reports, per matrix weight, the
kernel, shape, weight bytes, storage kind, plan-level parallel request, and
the resulting schedule (`serial` / `row-parallel-rayon` /
`column-parallel-rayon`), plus a host snapshot (threads, detected features,
ISA tiers) and arena/KV-per-token figures. Decisions call the same predicates
as the kernels: `k_quant_matmul::parallel_for_shape` (extracted from
`should_use_parallel`) and `simd::q8_decode_uses_row_parallel`; a boundary
test asserts schedule == `scheduler_name` for shapes around both thresholds.
`inspect plan` now prints the schedule, the scratch-region lifetimes
(offset/size/first..last op), and the per-tensor kernel/ownership map. The
serialized plan and its hash are unchanged (host state must not enter the
v0.5 semantic identity).

### 4. Load instrumentation (`7cd2dcb9`)

`loader::LoadTimings` + `load_gguf_with_k_strategy_report` separate mapping,
parsing, tensor materialization and the eager-conversion share; Llama records
packing time for the VNNI repack and the interleaved head. `bench-decode` now
reports `prefill_ns` and `first_token_ns` medians and a `load_report`
(loader phases, model build, packing, residency snapshots with RSS/peak/PSS
and minor/major faults at `load_start` and `model_built`). The decode median
still excludes load and prefill.

Measured (llama-1B, warm page cache):

| phase | q4_K_M | q8_0 |
|---|---:|---:|
| mmap | 4.5 µs | 4 µs |
| parse (header + tensor table + accounting) | 29.8 ms | 32.8 ms |
| tensor materialization | 0.43 ms | 0.37 ms |
| eager dequant (share of the above) | 0 ms | 0 ms |
| model build | 81 ms | 663 ms |
| — VNNI packing | 6 µs (no Q8 tensors) | **478 ms** |
| — interleaved head | 0.5 ms (head not wide enough / absent) | **99 ms** |

The instrumentation's first payoff is that the expensive phase is now
attributable: Q8 model build is almost entirely packing, while the load
itself (mmap + parse) is ~35 ms.

### 5. On-disk packed-weight cache (`0872c215`, opt-in)

`packed_cache.rs` persists the VNNI packed layout, keyed by source path,
size, mtime, GGUF metadata/tensor-table fingerprint, layout id and crate
version. Reads validate magic/version/header digest/bounds/non-overlap and
the exact entry length for the requested shape; `EMBER_PACKED_CACHE_VERIFY=1`
additionally checks a stored payload digest. Payloads stream into a sibling
temporary file and are published by rename, so concurrent writers/readers
never observe a partial file; every failure degrades to in-memory packing.
`QuantizedWeightVnni` storage is now owned-or-mapped, so a hit maps the file
once and hands out zero-copy ranges (this is what makes the hit fast).

Measured (llama-1B q8_0, warm page cache; `bench-decode --tokens 2`):

| scenario | model build | packing | cache |
|---|---:|---:|---|
| no cache | 861–884 ms | 620–648 ms | — |
| first run (writes 987 MiB) | 1248 ms | 1010 ms | 0 hits / 112 misses |
| cache hit | **163–182 ms** | **0.1 ms** | **112 hits** / 0 misses |

End-to-end `generate` (deterministic prompt, 16 tokens): byte-identical
output, wall 2.81 s → 2.12 s. The packed bytes are proven identical to the
in-memory repack by unit test, and the cached run's output equals both the
uncached packed run and the generic (unpacked) kernel run — that is the full
equality chain generic == packed == cached.

Scope note: the cache is wired into `bench-decode` and the default `generate`
path; other constructors can adopt `from_loader_with_max_seq_len_cached`.
The interleaved head packing (~99 ms) is not cached yet.

## Audited and rejected (with evidence)

- **Per-op thresholds for the always-parallel Q8 packed/interleaved kernels.**
  Every op routed through those paths is bandwidth-bound on llama-1B at 4
  threads: down/gate/up ≈ 46 GB/s, q/o ≈ 41 GB/s, k/v ≈ 33 GB/s, lm_head
  ≈ 37 GB/s (weights streamed per token ÷ median op time). The smallest such
  op (k/v, 512×2048 = 1.05M MACs) sits exactly at the existing
  `PARALLEL_Q8_DECODE_MIN_WORK` threshold and still reaches 33 GB/s;
  serializing it cannot help. No tiny op dispatches to a parallel kernel in
  the models in scope, so no threshold was added (no dead guard code).
- **Kernel micro-optimizations / prefetch / AVX-512 tier**: phase-2's
  keep/revert ledger already rejected these as noise on this host; the new
  profile reproduces the same 87% dot-kernel distribution.
- **Execution-mode default flip**: measured planned == reference throughput
  (37.5 vs 37.9 t/s) on q4_K_M, so there is no throughput case for changing
  the default in this milestone.
- **Buffer aliasing (`shared_with`)**: memory-only change, no evidence of a
  throughput benefit.

## Correctness and test evidence

- `cargo test --all-targets`: 443 lib tests + all integration targets pass.
  One pre-existing flake in the unrelated `cli_diff` track
  (`externals_evaluate_concurrently_not_sequentially`) fails intermittently
  under the full bin suite; it also fails on pristine sources and a different
  test in that family fails on other runs.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean;
  `cargo fmt --check`: clean.
- Real-model gates: `scripts/validate_k_parity.sh` — planned parity,
  inactive-hook parity, Gate-E allocation contract (5/token on the old
  allocating API) and the new into-route test (bit-identical, 1 allocation)
  pass. `production_q8_k_keeps_oracle_behavior_across_frozen_prompts` fails
  on a pre-existing load-budget check (1 GiB cap vs the q4_K_M embedding’s
  1.27 GiB eager expansion), unrelated to this work.
- Q8 equality chain (llama-1B q8_0, deterministic greedy 16 tokens):
  packed == unpacked, cached == uncached, byte-identical text.
- Packed-cache unit tests: round-trip byte identity, shape/identity
  invalidation, corrupt/truncated file fallback, concurrent writers,
  disabled no-op.
- `inspect plan` plan hash for a fixture model is unchanged
  (`ExecutionPlan` serialization was not modified).

## Final matrix (baseline binary vs final binary, ABBA)

Raw runs: `artifacts/performance-phase3/2026-09-11/pair-final2/` (llama-1B
ladder, 4 threads, package at 93–94 °C with the user's browser holding a
core; the control row bounds the noise band):

| config | baseline t/s | final t/s | delta | allocations/token |
|---|---:|---:|---:|---|
| q4_K_M planned | 27.19 | 26.90 | −1.1% | 5 → **2** |
| q8_0 fast path | 22.17 | 22.62 | +2.0% | 5 → **2** |
| q4_K_M reference (control, unchanged path) | 20.86 | 21.01 | +0.7% | 833 (unchanged) |

Throughput is unchanged within the ±2% noise band; the deterministic
allocation counts and the packed-cache phase timings are the milestone's
measured wins.

## Follow-up milestone (2026-09-11, same session)

Commits `92e8dfb9`, `7d000ad7`, `674b95b2`.

### Default decode mode flipped to `planned` (`92e8dfb9`)

The planned interpreter was already the leaner route at equal throughput;
the CLI defaulted to `reference` for historical reasons. Parity evidence
before the flip:

- 24 decode steps x 6 frozen prompts on Q4_K_M: greedy tokens identical,
  per-step logits inside the frozen 1e-3 Gate B envelope
  (`v04_planned_matches_reference_real_model`, extended run);
- 18/18 greedy runs byte-identical between modes across Q4_K_M / Q6_K /
  Q8_0 (English, code-ish and Arabic prompts);
- end-to-end: `bench-decode` with no `--execution` flag now reports
  2 allocations/token on Q4_K_M (was 833); `--execution reference` still
  selects the oracle path.

`planned-fused` still awaits its gates; KV snapshot commands keep their
explicit defaults. The execution concept is now part of the run-manifest
execution identity (`mode.execution`), closing a provenance gap: two runs
differing only in decode mode previously produced the same identity digest.
Recorded as the D1 amendment in `docs/v04-execution-contract.md`.

### Packed cache extended to the interleaved head, format v2 (`7d000ad7`)

With only VNNI cached, the interleaved lm-head repack became the dominant
load cost. The cache now stores both layouts in one file (entries carry a
kind and an optional second byte range), both as zero-copy mapped ranges:

| scenario | model build | VNNI pack | interleaved pack | entries |
|---|---:|---:|---:|---|
| no cache | 890 ms | 646 ms | 140 ms | — |
| first run (writes ~1.3 GiB) | 1221–2116 ms | 909–1624 ms | 209–376 ms | 0 hits / 113 misses |
| cache hit | **110–134 ms** | **0.07 ms** | **0.01 ms** | **113 hits** |

Validation additionally checks per-kind expected lengths, full payload
coverage and non-overlap; a v1 file is treated as stale and rebuilt.
Deterministic greedy output is byte-identical to the uncached path.
`Linear.interleaved` is boxed (one load-time allocation) because the packed
struct pushed `Linear` past clippy's `large_enum_variant` threshold in
Gemma-4's head enum.

### Parse phase measured (`674b95b2`)

Split into metadata vs tensor table: **parse 52.0 ms = metadata 51.6 ms +
tensor table 0.09 ms** on Llama-3.2-1B. Metadata decoding is the tokenizer
arrays (~128k tokens + ~280k merges, each an owned UTF-8-validated String);
the array reader already reserves exact capacity with O(1) budget checks, so
there is no cheap win — cutting it needs lazy key-on-demand metadata or
borrowed string ranges, which changes `GgufValue` for every consumer.

### Still deferred

The bounded packed/tiled K-quant layout experiment needs a cool host with a
stable clock: this session ran at 93–94 °C with a browser holding a core, so
any kernel-level A/B would have been noise (per the phase-2 standard).

## Remaining bottlenecks

1. **K-quant decode kernels** — 87% of cycles in `q4_k_dot_q8_k` /
   `q6_k_dot_q8_k`; bandwidth-bound at 33–46 GB/s. Any further kernel work
   needs a cool host and a current llama.cpp comparison (phase-2 conclusion,
   unchanged).
2. **No packed layout for K-quant** — Q8_0 has VNNI tiles/interleaved head;
   Q4_K/Q6_K stream row-contiguous blocks with no repack. A packed K layout
   is a new-layout project and must beat the bit-identical x4 path.
3. **GGUF metadata decoding ~51.6 ms** — the tokenizer arrays dominate
   (~128k tokens + ~280k merges as owned UTF-8-validated strings). Fixing it
   needs lazy key-on-demand metadata or borrowed string ranges into the
   mapping, which changes `GgufValue` for every consumer (inspect, manifests,
   probes). Not a cheap win; measured in this milestone.
4. **Arena score scratch sized by context** — 16 MiB at 128k context for a
   single decode token.
5. **Shared global rayon pool across GUI requests/agent tools** — real
   oversubscription risk, out of scope here.
6. **Packed cache policy** — addressed in the 2026-09-12 follow-up below (on by
   default; gemma4 gate/up covered). Open items there: no eviction, no
   free-space pre-check.

## Next milestone recommendation

1. Decide the packed-cache default: with both layouts cached, build drops to
   ~110 ms; weigh that against the ~1.3 GiB per model and the cold-write cost
   (a size/mtime/inode key is cheap, so a policy like "cache when the target
   dir is writable and has room" is viable). Extend it to gemma4 gate/up.
2. On a cool host: one bounded packed/tiled K-quant layout experiment measured
   against the bit-identical x4 path, plus a fresh llama.cpp comparison.
3. Lazy GGUF metadata — **closed (2026-09-12)** with a smaller change than the
   lazy-key design: no consumer reads the tokenizer arrays (tokenizers come
   from `tokenizer.json`), so the loader now validates them in place and
   stores `GgufValue::SkippedArray { element_type, elements }` instead of
   ~10^5-10^6 owned values. Measured (5 interleaved runs, taskset 0-3):
   metadata parse 37.7 -> 10.8 ms on Llama-1B Q8_0 (loader total 34 -> 17 ms),
   warm run wall 0.295 -> 0.21 s; Gemma 4 E2B metadata ~20 ms (no pre-change
   binary could load the file). Open: the remaining ~11 ms is the per-string
   length walk — bulk chunked parsing would be needed to go lower.
4. `planned-fused` for the default — **measured and closed (2026-09-12)**:
   fused decode is within noise of `planned` on current kernels (llama-1B
   Q4_K_M 36.6 -> 36.3 t/s at 4 threads; Q6_K 27.5 -> 28.3; median of 2
   interleaved 48-token runs), with identical greedy output on all four
   primary models. The default stays `planned`; see the contract amendment in
   `docs/v04-execution-contract.md`.

## Reproduction

```bash
cargo build --release
# decode + allocation + load report
RAYON_NUM_THREADS=4 taskset -c 0-3 target/release/ember --k-strategy x86 bench-decode \
  --model models/v03-ladder/llama-3.2-1b-q4_k_m.gguf --arch llama \
  --tokens 32 --warmups 1 --repetitions 3 --execution planned --allocations --profile-operators
# runtime schedule
target/release/ember inspect models/v03-ladder/llama-3.2-1b-q4_k_m.gguf plan
# packed cache (first run writes, second run hits)
EMBER_CACHE_DIR=/tmp/ember-packed EMBER_PACKED_CACHE=1 RAYON_NUM_THREADS=4 \
  taskset -c 0-3 target/release/ember bench-decode \
  --model models/v03-ladder/llama-3.2-1b-q8_0.gguf --arch llama --tokens 2 --warmups 0 --repetitions 1
```

## Follow-up milestone (2026-09-12): packed cache on by default

The first next-milestone recommendation is implemented.

### Decision: enabled by default

`PackedCache::for_loader` now enables the cache unless `EMBER_PACKED_CACHE=0`
(any other value, or unset, enables). Rationale: the repack is paid on every
process start, a validated hit removes essentially all of it, and every failure
path already degrades to in-memory packing. Writability and free space are not
probed up front: an unwritable or full cache directory disables writes for that
run with one warning, removes the partial temp file, and continues. The
per-model cost is ~1.3 GiB for a 1B Q8_0 model, written once.

### Gemma gate/up coverage

`Linear::prepare_packed_decode_cached` (`model.rs`) mirrors the llama helper: a
validated entry replaces the repack, otherwise the freshly packed layout is
recorded. `Gemma4::from_loader_cached` uses it for `ffn_gate`/`ffn_up`, and the
CLI generation and bench-decode gemma4 arms open the cache. The public
`from_loader` stays cache-free.

### Measurements (Llama-3.2-1B Q8_0, `RAYON_NUM_THREADS=4`, taskset 0-3, tokens=2)

| configuration | model build | cache state |
|---|---:|---|
| cache off (`EMBER_PACKED_CACHE=0`) | 670 ms median (n=3) | — |
| first run, fresh cache dir | 746 / 749 / 751 ms | 113 misses; writes 1.31 GiB |
| warm cache (second+ run) | 83 / 84 / 83 ms | 113 hits |

First-run overhead over no-cache is ~80 ms on this NVMe host; the warm win is
~590 ms (~8x). Back-to-back fresh writes once measured 5-20 s when dirty
writeback throttled the device; spacing runs with `sync` removes that, so the
write cost is IO-pressure dependent, not a fixed surcharge.

### Gemma 4: measured after the loader fixes

The loader-cap issues recorded in the first version of this section were fixed
as a follow-up: aggregate metadata values 1M -> 4M, per-tensor encoded bytes
1 GiB -> 4 GiB, RoPE `context * head_dim` 2^25 -> 2^27, plus two Gemma 4
geometry fixes (double-wide MLP on KV-shared layers; packed 2D Q8_0 PLE
tensor) and the missing `finish_write` call on this path. See the changelog.
`gemma-4-E2B-it-Q8_0.gguf` now loads, generates coherently, and uses the
cache; the 12B variant (no PLE tensors) passes the loader checks.

| configuration | model build | cache state |
|---|---:|---|
| cache off | 2,062 / 2,279 ms | — |
| first run, fresh cache dir | 2,184 ms | 70 misses; writes 1.10 GiB |
| warm cache | 1,542 / 1,593 / 1,601 ms | 70 hits |

That is ~0.5-0.7 s (~25%) off a build dominated by non-packing work — not the
~8x seen on Llama, where the repack is the dominant term.

### Cache-key cost

The first version of this follow-up hashed every metadata value with `{:?}`
when building the cache key; Gemma-scale headers formatted ~1.3M values per
cache-enabled load. The key now hashes scalars exactly and bounds string/array
contents (packed layouts depend only on tensor bytes and shapes, still bound by
the tensor-table fingerprint). Warm-run wall time, same protocol:

| model | before | after |
|---|---:|---:|
| Llama-3.2-1B Q8_0 | 342 ms | 274 ms |
| Gemma 4 E2B | 2,340 ms | 1,927 ms |

### Eviction

The directory is pruned after every successful publish, so default-on caching
cannot grow without bound: entries are evicted least-recently-used first until
the directory fits `EMBER_PACKED_CACHE_BYTES` (default 8 GiB, `0` disables),
and `.tmp-*` files older than a day (left by crashed writers) are removed.
The file just published counts toward the budget but is never evicted, so a
single oversized entry stays rather than being deleted and immediately
rewritten. Verified end to end: a 1.31 GiB Llama cache was evicted by a Qwen
run with a 1 GiB budget, leaving only the new 248 MB entry.

### Remaining

- No free-space pre-check (write failures degrade lazily, per the module docs);
  eviction bounds the directory but not the disk's free space.

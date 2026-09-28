# Ember Phase 2 Optimization Report

Date: 2026-08-27  
Baseline revision: `673de6a0da73af3a25989b59dffc8a508d210b36`  
Final source change: optional parallel Q8 VNNI repacking in `src/quant.rs`

This was an isolated, keep/revert pass. The worktree already contained unrelated
user changes (three deleted `data/test_*.npy` files and
`examples/tmp_vits_ids.rs`); those were not touched.

## Host and environment

| item | value |
|---|---|
| CPU | Intel Core i5-1135G7 (Tiger Lake, 11th gen) |
| topology | 1 socket, 4 physical cores, 8 logical CPUs; SMT siblings 0/4, 1/5, 2/6, 3/7 |
| caches | L1d 192 KiB, L2 5 MiB, L3 8 MiB |
| RAM | 15 GiB visible (16 GiB class host) |
| NUMA | one node (node0 CPUs 0-7) |
| OS | Arch Linux, kernel 7.1.5-arch1-2, PREEMPT_DYNAMIC |
| compiler | rustc 1.92.0, LLVM 21.1.3 |
| release profile | `lto = "thin"`; default codegen units; no `target-cpu=native` |
| governor | `powersave`, `intel_pstate`; turbo enabled |
| ISA | AVX2/FMA/F16C/SSSE3 and AVX-512 (AVX-512 K path remains opt-in) |
| normal measurement | `RAYON_NUM_THREADS=4`, `taskset -c 0-3`, release build |

Temperatures were not stable over the session. A spot check after the runs was
63--66 C, but the kernel thermal counters had accumulated
955,174 core and 1,841,543 package throttle events. Consequently, small
throughput differences are treated as noise, not as wins. Runs were short and
interleaved; large startup effects were retained only where the effect repeated.

## Initial baseline

Representative model files were the local v0.3 ladder:

| model | SHA-256 prefix/full | quantization | workload |
|---|---|---|---|
| `llama-3.2-1b-q8_0.gguf` | `da49f51ced8c15546e7779beb677fb53eb5d0b3b38ac4607ac60d58d77074823` | Q8_0 | 32 timed decode tokens, 1 warmup, 3 repetitions |
| `llama-3.2-1b-q4_k_m.gguf` | `26bac8efd811cb41a80db4393dbe5c8360abd54b98954ec766aa4ba7dacc0bc5` | Q4_K_M | 32 timed decode tokens, 1 warmup, 3 repetitions |
| `llama-3.2-1b-q6_k.gguf` | `4bf385159856b7c50a938b1228112318d9f99238a76880ea0f6381ab879982b3` | Q6_K | 24 timed decode tokens, 1 warmup, 2 repetitions |

The first clean baseline pass at four physical cores measured 29.97 t/s (Q8)
and 33.69 t/s (Q4). Repeated final-matrix baseline medians were 30.99 t/s
(Q8) and 33.61 t/s (Q4). A 26-token Arabic prompt generated with Q4 measured
about 88.6 tok/s prefill and 38.7 eval/s decode; this timing includes neither
model loading nor tokenization in the decode-only benchmark, and is not a TTFT
measurement.

Warm Q8 startup (model load plus the short benchmark command) was 1.24 s
median for the baseline. Cold startup after `POSIX_FADV_DONTNEED` was 2.18 s,
with 632 major faults. Peak RSS was approximately 1.64 GiB. `bench-decode`
explicitly excludes model load, prefill, tokenization, and sampling from its
reported decode timing.

## Initial profile

`perf record` on an 8-token Q4 decode (four physical cores) found:

| symbol | cycles |
|---|---:|
| `q4_k_dot_q8_k` | 50.51% |
| `q6_k_dot_q8_k` | 34.16% |
| `tensor::compute_rope_freqs` | 5.50% (2.61% self) |
| `k_quant_matmul::dot_column` | 1.74% |
| `parallel_body::recurse` | 1.31% |

Thus approximately 85% of sampled cycles were in the two K-quant decode
kernels. A representative `perf stat` included process startup and reported
5.272 task-clock seconds, 14.253 billion cycles, 28.572 billion instructions
(IPC 2.00), zero context switches, zero CPU migrations, and 23,428 page
faults. It is not a kernel-only counter measurement.

The Q4 operator profile after the final change remains dominated by LM head
(5.37 ms per timed-token aggregate), MLP down (5.18 ms), gate/up (4.41 ms
each), then Q/O (about 1.27/1.26 ms). The dominant bottleneck did not move.

## Experiment ledger

All matmul experiments used the existing pinned model harness and checked its
output checksums. Percentages below are median/paired comparisons, not claims
about a cool-machine sustained rate.

| ID | hypothesis | change | baseline | candidate | delta | correctness | decision |
|---|---|---|---:|---:|---:|---|---|
| E1 | Q4 four-row min unpack was repeated per row | hoist `mins8/mins16` in Q4 x4 | direct rows=17 parallel median 376.9 us | 359.3 us in one pass; interleaved result inconsistent | not repeatable | K tests pass; disassembly was instruction-identical | revert |
| E2 | Q6 four-row scale unpack was repeated per row | hoist `scale_words` in Q6 x4 | no stable paired gain | mixed direct results; end-to-end Q6 null/slower | noise | K tests pass | revert |
| E3 | 128-column leaf floor left 512-wide K/V underused | lower floor to 64 | rows=1 K 20.7--29.5 us | 25.6--27.6 us (often slower); rows=17 K/V improved but O worsened | shape-dependent, no end-to-end win | K tests pass | revert |
| E4 | pre-splitting K quants would accelerate prefill | `EMBER_PRESPLIT=1` | Q4 gate rows=17, 2.23 ms | 2.58 ms; O also slower | -15.6% | existing presplit parity tests pass | reject |
| E5 | AVX-512 K dot would reduce decode instructions | existing opt-in `EMBER_K_AVX512=1` | Q4 gate rows=1, 298.7 us | 333.1 us | -11.5% | checksums identical | keep opt-in off |
| E6 | OnceLock ISA lookup per output column was costly | cache AVX-512 choice per matmul | perf `dot_column` self 1.74% | wrapper self 2.05%; total cycles 5.60 -> 5.69B | regression | K tests pass | revert |
| E7 | software prefetch would hide K weight latency | prefetch next Q4/Q6 block | paired Q4/Q6 decode averages changed by about -0.3/-0.2% | no repeatable effect | noise | K tests pass | revert |
| E8 | native codegen could improve dispatch and scalar glue | separate `-C target-cpu=native` build | Q8 paired baseline about 31.69 t/s | about 32.20 t/s | ~+1.6% locally; Q4 null | binary built and ran; not a portable source change | deployment option only |
| E9 | one codegen unit would improve inlining | separate `-C codegen-units=1` build | Q8 about 31.86 t/s | about 31.81 t/s | null | build passed | reject |
| E10 | parallelizing every VNNI output tile would shorten load | unconditional tile `par_chunks_mut` | warm 1.22 s | 0.90 s | ~-26% warm | packed bytes and logits identical | reject default: cold regressed |
| E11 | grouping 16 tiles would preserve cold locality | unconditional grouped parallel repack | warm about 1.29 s | 0.95 s | ~-26% warm | packed bytes identical | reject default: cold regressed |
| E12 | warm deployments can opt into grouped repack safely | grouped repack behind `EMBER_PARALLEL_REPACK=1`, sequential default | warm 1.24 s | 0.93 s | ~-25% warm startup | 33 quant tests, full suite, exact logits | **keep as opt-in** |
| E13 | final tree must not regress inference | final source plus E12, decode ABBA | Q8 30.99 t/s; Q4 33.61 t/s | Q8 30.88; Q4 33.67 | -0.34%; +0.16% | generated text and logits unchanged | keep |

E1's apparent direct-kernel gain was specifically checked with `objdump`:
the baseline and candidate Q4 x4 instruction streams were identical apart
from LLVM anonymous-constant symbol naming. It was therefore not a real codegen
optimization.

## Kept optimization

### Optional parallel Q8 VNNI repacking (E12)

**Bottleneck.** Q8 Llama startup spends substantial CPU time transforming
row-contiguous Q8 weights into the 16-output VNNI layout. This is outside the
steady-state decode loop, but it contributes directly to model-ready latency.

**Implementation.** `QuantizedWeightVnni::from_quantized` now packs adjacent
16-output tiles in groups of 16. With `EMBER_PARALLEL_REPACK=1`, disjoint groups
are processed with Rayon. The default remains the original sequential order:
parallel page faults on an mmap-backed cold file were harmful. The packed output
layout and byte order are unchanged. A unit test compares sequential and
parallel output bytes, shapes, and block counts.

**Measured effect.** On the Q8 Llama file with a warm page cache and four
workers, startup fell from 1.22--1.26 s to 0.92--0.94 s (approximately 25--27%).
RSS stayed at approximately 1.64 GiB. Qwen Q8, whose path does not construct
this Llama VNNI layout, showed no meaningful effect.

**Correctness.** `cargo test --all-targets`, `cargo clippy --all-targets
--all-features -- -D warnings`, and Python tests all passed. Generated text was
bit-identical for a deterministic prompt. `--dump-logits` comparison between
baseline and opt-in builds gave max absolute difference 0.0 and identical
argmax (token 12366 for the test prompt).

**Tradeoff.** Cold Q8 startup became 2.45 s versus 2.17--2.19 s sequential;
major faults rose from 632 to 3,865--3,904. The flag is therefore intentionally
opt-in for warm page-cache/server deployments, not a universal default.

## Reverted experiments

- Q4/Q6 metadata hoists: LLVM already hoisted the Q4 sequence; Q6's smaller
  function did not produce a repeatable end-to-end result.
- Leaf floor 64: helped a narrow rows=17 K/V microbench but hurt other shapes,
  especially output 2048; the model-level result was noise.
- Presplit quants, AVX-512 K, and software prefetch: slower or neutral on this
  Tiger Lake host.
- Per-column ISA caching: extra plumbing increased the dispatch wrapper's
  measured self time.
- Unconditional parallel repacking: a substantial warm win but unacceptable
  cold mmap/page-fault regression. This is why only the gated form remains.
- `target-cpu=native` and one-codegen-unit builds: either machine-specific or
  noise-bound, and neither belongs in portable defaults.

No failed experimental kernel or threshold code remains in the source tree.

## OS-level findings

### Affinity, topology, and SMT

Four physical workers were the stable default. A short sweep produced Q4
13.70/24.44/37.78/38.89 t/s and Q8 12.30/21.82/32.61/33.52 t/s at
1/2/4/8 workers respectively. The 4->8 improvement was only about 3% and was
not thermally controlled enough to ship as policy. Four workers avoid SMT
contention and are a sensible deployment setting, but Ember does not hard-code
CPU ids. Pinned and unpinned perf samples both recorded zero migrations.

### Scheduler behavior

`nice -n 10` versus normal priority produced Q4 37.29 and 37.84 t/s versus
37.44 and 37.12 t/s in paired short runs: no signal. No real-time policy was
attempted. No affinity code was added.

### mmap, prefault, and page cache

The Q4 model measured 1.66 s cold versus 0.66 s warm in a direct startup pass
(2,760--2,774 versus zero major faults). The Q8 file's baseline was about
2.18 s cold versus 1.22--1.26 s warm. Explicitly reading/prefaulting the file
would move I/O cost outside the process rather than improve steady-state
throughput, so no prefault policy was shipped. The kept repack flag documents
this distinction and its cold-page cost.

### Huge pages / TLB

THP was set to `always`, but the live model mapping showed 4 KiB
`KernelPageSize`, zero `FilePmdMapped`, and `THPeligible: 1`. A perf sample
reported 2.138 billion dTLB loads and 110,428 dTLB load misses while including
startup; this is not evidence of a throughput-limiting TLB problem. No
`MADV_HUGEPAGE` or explicit huge-page complexity was added.

### NUMA

The host has one NUMA node. There was no cross-node placement question and no
NUMA policy was added.

### Page cache / readahead

`POSIX_FADV_DONTNEED` produced repeatable cold/warm separation. Parallel mmap
repack touched distant source ranges concurrently and amplified major faults;
the OS/deployment environment should own readahead and cache warming. Ember's
safe default remains sequential.

### Thermal behavior

The machine uses `intel_pstate` with powersave and a reported 0.4--4.2 GHz
range. The active thermal trip is around 87 C; historical session notes and
throttle counters show that long runs can downclock. All small decode deltas
were therefore rejected unless repeated and materially larger than observed
variance. The warm repack result is a startup wall-time result, not a claim of
higher sustained token throughput.

## Final combined benchmark

ABBA, warm page cache, four physical cores, release binaries, equivalent
workloads:

| workload | original median-of-paired medians | final with `EMBER_PARALLEL_REPACK=1` | delta |
|---|---:|---:|---:|
| Llama Q8 decode, 32 tokens | 30.99 t/s | 30.88 t/s | -0.34% (noise) |
| Llama Q4_K_M decode, 32 tokens | 33.61 t/s | 33.67 t/s | +0.16% (noise) |
| Q4 26-token prefill | 88.60 tok/s | 88.06 tok/s | -0.61% (noise) |
| Q8 startup, warm | 1.24 s | 0.93 s | **-25%** |
| Q8 startup, cold | 2.18 s | 2.45 s | **+12% (regression)** |

The final source's default behavior is sequential repacking, so ordinary users
see no cold-start regression. The warm startup improvement requires the explicit
flag and an appropriate deployment/cache policy.

## Final profile

The final Q4 profile still has Q4/Q6 K-quant dots as the dominant sampled
cycles (the startup-only repack change does not alter them). LM head and the
three MLP projections remain the largest timed operators. The system is still
compute/weight-stream limited in the K kernels, not dispatch- or allocator-
limited. The Q8 steady-state result is intentionally unchanged.

## Allocator and memory-system findings

The existing allocation report showed 4--6 caller-thread allocations per Q8
fast-path token, dominated by the returned approximately 516 KiB logits
buffer; the K planned path is similarly dominated by its API-owned logits
materialization. The profiled hot path is K-quant arithmetic and weight
traffic, not allocator churn, so no allocator dependency or unsafe buffer
reuse was added. The existing scratch arena remains unchanged.

A separate `-C target-cpu=native` build improved one short Q8 decode comparison
by about 1.6% but was null for Q4 and is machine-specific. A separate
`-C codegen-units=1` build was noise-bound. Neither was made a portable default.

## Remaining opportunities

### High confidence next work

1. Repeat K-quant kernel work on a cool host and compare against a current
   llama.cpp build with matching compiler flags and perf counters.
2. Investigate a cache-aware cold/warm repack policy only if a reliable resident
   page test or OS readahead API can avoid the observed cold regression.
3. Profile long-context attention separately; its cache traffic was small at the
   short benchmark contexts used here.

### Speculative ideas

- A dedicated AVX-512 four-row prefill kernel could be tested independently;
  the existing AVX-512 single-row path is slower here and the prefill path has
  different register pressure.
- Repacked/tiled K-quant GEMM may help prefill, but it is a new layout project
  and must be measured against the existing bit-identical x4 path.

### Intentionally rejected

- Hard-coded CPU ids, real-time scheduler policies, unconditional SMT use,
  unconditional THP, explicit page locking, and allocator replacement.
- Portable default `target-cpu=native`, `codegen-units=1`, or weaker numerical
  gates.
- Keeping sub-percent kernel/threshold changes under this thermally unstable
  host.

## Reproduction commands

```bash
cargo build --release
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
.venv/bin/python -m pytest tests -q

# steady-state decode
RAYON_NUM_THREADS=4 taskset -c 0-3 target/release/ember \
  --k-strategy x86 bench-decode \
  --model models/v03-ladder/llama-3.2-1b-q4_k_m.gguf --arch llama \
  --tokens 32 --warmups 1 --repetitions 3 --execution planned

# warm-startup optimization (opt-in; measure with a warm page cache)
EMBER_PARALLEL_REPACK=1 RAYON_NUM_THREADS=4 taskset -c 0-3 \
  target/release/ember bench-decode \
  --model models/v03-ladder/llama-3.2-1b-q8_0.gguf --arch llama \
  --tokens 8 --warmups 1 --repetitions 1 --execution reference
```

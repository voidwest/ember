# Packed Q8 lifecycle and selective-residency investigation

Date: 2026-07-28

Verdict: **durable replacement packing remains viable**, on the specific
Llama-3.2-1B Q8_0 / Tiger Lake workload measured here. This is evidence for
continuing the hypothesis, not a general paper claim.

## 1. Baseline state

The preserved baseline is commit
`694f0331446b26c31e343f4807ed106b453d37e7` plus the pre-existing dirty packed
backend diff documented in `packed-q8-phase0.md`. Before this session, ordinary
generic prefill restored all row-contiguous projection residency, so the
replacement-memory thesis had failed for a normal prefill/decode session.

The final tracked `git diff --binary HEAD` SHA-256 is
`3e14188cd141604fd2de5b5e4e9379784478166483b0b3a0f2ab43512b4a532b`.
Untracked research notes, the driver, and `src/residency.rs` are deliberately
preserved and are not included in Git's unstaged-diff hash.

No scheduler, tile, compensation, ISA, or quantization-format work was done in
this investigation. The tied embedding and LM head remain excluded from the
16-output packed projection representation. Qwen remains generic.

## 2. Exact question tested

Can explicit packing lifecycle order and an existing-kernel projection
selection keep source projection pages nonresident through decode, while
preserving deterministic output and most of the all-projection speedup?

The same release binary exposed these lifecycle modes:

- A: control: generic prefill, generic decode;
- B: pack and evict before generic prefill, packed decode;
- C: generic prefill, then pack and evict, packed decode;
- D: pack and evict before prefill, re-evict after prefill, packed decode;
- E: duplicate packed representation with eviction disabled.

Selective modes F-J use D's durable lifecycle:

- F: gate/up;
- G: all MLP projections;
- H: Q/K/V/O;
- I: gate/up plus Q/K/V/O;
- J: all eligible projections.

## 3. Experimental design

Each trial ran in a fresh process with 4 Rayon workers pinned to physical CPUs
0-3. It used a six-token generic prefill and produced 128 deterministic greedy
tokens. Every phase boundary recorded `/proc/self/smaps_rollup`,
`/proc/self/status`, and `/proc/self/stat`: RSS, peak RSS, anonymous PSS,
file-backed PSS, minor faults, and major faults.

Three repetitions were run in deterministically shuffled mode order. Every
child process began at or below 80 C package temperature. D and J deliberately
repeat the same configuration as a run-order validity control. Their medians
were 30.888 and 30.827 decode evaluations/s, a 0.2% difference. An earlier mode-grouped run
placed them 13% apart and was rejected for selective-throughput interpretation.

Procfs hooks ran only outside phase timers. They added 0.9-1.7% to internal
whole-process time. In a separate two-repetition measured-versus-timing-only
audit, hooks did not reduce decode or prefill timing: control medians were
24.73 versus 24.16 evaluations/s and D medians were 32.97 versus 32.68 evaluations/s. The
external process times below include hooks; phase and time-to-first-token work
timers exclude them.

Break-even is:

```text
ceil((variant predecode work - control predecode work)
     / (control decode ns/evaluation - variant decode ns/evaluation)) + 1
```

The added one is the token produced by generic prefill before decode savings
begin. The calculation describes this one long-lived process and is not yet a
cross-workload planner.

## 4. Code changes

- `src/residency.rs`: low-frequency Linux phase recorder plus a timing-only
  perturbation-audit mode.
- `src/main.rs`: `bench-lifecycle` CLI, modes A-E, selections F-J, explicit
  generic prefill and trait-dispatched packed decode, deterministic JSON.
- `src/llama.rs`: constructor that suppresses automatic packing, explicit
  projection-selection groups, separately timed packing and re-eviction.
- `src/model.rs` and `src/quant.rs`: separate existing packed-layout
  construction from source-page advice and return advice success/failure.
- `scripts/benchmark_packed_lifecycle.py`: fresh-process orchestration,
  affinity, randomized order, temperature gating, parity, break-even, raw JSON,
  and Markdown/JSON summaries.

The experimental path is selected only through explicit CLI options. Normal
automatic packing and `EMBER_LLAMA_PACKED_Q8=0` rollback behavior remain.

## 5. Files and line ranges changed

Line numbers refer to the formatted session state:

- `src/residency.rs:1-176`
- `src/main.rs:203-313`, `src/main.rs:476`,
  `src/main.rs:1559-1847`, `src/main.rs:2706-2738`
- `src/llama.rs:1086-1133`, `src/llama.rs:1737-1750`,
  `src/llama.rs:1891-2038`
- `src/model.rs:307-373`
- `src/quant.rs:226-252`
- `src/lib.rs:1-19`
- `scripts/benchmark_packed_lifecycle.py:1-533`
- `.gitignore:29-31`
- `docs/packed-q8-research-memo.md`

## 6. Correctness validation

All 30 randomized A-J trials produced
`fnv1a64:9f8e8158645ba677` for the generated token IDs. The rejected
mode-grouped pass and the two-mode perturbation audit produced the same hash.
The output began with ` Paris.` and was deterministic across lifecycle and
selection changes.

Validation after implementation:

- 109 active tests passed; 8 benchmark/sweep tests remained ignored;
- packed-kernel versus row-contiguous parity passed on the active AVX-512 test;
- strict Clippy passed;
- formatting passed;
- native release build passed;
- `git diff --check` passed.

## 7. Whole-model results

Values are medians of three randomized, temperature-gated fresh processes.
Whole-process time is external Python subprocess wall time.

| Mode | Pack ms | Prefill ms | Decode eval/s | vs A | TTFT ms | Process ms | Peak MiB | minflt | majflt | Break-even generated | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | 0.0 | 226.5 | 21.968 | - | 897.4 | 6,803.9 | 2,194.1 | 88,372 | 0 | 1 | yes |
| B | 434.0 | 221.4 | 30.992 | +41.1% | 1,312.3 | 5,603.2 | 3,104.6 | 142,218 | 0 | 33 | yes |
| C | 437.8 | 220.9 | 30.791 | +40.2% | 866.4 | 5,562.6 | 3,099.9 | 129,624 | 0 | 33 | yes |
| D | 449.9 | 221.0 | 30.888 | +40.6% | 1,342.6 | 5,601.7 | 3,099.7 | 139,323 | 0 | 36 | yes |
| E | 451.0 | 192.1 | 30.763 | +40.0% | 1,305.2 | 5,566.5 | 3,099.7 | 107,437 | 0 | 33 | yes |
| F | 260.4 | 224.9 | 27.659 | +25.9% | 1,159.5 | 5,874.8 | 2,694.8 | 108,722 | 0 | 30 | yes |
| G | 357.2 | 222.7 | 28.052 | +27.7% | 1,237.7 | 5,901.4 | 2,962.5 | 120,095 | 0 | 37 | yes |
| H | 79.8 | 227.9 | 23.254 | +5.9% | 958.3 | 6,541.1 | 2,283.5 | 102,237 | 0 | 26 | yes |
| I | 350.3 | 226.1 | 30.199 | +37.5% | 1,229.9 | 5,569.4 | 2,827.4 | 127,616 | 0 | 29 | yes |
| J | 452.9 | 225.2 | 30.827 | +40.3% | 1,331.5 | 5,585.8 | 3,104.5 | 140,384 | 0 | 35 | yes |

Mode C can emit token one before its 438 ms packing pause; its low TTFT
therefore does not imply low first-to-second-token latency. B's prefill restores
source residency. E keeps it deliberately. Neither duplicate mode improved
decode materially over durable D/J.

## 8. Operator-level/selective result

No operator kernel profiler was added or changed. The existing operator result
remains: gate/up and Q/O/K/V benefit strongly, while down is weak and LM head is
unchanged.

The selection ablation is consistent with down being a poor packing target on
this model:

- F gate/up packed 544 MiB and retained 64.2% of J's decode gain.
- G added the 272 MiB down projections but retained 68.7% of J's gain. The
  incremental median was only 0.393 decode evaluations/s.
- H attention-only packed 170 MiB and retained only 14.5% of J's gain. This is
  a negative result and satisfies the selective stop condition for that policy.
- I packed 714 MiB, omitted down, and retained 92.9% of J's decode gain.
- J packed 986 MiB.

The I result is promising but is one model/CPU observation. It does not yet
establish a shape policy or explain the non-additive group timings.

## 9. Startup and prefill impact

All eligible packing cost 434-453 ms in B-E/J. Packing I cost 350 ms. Generic
prefill was approximately 221-226 ms except E at 192 ms; E retains the
source pages warmed by packing and is a deliberately rejected duplicate-memory
baseline.

D/J moves packing into TTFT and breaks even at approximately 35-36 generated
tokens for this prompt and session. I reduces the measured break-even to 29
tokens. C preserves control-like TTFT by packing after token one, but introduces
an approximately 438 ms gap before packed decode can begin.

## 10. RSS and page-residency result

PSS values below are sampled after generic prefill and after decode. Modes with
post-prefill re-eviction use the latter as the durable state.

| Mode | Packed MiB | Post-prefill RSS/anon/file MiB | Post-decode RSS/anon/file MiB |
|---|---:|---:|---:|
| A | 0 | 2,194.1 / 925.4 / 1,265.1 | 2,194.1 / 925.4 / 1,265.1 |
| B | 986 | 3,104.6 / 1,835.9 / 1,265.0 | 3,104.6 / 1,835.9 / 1,265.0 |
| C | 986 | 2,198.2 / 929.6 / 1,265.0 | 2,113.8 / 1,831.0 / 278.9 |
| D | 986 | 3,099.7 / 1,830.9 / 1,265.1 | 2,113.6 / 1,830.9 / 279.0 |
| E | 986 | 3,099.7 / 1,830.9 / 1,265.1 | 3,099.7 / 1,830.9 / 1,265.1 |
| F | 544 | 2,694.8 / 1,426.1 / 1,265.3 | 2,151.2 / 1,426.1 / 721.7 |
| G | 816 | 2,962.5 / 1,693.8 / 1,265.0 | 2,146.4 / 1,693.8 / 449.0 |
| H | 170 | 2,283.5 / 1,014.5 / 1,265.0 | 2,113.4 / 1,014.5 / 1,095.0 |
| I | 714 | 2,827.4 / 1,558.8 / 1,265.0 | 2,113.7 / 1,558.8 / 551.3 |
| J | 986 | 3,104.5 / 1,835.9 / 1,265.0 | 2,118.4 / 1,835.9 / 278.9 |

The central mechanism survived this experiment:

- D reduced file PSS from 1,265.1 to 279.0 MiB after prefill.
- D's packed decode left file PSS at 279.0 MiB; it did not re-fault projection
  sources.
- C reached the same durable state by packing after prefill.
- B immediately returned to duplicate residency during generic prefill.
- E retained approximately 986 MiB more file-backed PSS than D by design.

Final RSS is similar across replacement selections because packed anonymous
bytes displace roughly equal encoded source bytes. Selection primarily reduces
packing time and temporary peak RSS, not durable total RSS, when eviction is
effective.

The explicit phase split intentionally samples the fully built packed state
before eviction, so C/D/J peak near 3.10 GiB. The pre-existing production path
evicts each source after its weight is packed and may have a lower transient
peak; this experiment does not claim otherwise.

Major faults were zero in every run because the model was already in the host
page cache. These data support warm-cache minor-fault and residency claims
only, not cold-storage behavior.

## 11. Hardware and software

- CPU: Intel Core i5-1135G7, Tiger Lake, 4 physical / 8 logical cores
- affinity: CPUs 0-3, SMT siblings excluded
- ISA: AVX-512F/BW/VL and AVX-512 VNNI
- cache: 192 KiB aggregate L1d, 5 MiB aggregate L2, 8 MiB shared L3
- governor: `powersave`, Intel P-state active
- trial package temperature: at most 80 C at child start
- kernel: Linux 7.1.4-arch1-1
- Rust: 1.95.0, LLVM 22.1.2
- model: `Llama-3.2-1B-Instruct-Q8_0.gguf`, 1,321,083,008 bytes
- model SHA-256:
  `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3`
- tokenizer SHA-256:
  `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`

## 12. Ablations

The experiment isolates lifecycle A-E and selection F-J while using the same
packed layout and AVX-512 kernel in every packed mode. It does not alter vector
width, tile size, compensation, reduction, scheduler, quantization, or ISA.

## 13. Negative and rejected results

- Generic prefill after packing does restore source residency (B).
- Duplicate residency (E) is not justified by a decode gain over D/J.
- Attention-only selection (H) retains only 14.5% of the full decode gain.
- Adding down to gate/up (G versus F) has a weak incremental return.
- Delayed packing (C) avoids TTFT cost but creates a long post-first-token
  pause and does not reduce temporary peak RSS in this phase-separated harness.
- The first smoke test accidentally called Llama's inherent generic method for
  decode; it re-faulted the source pages. It was rejected, dispatch was fixed,
  and no smoke number entered the result table.
- The first grouped A-J matrix was rejected for selective throughput because
  identical D/J configurations differed by 13%.
- Procfs hooks add measurable external wall time, so both their duration and
  timing-only audit are disclosed.

## 14. Exact reproduction commands

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release

python3 scripts/benchmark_packed_lifecycle.py \
  --binary target/release/ember \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --tokens 128 --threads 4 --cpus 0-3 \
  --repetitions 3 --random-seed 1729 \
  --max-start-temperature-c 80 \
  --measurement-overhead-percent 2 --force-selective \
  --output-dir data/benchmarks/packed-lifecycle-20260728-tgl4-r3-randomized
```

One direct mode, using the same binary:

```bash
taskset -c 0-3 env RAYON_NUM_THREADS=4 EMBER_LLAMA_PACKED_Q8=0 \
  target/release/ember bench-lifecycle \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt 'The capital of France is' --tokens 128 \
  --lifecycle pack-before-prefill-reevict \
  --selection attention-gate-up
```

Generated raw JSON, metadata, summary JSON, and report are under
`data/benchmarks/packed-lifecycle-20260728-tgl4-r3-randomized/`. That directory
is intentionally ignored because benchmark artifacts are large/local; this
note and the driver are unignored.

## 15. Updated research thesis

Replacement-style packed residency is technically viable across one complete
generic-prefill/packed-decode lifecycle when the runtime re-evicts packed
projection sources at the phase transition. Selective replacement can reduce
packing time and transient peak while preserving most decode gain: on this
case, excluding down retained 92.9% of the measured gain.

This remains a hypothesis beyond one model and one client Intel CPU. The result
does not show that generic prefill can execute without source pages; it shows
that those pages need not remain resident during packed decode.

## 16. Novelty risks

`madvise`, prepacking, selective operator packing, and amortization are
individually established techniques. XNNPACK's weight cache remains a close
replacement-memory threat. The result becomes research-relevant only if the
lifecycle/shape policy predicts across models and microarchitectures, improves
the complete speed-memory-startup tradeoff, and transfers beyond Ember.

The current experiment has no cold-cache, energy, hardware-counter,
multi-model, AMD, ARM, AVX2, or lower-bit evidence. It does not establish why
attention-only is weak or why the combined I mode is stronger than the sum of
its isolated group impressions.

## 17. IEEE TC readiness

**Engineering note only.**

The session found a viable mechanism and a promising selective configuration,
but the evidence is still one model, Q8_0, one client Intel system, and three
repetitions under a laptop thermal policy. It is below workshop level until at
least a second model size, the repeated-prompt lifecycle, and an evidence-backed
operator-regime explanation exist. IEEE TC remains only a research hypothesis.

## 18. Highest-value next experiment

Run the same A/D/I/J lifecycle bundle on Llama-3.2-3B Q8_0 on this host before
changing the kernel. It tests whether durable re-eviction, down exclusion,
packing cost, peak RSS, and break-even scale beyond 1B. If memory pressure makes
the phase-separated peak unsafe, use the already-existing per-weight eviction
ordering but preserve the same phase metrics; do not add a new kernel.

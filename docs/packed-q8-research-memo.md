# Packed quantized projections: living research memo

Date initialized: 2026-07-28

Status: engineering note only

This memo treats publication as a hypothesis. It must be updated with negative
results and must not turn a single favorable decode benchmark into a general
architecture claim.

## Candidate thesis under test

A CPU inference runtime may recover a substantial fraction of the throughput
benefit of aggressively packed quantized kernels without retaining two
resident model-sized weight representations, if the packed representation
replaces or selectively displaces the source representation across the full
prefill/decode lifecycle.

The emphasized condition is now met in one narrow lifecycle experiment:
ordinary multi-token prefill touches the generic row-contiguous weights and
makes those pages resident again, but an explicit post-prefill re-eviction
restores replacement-style residency, and packed decode does not fault those
projection sources back. Generality is not established.

## Current evidence

Positive:

- On Llama-3.2-1B Q8_0 and Sapphire Rapids, packed decode improved throughput
  by 23.5% at four physical cores and 31.3% at eight physical cores.
- On local Tiger Lake, a matched decode-only benchmark improved from 22.085 to
  33.009 tok/s.
- Packed-vs-control kernel output is bit-identical on active tests.
- Greedy Llama output is byte-identical across the same-binary switch.
- Qwen3 remains byte-identical and on its generic path.
- The weak down-projection regime recurs across the prior server data and the
  local operator profile.
- On a three-repetition, randomized, temperature-gated Tiger Lake lifecycle
  run, post-prefill re-eviction reduced file PSS from about 1,265 to 279 MiB.
  File PSS remained 279 MiB after 127 packed decode evaluations.
- Durable all-projection packing improved median decode from 21.968 to
  30.888 tok/s while ending near 2,114 MiB RSS rather than the duplicate
  baseline's approximately 3,100 MiB.
- Excluding down packed 714 rather than 986 MiB, retained 92.9% of the
  all-projection decode gain, reduced median packing from 453 to 350 ms, and
  reduced the measured break-even from 35 to 29 generated tokens.

Negative or limiting:

- Ordinary prefill re-resides the original GGUF projection pages.
- Without re-eviction, post-prefill packed PSS has about 910 MiB more anonymous
  residency while file-backed PSS is the same as control.
- Attention-only packing retained only 14.5% of the all-projection whole-model
  decode gain on the controlled local run.
- Explicit delayed packing preserved control-like TTFT but created an
  approximately 438 ms pause after token one.
- The phase-separated measurement mechanism observes a temporary duplicate
  peak near 3.10 GiB before eviction.
- A 32-token local generation did not amortize observed process startup.
- The local llama.cpp build used less RSS than packed Ember, opposite to the
  earlier AWS comparison.
- The current evidence covers one packed quantization format, one packed model
  architecture, and two Intel microarchitecture families.
- Persistent scheduler workers regressed end-to-end decode.
- Projection-task parallel gate/up only helped a narrow isolated two-core
  regime.

## Unsupported assumptions

1. That post-prefill `MADV_DONTNEED` remains effective across repeated
   prompts, other kernels/filesystems, and memory-pressure policies. It worked
   for one prompt plus 127 decode evaluations on the local host.
2. That the weak down-projection gain is caused by bandwidth or cache behavior;
   only timing and shape evidence exists so far.
3. That the 16-row layout is useful with AVX2, Arm dot product, or i8mm.
4. That the layout remains beneficial for other model sizes, vocabulary sizes,
   or quantization formats.
5. That the AWS llama.cpp RSS difference reflects a stable architectural
   distinction rather than version, buffer policy, or workload residency.
6. That packing startup can be amortized for short interactive requests.
7. That page advice is equally effective across kernels, filesystems, and
   virtualization environments.

## Closest-work and novelty threats

This is an initial audit, not a complete systematic literature review.

- FBGEMM already combines low-precision CPU inference, fused quantization, and
  shape/size-specific kernel generation. Packing or specializing by shape is
  not novel by itself:
  <https://arxiv.org/abs/2101.05615>
- Gope et al. explicitly use an interleaved group weight layout and amortize
  operand loading/unpacking across multiple output rows for Arm LLM kernels.
  Output-row reuse and interleaving are therefore strong prior-art threats:
  <https://arxiv.org/abs/2501.00032>
- T-MAC includes offline weight permutation, weight interleaving, layout
  ablations, cross-platform CPU evaluation, and energy measurements:
  <https://arxiv.org/abs/2407.00088>
- Intel's CPU LLM work evaluates a specialized INT4 runtime across multiple
  model families. A paper limited to one Llama Q8 model would compare poorly:
  <https://arxiv.org/abs/2311.00502>
- XNNPACK's documented weights cache is the closest threat to a
  memory-efficient replacement claim. It packs static weights, can persist the
  packed cache in an mmap-backed file, avoids reading original mmap pages on a
  cache hit, and shares packed pages across processes:
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/xnnpack/README.md#using-the-xnnpack-weights-cache>
- llama.cpp already contains repack types for some quantization kernels.
  Repacking for wider contiguous loads is not enough for novelty:
  <https://github.com/ggml-org/llama.cpp/issues/15351>
- oneDNN's primitive model explicitly assumes that setup and precomputation
  costs are amortized through reuse. Packing amortization as a concept is not
  novel without an LLM-specific mechanism or predictive result:
  <https://uxlfoundation.github.io/oneDNN/dev_guide_basic_concepts.html>

The potential gap is narrower: an interpretable, operator-selective,
session-aware mechanism that preserves quantized correctness and controls
resident source/packed duplication across prefill and decode. That gap is
still a hypothesis.

## Candidate formulation ranking

Scores are 1 (weak) to 5 (strong). They describe potential after the required
experiments, not current accomplishments.

| Rank | Formulation | Novelty | Technical strength | Feasibility | Generality | IEEE TC fit | Main risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | B: shape-aware selective packing | 3 | 4 | 4 | 4 | 4 | Rules may collapse to model-specific thresholds or known blocking practice |
| 2 | A: memory-efficient replacement packing | 2 | 4 | 4 | 4 | 3 | XNNPACK cache is close prior art; current lifecycle evidence is one host/model |
| 3 | D: packing amortization planner | 3 | 3 | 5 | 4 | 3 | Could be judged an obvious cost model without a new lifecycle mechanism |
| 4 | C: cross-ISA packed layout | 3 | 5 | 1 | 5 | 4 | Arm and T-MAC prior art is strong; one common layout may compromise every ISA |

The most plausible direction is a combination of B and D, provided operator
regimes predict across models and CPUs and the policy improves whole-session
speed/memory/startup tradeoffs. Formulation A can support that story only
after source residency remains controlled during ordinary prefill.

## Required experimental sequence

### Completed immediate lifecycle ablation

Modes A-E and selective modes F-J now record all requested phase boundaries,
RSS/PSS, anonymous/file-backed residency, faults, timing, parity, and
break-even. The result supports post-prefill re-eviction and rejects
attention-only selection on Llama-1B/Tiger Lake. See
`docs/packed-q8-lifecycle.md`.

The next lifecycle questions are repeated prompts, fallback re-fault cost, cold
page-cache behavior, and whether the same selection survives Llama-3B.

### Shape characterization

Extend the profiler only enough to record:

- model, architecture, layer, operator;
- input/output dimensions and aspect ratio;
- MACs, mapped bytes, packed bytes;
- control/packed time and speedup;
- worker topology and selected representation.

The first testable model should predict why `[8192, 2048]` down benefits less
than `[2048, 8192]` gate/up despite equal MAC and encoded byte counts. Do not
name a bandwidth/cache mechanism until counters or controlled working-set
experiments support it.

### Generalization gates

- At least one additional Llama size before claiming a model-size trend.
- Correct architecture-specific parity before enabling any Qwen packed path.
- At least Tiger Lake, Sapphire Rapids, and Zen 4/5 before an x86-wide claim.
- AVX2-only behavior separated from the AVX-512 representation.
- Arm only after a native dot-product/i8mm design is justified.
- Q4 only if Ember can support a comparable path without building a new
  quantization backend solely to fill a paper table.

## Reviewer objections to keep active

1. "This is standard GEMM weight prepacking with `madvise`."
2. "The memory advantage disappears after the first real prefill."
3. "The result is a Llama-1B/Sapphire-Rapids tuning artifact."
4. "The output tile merely mirrors an existing ggml/XNNPACK/FBGEMM layout."
5. "RSS is page-cache state, not allocated model storage."
6. "The comparison uses a mismatched llama.cpp build or thread policy."
7. "Packing time makes the method unattractive for interactive workloads."
8. "The down result contradicts a simple MAC- or byte-count policy."
9. "The policy is fitted to seven shapes from one model."
10. "No hardware evidence supports the proposed cache/bandwidth explanation."
11. "Q8-only scope misses the lower-bit formats used in practical CPU LLMs."
12. "The contribution is an Ember optimization that cannot transfer to another
    runtime."

## IEEE Transactions on Computers readiness

Current verdict: **engineering note only**.

The subject can fit a venue concerned with computer architecture, performance
analysis, and hardware/software interaction. The current artifact does not
meet that bar. It now has a durable mechanism on one workload, controlled
lifecycle/selection ablations, and a reproducible bundle, but still lacks
multiple models, multiple CPU vendors/ISAs, robust statistical treatment,
repeated-session behavior, and hardware-level explanation.

Promotion criteria:

- Workshop-level: lifecycle ablation, a second model size, reproducible bundle,
  and an evidence-backed explanation of the down regime.
- Specialist-journal-level: multiple models and x86 microarchitectures, robust
  startup/prefill/decode/RSS tradeoff, and an interpretable selection policy.
- IEEE TC candidate: a transferable principle or runtime mechanism, at least
  three materially different CPU microarchitectures, portability limits,
  controlled component ablations, energy/memory evidence, and comparison with
  close packed-weight systems.

## Highest-value next experiment

Run A/D/I/J unchanged on Llama-3.2-3B Q8_0 before implementing another kernel
mechanism. The decision it must resolve is:

> Does durable re-eviction and down-excluding selective packing preserve its
> speed, peak-memory, startup, and break-even advantage at a second model size?

If the selection or residency result does not survive 3B, narrow or retire the
general selective/replacement thesis rather than tuning a new policy to 1B.

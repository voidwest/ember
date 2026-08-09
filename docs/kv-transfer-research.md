# Cross-model KV transfer: research and design note

Status: experimental seam and research plan only

This note describes a possible future cross-model key/value (KV) cache transfer
workflow. It does not claim that cross-model transfer works in Ember. Ember does
not currently implement a ridge mapper, a neural mapper, mapper fitting, mapper
execution, transformed-snapshot construction, or cross-model replay.

A closed-form regression is still a **fitted mapper** even when it does not use
backpropagation. No such fitted parameters ship in this repository.

## Motivation and closest workflow

Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse* (arXiv:2608.03893v1, 2026) study a relevant
future workflow:

1. run the same tokenized calibration prefixes through a source and target
   model;
2. compare every source layer with every target layer;
3. select the top predictive source layers for each target layer;
4. remove RoPE from keys and fit independent per-target-head ridge maps for
   keys and values;
5. apply the maps to a source prefix, apply the target model's RoPE to mapped
   key content, and populate a target-compatible cache;
6. decode with the target without running its ordinary prefill; and
7. evaluate reconstruction, attention behavior, downstream behavior,
   multi-turn drift, and end-to-end latency.

The preprint reports useful evidence for several matched-KV, within-family GPU
model pairs. It also reports sharp failures on other pairs. Those results are
external evidence, not Ember results, and do not establish that the workflow is
accurate or faster on Ember's CPU path, quantized models, f16 cache, model
families, or prompts. The preprint's strongest experiments use equal source and
target KV-head counts and equal head dimensions. Mismatched geometry and
cross-family transfer remain research questions.

## What exists in Ember now

The current code establishes two narrow, non-mapper boundaries.

### Verified native snapshots

`ember.kv-snapshot.v1` stores compact f16 key and value payloads in
`[layer][head][position][dimension]` order, plus checksummed metadata and
provenance. The `ember kv export|inspect|verify|replay|compare` commands currently cover
Llama/Qwen-family native snapshots and explicit measurements. Replay is deliberately strict and is a
same-model continuation facility: model and tokenizer identity, cache geometry,
precision, layout, RoPE semantics, value state, execution mode, and execution
fingerprint must be compatible.

The manifest reserves `origin: transformed` and transform-provenance fields for
a future explicit transform. Their presence is not a transform implementation.
There is currently no constructor that accepts mapper output and publishes a
verified transformed snapshot.

### Deterministic measurement and causal-control harness

`ember kv compare` now supplies the measurement harness required before any
mapper work. Its strict two-snapshot mode requires exact target/prefix
coordinate alignment and reports global plus every layer/head K and V cosine,
MSE, optional directional R2, maximum absolute error, and f16 bit mismatches.
Optional thresholds produce a deterministic complete failure list and first
exceedance. The typed, timing-free JSON is intended for downstream research
scripts, not as evidence that transfer works.

With the exact target model, comparison can additionally run two deliberately
separate continuation protocols:

- a reference-greedy teacher-forced path that evaluates identical input tokens
  in both caches and measures the semantic per-layer attention O-projection
  output and full logits; and
- two clean independent greedy rollouts that measure token-sequence agreement
  and first divergence without interpreting post-divergence activations as
  same-input differences.

The first measurable row is after evaluating the stored/common resume token;
a KV-only snapshot cannot reconstruct the prefix-boundary logit or attention
row that originally selected that token.

For a causal negative/control condition, the command can zero or scale one
selected K/V head across the initialized prefix **in memory**. A receipt pins
the native source snapshot, typed operation, exact factor bits, and affected
element counts. This is not a mapper: no parameters are learned, the edited
cache cannot be published as `ember.kv-snapshot.v1`, reserved mapper provenance
is not populated, and ordinary replay still admits native origins only. This
makes native-vs-altered localization testable without prematurely defining a
transformed-artifact schema.

The current attention hook is the target layer's complete O-projection result
before residual addition, not attention probabilities or a per-head weighted-V
capture. Observer hooks can use a different internal route from ordinary
serving decode (notably planned Q8), so paired forced metrics and independent
ordinary greedy behavior remain distinct in the report. See
`docs/kv-snapshots.md` for the command and JSON contract.

### Allocation-bearing RoPE content seam

`src/kv_transfer/rope.rs` contains tested, offline utilities for:

- applying forward or inverse headwise RoPE for adjacent-pair and split-half
  layouts;
- reading verified f16 stored keys into f32 and removing RoPE; and
- applying a compatible target manifest's RoPE to f32 content keys.

These utilities are not called by ordinary inference. Production scalar/SIMD
RoPE call sites were not rerouted, so the seam does not silently change decode
numerics. The high-level conversion currently accepts only full-dimension,
`uniform-theta`, absolute-zero-based RoPE snapshots. It returns f32 values and
does not quantize them, create a transformed snapshot, fit a mapper, or inject a
cache into another model.

Current tests distinguish the two pair layouts and both directions, check head
boundaries, exercise an f16 snapshot round trip with a tight tolerance, reject
post-RoPE K normalization, and reject malformed shapes. They are structural and
numerical unit tests, not cross-model evidence or a trusted-reference activation
check.

## Exact meaning of key “content”

This document uses one narrow definition. For model `m`, layer `l`, head `h`,
and absolute zero-based position `p`, let the runtime cache key be

```text
K_stored[m,l,h,p] = f16(R[m,l,p](K_content[m,l,h,p]))
```

where `R` is the architecture's forward RoPE rotation. The offline content seam
first converts the stored f16 value to f32 and defines

```text
K_content_from_snapshot = R^-1(f32(K_stored)).
```

Thus **content means post-K-normalization and pre-RoPE**, not necessarily the raw
K-projection output. The f16 store has already rounded the original f32 key, so
this operation does not recover the exact pre-cache f32 activation.

The current conventions are:

| Ember path | RoPE coordinate pairs | K normalization relative to RoPE | Content represented after inverse RoPE |
|---|---|---|---|
| Llama | adjacent `(2d, 2d+1)` | nominally after, but ordinary supported Llama has no K-norm tensor | raw K projection, subject to cache f16 rounding |
| Qwen2/Qwen2.5 | split-half `(d, d + head_dim/2)` | before; ordinary Qwen2.5 snapshots have no K-norm tensor | raw K projection, subject to cache f16 rounding |
| Qwen3 | split-half `(d, d + head_dim/2)` | learned headwise RMSNorm before RoPE, epsilon `1e-6` | normalized K content; the norm must remain in the representation |
| Gemma 4 path | split-half `(d, d + head_dim/2)` | learned headwise RMSNorm before RoPE, using model metadata epsilon | theoretical normalized K content, but not supported by the current snapshot/content boundary |

Forward rotation uses

```text
(a, b) -> (a*c - b*s, a*s + b*c)
```

and inverse uses the transpose

```text
(a, b) -> (a*c + b*s, -a*s + b*c).
```

The target-side future operation would be

```text
K_target_stored_hat[p] = f16(R_target[p](mapper_K(K_source_content[p]))).
```

The source must be unrotated with the source model's table and the prediction
must be rotated with the target model's table. Recomputing a relative angle is
not an exact substitute for using the two resident/precomputed f32 tables.
Inverse followed by forward is also not a bit-exact identity in f32. Any exact
no-op or restoration operation must bypass the transform and preserve the
original stored bytes.

Q/K normalization is deliberately **not inverted**. Applying a combined
QK-norm-plus-RoPE inference helper to content would normalize Qwen3/Gemma keys
a second time. RMSNorm is not bit-idempotent, and a safe general inverse is not
available when learned weights can be zero or epsilon can be zero.

Values carry no RoPE. Their semantic state still belongs in the contract:
current Llama/Qwen snapshots store projection output, whereas Gemma normalizes V
before cache storage. A future mapper must fit and emit exactly the value state
the target consumes; “values need no RoPE” does not make value representations
architecture-independent.

## Cases that must fail closed today

The current content conversion correctly rejects or does not expose the
following cases:

- K normalization after RoPE when a K-norm tensor is present. Learned
  per-dimension weights do not commute with rotation, so inverse RoPE alone is
  not the defined content transform.
- non-uniform, scaled, factor-modified, or partial-dimension RoPE. In
  particular, Gemma local/global layers can use different head dimensions and
  theta values, global layers can use `rope_freqs.weight` factors, and shared-KV
  layers reuse an earlier layer's already transformed cache. A single
  uniform-theta snapshot descriptor is insufficient.
- non-absolute or block-reset position semantics. Packed independent sequences
  can reset RoPE at block boundaries; a prefix-wide `position = 0..T-1`
  assumption would be wrong.
- odd/zero dimensions, inconsistent payload shapes, non-finite tables, and
  positions beyond the target table/context limit.
- interpreting a flattened GQA tensor as one head. Q and K have different head
  counts, and every transform must receive explicit KV-head count and head
  dimension.
- exact recovery of pre-cache f32 values from the f16 snapshot payload.
- cross-model replay through the native compatibility path. A native source
  snapshot intentionally retains its source model identity and is rejected by
  a different target.

Other future preflight failures should include unequal token IDs or positions,
unknown tokenizer identity, incompatible chat-template/BOS handling, unsupported
value state, unsupported hybrid/sliding-window/shared-KV semantics, missing
source/target model hashes, and a mapper whose own provenance does not match the
source and target artifacts. Equal prompt strings are not enough: paired caches
must correspond to the same token sequence and position convention.

Matched KV geometry should be the first research scope. A rectangular linear
map can be written for unequal head counts or dimensions, but the current Ember
seam does not establish head correspondence and the cited preprint does not
validate mismatched-KV transfer.

## Proposed research workflow

This is a staged validation plan, not an implementation commitment.

### Gate 0: prove the native boundary

Before cross-model work, export, verify, import, and replay a native cache for
each exact model/quantization/execution mode in scope. Compare continuation
logits and greedy tokens with uninterrupted same-model inference. Record model,
tokenizer, prompt-token, plan, execution fingerprint, cache, and binary hashes.
A native replay failure is a runtime/snapshot defect and must not be attributed
to mapper quality.

### Gate 1: establish paired calibration data

Use identical token IDs and explicit positions for source and target. Split by
sequence, not by token, so tokens from one prefix cannot leak across train and
evaluation. Include domain-held-out and length-held-out sets. Capture native f16
snapshots because that is the representation a serving mapper would receive;
optionally capture pre-cache f32 activations as a separate diagnostic with a
clearly different representation label.

Ember's existing experiment hooks expose residual-stream semantic sites, not
Q/K projections, target queries, attention probabilities, or pre-cache values.
Those research captures would require new opt-in instrumentation and artifact
schemas. They must not be inferred from residual captures.

### Gate 2: map linear structure before fitting a production mapper

For each source layer/target layer pair, fit a small single-source linear probe
on training sequences and report held-out results separately for:

- RoPE-stripped keys;
- stored/rotated keys as a negative or ablation condition; and
- values in their declared architecture-specific state.

Produce layer-by-layer, head-by-head and head-averaged R-squared/cosine matrices.
Use training-only matrices to select candidate source layers. Preserve negative
results: a diagonal-looking heatmap is evidence of linear predictability in that
sample, not proof of cache substitutability.

### Gate 3: fit a minimal closed-form baseline

A paper-like baseline would, for each target layer, choose `k` source layers,
concatenate their source KV-head features, and fit separate centered ridge
regressions for each target head and for K and V. Record at least:

- ordered source and target model/tokenizer hashes;
- calibration dataset identity, licensing, split hashes, token IDs, positions,
  and sequence lengths;
- source-layer selection rule and selected layers per target layer;
- K/V representation states and RoPE metadata;
- feature/output shapes, bias convention, ridge lambda, solver, accumulation
  dtype, mapper precision, and mapper SHA-256; and
- software, CPU, thread, memory, and timing information.

Select `k`, lambda, compression, and any acceptance threshold without using the
reported downstream test set. Closed-form fitting avoids gradient descent; it
does not remove calibration dependence or selection bias.

### Gate 4: reconstruction and attention diagnostics

The same-model `kv compare` harness freezes the basic metric, hook, indexing,
and greedy-divergence semantics for this gate, but it does not make a
cross-model candidate. Evaluate future mapper outputs on held-out sequences
before generation. At minimum report per-layer,
per-head and aggregate:

- K-content and V R-squared, cosine, mean absolute error, relative error, and
  worst-case norms;
- post-target-RoPE key error and f16 cache-rounding error separately;
- target attention-probability divergence and attention-output cosine using the
  target's real queries;
- target logit cosine, max/mean absolute difference, KL or cross-entropy delta,
  top-1 agreement, and top-k overlap for the first continuation token; and
- error by position, context length, token class/domain, layer, and head.

R-squared is useful for within-pair layer selection but should not be the sole
acceptance metric. The cited preprint found downstream retention tracked
attention-output cosine better than calibration R-squared. A useful mechanism
diagnostic is where residual error lands:

- for K, project K error into right-singular directions of the target query
  matrix and weight by squared query singular values; and
- for V, weight positionwise V error by squared ground-truth target attention
  weights.

These are diagnostics, not causal explanations. Report them together with
behavior and include confidence intervals or bootstrap variation across
sequences.

### Gate 5: construct and verify a transformed artifact

Only after the preceding gates should a future offline tool create a target
snapshot. It should map source f16 cache values through the declared content
space, apply target RoPE, round through the same target f16 cache path, and emit
`origin: transformed` with the source snapshot hash, target model identity,
mapper hash, selected layers, and transform type.

The transformed artifact must pass ordinary schema/payload verification and a
separate transform-provenance verification before target import. Do not add an
“ignore compatibility” switch to native replay. Cross-model acceptance should
be an explicit, provenance-bearing construction path.

### Gate 6: behavioral and serving evaluation

Compare at least four conditions: target standalone prefill, exact native target
snapshot replay, mapped source-to-target cache, and a declared negative control
(for example a random/mean mapper or wrong layer selection). Evaluate:

- prefix-conditioned perplexity on identical continuation tokens;
- multiple-choice/log-likelihood tasks and generation tasks appropriate to the
  model;
- greedy and sampled rollouts, not only the first token;
- repeated handoffs and drift by turn;
- context lengths both inside and beyond the calibration distribution;
- both transfer directions independently; and
- end-to-end wall time, CPU time, peak/current RSS/PSS, page faults, bytes read,
  mapper load time, transform time, target decode time, and target re-prefill
  time.

A speed claim requires mapper load plus transform and cache-materialization
costs. A quality claim requires downstream target behavior. Reconstruction alone
is not behavioral equivalence, and generation smoke is not numerical parity.

## CPU mapper size and execution caveats

For the dense per-target-head design in the cited workflow, ignoring biases,
total K+V parameters are

```text
2 * L_target * n_kv_target
  * (k * n_kv_source * d_head_source)
  * d_head_target.
```

The preprint's selected dense maps contain roughly 1.01--3.36 billion parameters
(about 4--12 GB at f32) for its evaluated pairs. It reports GPU fitting on an
8xH100 node and GPU serving speedups. Those sizes are not “small” for a
CPU-first runtime. They can rival or exceed a quantized Ember model, multiply
memory traffic for every prefix token, incur page-fault/NUMA costs, and make a
cold mapper slower than receiver re-prefill for short contexts. A directional
mapper is needed for each ordered model pair, so a fleet can grow quadratically
in pair count.

Keeping weights on host memory is not an advantage for a runtime whose compute
already runs on the host. Disk/mmap residency, memory bandwidth, cache behavior,
thread scaling, and time to first mapped token must be measured rather than
borrowed from PCIe estimates. Streaming one target layer at a time lowers peak
resident memory but can reread large mappings and trades RAM for I/O. Mapper
weights, source features, mapped f32 outputs, and the final f16 cache must all be
included in peak-memory accounting.

The following are **unimplemented hypotheses** for making a CPU mapper smaller:

- **Low-rank:** factor each dense matrix as `A @ B`, choose rank on held-out
  attention/output behavior, and compare randomized/truncated-SVD or reduced
  rank ridge against the dense solution. Low reconstruction error alone is not
  a rank-selection criterion.
- **Shared/basis maps:** share a basis across heads, K/V, adjacent target layers,
  or multiple ordered model pairs, with small per-head/layer coefficients or
  residual adapters. Sharing reduces parameters but can erase exactly the
  head/layer specificity the mapper relies on.
- **Quantized maps:** store/execute mapper weights in Q8_0 first, then cautiously
  test Q6/Q4 or mixed precision. Ember's model-weight quantization kernels are
  useful engineering references, not evidence that mapper quantization is
  harmless. Quantization error must be evaluated after attention and through
  rollout.
- **Structured sparsity:** prune whole source layers, source heads, feature
  blocks, or matrix tiles so CPU kernels can skip work. Unstructured zeros are
  not a speedup without a measured sparse kernel. Layer selection is already a
  form of structured sparsity and should be the first baseline.
- **Diagonal/block-diagonal plus low-rank residual:** exploit any stable head or
  feature alignment while retaining a small cross-head correction. This is a
  hypothesis about structure and needs pairwise evidence.
- **On-demand/mmap storage:** retain only an active directional pair and map
  target layers sequentially. Measure cold and warm page residency separately;
  do not report only steady-state throughput.

Each compression axis needs an accuracy-size-latency frontier against both the
dense mapper and target re-prefill. Combining several techniques without single
factor ablations would make a negative result uninterpretable.

## Required artifact set for any future claim

A serious transfer run should preserve:

- source and target native snapshot manifests and hashes;
- the exact shared token-ID sequences and split manifest;
- content-conversion metadata and version;
- layer-selection matrices and held-out diagnostics;
- mapper manifest, weights, precision, hash, and fitting command;
- transformed snapshot and complete provenance chain;
- target standalone, native-replay, mapped, and negative-control outputs;
- attention/logit/behavior metrics with missing-artifact reporting; and
- matched latency/RSS/PSS/page-fault reports and commands.

Until those artifacts exist, Ember has a verified same-model KV snapshot seam
and a tested limited RoPE coordinate conversion utility. It has **no learned or
fitted KV mapper**, no cross-model cache construction path, and no evidence of
cross-model accuracy retention or CPU speedup.

## Reference

Taekyung Heo et al. “Cross-Model KV Cache Transfer in LLM Families: A
Closed-Form Linear Mapping for Prefill Reuse.” arXiv:2608.03893v1, 2026.
<https://arxiv.org/abs/2608.03893>

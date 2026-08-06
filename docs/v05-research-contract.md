# Ember v0.5 Research Contract

Status: **frozen** for the v0.5.0 release.
Schema identifiers defined by this contract: `ember.experiment.v1`,
`ember.bundle.v1`, `ember.hook.v1` (semantic hook sites), reusing
`v04-plan/1` (execution-plan schema, `PLAN_SCHEMA_VERSION = 1`).

This document defines the semantics a researcher can rely on when running
`ember experiment run|validate|inspect|verify|compare|reproduce` in
Ember 0.5.x. It is the reference for the implementation; the implementation
may not silently deviate from it.

## 0. Scope and thesis

Ember v0.5 packages exact token selection, semantic hidden-state capture,
activation intervention, execution provenance, and verification into
reproducible experiment bundles that can be run without writing Rust.

The release is not primarily a performance release. v0.4 performance and
memory behavior are preserved; the experiment machinery must not contaminate
ordinary execution (section 16, Gate H).

## 1. Public semantic hook sites

There are exactly six public semantic hook sites. They correspond
one-to-one to the existing v0.4 execution hook stages; the public
identifiers describe model semantics, not Rust implementation details.
The mapping below is precise and was verified against the reference path
(`forward_last_logits_with_cache_hooked`, src/llama.rs) and the planned
interpreter (`forward_last_logits_planned`, src/planned_decode.rs).

| Public id (kebab) | Rust variant | v0.4 stage id | Tensor exposed |
|---|---|---|---|
| `residual-pre-attention` | `ResidualPreAttention` | `before-layer` | The residual stream entering the transformer block, **before** the input RMS norm. |
| `attention-output` | `AttentionOutput` | `after-attention` | The attention output projection result, **before** the attention residual add. |
| `mlp-output` | `MlpOutput` | `after-mlp` | The MLP down-projection result, **before** the MLP residual add. |
| `residual-post-mlp` | `ResidualPostMlp` | `after-layer` | The residual stream leaving the block, **after** both residual adds. |
| `final-norm-output` | `FinalNormOutput` | `before-logits` | The final RMS-norm output feeding the LM head. |
| `logits` | `Logits` | `after-logits` | The raw logits from the LM head. |

Exact meanings per site:

1. `residual-pre-attention` — for layer `l`, the tensor `x` at block entry.
   During prefill this is the full `[seq, embed]` block input; during decode
   it is the `[1, embed]` row for the evaluated token position. It is the
   tensor the input RMS norm consumes. In the planned path it resolves to
   the first op's input region (input norm or fused qkv input).
2. `attention-output` — for layer `l`, the tensor produced by the attention
   output projection (`o_proj` result, attention scores applied to `V` then
   projected). It is observed **pre-residual**: the value at this site does
   not include the residual stream. The residual add happens after the hook
   returns. In the planned-fused builder, fusion F5 eliminates the standalone
   `o` tensor; the site then de-fuses (section 9, 10).
3. `mlp-output` — for layer `l`, the MLP down-projection result
   (SwiGLU `gate(x)*up(x)` projected by `down_proj`), **pre-residual**.
4. `residual-post-mlp` — for layer `l`, the block output: the residual
   stream after both the attention and MLP residual adds. This is the input
   to layer `l+1`.
5. `final-norm-output` — the final RMS norm output (no layer). Shape
   `[1, embed]` in both phases (prefill hooks expose only the final row).
6. `logits` — the LM head output, shape `[1, vocab]`.

**Sites that do not exist in v0.5.** There is no public site between the
attention residual add and the MLP input norm (`residual-post-attention` /
`residual-pre-mlp` in the conceptual six-site model). In both v0.4 execution
paths the attention residual add is immediately followed by the post-attention
RMS norm with no hook between them. A spec that requests such a site fails
validation.

**Tensor dtype and shape conventions.** All six sites expose f32 tensors.
Layer sites are `[seq, embed]` during prefill and `[1, embed]` during decode.
`final-norm-output` is `[1, embed]`; `logits` is `[1, vocab]`. Rows are
indexed by absolute token position (section 6).

## 2. Observation timing

- All sites are observed **after** the named computation completes and
  **before** the next computation begins.
- `attention-output` and `mlp-output` are observed **before** their
  respective residual adds.
- `residual-post-mlp` is observed after the final residual add of the block.
- During prefill (multi-token evaluation), the layer sites expose the full
  `[seq, embed]` tensor at the end of the block's computation. During decode
  (single-token evaluation), they expose the `[1, embed]` row of the
  evaluated position.
- `logits` is observed after the LM head projection completes, before any
  sampling.

## 3. Intervention timing

Interventions apply at the same moments as observations, in-place:

- An intervention at `attention-output` replaces the projection output
  **before** the attention residual add; the add then uses the replaced
  value. This is the documented semantic for activation patching.
- An intervention at `mlp-output` replaces the MLP projection output
  **before** the MLP residual add.
- An intervention at `residual-pre-attention` replaces the block input
  **before** the input norm.
- An intervention at `residual-post-mlp` replaces the block output **after**
  both residual adds; the next block consumes the replaced value.
- An intervention at `final-norm-output` replaces the norm output before the
  LM head.
- An intervention at `logits` replaces the logits before sampling.

Interventions are applied once per evaluation (prefill and each decode
step), gated by the resolved token selector (section 6). A replace
intervention with a single-row source applies the source row to every
selected row.

## 4. Layer numbering

Layers are numbered `0..n_layers - 1` in model order, matching the v0.4
`layer_index`, the plan's `LayerPlan.layer_index`, and the GGUF block order.
Layer selectors are resolved against `n_layers` from the loaded model
metadata; out-of-range selectors fail validation.

## 5. Token-position conventions

- Token positions are 0-based indices into the **model input sequence**:
  the tokenizer output for the prompt, **including** any BOS token the
  tokenizer prepends (Ember always encodes with `add_special_tokens=true`,
  and the wrapper prepends `<bos>` when the tokenizer defines one, with a
  zero-width `(0, 0)` offset).
- Position 0 is therefore the BOS token for tokenizers that define one
  (Llama, Qwen3, Gemma4 all do).
- `prompt-final` selects position `seq_len - 1` where `seq_len` is the
  length of the complete model input (BOS included).
- Generated-token positions: the first decode evaluation (the evaluation
  whose input is the first generated token) runs at `start_pos = seq_len`;
  the token generated at decode step `k` (1-based) sits at position
  `seq_len + k - 1`.
- During prefill, token rows of layer-site tensors are indexed by absolute
  position directly. During decode, the single row corresponds to the
  evaluated position `start_pos`.

## 6. Token selection

Selection is exact and fail-closed (section 17). The typed selector is:

- `prompt-final` — the last token of the complete model input.
- `absolute-token { index }` — the token at position `index`.
- `relative-token { offset_from_end }` — position `seq_len - 1 - offset`
  (offset 0 equals `prompt-final`). Out-of-range offsets fail.
- `generated-step { step }` — the token generated at decode step `step`
  (1-based), observed at the decode evaluation processing it. Fails when
  generation produced fewer than `step` tokens.
- `matched-span { text, occurrence, subtokens }` — exact text match
  (section 7).
- `byte-span { start, end, subtokens }` — byte span into the prompt text
  (section 7).

Subtoken selection: `first`, `final`, `all` — the first / final / all
tokens whose byte coverage intersects the matched span (see section 7 for
coverage rules).

Every selection records: original text, normalized text (when requested),
tokenizer IDs, token pieces, byte offsets, the matched byte span, selected
indices, the selection rule, ambiguity status, round-trip status, and any
fallback used.

## 7. Span matching and byte alignment

- The tokenizer wrapper exposes **character** offsets (validated against
  the input's char count). The token-selection layer converts them to byte
  offsets deterministically by walking the UTF-8 string once
  (char index → byte offset via cumulative byte lengths).
- Matched spans are searched in the **original input text** (no silent
  normalization; section 18). The match is an exact substring match on
  Unicode scalar sequences.
- Occurrence `o` selects the `o`-th non-overlapping occurrence (0-based).
  If the text appears fewer than `o + 1` times the selection fails.
- If multiple tokenizations of the same text are ambiguous (the tokenizer
  provides no offset for the span), selection fails unless `occurrence`
  resolves it.
- Coverage: a token **covers** the span when the token's byte interval
  intersects the span's byte interval and the union of selected token
  intervals covers the span's interval. `exact` coverage means the union
  equals the span; `enclosing` means the union strictly contains it
  (token boundaries may exceed the text span). Both are recorded;
  boundary expansion is recorded, never guessed away.
- Round-trip: the decoded concatenation of selected pieces must reproduce
  the span text where exact reconstruction is possible; otherwise the
  record marks `round_trip: "partial"` with the reason.
- Multi-token targets, targets at prompt start/end, leading-space
  tokenization (e.g. `Ġ` prefixes), punctuation-adjacent targets, Arabic
  diacritics and combining marks are all supported without normalization.

## 8. Capture timing

Captures resolve to a plan **before** inference (capture plan), then fire
during execution at the documented observation timing (section 2). Capture
copies are made at the hook; captured payloads are **owned** buffers and
never retain references into scratch arenas or KV buffers (they remain
valid after scratch reuse). `SummaryOnly` captures compute deterministic
statistics (shape, finite count, min, max, mean, L2 norm) and cannot be
intervention sources. Default storage is `SelectedRows` (only selected
token rows); `FullTensor` is explicit and reported as a cost.

## 9. Fused execution and requested tensors

The frozen fusion set is F1-F5 (v0.4). The only fusion that eliminates a
hook-observable tensor is **F5** (output projection fused with the
residual add), which eliminates the standalone `attention-output` tensor
for a layer. At runtime (the v0.4 decode interpreter), the plan is
requested with mode `Planned`, so the executing plan is always unfused and
all six sites are materialized; the fused plan exists in the plan builder
(`execution_plan(mode = PlannedFused, ...)`) and in `ember inspect-plan`.

## 10. De-fusion policy

When a capture or intervention requires a tensor that a fusion would
eliminate:

- The execution plan is built with the hook stages required by the
  capture/intervention plan. The builder de-fuses F5 for a layer when
  `after-attention` (the `attention-output` site) is requested for that
  layer: the `o` projection output stays materialized and the hook fires
  on it (Gate C).
- Every de-fusion decision is recorded in provenance (stage, layer,
  fusion state, reason).
- If a requested tensor cannot be materialized by any route, the run fails
  before inference rather than silently capturing the wrong tensor.

## 11. Exact restoration

Restoration is the operation pair: capture the target tensor at a semantic
site (owned copy), apply an intervention, then write the captured values
back at the **same** site in a follow-up evaluation. Exact restoration is
defined as: after restoration, downstream outputs (generated token IDs and
logits at the restored positions) match the unintervened baseline within
the frozen restoration envelope (bit-identical where the underlying
execution is deterministic; see Gate D). A run that captures the original,
applies an intervention, and restores must record all three events.

## 12. Stable public API vs diagnostic fields

Stable (schema-versioned, `ember.hook.v1`): hook site ids, layer numbering,
token-position conventions, site semantics (sections 1-6), capture plan
fields, intervention operation set (`replace`, `zero`, `scale`,
`interpolate`, `add-delta`, `restore-original`), bundle layout, bundle
schema version, semantic manifest fields, semantic hash definition.

Diagnostic (may evolve within 0.5.x): kernel names, dispatch details,
scratch layout, timing fields, operator profiles, `ExecutionPlan`
internals other than `plan_hash`, `runtime.json` contents.

## 13. Bundle schema compatibility

- `ember.bundle.v1`: unknown major versions fail; minor-version evolution
  is forward-compatible only for optional recognized fields.
- The semantic manifest never reinterprets a hook id under new semantics:
  hook meaning changes bump the hook schema major version.
- Deterministic content (semantic manifest) is hashed with canonical
  serialization: stable key ordering, stable array ordering, documented
  float serialization (Rust `serde_json` f32/f64 shortest round-trip
  formatting), UTF-8, no platform-dependent separators in semantic
  identifiers, no hash-map iteration order.

## 14. Deterministic vs machine-dependent metadata

`semantic-manifest.json` holds only fields expected to be equal across
equivalent reruns (section 7 of the release objective): schema versions,
experiment name and semantic configuration, model SHA-256, tokenizer
SHA-256, resolved token selectors, selected token IDs, execution mode,
plan hash, hook sites, capture/intervention definitions, generated token
IDs, output text, tensor payload checksums, deterministic warnings, Ember
version/commit. `runtime.json` holds timestamp, hostname, OS, CPU model and
features, thread count, wall-clock timings, peak RSS, compiler version,
PID, local paths — verifiable but excluded from the semantic hash.
`BundleIdentity { semantic_hash, payload_hash }` splits the two.

## 15. Fail-closed behavior

- **Ambiguous token selection** fails (unless `occurrence` resolves it).
- **Absent spans** fail. **Missing occurrences** fail.
- **Incompatible intervention sources** fail: model SHA mismatch,
  tokenizer SHA mismatch (where token semantics depend on it), hook-site
  mismatch, layer mismatch, rank/shape mismatch, selected-token-count
  mismatch, dtype mismatch, schema mismatch, source checksum mismatch,
  source bundle unverified. Expert overrides exist only where semantically
  defensible (documented in provenance); **shape incompatibility is never
  overridable**.
- **Unknown schema major versions** fail. **Unknown required fields**
  fail. **Duplicate ids** fail. **Invalid layer ranges** fail.
  **Unsupported hook sites and execution modes** fail before inference.
- Malformed specs are never silently repaired; errors carry the exact
  field path.
- A bundle is written atomically: staging dir → rename; incomplete
  bundles never look successful, and `verify` rejects them.

## 16. v0.5 gates

- **Gate A — specification correctness.** Valid v1 specs parse and resolve
  deterministically; invalid field paths produce precise errors; unknown
  schema majors fail; duplicate ids fail; ambiguous token matches fail;
  invalid layer selectors fail; unsupported hook sites and execution modes
  fail before execution; resolved specs serialize deterministically.
- **Gate B — token-selection correctness.** Across ASCII and Arabic
  prompts: prompt-final selects the actual final prompt token; explicit
  indices resolve; final-subtoken selection matches byte coverage;
  occurrence selection is deterministic; absent spans fail; ambiguous spans
  fail unless resolved; combining marks do not trigger silent
  normalization; punctuation-adjacent targets align; records reconstruct
  the selected span where exact reconstruction is possible. Tests include
  Arabic diacritics, Arabic punctuation, repeated Arabic targets, mixed
  Arabic/Latin text, leading-space tokenization, target at prompt
  start/end, multi-token targets, one-token targets, normalization-
  sensitive boundaries.
- **Gate C — capture semantics.** For every public site: captured tensors
  match direct internal captures from the reference path; layer ids and
  token rows are correct; selected-row capture equals the corresponding
  row of a full capture; planned and planned-fused results satisfy the
  existing parity envelopes; fusion deactivates when required; provenance
  records the executed route; payloads remain valid after scratch reuse.
- **Gate D — intervention semantics.** Replace/zero/scale/interpolate
  execute at the correct site; cross-bundle replacement rejects
  incompatible sources; shape and model mismatches fail; exact restoration
  reproduces the baseline; intervention-disabled execution is bit-identical;
  reference/planned/planned-fused agree within existing gates.
- **Gate E — bundle determinism.** Two equivalent runs on the same
  supported environment produce identical resolved specs, token-selection
  records, generated token ids, semantic manifests, semantic hashes,
  capture payloads (where execution is bit-identical), and matching
  payload hashes except for documented architecture-dependent float cases.
  Timestamps, hostname, timing, local paths never alter the semantic hash.
- **Gate F — verification.** Valid bundles verify; one-byte payload
  corruption fails; altered manifest values fail; removed files fail;
  extra unindexed payloads fail or produce an explicit policy error;
  invalid relative paths fail; unsupported bundle schema fails; incomplete
  staging bundles fail; model mismatch under deep verification fails;
  verification is fully offline.
- **Gate G — comparison.** Identical bundles report semantic identity;
  timing-only differences are separated; capture perturbations produce
  correct tensor metrics; token divergence reports the first differing
  position; intervention differences are identified precisely;
  machine-readable output is deterministic.
- **Gate H — performance and memory.** Relative to the final v0.4 release
  under the same protocol: ordinary inference does not regress more than
  3% median decode throughput; peak RSS without captures does not regress
  more than 3%; compressed Q4_K/Q6_K residency remains intact; Q8_0 stays
  untouched; selected-row capture overhead is measured and reported;
  full-sequence capture creates no unexplained copies; no steady-state
  allocations enter ordinary decode; bundle-system binary-size increase is
  documented. This gate is not weakened after seeing final results without
  a committed evidence-based amendment.
- **Gate I — clean-machine workflow.** On a clean supported machine,
  following only the documentation: install/build, validate a provided
  spec, run the reference example, inspect the bundle, verify it, run an
  intervention, compare bundles — no source edits, no knowledge of Ember's
  internal Rust types.

## 17. Compatibility policy

Versioned independently: experiment spec schema (`ember.experiment.v1`),
bundle schema (`ember.bundle.v1`), semantic hook schema (`ember.hook.v1`),
execution-plan schema (`v04-plan/1`). `v1` means stable within Ember
0.5.x, not that every future field is frozen forever. Unknown major:
fail. Newer minor with only optional recognized-compatible fields:
handle only if explicitly supported. Missing required field: fail.
Renamed semantic hook: require migration. Changed hook meaning:
increment hook schema major. Verification never reinterprets an old hook
id with new semantics silently.

## 18. Security assumptions

Experiment specs and bundles are untrusted input: reject path traversal;
reject absolute paths inside bundle indexes; enforce payload-size checks
before allocation; validate tensor dimensions before multiplication;
detect integer overflow; never execute embedded commands; never load
arbitrary dynamic libraries; do not follow bundle symlinks during
verification by default; write output atomically; never overwrite existing
bundles unless explicitly requested; sanitize experiment ids used in
paths; localize `unsafe`; every `unsafe` block carries a safety comment.

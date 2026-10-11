# ADR-0002: Attention-head sites (proposed)

- **Status:** Proposed. It needs the owner's decision, because it changes the
  frozen `ember.hook.v1` site list.
- **Date:** 2026-10-08
- **Scope:** `ember.hook.v1`, `ember.experiment.v1`, the core hook framework,
  and every Llama execution path

## Context

`ember.hook.v1` has exactly six public sites (research contract, section 1).
The nearest site to the attention heads is `attention-output`, the result of
`o_proj`. That result mixes all heads. Thus Ember cannot capture one head,
patch one head, or show where a head attends. Most circuit research needs
these operations, for example the search for induction heads.

Research contract section 19 names `ember.hook.v1` as a 1.x serialized
anchor. A seventh site is an addition to a frozen list. This document
describes the change, its cost, and the rules that keep earlier bundles
valid. The owner decides whether 1.x takes the change.

## What the code has today

These facts come from the current tree (2026-10-08).

- **The per-head output `z` exists in every Llama path.** `z` is the
  concatenation of the head outputs before `o_proj`, with shape
  `[seq, n_heads * head_dim]`. It is a real f32 buffer in the reference path
  (`llama.rs`, the attention scratch that `o_proj` reads), in the Q8 fast
  decode path, in the planned interpreter (plan tensor `layer{l}.attn`), in
  batched decode (`llama_batch.rs`), and in prefix resume (which uses the
  reference hooks).
- **No fusion merges attention with `o_proj`.** F5 fuses `o_proj` with the
  residual add and takes `z` as its input. Thus a hook on `z` needs no
  de-fusion rule.
- **Attention probabilities are never a tensor.** Each (row, head) computes
  its score row, applies softmax in place in thread-local scratch, and
  continues. Planned decode keeps one `[n_heads, max_seq]` score region for
  the current token only.
- **Plan hashes already change between binaries.** `plan_hash` covers the
  whole plan, which includes `ember_version` and `git_commit`. Thus
  `exact-semantic` reproduction already holds only on the same binary. The
  change must not alter plans of specs that do not use the new site, and it
  must not break `verify` of earlier bundles.

## Proposal

### 1. A new per-layer site: `attention-heads`

- **Tensor.** `z` for layer `l`, before `o_proj`. The shape is
  `[seq, n_heads * head_dim]` in prefill and `[1, n_heads * head_dim]` in
  decode. Head `h` is the column slice `[h * head_dim, (h + 1) * head_dim)`.
  "Head" means the query head index `0..n_heads` (with GQA, several query
  heads share one KV head).
- **Width.** The width is `n_heads * head_dim`. It is not always the
  embedding width, because `head_dim` comes from `attention.key_length`. The
  contract text "layer sites are `[seq, embed]`" needs an amendment for this
  site.
- **Timing.** Observation and intervention occur after the attention op and
  before `o_proj`. `o_proj` reads the changed value. This is the same rule as
  the other pre-projection sites.

### 2. A `heads` selector on captures and interventions

- `heads = [3, 7]` limits an intervention to those column slices. Without
  `heads`, the whole row changes, as at every other site.
- On a capture, `heads` stores only those slices. Rows stay in head order.
- `heads` is valid only at `attention-heads`. An index at or above
  `n_heads` fails before the run.
- Source rows for `replace`, `interpolate` and `add-delta` must have the
  width of the selected slices.

### 3. Attention probabilities: a later, separate decision

A capture-only `attention-pattern` site is possible but costly. In prefill,
it needs `n_heads * S * T * 4` bytes for each layer: 32 MiB for each layer of
Llama-3.2-1B at 512 tokens. The safe method recalculates `QK^T` and the
softmax from `q` and the KV cache after the attention op, with the same dot
and softmax code. Then the hot kernels do not change. This document does not
propose it for the first step.

## Rules that keep earlier bundles valid

1. **Site records only when active.** The plan builder emits one
   `HookSiteRecord` for every stage on every layer, active or not
   (`plan_build.rs`). Emit the new record only when a spec requests the site.
   Otherwise every plan, and every plan hash, changes.
2. **Optional fields only.** `heads` and every other new field use
   `#[serde(default, skip_serializing_if = "Option::is_none")]`. `verify`
   re-serializes the plan and the semantic manifest to recompute hashes, so
   a field that serializes when absent breaks earlier bundles
   (`tests/bundle_compatibility.rs`, `tests/fixtures/bundle-v050`).
3. **Opt-in stage.** The default `uses_activation_stage` returns true.
   Built-in experiments that keep the default (`ActivationStats`, the KV
   attention capture) must not start to request the new stage.
4. **No hook-schema version bump.** `verify` compares `HOOK_SCHEMA_VERSION`
   for equality. A bump makes the new binary reject every earlier bundle.
   The new site is an addition to `ember.hook.v1`. An older binary fails
   closed on a bundle that uses it, because the site name is unknown.
5. **gemma4 fails closed.** The v05 runner runs only Llama-family models
   today. A core hook on gemma4 must return an error, not skip.

## Work

- **Core:** `hook_types.rs`, `experiments/mod.rs` (stage, hook, capture
  trait, macros, active and disabled hooks), `plan_build.rs` (stage lists,
  active-only site record), `planned_decode.rs` (resolved hook sites),
  `llama.rs` (the attention function needs a hooks argument, and the Q8 fast
  path), `llama_batch.rs` (a per-sequence hook loop between attention and
  the `o_proj` quantization), `gemma4.rs` (fail closed).
- **Research layer:** `v05/hook.rs`, `runner.rs` (stage-to-site map, site
  order, a new `fire_site` method, head slicing), `capture.rs`,
  `intervention.rs`, `spec.rs` (validation of `heads`), `lens.rs` and
  `attribution.rs` (refuse the site), `run.rs` (hook route).
- **Hook forwarders:** `cli_experiment.rs` (`V05Adapter`) and
  `cli_experiment_shared.rs` (`SharedPass`). A forwarder that misses the new
  method drops it silently through the trait default. Add an end-to-end
  test through `SharedPass`.
- **Model facts:** `ModelFacts` and `ModelContext` need `n_heads` and
  `head_dim`.
- **Tests:** a seven-site version of `tests/six_site_observer.rs`; equality
  of `z` across reference, planned, planned-fused and batched decode; a head
  ablation that equals a manual `o_proj` of the changed `z`; old-bundle
  fixtures that still verify; plans without the site that do not change.
- **Docs:** research contract sections 1, 9, 10 and 12; `interventions.md`;
  `experiments.md`.

## Risks

- **Hook timing.** The hook must fire between attention and `o_proj` in
  five Llama paths. A path that misses it gives different results in a
  different mode. The cross-mode equality test is the guard.
- **Performance.** The disabled-hook path is an inlined no-op, and the
  planned path checks `if let Some(hooks)`. Measure decode speed before and
  after with the hooks off.
- **Batched decode.** Each sequence must see only its own row.

## Decision needed

1. Does 1.x add a seventh site to `ember.hook.v1`, as an addition under the
   rules above? The alternative is a new `ember.hook.v2`, with a migration
   note.
2. Is `attention-pattern` out of scope for the first step?

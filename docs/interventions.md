# Interventions (v0.5)

Interventions use the same semantic addressing as captures (site, layer,
token selector, input selector) and apply in declaration order at the
documented hook timing (see `docs/v05-research-contract.md`).

## Operations

- `replace`: copy the source row into the target row(s).
- `zero`: fill the target row(s) with zeros.
- `scale { factor }`: multiply the current values by `factor`
  (finite-checked).
- `interpolate { alpha }`: `target := (1 - alpha) * target + alpha *
  source` (finite-checked).
- `add-delta`: `target := target + source`.
- `restore-original`: write back the exact pre-intervention snapshot of
  the target row.

The pre-intervention snapshot of every intervened row is taken at the
first fire and checksummed; `restore-original` reproduces it exactly.

## Sources

- `inline-vector { values }`: one row, broadcast to every selected row.
- `capture-from-current-run { capture_id }`: rows from a capture in the
  same run; the capture must have fired earlier in execution order.
- `capture-from-bundle { bundle_path, capture_id, input_id, layer }` :
  rows from a verified bundle. The source bundle must pass full offline
  verification, and the model/tokenizer hashes must match unless an
  explicit expert compatibility override is set (recorded prominently in
  provenance).
- `zero`: an all-zero row.

No arbitrary executable transformations exist.

## Fail-closed validation

Before execution: model SHA compatibility (default fails on mismatch),
tokenizer SHA compatibility, hook-site compatibility, layer
compatibility, tensor rank/shape, selected-token count, dtype
conversion, source-capture checksum, source-bundle verification status.
Shape mismatch is never overridable.

## De-fusion

When an intervention targets a site whose tensor would be eliminated by
a fused execution plan, execution de-fuses automatically. Every
de-fusion decision is recorded in the bundle (`traces/events.jsonl`:
per-layer fusion state; capture index: hook route per tensor). The
frozen v0.4 fusion set F1–F5 is preserved for runs without
interventions.

## Restoration workflow

1. Capture the original tensor (or rely on the automatic snapshot).
2. Apply an intervention.
3. Apply `restore-original` at the same site.
4. Compare against the unintervened baseline: the reference example
   (`examples/experiments/morphology-restoration.toml`) restores
   bit-exactly: `compare` reports identical tokens, text, top-1, and
   every capture `exact`.

## Cross-bundle replacement

A capture from a saved bundle can back an intervention in a later run.
The source bundle must pass full offline verification; model, tokenizer,
hook site, layer and shape compatibility are checked before execution.
Model/tokenizer mismatches require an explicit expert override recorded in
provenance. Source layer must match every targeted layer; shape and layer
mismatches are not overridable.

For the [morphology workflow](../examples/experiments/README.md), first create
and verify `runs/morphology-baseline`. In a copy of the intervention spec,
keep `layers = [7]` and replace its current-run source with:

```toml
source = { kind = "capture-from-bundle", bundle_path = "runs/morphology-baseline", capture_id = "target-final-subtoken", input_id = "example-001", layer = 7 }
```

Give the copied spec a new experiment name and output directory. Validate,
run, verify and compare it through the same documented commands. For exact
restoration, append the `restore-original` declaration from the restoration
spec at the same site and layer. A single source row broadcasts to the selected
rows; otherwise the source row count must exactly match the target selection,
in selector order. Column count must match.

### Historical cross-layer result

The retained 2026-08-04 artifacts under
`artifacts/benchmark-v05/capture-from-bundle/` used a layer-3 source for a
layer-8 target. Their observed restoration result belongs to that older writer;
those intervention specifications are rejected by the current layer-compatibility
check and are not current runnable examples. Preserve the artifacts and their
identities as historical evidence. See the
[migration notes](migration-to-1.0.md) before rerunning old experiments.

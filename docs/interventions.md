# Interventions (v0.5)

Interventions use the same semantic addressing as captures (site, layer,
token selector, input selector) and apply in declaration order at the
documented hook timing (see `docs/v05-research-contract.md`).

## Operations

- `replace` — copy the source row into the target row(s).
- `zero` — fill the target row(s) with zeros.
- `scale { factor }` — multiply the current values by `factor`
  (finite-checked).
- `interpolate { alpha }` — `target := (1 - alpha) * target + alpha *
  source` (finite-checked).
- `add-delta` — `target := target + source`.
- `restore-original` — write back the exact pre-intervention snapshot of
  the target row.

The pre-intervention snapshot of every intervened row is taken at the
first fire and checksummed; `restore-original` reproduces it exactly.

## Sources

- `inline-vector { values }` — one row, broadcast to every selected row.
- `capture-from-current-run { capture_id }` — rows from a capture in the
  same run; the capture must have fired earlier in execution order.
- `capture-from-bundle { bundle_path, capture_id, input_id, layer }` —
  rows from a verified bundle. The source bundle must pass full offline
  verification, and the model/tokenizer hashes must match unless an
  explicit expert compatibility override is set (recorded prominently in
  provenance).
- `zero` — an all-zero row.

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
   bit-exactly — `compare` reports identical tokens, text, top-1, and
   every capture `exact`.

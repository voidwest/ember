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

## Cross-bundle replacement (real-model workflow)

A capture from a saved bundle can back an intervention in a later run.
The source bundle must pass full offline verification; model, tokenizer,
hook-site, layer, and shape compatibility are checked before execution
(model/tokenizer mismatches fail closed unless an expert override is
recorded). The recorded real-model workflow lives under
`artifacts/benchmark-v05/capture-from-bundle/`:

```bash
# 1. baseline: capture prompt-final attention-output across all layers
ember experiment run artifacts/benchmark-v05/capture-from-bundle/baseline.toml
ember experiment verify runs/cfb-baseline

# 2. cross-bundle replace: layer 8's prompt-final attention row is
#    replaced with the baseline bundle's layer-3 row
ember experiment run artifacts/benchmark-v05/capture-from-bundle/intervention.toml
ember experiment verify runs/cfb-intervention
ember experiment compare runs/cfb-baseline runs/cfb-intervention

# 3. restoration: the same replace followed by restore-original at
#    layer 8; compare reports every capture exact and outputs equal
ember experiment run artifacts/benchmark-v05/capture-from-bundle/restoration.toml
ember experiment verify runs/cfb-restoration
ember experiment compare runs/cfb-baseline runs/cfb-restoration
```

Observed (2026-08-04, Llama-3.2-1B-Instruct-Q8_0): the replace leaves
layers 0-8 bit-exact (captures fire before interventions at the same
site) and diverges from layer 9 onward; the restoration leg reproduces
the baseline with all 16 capture layers exact, generated tokens/text
equal, and final top-1 equal.

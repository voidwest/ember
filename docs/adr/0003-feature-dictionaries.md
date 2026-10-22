# ADR-0003: Feature dictionaries

- **Status:** Proposed on 2026-10-22. The owner accepts it by merging the
  pull request that implements it.
- **Date:** 2026-10-22
- **Scope:** `ember.experiment.v1` (a `[[dictionaries]]` section and a
  `features` field on captures and interventions), two new artifact schemas
  (`ember.dictionary.v1`, `ember.features.v1`), the v0.5 runner, and
  `ember experiment verify`

## Context

A sparse autoencoder (SAE) maps an activation `x` of width `d_in` to `d_sae`
feature activations `f`, which are mostly zero, and maps `f` back to a
reconstruction `x̂`. Much current interpretability work reads and edits a
model through such a dictionary: which features fire on a token, and what
the output does when one feature is removed. Ember can capture and edit raw
activations, but it cannot do either of these in feature space.

Public SAEs for the models that Ember runs exist in the SAELens format: a
folder with `cfg.json` and `sae_weights.safetensors` (`W_enc [d_in, d_sae]`,
`b_enc [d_sae]`, `W_dec [d_sae, d_in]`, `b_dec [d_in]`, and `threshold
[d_sae]` for JumpReLU).

## What the code has today

- **Steering is the precedent.** It added two operations and two sources
  inside `ember.experiment.v1`, with no schema bump. A pinned file
  (`vector-file`, with a required SHA-256) is read before execution, and the
  resolved direction is written to `artifacts/directions/` with its own
  record schema, `ember.direction.v1`, which verify checks.
- **The operation set is a stable item** (research contract, section 12).
  `InterventionOperation` is a small `Copy` enum.
- **A capture is raw site rows.** Derived data lives under `artifacts/` with
  a schema and a verify check of its own (directions, attribution, probe).
- **Verify is offline.** It checks hashes, shapes and internal consistency.
  It does not run the model. `reproduce` runs the model again.
- **The `heads` selector** (ADR-0002) is the pattern for a narrowing field:
  the operation sees a view of the row, and the result is written back.

## Decision

### 1. A `[[dictionaries]]` section

```toml
[[dictionaries]]
id = "l9"
path = "/path/to/blocks_9_hook_resid_post_8X_2048_jumprelu"
sha256 = "<SHA-256 of sae_weights.safetensors>"
config_sha256 = "<SHA-256 of cfg.json>"
site = "residual-post-mlp"
layer = 9
```

- Both files are pinned. `cfg.json` selects the architecture and its
  options, so the weights alone do not say how to encode.
- The format is SAELens. Version 1 accepts the `standard` (ReLU) and
  `jumprelu` architectures, a ReLU activation (`activation_fn` or the older
  `activation_fn_str`, with no `activation_fn_kwargs`), F32 or F16 tensors,
  `normalize_activations = "none"`, and an explicit `apply_b_dec_to_input`.
  It refuses TopK and other activations, `finetuning_scaling_factor`, a
  `hook_head_index`, BF16, a weights file above 2 GiB, and a hash mismatch.
- The site must be one of the `d_model`-wide per-layer sites. `d_in` must
  equal the model's width.
- `cfg.json` is read for the architecture and its options. Its `hook_name`
  and `model_name` are recorded, but Ember does not translate SAELens hook
  names into its own sites: the author states the site and the layer.

### 2. Feature interventions use the existing operations

An intervention with `features = { dictionary = "l9", ids = [..] }` applies
its operation to the selected feature activations `f_S`, not to the raw
row. The row then changes by the decoded difference only:

```text
f_S  = encode(x)_S
f'_S = operation(f_S, source)
x'   = x + (f'_S - f_S) · W_dec[S]
```

- The reconstruction error `x - x̂` stays in the row, so the features that
  the edit does not touch keep their effect.
- An edit that changes nothing is bit-identical: each output element adds an
  f64 sum of zero terms to the f32 value, which returns the same value.
- `zero`, `scale`, `replace`, `add-delta` and `interpolate` are allowed.
  `steer`, `ablate-projection` and `restore-original` are refused with
  `features`.
- An `inline-vector` source holds feature values (one per selected feature).
  A capture source (`capture-from-current-run`, `capture-from-bundle`) holds
  raw site rows, which Ember encodes with the same dictionary. Thus "patch
  feature 1234 from the clean run" is `replace` with a bundle source.
- The intervention must act at the dictionary's site and at its one layer.

No operation kind is added, so the stable operation set does not change.

### 3. Feature captures are a derived artifact

A capture with `features = { dictionary = "l9", top_k = 32 }` stores its raw
rows as before (it must use `f32` and keep rows), and Ember also writes
`artifacts/features/<capture-id>.json` (`ember.features.v1`). For every
stored row it records the input, layer and position, the top-k features
(index and value), `l0` (the count of active features), the row norm, and
the relative reconstruction error `|x - x̂| / |x|`.

### 4. Bundles record the dictionary, not its weights

A dictionary is hundreds of megabytes, and a sweep writes one bundle per
point. Each bundle therefore records `artifacts/dictionaries/<id>.json`
(`ember.dictionary.v1`): the two pinned hashes, the path as given, `d_in`, `d_sae`,
the architecture and its options, the site and the layer, and the
`cfg.json` model and hook names. `reproduce` needs both files at the
recorded path with the recorded hashes.

### 5. What verify checks

Verify cannot recompute feature values offline. It checks:

- that every dictionary record agrees with the spec's `[[dictionaries]]`
  entry (both hashes, the path, the site and the layer), and every feature
  artifact with its capture;
- that the feature rows correspond one-to-one to the capture's stored rows,
  with the same inputs, layers and positions;
- that each recorded row norm equals the norm of the stored raw row,
  recomputed in the same order, to within 1e-12 relative: `serde_json`'s
  default parser can read a written f64 back one unit in the last place off;
- that top-k lists are sorted, have valid indices below `d_sae`, and agree
  with `l0`.

### 6. Numerics

Encode and decode accumulate in f64 in index order and round once to f32.
The selected-feature encode and the full encode use the same order, so a
capture and an intervention see the same feature values. JumpReLU follows
SAELens: a feature is active when its pre-activation is above its threshold
and above zero.

## Rules that keep earlier bundles valid

- The new fields use `skip_serializing_if`, so a spec, plan or manifest
  without them serializes byte for byte as before (ADR-0002, rule 2).
- `HOOK_SCHEMA_VERSION` and the experiment and bundle schema strings do not
  change.
- Earlier verifiers refuse a bundle with a `features` field, because the
  manifest types deny unknown fields. That is the correct failure for a
  bundle that they cannot check.

## Out of scope for version 1

- Transcoders and skip transcoders: they read the MLP input, which is not a
  public site.
- Sweeps over features, and attribution in feature space.
- Copying weights into bundles, and formats other than SAELens.

## Risks

- **A dictionary trained on another model or precision.** Version 1 records
  the reconstruction error of every captured row, so the author can see the
  fit. It does not refuse a poor fit.
- **Memory.** A d_sae 16384 dictionary at d_in 2048 holds 268 MB of f32
  weights. It loads once per session and is shared between inputs and sweep
  points. While it loads, Ember holds the file and its f32 copy, so memory
  peaks near twice the file size (three times for F16).
- **Speed.** A full encode is `d_in × d_sae` multiply-adds per row (34 M for
  the example above). Captures encode only their selected rows, and an
  intervention encodes only its selected features.

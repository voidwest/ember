# Experiment specification schema v1

`ember.experiment.v1` is the user-authored experiment language. It is a
strict TOML document: unknown fields, unknown schema majors, duplicate
IDs, invalid references, and impossible settings fail before inference.
Semantic validation reports a field path; TOML decoding errors include source
location and field/type details. Nested token, operation, and source objects
also reject unknown fields, including objects with a unit `kind`.

`v1` means the schema is stable within Ember 0.5.x: it does not freeze
every possible future field forever.

## Top-level

```toml
schema = "ember.experiment.v1"

[experiment]
name = "layerwise-target-capture"   # required; path-safe id
description = "..."                 # optional
seed = 42                           # optional; 0 = no stochastic sampling

[model]
path = "models/model.gguf"          # required
expected_sha256 = "..."             # optional; verified at load
tokenizer = "tokenizer.json"        # optional; resolved from --arch
tokenizer_expected_sha256 = "..."   # optional
arch = "auto"                       # optional: auto|gpt2|llama|qwen3|gemma4

[execution]
mode = "reference"                  # optional: reference|planned|planned-fused
threads = 8                         # optional; 0 = auto
deterministic = true                # optional; requires temperature 0 or a seed

[generation]
max_new_tokens = 0                  # optional
temperature = 0.0                   # optional

[[inputs]]
id = "example-001"                  # required; path-safe id
text = "..."                        # required

[[captures]]
id = "prompt-final"                 # required; unique across captures
site = "residual-post-mlp"          # required; see hook sites
layers = "all"                      # optional: all|[list]|{start,end,step} (end exclusive)
storage = "selected-rows"           # optional: selected-rows|full-tensor|summary-only
dtype = "f32"                       # optional: f32|f16
# heads = [0, 3]                    # attention-heads only: ascending heads to store
# features = { dictionary = "l9", top_k = 32 }  # also record each row's strongest features

[captures.tokens]                   # required
kind = "prompt-final"               # see token selectors

[[interventions]]
id = "iv-1"                         # required; unique, no capture-id collision
site = "attention-output"           # required
layers = [0]                        # optional
operation = { kind = "zero" }       # required; see operations
# source = { kind = "zero" }        # required by replace/interpolate/add-delta/steer/ablate-projection
# heads = [1]                       # attention-heads only: ascending heads to change
# features = { dictionary = "l9", ids = [12, 40] }  # act on these features, not the row
[interventions.tokens]              # required
kind = "prompt-final"

[output]
directory = "runs/example"          # required
tensor_format = "safetensors"       # optional; safetensors is the only v0.5 format
overwrite = false                   # optional
```

Every omitted default is recorded in the resolved specification and
serialized into the bundle as `resolved-experiment.json` (its `defaults`
array), so a bundle always states exactly what ran. Nested omissions include capture/intervention
layer and input selectors, range `step`, span `normalization`, storage/dtype,
optional intervention source, and each compatibility flag. An omitted
`heads` selector is not recorded: it is absent from the resolved spec, so
specs without it resolve exactly as before it existed. Explicit values
equal to defaults are not labeled as omitted. Omitted empty capture/intervention
lists and automatic tokenizer selection are also recorded. The resolved
settings are identical whether a default is omitted or explicitly supplied.


## Sweeps

An optional `[sweep]` table turns the spec into a template for a layer
sweep (`layers`, optional `positions`, optional `heads`, optional `alphas`,
optional `interventions`, optional `move_bundle_sources`, optional
`[sweep.effect]`; `layers` may be omitted when `heads` or `alphas` is
given); see `docs/experiments.md#layer-sweeps`. Unknown `[sweep]` fields fail, and a
spec with `[sweep]` is only accepted as a sweep.

## Attribution

An optional `[attribution]` table (`clean`, `corrupted`, `target`, `foil`,
optional `sites`, `layers`, `positions`, `verify_top_k`) adds the
attribution-patching workflow; see `docs/experiments.md#attribution-patching`.
The inputs must exist and differ, sites must be per-layer and distinct,
`verify_top_k >= 1`, and the spec must declare no interventions. Omitted
fields are recorded as defaults. It cannot be combined with `[sweep]`.

## Probe bridge

An optional `[probe]` table (`site`, `layer`, `target`, and `train` or
`file`; optional `tokens`, `test`, `ridge_lambda`, `ablate`, `steer_alphas`,
`steer_normalize`, `intervene_tokens`) adds the probe-bridge workflow; see
`docs/experiments.md#probe-bridge-can-the-model-use-it`. Labels are 0 or 1,
training needs both labels, `ridge_lambda > 0`, at least one of `ablate` and
`steer_alphas`, the behavioural inputs must not be probe examples, the site
is per-layer, and the spec declares no interventions and no `[attribution]`.

## Feature dictionaries

Optional `[[dictionaries]]` entries (ADR-0003) name sparse autoencoders in
the SAELens format, which captures and interventions then use by id:

```toml
[[dictionaries]]
id = "l9"                           # required; path-safe, unique
path = "dictionaries/l9"            # required; folder with cfg.json and sae_weights.safetensors
sha256 = "<64 hex>"                 # required; SHA-256 of sae_weights.safetensors
config_sha256 = "<64 hex>"          # required; SHA-256 of cfg.json
site = "residual-post-mlp"          # required; residual-pre-attention|attention-output|mlp-output|residual-post-mlp
layer = 9                           # required; the one layer it was trained at
```

Version 1 reads the `standard` and `jumprelu` architectures with a ReLU
activation, `normalize_activations = "none"`, F32 or F16 tensors, and an
explicit `apply_b_dec_to_input`; it refuses TopK, `finetuning_scaling_factor`
and `hook_head_index`. A capture or an intervention with `features` must
be at the dictionary's site and set `layers = [<its layer>]`, and cannot
also select `heads`. A feature capture needs `dtype = "f32"` and stored rows
(not `summary-only`), and `1 <= top_k <= 4096`. Feature `ids` are ascending
without repeats; `steer`, `ablate-projection` and `restore-original` do not
take `features`. When the run starts, the two files must match `sha256` and `config_sha256`,
`d_in` must equal the model's width, the layer must exist, and every feature
id must be below `d_sae`. See `docs/interventions.md#features` and
`docs/experiments.md#feature-dictionaries`.

## Operations and sources

```toml
operation = { kind = "replace" }                 # source required
operation = { kind = "zero" }
operation = { kind = "scale", factor = 0.5 }
operation = { kind = "interpolate", alpha = 0.5 } # source required
operation = { kind = "add-delta" }               # source required
operation = { kind = "restore-original" }
operation = { kind = "steer", alpha = 4.0, normalize = "unit" } # none|unit|match-residual-norm; direction source
operation = { kind = "ablate-projection" }       # direction source

source = { kind = "inline-vector", values = [0.1, 0.2] }
source = { kind = "capture-from-current-run", capture_id = "cap" }
source = { kind = "capture-from-bundle", bundle_path = "runs/x", capture_id = "cap", input_id = "i", layer = 7 }  # semantic_hash = "<64 hex>" pins the source
source = { kind = "zero" }
source = { kind = "vector-file", path = "dir.npy", sha256 = "<64 hex>" }            # tensor = "name" for safetensors
source = { kind = "contrastive", positive = ["..."], negative = ["..."] }          # tokens = {...}, default prompt-final
```

Direction operations (`steer`, `ablate-projection`) take `inline-vector`,
`vector-file` or `contrastive` sources; `vector-file`/`contrastive` are
refused for other operations. `vector-file` requires a 64-hex `sha256`;
contrastive prompt lists must be non-empty and their selector cannot be
`generated-step`. See `docs/interventions.md`.

## Semantic hook sites

| id | meaning | layers |
|---|---|---|
| `residual-pre-attention` | residual stream entering the block, before the input RMS norm | per-layer |
| `attention-heads` | per-head attention outputs, concatenated, before the output projection (`n_heads * head_dim` wide) | per-layer |
| `attention-output` | attention output after the output projection (pre-residual) | per-layer |
| `mlp-output` | MLP output after the down projection (pre-residual) | per-layer |
| `residual-post-mlp` | residual stream after the MLP addition (block output) | per-layer |
| `final-norm-output` | the final layer's normalized output, before the LM head | none (0) |
| `logits` | raw pre-softmax logits of the final row | none (0) |

Non-per-layer sites reject explicit layer selectors other than `all`.

`heads = [..]` is valid only at `attention-heads`. It must be non-empty,
ascending and without repeats; a head at or above the model's head count
fails before inference. A capture stores the selected heads' columns in
head order; an intervention changes only those columns, and its source or
direction has the selected width (`len(heads) * head_dim`). A capture source
may instead have the full `n_heads * head_dim` width; the selected heads are
then taken from it by head index.

## Token selectors

```toml
[captures.tokens]
kind = "prompt-final"

kind = "absolute-token", index = 5
kind = "relative-token", offset_from_end = 2
kind = "generated-step", step = 1        # 1-based decode steps
kind = "matched-span", text = "كِتَاب", occurrence = 0, subtokens = "first"
kind = "byte-span", start = 12, end = 24, subtokens = "all"
```

`subtokens` is required for matched/byte spans: `first`, `final`, or `all`.
`occurrence` is required for matched spans and is zero-based. `normalization`
defaults to `"none"`; use `normalization = "nfc"` for an explicit opt-in; Arabic is never normalized
silently. See `docs/token-selection.md`.

## Input selectors

Captures and interventions apply to an input selector:

```toml
inputs = "all"          # default
inputs = ["i1", "i2"]   # explicit list
```

## Capture storage

- `selected-rows` (default): only the selected token rows are stored.
- `full-tensor`: the complete sequence tensor at the site (explicit and
  potentially large; the cost is reported).
- `summary-only`: deterministic statistics only (shape, finite count,
  min, max, mean, L2 norm). Summary-only captures cannot back an
  intervention source.

## Validation rules (fail closed)

- unknown schema majors and unknown minor variants fail;
- unknown TOML fields fail;
- duplicate input/capture/intervention ids fail; intervention ids must
  not collide with capture ids;
- invalid layer ranges fail (`start >= end`, out-of-range
  indices for the model's layer count);
- unsupported hook sites, execution modes, and tensor formats fail
  before inference;
- `deterministic = true` with `temperature > 0` and no `seed` fails;
- `matched-span`/`byte-span` selectors require non-empty input text;
- `generated-step` selectors require `generation.max_new_tokens > 0`;
- `heads` outside `attention-heads`, an empty or unordered `heads` list,
  or a head beyond the model fails;
- `features` naming no `[[dictionaries]]` entry, at another site or layer
  than the dictionary's, with `heads`, with unordered ids, or (for a capture)
  with f16 or summary-only storage fails;
- cross-bundle sources reference a capture that exists in the current
  spec only for `capture-from-current-run`; bundle sources are validated
  against the source bundle at run time.

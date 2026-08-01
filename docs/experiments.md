# Research experiments

> **Experimental v0.1 API:** Ember's experiment interface is intentionally
> unstable. It may change between v0.1 releases and is not a dynamic plugin
> ABI or a semver compatibility commitment.

Ember supports one statically compiled Rust experiment during an ordinary
LLaMA, Qwen3, or Gemma 4 generation run. The interface is deliberately small:
an experiment can observe stable execution metadata, inspect selected owned
activations, and explicitly modify those activations in place.

The v0.1 release contains exactly two built-in proof points:

- `activation-stats`, an observation-only artifact recorder;
- `zero-layer-output`, an explicit intervention.

The MVP stops there. It does not include a registry or generalized experiment
configuration.

## Observation example

`activation-stats` records value summaries at every tensor-bearing hook:

```bash
cargo run --release -- \
  --arch qwen3 \
  --model Qwen3-0.6B-Q8_0.gguf \
  --tokenizer tokenizer-qwen3.json \
  --prompt "The capital of France is" \
  --max-tokens 4 \
  --temperature 0 \
  --activation-stats activation-stats.json
```

The JSON artifact contains model and generation metadata followed by ordered
records. Each record contains:

- prefill or decode phase;
- semantic hook stage and optional layer index;
- start position, input token count, and resulting sequence length;
- tensor shape and f32 dtype;
- L2 norm, absolute maximum, and a stable lightweight fingerprint.

The fingerprint reuses Ember's structured-tracing algorithm. It samples every
64th value and is intended for divergence detection, not cryptographic
identity. Norm calculation reads every value. Neither operation mutates the
tensor, but both add work and the experiment stores one record per hook.

Inspect the residual stream without loading the artifact into a custom tool:

```bash
jq '.records[] | select(.stage == "after_layer") |
    {phase, layer_index, sequence_length, l2_norm, fingerprint}' \
  activation-stats.json
```

Generated text keeps the normal stdout format. Activation notices and the
artifact summary are written to stderr. Existing run-manifest provenance marks
the experiment with `"modifies_execution": false`.

## Intervention example

`zero-layer-output` zeros one residual contribution at a selected layer:

```bash
cargo run --release -- \
  --arch gemma4 \
  --model models/gemma-4-E2B-it.Q8_0.gguf \
  --tokenizer tokenizer-gemma4.json \
  --prompt "The capital of France is" \
  --max-tokens 8 \
  --temperature 0 \
  --zero-layer-output 4:attention
```

The accepted stages are `attention`, `mlp`, and `layer`. Layer indices are
zero-based. Ember rejects malformed stages and validates the layer index after
the model is loaded.

The intervention warning and completion summary are written to stderr:

```text
research experiment active: zero-layer-output layer=4 stage=attention; execution will be modified
experiment zero-layer-output: 8 intervention(s) at layer 4 stage attention
```

When a run manifest is requested, its existing `execution` object records the
experiment name, layer, stage, and the fact that execution was modified. No
experiment field is emitted for a normal run.

## Observation and intervention

Hook methods do nothing by default. `activation-stats` observes activations
through `TensorAccess::values()`. An experiment must explicitly call
`values_mut()` or `zero()` to intervene.

`TensorAccess` borrows Ember's existing contiguous f32 allocation. It cannot:

- resize or reallocate the tensor;
- mutate its shape;
- transfer ownership of the storage;
- access weights, KV-cache storage, QKV scratch buffers, or backend internals.

Ember does not create tensor copies for experiments. Mutation hooks are
offered only where the model owns a mutable intermediate: layer input,
attention residual contribution, MLP residual contribution, completed layer
output, final hidden state, and logits.

## Lifecycle and hook order

A successful generation has this lifecycle:

```text
on_model_loaded
before_prefill

prefill evaluation:
  for each layer:
    before_layer
    after_attention
    after_mlp
    after_layer
  before_logits
  after_logits

each single-token decode evaluation:
  the same per-layer and logits hooks with phase=Decode

on_generation_complete
```

`ExecutionContext` distinguishes prefill and decode and reports the input
range, resulting sequence length, tracing state, model family, identifier, and
basic architecture metadata. `LayerContext` adds the zero-based layer index.
Tensor shape and dtype are available from `TensorAccess`.

The family-specific semantic boundaries remain explicit:

- LLaMA and Qwen invoke `after_attention` after the O projection and
  `after_mlp` after the down projection, immediately before their residual
  additions.
- Gemma 4 invokes those hooks after its post-attention and post-FFN
  normalization, immediately before the corresponding residual additions.
- Gemma 4 invokes `after_layer` after PLE and layer-output scaling.
- Qwen retains split-half RoPE and its QK-normalization ordering.

LLaMA's active decode hooks operate directly on its preallocated workspace.
They do not force generic decoding. Gemma's packed/generic MLP dispatch and
all existing tracing dispatch rules are also preserved.

## Important current limitation

> **Active experiments do not participate in hidden-state extraction,
> probing, `--dump-layers`, `--dump-logits`, demo mode, interactive mode, or
> benchmark subcommands in the MVP.**

Those options reject `--zero-layer-output`. Their normal, no-experiment
behavior is unchanged. Supporting experiment-aware hidden-state extraction
would require defining which representation is authoritative and how
interventions appear in extraction metadata; the MVP deliberately does not
guess.

## Implementing another built-in experiment

New experiments are compiled into Ember:

1. Add an implementation under `src/experiments/`.
2. Implement `Experiment`, leaving unused hooks at their defaults.
3. Keep owned experiment-specific records inside the implementation.
4. Use `TensorAccess` only at an approved mutation hook.
5. Add a narrow typed CLI option and explicit construction path.
6. Add hook-order, family, intervention, parity, allocation, and benchmark
   coverage.

Do not add a registry, discovery mechanism, configuration DSL, or generalized
pipeline. The MVP supports exactly one active experiment.

Hook errors should return `ExperimentError`. Ember adds the experiment name,
hook, execution phase, and layer where applicable before propagating the
failure through its normal error path.

## Guarantees

With no experiment option:

- model outputs and generation remain unchanged;
- tracing and trace fingerprints remain unchanged;
- hidden-state and layer-dump surfaces remain unchanged;
- packed and generic kernel selection remains unchanged;
- no experiment object is allocated;
- disabled per-layer hook calls are compiled away.

An active experiment receives the documented hook order and borrowed metadata
without per-hook token or model-metadata clones.

## Non-guarantees

Experiments can intentionally invalidate model outputs. They can also allocate
or perform expensive work inside their own hook implementations. Ember does
not claim numerical validity, latency stability, or benchmark comparability
for an intervention run.

Treat all benchmark numbers collected with an active experiment as a distinct
workload. Even an observation-only experiment can scan tensors, allocate
records, and write artifacts. Run provenance and stderr identify the
experiment, but they do not make its timing comparable to unmodified
inference.

Dynamic `.so`/`.dll` loading, WASM, Python plugins, runtime discovery,
multiple concurrent experiments, async hooks, arbitrary weight mutation,
custom tokenizers, and custom backends are intentionally unsupported in the
MVP.

## v0.2: capture, patching, and artifact comparison (experimental)

The v0.2 additions keep the single-experiment invariant and add two
run-level facilities:

- **Selective activation capture** (`--capture-activations capture.toml`):
  a run-level recorder that rides alongside the single experiment (or runs
  alone). It copies tensor values only for explicitly selected records at the
  six semantic stages, filtered by layer, phase, and decode position, with an
  optional record cap. Capture runs **after** the experiment, so captured
  values reflect post-intervention state. See
  [activation-artifacts.md](activation-artifacts.md).
- **`activation-patch`** (`--activation-patch manifest.json` +
  repeatable `--patch-target LAYER:STAGE:PHASE[:POSITION]`): replaces one
  live activation in place with a captured tensor. Source resolution is
  unambiguous (exactly one record per target), validation covers family,
  layer, width, dtype, and byte order, and every target must be applied or
  generation fails. See [activation-patching.md](activation-patching.md).
- **`compare-artifacts`** subcommand: deterministic record-by-record
  comparison of two artifacts with hard refusal on ambiguous alignment and
  exact-or-reported semantics for every field except `created_at_unix`.
  See [activation-artifacts.md](activation-artifacts.md).

Dispatch-path provenance: the runner records the kernel path (fast/workspace
vs generic) per evaluation and per captured record, and the manifest lists
phase-specific dispatch observations, since a single run can mix generic
prefill with fast/workspace decode.

The v0.1 guarantees below still hold: with no experiment and no capture, the
disabled per-layer hook calls compile away and outputs are unchanged. With
capture active but no record selected at a hook, the overhead is a branch
plus no-op. Patching allocates nothing inside the hook after initialization.

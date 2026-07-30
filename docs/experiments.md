# Research experiments

> **Experimental v0.1 API:** Ember's experiment interface is intentionally
> unstable. It may change between v0.1 releases and is not a dynamic plugin
> ABI or a semver compatibility commitment.

Ember supports one statically compiled Rust experiment during an ordinary
LLaMA, Qwen3, or Gemma 4 generation run. The interface is deliberately small:
an experiment can observe stable execution metadata, inspect selected owned
activations, and explicitly modify those activations in place.

## Example

The built-in `zero-layer-output` research example zeros one residual
contribution at a selected layer:

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

Generated text keeps the normal stdout format. The activation warning and
completion summary are written to stderr:

```text
research experiment active: zero-layer-output layer=4 stage=attention; execution will be modified
experiment zero-layer-output: 8 intervention(s) at layer 4 stage attention
```

When a run manifest is requested, its existing `execution` object records the
experiment name, layer, stage, and the fact that execution was modified. No
experiment field is emitted for a normal run.

## Observation and intervention

Hook methods do nothing by default. An experiment observes an activation with
`TensorAccess::values()` and must explicitly call `values_mut()` or `zero()` to
intervene.

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
workload. Run provenance and stderr make the intervention explicit, but they
do not make it comparable to unmodified inference.

Dynamic `.so`/`.dll` loading, WASM, Python plugins, runtime discovery,
multiple concurrent experiments, async hooks, arbitrary weight mutation,
custom tokenizers, and custom backends are intentionally unsupported in the
MVP.

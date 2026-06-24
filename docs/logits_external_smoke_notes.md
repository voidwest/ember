# External Logits Smoke Notes

## Decision

Chosen path: **B, llama-cpp-python/libllama binding**.

No standalone llama.cpp CLI binary was available under
`/tmp/llama.cpp-master/build/bin` in this workspace. The installed
`llama-cpp-python` package exposes `Llama.eval(tokens)` and stores logits when
the model is created with `logits_all=True`. That provides a stable local path
for a tiny logits-only external-backend smoke without generation and without
hidden-state extraction.

## What This Smoke Does

`scripts/llama_cpp_python_logits_extract.py` implements Ember's
`llama-cpp-external` request shape and writes:

- `samples.jsonl`
- `tokenization.jsonl`
- `positions.jsonl`
- `logits.npy`
- `manifest.json`
- `report.json`
- `checksums.json`
- `metadata.llamacpp-python-logits.json`

The helper evaluates each tiny prompt with llama-cpp-python/libllama and stores
the final selected-token logits as an `[n_samples, vocab_size]` f32 NPY array.

The run marks:

- `real_llama_cpp = true`
- `binding = "llama-cpp-python"`
- `standalone_llama_cpp_binary = false`
- `real_tokenization = true`
- `real_logits = true`
- `no_generation = true`
- `no_hidden_states = true`
- `not_research_output = true`

## What This Smoke Does Not Do

- It does not use a standalone llama.cpp CLI binary.
- It does not generate text.
- It does not compute or compare hidden states.
- It does not patch llama.cpp.
- It does not prove logits parity by itself.
- It is not research output.

## Local Flow

Use the guarded runner so machine-local paths stay under `/tmp`:

```bash
MODEL_PATH=/path/to/model.gguf \
OUT_ROOT=/tmp/sarf-atlas-ember-smoke \
bash sarf-atlas/smoke/run_logits_python_smoke.sh
```

The runner generates:

```text
/tmp/sarf-atlas-ember-smoke/llama_cpp_python_logits_smoke.local.toml
/tmp/sarf-atlas-ember-smoke/runs/llama-cpp-python-logits-smoke/
```

It then runs:

```bash
cargo run -- extract --config /tmp/sarf-atlas-ember-smoke/llama_cpp_python_logits_smoke.local.toml
cargo run -- validate-run /tmp/sarf-atlas-ember-smoke/runs/llama-cpp-python-logits-smoke
```

## Parity Status

`gguf-parity-tools compare-logits` is available, but it requires a candidate
and reference `.npy` or `.npz` logits artifact. Do not compare an artifact
against itself as a parity claim.

A real parity check needs a separate logits reference produced by another
trusted path for the same model, prompt rendering, tokenization policy,
selected token position, dtype, and vocabulary order.

## Native Reference

Ember also has a native logits-only reference smoke:

```bash
MODEL_PATH=/path/to/model.gguf \
TOKENIZER_JSON=/path/to/tokenizer.json \
OUT_ROOT=/tmp/sarf-atlas-ember-smoke \
bash sarf-atlas/smoke/run_native_logits_reference_smoke.sh
```

This runner generates a temporary config and calls:

```bash
cargo run -- native-logits-reference \
  --config /tmp/sarf-atlas-ember-smoke/native_logits_reference_smoke.local.toml
```

The native reference path uses Ember's `forward_last_logits_with_cache`
interface. It does not call the hidden-state extraction backend, does not write
layer shards, and does not generate tokens.

Once both artifacts exist, a tiny local comparison can be run with:

```bash
PYTHONPATH=/path/to/gguf-parity-tools \
python3 -m parity_tools compare-logits \
  --candidate /tmp/sarf-atlas-ember-smoke/runs/llama-cpp-python-logits-smoke/logits.npy \
  --reference /tmp/sarf-atlas-ember-smoke/runs/native-logits-reference-smoke/logits.npy \
  --out /tmp/sarf-atlas-ember-smoke/runs/logits-compare
```

Do not treat this as a research claim. It is a tiny engineering smoke over the
three prompt fixtures.

Observed local smoke result:

- shape matched: `[3, 151936]`
- status: `pass`
- all top-1 logits matched
- max absolute difference: `0.3790297508239746`
- mean absolute difference: `0.05929932991663615`

These values are small enough for the default smoke harness to pass, but they
are not manuscript evidence and should not be generalized beyond this tiny
local Qwen3 smoke.

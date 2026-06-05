# Activation Reference Checks

## Purpose

Golden logits are necessary because they validate Ember's final output logits
against a trusted implementation for a matched model, tokenizer, prompt, and
quantization path. They are not sufficient for hidden-state probing because a
final-logit match does not prove that every intermediate layer state matches the
reference implementation closely enough for layer-level geometry, CCA/RSA, probe
directions, or causal interventions. Hidden-state probing needs activation-level
validation before treating layer-local findings as numerically trusted.

## What To Compare

Activation checks should compare one hidden-state vector at a time under a fully
matched setup:

- same prompt
- same tokenizer
- same model
- same layer
- same token position
- hidden state vector

The initial target should be the final prompt token, then expand to selected
root, pattern, and prompt-mean positions once single-vector checks are stable.

## Candidate Reference Sources

Hugging Face Transformers is the most practical source for full-precision
architecture sanity checks. It can validate tokenizer alignment, layer indexing,
position selection, and broad implementation behavior, but it may not match GGUF
Q8 numerics.

llama.cpp, or another audited GGUF execution path, is preferable for quantized
checks if it can expose comparable per-layer hidden states for the same GGUF,
tokenizer, prompt, and quantization path. This is the stronger reference for
Ember's Q8 hidden states, but may require instrumentation or a local patch.

## Metrics

For each compared layer, record:

- cosine similarity
- mean absolute difference
- max absolute difference
- optional top changed dimensions by absolute difference

The report should also record model path, tokenizer path, prompt, token IDs,
token position, layer count, hidden size, dtype, quantization path, Ember
activation path, reference activation path, and reference implementation.

## Expected Caveats

- quantization differences can produce real activation drift even when model
  architecture and prompt handling are correct
- layer norm placement, epsilon, and accumulation dtype can change hidden states
- RoPE scaling, base, position indexing, and long-context metadata must match
- tokenizer mismatch can invalidate the whole comparison
- BOS/EOS handling and chat-template insertion can shift token positions
- layer numbering must be explicit: embeddings, block outputs, and final norm
  are different comparison points

## Proposed Ember CLI Shape

Possible additions:

```bash
target/release/ember \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --dump-hidden-states data/reference/llama32_1b_ember_hidden.npy \
  --dump-hidden-metadata data/reference/llama32_1b_ember_hidden_metadata.json
```

For narrower extraction:

```bash
target/release/ember \
  --arch llama \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --dump-layer-hidden 12 \
  --dump-hidden-position last \
  --dump-hidden-states data/reference/llama32_1b_l12_last.npy
```

The exact flag names can change, but the output should make layer index,
position selection, token IDs, and comparison point unambiguous.

## Proposed Python Comparison Script

Proposed script:

```bash
python probes/check_reference_activations.py \
  --ember data/reference/llama32_1b_ember_hidden.npy \
  --reference data/reference/llama32_1b_reference_hidden.npy \
  --label llama32_1b_last_token \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --position last \
  --output-json data/reference/llama32_1b_activation_report.json \
  --output-md data/reference/llama32_1b_activation_report.md
```

The script should fail on invalid JSON/NPY shapes, report missing metadata
explicitly, and avoid declaring pass/fail unless thresholds are configured and
the reference source is appropriate for the claimed validation level.

## Minimum Viable Milestone

Start with:

- one model
- one prompt
- last token
- all layers
- JSON summary
- Markdown summary

The milestone is complete when the report records per-layer cosine similarity,
mean absolute difference, max absolute difference, paths, prompt/token metadata,
reference source, and caveats. This validates an activation comparison workflow;
it does not by itself validate every prompt template, pooling position, model
family, or quantization path.

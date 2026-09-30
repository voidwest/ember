# Ember experiments (v0.5)

The v0.5 experiment workflow turns exact token selection, semantic
hidden-state capture, activation intervention, execution provenance, and
offline verification into reproducible experiment bundles that can be
run without writing Rust.

## Five-minute quick start

Prerequisites: a release build (`cargo build --release`) and a supported
GGUF model with its `tokenizer.json`. The reference example is pinned to
Llama-3.2-1B-Instruct-Q8_0 (see `examples/experiments/README.md` for the
expected file and checksum).

```bash
# 1. validate the specification (no inference)
ember experiment validate examples/experiments/morphology-layerwise-capture.toml

# 2. run it: captures prompt-final and target-final-subtoken across all
#    16 layers, writes runs/morphology-baseline
ember experiment run examples/experiments/morphology-layerwise-capture.toml

# 3. inspect the bundle (captures, identity, hashes)
ember experiment inspect runs/morphology-baseline

# 4. verify it fully offline
ember experiment verify runs/morphology-baseline

# 5. run the intervention (zeroes layer 8) and compare
ember experiment run examples/experiments/morphology-intervention.toml
ember experiment compare runs/morphology-baseline runs/morphology-intervention

# 6. run the exact-restoration leg and confirm the baseline is reproduced
ember experiment run examples/experiments/morphology-restoration.toml
ember experiment compare runs/morphology-baseline runs/morphology-restoration

# 7. reproduce the baseline bundle from the model file
ember experiment reproduce runs/morphology-baseline --model Llama-3.2-1B-Instruct-Q8_0.gguf
```

## CLI surface

```text
ember experiment validate <spec.toml> [--json]
ember experiment run <spec.toml> [--execution reference|planned|planned-fused]
                                [--threads <n>] [--output <dir>] [--retain-incomplete]
                                [--json]
ember experiment inspect <bundle> [--json]
ember experiment verify <bundle> [--model <model.gguf>] [--tokenizer <tokenizer.json>]
                                 [--json]
ember experiment compare <bundle-a> <bundle-b> [--json]
ember experiment reproduce <bundle> --model <model.gguf> [--output <dir>] [--json]
ember experiment tokenize --model <model.gguf> --arch <arch> --tokenizer <tokenizer.json>
                          --text "<text>" [--match-span "<span>"] [--json]
ember experiment lens <bundle> --model <model.gguf> [--tokenizer <tokenizer.json>]
                            [--top-k <n>] [--json] [--out <lens.json>]
```

CLI output states the experiment schema, model and tokenizer identity,
execution mode, plan hash, capture/intervention counts, output directory,
semantic hash, and the verification result. Per-tensor detail lives in
the bundle and is exposed through `inspect`.

## Logit lens

`ember experiment lens` reads the residual-stream captures of a bundle and
asks, for every captured row, which next token the model would predict if
the network stopped at that depth. Each row goes through the model's own
final RMS norm and LM head (untied `output.weight` or tied embeddings); the
lens calls the same `Llama::final_norm` / `Llama::lm_head` functions the
forward pass uses, so it is not a re-implementation.

```bash
ember experiment run examples/experiments/morphology-layerwise-capture.toml
ember experiment lens runs/morphology-baseline \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --top-k 5
ember experiment lens runs/morphology-baseline \
  --model Llama-3.2-1B-Instruct-Q8_0.gguf --json --out /tmp/lens.json
```

Fail-closed rules (the same as `reproduce`):

- The bundle is fully verified first; the anchors `--expect-semantic-hash`
  and `--trusted-key` (with `--expect-evidence`, or the sibling
  `<bundle>.evidence.json`) are accepted.
- The model must hash to the bundle's recorded model SHA-256, and its
  layer count, width and vocabulary must match the manifest.
- The tokenizer (`--tokenizer`, else the path the bundle's bound spec
  names) must hash to the recorded tokenizer SHA-256; it is hashed and
  parsed from one read.
- Nothing is written into the bundle. `--out` must point outside it.

Per captured `(capture, input, position)` the report gives the token at the
position and the token that actually followed it (the next prompt token, or
the model's generated token), then for every captured layer: the top-k
tokens with probabilities, the rank (1-based, ties to the lower id as in
greedy argmax) and probability of the actual next token, the entropy (nats),
and `KL(final || layer)` (nats) against the final-depth row of the same
position.

Sites: only residual-stream sites are projected. `residual-post-mlp` at
layer `L` is the stream after `L + 1` blocks (reported as `depth`);
`residual-pre-attention` at layer `L` is the stream after `L` blocks.
`attention-output` and `mlp-output` (projections before their residual
add), `final-norm-output` and `logits` are listed as skipped with the
reason. KL needs the final-depth row, i.e. a `residual-post-mlp` capture of
the last layer.

Final-layer equality: the last layer's lens *is* the model's final-logits
computation. A unit test (`v05::lens::tests::final_layer_lens_equals_model_logits`)
runs a synthetic Llama (untied and tied heads) through the experiment
execution path and checks that the captured last-layer row, projected by
the lens, is bit-identical to the logits the model produced, for prefill
and for a decode step. The report's `final_layer_check` counts final-depth
rows whose next token was generated and how many have that token as lens
top-1 (all of them for a greedy run).

Limits: rows at prompt positions reproduce the prefill route exactly. Rows
at generated positions were produced by the decode route, which for Q8_0
models uses fused single-token kernels (and, in planned modes, the plan
interpreter); these agree with the lens projection within kernel
tolerance, not bit-for-bit. F16 captures are a rounded stream. The lens is
a read-out of the as-run stream: with interventions in the bundle it
projects the intervened state.

## Workflow semantics

1. **Resolve**: the TOML spec is parsed strictly (unknown fields and
   unknown schema majors fail); defaults are applied and recorded.
2. **Load and validate**: model and tokenizer SHA-256 are verified
   against the spec when provided; mismatches fail closed.
3. **Tokenize and align**: token selection is exact, byte-based, and
   fail-closed (see `docs/token-selection.md`).
4. **Execute**: every input is generated through the existing v0.4
   execution machinery; captures and interventions fire at the six
   public semantic hook sites (see `docs/v05-research-contract.md`).
5. **Bundle**: a deterministic `ember.bundle.v1` is staged and
   atomically renamed into place only after all payloads, checksums, and
   the manifest are complete (see `docs/bundle-schema-v1.md`).
6. **Self-verify**: the run command verifies the bundle it just wrote.

## Determinism and identity

Two equivalent runs on the same environment produce identical semantic
manifests and identical semantic hashes. The semantic hash covers every
deterministic file (specs, token selection records, generated tokens,
payload checksums, plan). Timestamps, hostnames, paths, and timing live
in `runtime.json`, which is verifiable but excluded from the identity
(see `docs/reproducibility.md`).

## Performance isolation

The experiment machinery is inert unless the `experiment` subcommand
runs: no spec is parsed, no bundle metadata allocated, and no hooks fire
during ordinary `ember run` inference. The reference example runs in a
few seconds on the pinned Q8_0 model.

## References

- `docs/experiment-schema-v1.md`: the specification language.
- `docs/bundle-schema-v1.md`: the bundle layout and identity rules.
- `docs/token-selection.md`: token selection and Arabic alignment.
- `docs/interventions.md`: interventions and restoration.
- `docs/reproducibility.md`: verification, comparison, reproduction.
- `docs/v05-research-contract.md`: the frozen research contract and
  gates.

## Related v0.1/v0.2 interfaces

The earlier experiment interfaces remain available: `--activation-stats`,
`--zero-layer-output`, `--capture-activations`, `--activation-patch`, and
`compare-artifacts` (see `docs/activation-artifacts.md` and
`docs/activation-patching.md`). The v0.5 workflow supersedes them for new
research; the old interfaces keep their v0.2 semantics unchanged.

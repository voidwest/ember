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
```

CLI output states the experiment schema, model and tokenizer identity,
execution mode, plan hash, capture/intervention counts, output directory,
semantic hash, and the verification result. Per-tensor detail lives in
the bundle and is exposed through `inspect`.

## Workflow semantics

1. **Resolve** — the TOML spec is parsed strictly (unknown fields and
   unknown schema majors fail); defaults are applied and recorded.
2. **Load and validate** — model and tokenizer SHA-256 are verified
   against the spec when provided; mismatches fail closed.
3. **Tokenize and align** — token selection is exact, byte-based, and
   fail-closed (see `docs/token-selection.md`).
4. **Execute** — every input is generated through the existing v0.4
   execution machinery; captures and interventions fire at the six
   public semantic hook sites (see `docs/v05-research-contract.md`).
5. **Bundle** — a deterministic `ember.bundle.v1` is staged and
   atomically renamed into place only after all payloads, checksums, and
   the manifest are complete (see `docs/bundle-schema-v1.md`).
6. **Self-verify** — the run command verifies the bundle it just wrote.

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

- `docs/experiment-schema-v1.md` — the specification language.
- `docs/bundle-schema-v1.md` — the bundle layout and identity rules.
- `docs/token-selection.md` — token selection and Arabic alignment.
- `docs/interventions.md` — interventions and restoration.
- `docs/reproducibility.md` — verification, comparison, reproduction.
- `docs/v05-research-contract.md` — the frozen research contract and
  gates.

## Related v0.1/v0.2 interfaces

The earlier experiment interfaces remain available: `--activation-stats`,
`--zero-layer-output`, `--capture-activations`, `--activation-patch`, and
`compare-artifacts` (see `docs/activation-artifacts.md` and
`docs/activation-patching.md`). The v0.5 workflow supersedes them for new
research; the old interfaces keep their v0.2 semantics unchanged.

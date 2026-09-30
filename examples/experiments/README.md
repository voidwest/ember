# Reference morphology workflow

A minimal model-internals workflow demonstrating exact token alignment,
representation-location selection, semantic layerwise capture,
intervention, restoration, and provenance. It does **not** claim to
reproduce any paper result.

## Expected model

Pinned to Llama-3.2-1B-Instruct (Q8_0), a documented open GGUF model:

```text
Llama-3.2-1B-Instruct-Q8_0.gguf
sha256 432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3
```

with `tokenizer.json` (the matching Llama-3.2 tokenizer):

```text
sha256 6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b
```

The example files embed these hashes; run them from the repository root
with the model and tokenizer present there. Ember does not download
models automatically. The matching model is available from
[bartowski at revision 067b946](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q8_0.gguf);
`scripts/download_models.sh` now uses that exact revision for the 1B model.
Do not substitute another publisher's file with the same filename.

The matching `tokenizer.json` is tracked in this repository, so a fresh clone
already has it and no gated access is required. To re-fetch or verify it
without Meta's gated `meta-llama` repository, run
`scripts/download_models.sh tokenizer`: it downloads the byte-identical public
copy pinned to `unsloth/Llama-3.2-1B-Instruct` revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c` and checks the SHA-256 above.

## The workflow

1. `morphology-layerwise-capture.toml` — one Arabic prompt containing an
   explicitly marked target word (`كِتَاب`); captures the prompt-final
   and the target's final-subtoken representation at `residual-post-mlp`
   across all 16 layers (one row per layer), writes
   `runs/morphology-baseline`.
2. `morphology-intervention.toml` — the same capture plus a `replace`
   intervention at layer 7's `residual-post-mlp`: replace the prompt-final
   row with the target word's final-subtoken row, writes
   `runs/morphology-intervention`.
3. `morphology-restoration.toml` — the same intervention followed by
   `restore-original` at the same site, writes
   `runs/morphology-restoration`.
4. `morphology-layer-sweep.toml` — the intervention of step 2 at every
   layer (`[sweep] layers = "all"`): one bundle per layer plus a baseline
   and `sweep.json`/`sweep.csv`, written to `runs/morphology-layer-sweep`
   (see `docs/experiments.md#layer-sweeps`).

## Commands

```bash
# validate
ember experiment validate examples/experiments/morphology-layerwise-capture.toml

# baseline
ember experiment run examples/experiments/morphology-layerwise-capture.toml
ember experiment verify runs/morphology-baseline
ember experiment inspect runs/morphology-baseline

# intervention and comparison
ember experiment run examples/experiments/morphology-intervention.toml
ember experiment compare runs/morphology-baseline runs/morphology-intervention

# restoration reproduces the baseline exactly
ember experiment run examples/experiments/morphology-restoration.toml
ember experiment compare runs/morphology-baseline runs/morphology-restoration

# or, instead of steps 1-3 (fresh output directories): one process, and the
# baseline computes the prompt prefix once for both variants
ember experiment run examples/experiments/morphology-layerwise-capture.toml \
  --variant examples/experiments/morphology-intervention.toml \
  --variant examples/experiments/morphology-restoration.toml

# the intervention at every layer, then verify the whole sweep
ember experiment run examples/experiments/morphology-layer-sweep.toml
ember experiment verify runs/morphology-layer-sweep

# reproduction
ember experiment reproduce runs/morphology-baseline --model Llama-3.2-1B-Instruct-Q8_0.gguf

# token alignment diagnostics
ember experiment tokenize --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --arch llama --tokenizer tokenizer.json \
  --text "في الجملة التالية، الكلمة المميزة هي: كِتَاب. اشرح معناها." \
  --match-span "كِتَاب"
```

On the reference machine the baseline reproduces `exact-semantic` and
the restoration leg compares bit-exact (tokens, text, top-1, and every
capture `exact`).

For an automated local acceptance run, use a new output directory:

```sh
.venv/bin/python scripts/validate_morphology_example.py --workdir /path/to/new-example-run
```

This follows the same three specs, deep-verifies all bundles against the
pinned model/tokenizer, requires an observable intervention, checks all 32
restored captures and generated outputs exactly, and reproduces the baseline
with an `exact-semantic` verdict. It preserves commands, JSON reports, and
stderr logs for review. A successful local run does not substitute for the
1.0 clean-machine or external-user acceptance gates.

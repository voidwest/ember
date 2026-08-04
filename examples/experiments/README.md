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
models automatically.

## The workflow

1. `morphology-layerwise-capture.toml` — one Arabic prompt containing an
   explicitly marked target word (`كِتَاب`); captures the prompt-final
   and the target's final-subtoken representation at `residual-post-mlp`
   across all 16 layers (one row per layer), writes
   `runs/morphology-baseline`.
2. `morphology-intervention.toml` — the same capture plus a `zero`
   intervention at layer 8's `attention-output`, writes
   `runs/morphology-intervention`.
3. `morphology-restoration.toml` — the same intervention followed by
   `restore-original` at the same site, writes
   `runs/morphology-restoration`.

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

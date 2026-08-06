#!/usr/bin/env bash
# Ember conference demo — end-to-end reproducible model-internals workflow.
#
# Story: 1) fast interactive generation (Q8_0), 2) the v0.5 reproducible
# experiment workflow (capture -> intervene -> restore -> compare ->
# reproduce) with machine verdicts, 3) the v0.4 planned-decode speedup
# (reference vs planned vs planned-fused on a K-quant model).
#
# Run from the repository root with the model files present:
#   bash scripts/conference_demo.sh
# Pre-build before a live talk to skip the compile step:
#   cargo build --release && SKIP_BUILD=1 bash scripts/conference_demo.sh
#
# Expected models (see examples/experiments/README.md):
#   Llama-3.2-1B-Instruct-Q8_0.gguf + tokenizer.json   (workflow demo)
#   Llama-3.2-1B-Instruct.Q6_K.gguf                    (speedup demo)
set -u

BIN=./target/release/ember
Q8="Llama-3.2-1B-Instruct-Q8_0.gguf"
Q6="Llama-3.2-1B-Instruct.Q6_K.gguf"
TOK="tokenizer.json"
# The example specs hardcode these output directories (see
# examples/experiments/*.toml); they are regenerable demo artifacts.
BASE=runs/morphology-baseline
INTV=runs/morphology-intervention
REST=runs/morphology-restoration
REPRO=runs/morphology-baseline.reproduced

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '  \033[32m%s\033[0m\n' "$*"; }
fail() { printf '  \033[31m%s\033[0m\n' "$*"; exit 1; }
need() { [ -f "$1" ] || fail "missing $1 — place the model files in the repo root first"; }

need "$Q8"; need "$TOK"
# clean regenerable demo artifacts from previous runs
rm -rf "$BASE" "$INTV" "$REST" "$REPRO"

echo "ember conference demo"
echo "  machine: $(uname -m), $(nproc) threads"
echo "  ember:   $($BIN --version 2>/dev/null | head -1 || echo '(build first)')"

# ---------------------------------------------------------------------------
say "0. build (release)"
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  ok "skipping build (SKIP_BUILD=1)"
else
  cargo build --release -q || fail "build failed"
fi

# ---------------------------------------------------------------------------
say "1. fast generation — 4 prompts x 20 tokens on Q8_0 (~10s)"
# ---------------------------------------------------------------------------
time "$BIN" --demo --arch llama --model "$Q8" --tokenizer "$TOK" --max-tokens 20

# ---------------------------------------------------------------------------
say "2. the v0.5 reproducible experiment workflow"
# ---------------------------------------------------------------------------
say "  2a. validate the spec (declarative, schema-checked)"
"$BIN" experiment validate examples/experiments/morphology-layerwise-capture.toml

say "  2b. baseline run — capture all 16 layers (prompt-final + target word)"
time "$BIN" experiment run examples/experiments/morphology-layerwise-capture.toml

say "  2c. intervention — replace layer 7 with the target-word representation"
time "$BIN" experiment run examples/experiments/morphology-intervention.toml

say "  2d. restoration — replace then restore the snapshot (same run)"
time "$BIN" experiment run examples/experiments/morphology-restoration.toml

# ---------------------------------------------------------------------------
say "3. the causal story"
# ---------------------------------------------------------------------------
say "  3a. intervention vs baseline — layers 8-15 diverge (cascade), tokens diverge at step 1"
"$BIN" experiment compare "$BASE" "$INTV" 2>&1 | grep -E "^(comparing|  (semantic|plan hash|prompts|tokenization))|outputs:|divergence|^  prompt-final @ residual-post-mlp layer (8|9|1[0-5]):|^interventions:|replace-layer7"

say "  3b. restoration vs baseline — bit-exact (divergence none, all layers exact)"
"$BIN" experiment compare "$BASE" "$REST" 2>&1 | grep -E "^(comparing|  (semantic|plan hash|prompts|tokenization))|outputs:|divergence|restore-layer7"

# ---------------------------------------------------------------------------
say "4. reproducibility — re-run the baseline and classify"
# ---------------------------------------------------------------------------
rm -rf "$REPRO"
time "$BIN" experiment reproduce --model "$Q8" "$BASE" 2>&1 | grep -E "verdict|tokens|captures|semantic hash"
"$BIN" experiment verify "$BASE" 2>&1 | tail -1

# ---------------------------------------------------------------------------
say "5. v0.4 speedup — planned decode vs the reference path on Q6_K"
# ---------------------------------------------------------------------------
if [ -f "$Q6" ]; then
  for mode in reference planned planned-fused; do
    tps=$("$BIN" bench-decode --arch llama --model "$Q6" --execution "$mode" --tokens 16 --repetitions 2 \
      | grep -o '"median_tokens_per_second": [0-9.]*' | grep -o '[0-9.]*$')
    printf '  %-14s %6.2f tps\n' "$mode" "$tps"
  done
else
  echo "  (skip: $Q6 not present)"
fi

say "demo complete"

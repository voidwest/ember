#!/usr/bin/env bash
# Ember v0.3 causal workflow on the compressed K-quant path:
# capture -> compare -> patch -> frozen verdict.
#
# Same semantics as scripts/research_example_capture_patch.sh (v0.2) but
# runs the compressed-resident path (--k-strategy, default auto):
#   A. normal run with capture            (baseline activations + logits)
#   B. zero-layer-output intervention     (same capture; values diverge)
#   C. patched run                        (A's activation patched back in)
#
# Frozen criterion: run C's captured logits must be bit-identical
# (sha256-equal) to run A's, and A vs B must differ. This validates that
# capture, compare, and patch semantics are preserved on the compressed
# path.
#
# Usage:
#   scripts/validate_k_causal.sh [--model M.gguf] [--arch ARCH]
#     [--tokenizer T.json] [--prompt "P"] [--layer N] [--stage after-mlp]
#     [--tokens N] [--k-strategy STRATEGY] [--workdir DIR]
#
# Requires target/release/ember (cargo build --release).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBER="${EMBER:-$REPO_ROOT/target/release/ember}"
MODEL="${MODEL:-$REPO_ROOT/Llama-3.2-1B-Instruct.Q4_K_M.gguf}"
ARCH="${ARCH:-auto}"
TOKENIZER="${TOKENIZER:-$REPO_ROOT/tokenizer.json}"
PROMPT="${PROMPT:-The capital of France is}"
LAYER="${LAYER:-0}"
STAGE="${STAGE:-after-mlp}"
TOKENS="${TOKENS:-4}"
K_STRATEGY="${K_STRATEGY:-auto}"
WORKDIR="${WORKDIR:-/tmp/ember-v03-k-causal}"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--arch|--tokenizer|--prompt|--layer|--stage|--tokens|--k-strategy|--workdir)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 1; }
      case "$1" in
        --model) MODEL="$2" ;;
        --arch) ARCH="$2" ;;
        --tokenizer) TOKENIZER="$2" ;;
        --prompt) PROMPT="$2" ;;
        --layer) LAYER="$2" ;;
        --stage) STAGE="$2" ;;
        --tokens) TOKENS="$2" ;;
        --k-strategy) K_STRATEGY="$2" ;;
        --workdir) WORKDIR="$2" ;;
      esac
      shift 2
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$STAGE" in
  after-attention) ZLO_STAGE=attention ;;
  after-mlp)      ZLO_STAGE=mlp ;;
  after-layer)    ZLO_STAGE=layer ;;
  *) echo "unsupported stage '$STAGE' (use after-attention, after-mlp, or after-layer)" >&2; exit 1 ;;
esac
case "$K_STRATEGY" in
  eager-f32|scalar|x86|auto) ;;
  *) echo "unsupported --k-strategy '$K_STRATEGY'" >&2; exit 1 ;;
esac

[[ -x "$EMBER" ]] || { echo "Ember executable not found: $EMBER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER" ]] || { echo "tokenizer not found: $TOKENIZER" >&2; exit 1; }
[[ "$LAYER" =~ ^[0-9]+$ ]] || { echo "--layer must be a non-negative integer" >&2; exit 1; }
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "--tokens must be a positive integer" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$WORKDIR"
for name in a b c; do
  [[ ! -e "$WORKDIR/run-$name" ]] || {
    echo "refusing to mix with an existing run directory: $WORKDIR/run-$name" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$WORKDIR" "$LAYER" "$STAGE" <<'PYEOF'
import json
import os
import sys
import tempfile
from pathlib import Path

workdir, layer, stage = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for name in "abc":
    content = (
        "schema_version = 1\n"
        f"output_dir = {json.dumps(str(workdir / f'run-{name}'))}\n"
        f"layers = [{layer}]\n"
        f"stages = [{json.dumps(stage)}, \"after-logits\"]\n"
        'phase = "both"\n'
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.toml.", dir=workdir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, workdir / f"{name}.toml")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
PYEOF

run_ember() {
  "$EMBER" --arch "$ARCH" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --max-tokens "$TOKENS" --temperature 0 \
    --k-strategy "$K_STRATEGY" "$@"
}

echo "== run A: baseline with capture (k-strategy=$K_STRATEGY) =="
run_ember --capture-activations "$WORKDIR/a.toml"

echo "== run B: zero-layer-output $LAYER:$ZLO_STAGE with capture =="
run_ember --zero-layer-output "$LAYER:$ZLO_STAGE" --capture-activations "$WORKDIR/b.toml"

echo "== compare A vs B (expect Differs) =="
"$EMBER" compare-artifacts --left "$WORKDIR/run-a/manifest.json" \
  --right "$WORKDIR/run-b/manifest.json" --json --output "$WORKDIR/ab.json" >/dev/null

echo "== run C: patch A's $LAYER:$STAGE activation back in (prefill + every decode position) =="
TARGETS=$("$PYTHON_BIN" - "$WORKDIR/run-a/manifest.json" "$TOKENS" "$LAYER" "$STAGE" <<'PYEOF'
import json, sys
manifest_path, tokens, layer, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def reject(value):
    raise ValueError(f"non-standard JSON constant {value!r}")
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle, parse_constant=reject)
prompt_len = len(manifest["run"]["input_token_ids"])
positions = range(prompt_len, prompt_len + tokens - 1)
targets = [f"{layer}:{stage}:prefill"]
targets += [f"{layer}:{stage}:decode:{p}" for p in positions]
print("\n".join(targets))
PYEOF
)
mapfile -t TARGET_VALUES <<< "$TARGETS"
PATCH_ARGS=(--activation-patch "$WORKDIR/run-a/manifest.json")
for target in "${TARGET_VALUES[@]}"; do
  [[ -n "$target" ]] && PATCH_ARGS+=(--patch-target "$target")
done
run_ember "${PATCH_ARGS[@]}" --capture-activations "$WORKDIR/c.toml"

echo "== compare A vs C (expect Identical: frozen restoration criterion) =="
"$EMBER" compare-artifacts --left "$WORKDIR/run-a/manifest.json" \
  --right "$WORKDIR/run-c/manifest.json" --json --output "$WORKDIR/ac.json" >/dev/null

echo "== summary =="
"$PYTHON_BIN" - "$WORKDIR" <<'PYEOF'
import json, sys
workdir = sys.argv[1]

def reject(value):
    raise ValueError(f"non-standard JSON constant {value!r}")

def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject)

def manifest(run):
    return load(f"{workdir}/run-{run}/manifest.json")

def text(ids):
    return " ".join(str(t) for t in ids)

a, b, c = manifest("a"), manifest("b"), manifest("c")
ab = load(f"{workdir}/ab.json")
ac = load(f"{workdir}/ac.json")

print(f"model:        {a['model']['family']} {a['model']['architecture']} "
      f"({a['model']['identifier']})")
print(f"k-strategy:   {a['run'].get('k_strategy')} | "
      f"kernels: {sorted({t['kernel'] for t in a['execution']['tensors']})}")
print(f"prompt hash:  {a['run']['prompt_hash']}")
print(f"run A ids:    {text(a['run']['generated_token_ids'])}")
print(f"run B ids:    {text(b['run']['generated_token_ids'])}")
print(f"run C ids:    {text(c['run']['generated_token_ids'])}")
print(f"A vs B:       status={ab['status']} identical={ab['identical_record_count']}/"
      f"{ab['aligned_record_count']} differing={ab['differing_record_count']}")
print(f"A vs C:       status={ac['status']} identical={ac['identical_record_count']}/"
      f"{ac['aligned_record_count']} differing={ac['differing_record_count']}")
print(f"intervention: {b['experiment']['name']} {b['experiment']['arguments']}")
print(f"patch:        {c['experiment']['name']} {c['experiment']['arguments']}")
print(f"artifacts:    {workdir}/run-{{a,b,c}}/manifest.json")

if ac["status"] not in ("identical", "tensor-identical") or ac["identical_record_count"] != ac["aligned_record_count"] or ac["aligned_record_count"] == 0:
    print("FAIL: frozen restoration criterion not met (A vs C must restore every captured tensor bit-exactly)", file=sys.stderr)
    sys.exit(1)
if ab["status"] != "differs":
    print("FAIL: intervention had no observable effect (A vs B must differ)", file=sys.stderr)
    sys.exit(1)
print("PASS: frozen restoration criterion met on the compressed path (every captured tensor bit-identical).")
PYEOF

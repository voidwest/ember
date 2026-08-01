#!/usr/bin/env bash
# Ember v0.2 research workflow: capture -> intervene -> compare -> patch -> restore.
#
# Runs the complete activation-capture/patch cycle on one model and prompt:
#   A. normal run with capture            (baseline activations + logits)
#   B. zero-layer-output intervention     (same capture; values diverge)
#   C. patched run                        (A's activation patched back in)
#
# The frozen restoration criterion is enforced: run C's captured logits must
# be bit-identical (sha256-equal) to run A's. Generated text equality alone
# is never sufficient evidence — this script checks the artifacts.
#
# Usage:
#   scripts/research_example_capture_patch.sh [--model M.gguf] [--arch ARCH]
#     [--tokenizer T.json] [--prompt "P"] [--layer N] [--stage after-mlp]
#     [--tokens N] [--workdir DIR]
#
# Defaults target the small Qwen3-0.6B model with an after-mlp patch at
# layer 4. Requires target/release/ember (cargo build --release).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBER="${EMBER:-$SCRIPT_DIR/target/release/ember}"
MODEL="${MODEL:-Qwen3-0.6B-Q8_0.gguf}"
ARCH="${ARCH:-qwen3}"
TOKENIZER="${TOKENIZER:-tokenizer-qwen3.json}"
PROMPT="${PROMPT:-The capital of France is}"
LAYER="${LAYER:-4}"
STAGE="${STAGE:-after-mlp}"
TOKENS="${TOKENS:-4}"
WORKDIR="${WORKDIR:-/tmp/ember-v02-example}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# zero-layer-output stage name for the intervention run
case "$STAGE" in
  after-attention) ZLO_STAGE=attention ;;
  after-mlp)      ZLO_STAGE=mlp ;;
  after-layer)    ZLO_STAGE=layer ;;
  *) echo "unsupported stage '$STAGE' (use after-attention, after-mlp, or after-layer)" >&2; exit 1 ;;
esac

mkdir -p "$WORKDIR"
for name in a b c; do
  cat > "$WORKDIR/$name.toml" <<EOF
schema_version = 1
output_dir = "$WORKDIR/run-$name"
layers = [$LAYER]
stages = ["$STAGE", "after-logits"]
phase = "both"
EOF
done

run_ember() {
  "$EMBER" --arch "$ARCH" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --max-tokens "$TOKENS" --temperature 0 "$@"
}

echo "== run A: baseline with capture =="
run_ember --capture-activations "$WORKDIR/a.toml" 2>/dev/null

echo "== run B: zero-layer-output $LAYER:$ZLO_STAGE with capture =="
run_ember --zero-layer-output "$LAYER:$ZLO_STAGE" --capture-activations "$WORKDIR/b.toml" 2>/dev/null

echo "== compare A vs B (expect Differs) =="
"$EMBER" compare-artifacts --left "$WORKDIR/run-a/manifest.json" \
  --right "$WORKDIR/run-b/manifest.json" --json --output "$WORKDIR/ab.json" >/dev/null

echo "== run C: patch A's $LAYER:$STAGE activation back in (prefill + every decode position) =="
TARGETS=$(python3 - "$WORKDIR/run-a/manifest.json" "$TOKENS" "$LAYER" "$STAGE" <<'PYEOF'
import json, sys
manifest_path, tokens, layer, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
manifest = json.load(open(manifest_path))
prompt_len = len(manifest["run"]["input_token_ids"])
# decode runs positions prompt_len .. prompt_len+tokens-2 (final eval is skipped)
positions = range(prompt_len, prompt_len + tokens - 1)
targets = [f"{layer}:{stage}:prefill"]
targets += [f"{layer}:{stage}:decode:{p}" for p in positions]
print(" ".join(f"--patch-target {t}" for t in targets))
PYEOF
)
# shellcheck disable=SC2086
run_ember --activation-patch "$WORKDIR/run-a/manifest.json" $TARGETS \
  --capture-activations "$WORKDIR/c.toml" 2>/dev/null

echo "== compare A vs C (expect Identical: frozen restoration criterion) =="
"$EMBER" compare-artifacts --left "$WORKDIR/run-a/manifest.json" \
  --right "$WORKDIR/run-c/manifest.json" --json --output "$WORKDIR/ac.json" >/dev/null

echo "== summary =="
python3 - "$WORKDIR" <<'PYEOF'
import json, sys
workdir = sys.argv[1]

def manifest(run):
    return json.load(open(f"{workdir}/run-{run}/manifest.json"))

def text(ids, tokenizer_file):
    # best-effort rendering via the ember tokenizer is not available here;
    # report token ids instead (deterministic and sufficient for the check)
    return " ".join(str(t) for t in ids)

a, b, c = manifest("a"), manifest("b"), manifest("c")
ab = json.load(open(f"{workdir}/ab.json"))
ac = json.load(open(f"{workdir}/ac.json"))

print(f"model:        {a['model']['family']} {a['model']['architecture']} "
      f"({a['model']['identifier']})")
print(f"prompt hash:  {a['run']['prompt_hash']}")
print(f"run A ids:    {text(a['run']['generated_token_ids'], '')}")
print(f"run B ids:    {text(b['run']['generated_token_ids'], '')}")
print(f"run C ids:    {text(c['run']['generated_token_ids'], '')}")
print(f"A vs B:       status={ab['status']} identical={ab['identical_record_count']}/"
      f"{ab['aligned_record_count']} differing={ab['differing_record_count']}")
print(f"A vs C:       status={ac['status']} identical={ac['identical_record_count']}/"
      f"{ac['aligned_record_count']} differing={ac['differing_record_count']}")
print(f"intervention: {b['experiment']['name']} {b['experiment']['arguments']}")
print(f"patch:        {c['experiment']['name']} {c['experiment']['arguments']}")
print(f"artifacts:    {workdir}/run-{{a,b,c}}/manifest.json")

if ac["status"] != "identical":
    print("FAIL: frozen restoration criterion not met (A vs C must be identical)", file=sys.stderr)
    sys.exit(1)
if ab["status"] != "differs":
    print("WARN: intervention had no observable effect (A vs B identical)", file=sys.stderr)
print("PASS: frozen restoration criterion met (logits bit-identical).")
PYEOF

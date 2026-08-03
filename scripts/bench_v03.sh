#!/usr/bin/env bash
# v0.3 benchmark matrix: Ember arms vs pinned llama.cpp on the ladder.
#
# Arms per rung (Q8_0 baseline, Q6_K, Q4_K_M x both families):
#   ember bench-decode --k-strategy eager-f32 | scalar | x86 (auto)
#   ember bench-lifecycle --lifecycle control (residency, compressed path)
#   llama-bench (pinned b9999) + /usr/bin/time peak RSS
#
# Outputs: one JSON per run under $OUT, plus bench-summary.json.
#
# Usage: scripts/bench_v03.sh [--out DIR] [--tokens N] [--threads N]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBER="${EMBER:-$REPO_ROOT/target/release/ember}"
LADDER="${LADDER:-$REPO_ROOT/models/v03-ladder}"
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/.cache/ember/llama.cpp/build/bin/llama-bench}"
OUT="${OUT:-$REPO_ROOT/artifacts/benchmark-v03}"
TOKENS="${TOKENS:-64}"
THREADS="${THREADS:-$(nproc)}"
TIME_BIN="$(command -v /usr/bin/time || true)"

[[ -x "$EMBER" ]] || { echo "Ember executable not found: $EMBER" >&2; exit 1; }
[[ -x "$LLAMA_BENCH" ]] || { echo "llama-bench not found at $LLAMA_BENCH" >&2; exit 1; }
[[ -z "$TIME_BIN" ]] && { echo "/usr/bin/time is required for peak RSS" >&2; exit 1; }
mkdir -p "$OUT"

LLAMA_CPP_COMMIT="$(cat "$(dirname "$(dirname "$LLAMA_BENCH")")/COMMIT" 2>/dev/null || echo unknown)"

summary_file="$OUT/bench-summary.json"
echo "[]" > "$summary_file"

append_summary() {
  "$REPO_ROOT/.venv/bin/python" - "$summary_file" "$1" <<'PYEOF'
import json, sys
path, entry = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest.append(json.loads(entry))
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
PYEOF
}

run_ember_decode() {
  local family="$1" rung="$2" arch="$3" strategy="$4"
  local model="$LADDER/$family-$rung.gguf"
  local target="$OUT/$family-$rung.ember-$strategy.json"
  echo "== ember bench-decode: $family-$rung ($strategy) =="
  "$EMBER" --k-strategy "$strategy" bench-decode --model "$model" --arch "$arch" \
    --tokens "$TOKENS" --warmups 1 --repetitions 3 \
    > "$target"
  append_summary "$(cat "$target")"
}

run_ember_lifecycle() {
  local family="$1" rung="$2" tokenizer="$3"
  local model="$LADDER/$family-$rung.gguf"
  local target="$OUT/$family-$rung.ember-lifecycle.json"
  echo "== ember bench-lifecycle: $family-$rung (control, compressed) =="
  "$EMBER" bench-lifecycle --model "$model" --tokenizer "$tokenizer" \
    --tokens "$TOKENS" --lifecycle control --selection all \
    > "$target"
  append_summary "$(cat "$target")"
}

run_llama_bench() {
  local family="$1" rung="$2"
  local model="$LADDER/$family-$rung.gguf"
  local target="$OUT/$family-$rung.llama-bench.json"
  local rss=""
  echo "== llama-bench: $family-$rung =="
  if [[ -n "$TIME_BIN" ]]; then
    rss="$(/usr/bin/time -v "$LLAMA_BENCH" -m "$model" -p 8 -n 16 -t "$THREADS" -r 3 -o json 2>&1 | tee "$target" | grep "Maximum resident" | awk '{print $NF}')"
  else
    "$LLAMA_BENCH" -m "$model" -p 8 -n 16 -t "$THREADS" -r 3 > "$target"
  fi
  append_summary "$(printf '{"benchmark":"llama-bench","model":"%s","commit":"%s","threads":%s,"peak_rss_kb":%s,"raw":%s}' \
    "$family-$rung" "$LLAMA_CPP_COMMIT" "$THREADS" "${rss:-null}" "$(python3 -c "import json,sys;print(json.dumps(open('$target').read()))")")"
}

for spec in "llama-3.2-1b tokenizer.json llama" "qwen2.5-1.5b tokenizer-qwen2.5.json qwen3"; do
  set -- $spec
  family="$1"; tokenizer="$2"; arch="$3"
  for rung in q8_0 q6_k q4_k_m; do
    run_ember_decode "$family" "$rung" "$arch" eager-f32
    run_ember_decode "$family" "$rung" "$arch" scalar
    run_ember_decode "$family" "$rung" "$arch" x86
    run_ember_lifecycle "$family" "$rung" "$tokenizer"
    run_llama_bench "$family" "$rung"
  done
done

echo "== benchmark matrix complete: $OUT/bench-summary.json =="
"$REPO_ROOT/.venv/bin/python" - "$summary_file" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for entry in manifest:
    if entry.get("benchmark") == "decode":
        print(f"{entry['model']:22s} ember-{entry['k_strategy']:14s} "
              f"{entry['median_tokens_per_second']:8.1f} tps  "
              f"compressed {entry.get('k_compressed_bytes', 0)/1e6:7.1f} MB")
    elif entry.get("benchmark") == "llama-bench":
        print(f"{entry['model']:22s} llama-bench   peak_rss {entry.get('peak_rss_kb')} KB")
PYEOF

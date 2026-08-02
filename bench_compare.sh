#!/usr/bin/env bash
# Repeatable Ember-vs-llama.cpp decode benchmark with strict output parsing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBER="${EMBER:-$SCRIPT_DIR/target/release/ember}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp-export/build/bin/llama-cli}"
MODEL="${MODEL:-$SCRIPT_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf}"
PROMPT="${PROMPT:-The capital of France is}"
REPS="${REPS:-5}"
TOKENS="${TOKENS:-128}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-1}"

for executable in "$EMBER" "$LLAMA_CLI"; do
  [[ -x "$executable" ]] || { echo "executable not found: $executable" >&2; exit 1; }
done
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "REPS must be a positive integer" >&2; exit 1; }
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "TOKENS must be a positive integer" >&2; exit 1; }
[[ "$COOLDOWN_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "COOLDOWN_SECONDS must be a non-negative number" >&2
  exit 1
}

TMPDIR_BENCH="$(mktemp -d "${TMPDIR:-/tmp}/ember-bench-compare.XXXXXX")"
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

run_ember_variant() {
  local label="$1"
  shift
  local output="$TMPDIR_BENCH/${label}.txt"
  : > "$output"
  echo "=== ember $label (${REPS} reps, ${TOKENS} tokens) ==="
  local i
  for ((i = 1; i <= REPS; i++)); do
    "$EMBER" --model "$MODEL" --arch llama --prompt "$PROMPT" \
      --max-tokens "$TOKENS" --temperature 0 --benchmark "$@" 2>&1 \
      | awk -v rep="$i" '/^(prefill|decode):/ { print "rep=" rep, $0; found=1 } END { exit(found ? 0 : 1) }' \
      | tee -a "$output"
    sleep "$COOLDOWN_SECONDS"
  done
  python3 - "$label" "$REPS" "$output" <<'PY'
import re
import statistics
import sys
from pathlib import Path

label, expected, path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
text = path.read_text(encoding="utf-8")
prefill = [float(value) for value in re.findall(r"prefill:.*?->\s*([0-9]+(?:\.[0-9]+)?)\s+tok/s", text)]
decode = [float(value) for value in re.findall(r"decode:.*?->\s*([0-9]+(?:\.[0-9]+)?)\s+(?:eval|tok)/s", text)]
if len(prefill) != expected or len(decode) != expected:
    raise SystemExit(
        f"failed to parse {expected} {label} repetitions: prefill={len(prefill)}, decode={len(decode)}"
    )
print(f"ember {label} median: prefill={statistics.median(prefill):.0f} tok/s  "
      f"decode={statistics.median(decode):.0f} eval/s")
PY
}

run_ember_variant standard
echo
run_ember_variant fast --fast

echo
echo "=== llama.cpp (${REPS} reps, ${TOKENS} tokens) ==="
LLAMA_OUTPUT="$TMPDIR_BENCH/llama.txt"
: > "$LLAMA_OUTPUT"
for ((i = 1; i <= REPS; i++)); do
  printf '/exit\n' | "$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" \
    -n "$TOKENS" --temp 0 -ngl 0 --simple-io --log-disable --no-conversation 2>/dev/null \
    | awk -v rep="$i" '/Generation:/ { print "rep=" rep, $0; found=1 } END { exit(found ? 0 : 1) }' \
    | tee -a "$LLAMA_OUTPUT"
  sleep "$COOLDOWN_SECONDS"
done

python3 - "$REPS" "$LLAMA_OUTPUT" <<'PY'
import re
import statistics
import sys
from pathlib import Path

expected, path = int(sys.argv[1]), Path(sys.argv[2])
text = path.read_text(encoding="utf-8")
values = [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*t/s", text)]
if len(values) != expected:
    raise SystemExit(f"failed to parse {expected} llama.cpp repetitions: decode={len(values)}")
print(f"llama.cpp median: decode={statistics.median(values):.0f} tok/s")
PY

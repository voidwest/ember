#!/usr/bin/env bash
# Quick repeatable ember vs llama.cpp benchmark
#
# Paths resolve relative to this script's location; override any of them
# with environment variables (EMBER, LLAMA_CLI, MODEL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBER="${EMBER:-$SCRIPT_DIR/target/release/ember}"
# llama.cpp export lives outside this repo; set LLAMA_CLI to your build
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp-export/build/bin/llama-cli}"
MODEL="${MODEL:-$SCRIPT_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf}"
PROMPT="The capital of France is"
REPS=5
TOKENS=128

echo "=== ember standard (${REPS} reps, ${TOKENS} tokens) ==="
for i in $(seq 1 $REPS); do
  $EMBER --model "$MODEL" --arch llama --prompt "$PROMPT" \
    --max-tokens $TOKENS --temperature 0 --benchmark 2>&1 \
    | grep -E "^(prefill|decode):" | while read -r line; do
      echo "rep=$i $line"
    done
  sleep 1
done | awk '
/prefill:/ { pp[++n]=$NF; gsub(/tok\/s/,"",pp[n]); pp_sum+=pp[n] }
/decode:/  { tg[++m]=$NF; gsub(/tok\/s/,"",tg[m]); tg_sum+=tg[m] }
END {
  asort(pp); asort(tg)
  mid_pp=int((n+1)/2); mid_tg=int((m+1)/2)
  printf "ember std median: prefill=%.0f tok/s  decode=%.0f tok/s\n", pp[mid_pp], tg[mid_tg]
}'

echo ""
echo "=== ember fast (${REPS} reps, ${TOKENS} tokens) ==="
for i in $(seq 1 $REPS); do
  $EMBER --model "$MODEL" --arch llama --prompt "$PROMPT" \
    --max-tokens $TOKENS --temperature 0 --benchmark --fast 2>&1 \
    | grep -E "^(prefill|decode):" | while read -r line; do
      echo "rep=$i $line"
    done
  sleep 1
done | awk '
/prefill:/ { pp[++n]=$NF; gsub(/tok\/s/,"",pp[n]); pp_sum+=pp[n] }
/decode:/  { tg[++m]=$NF; gsub(/tok\/s/,"",tg[m]); tg_sum+=tg[m] }
END {
  asort(pp); asort(tg)
  mid_pp=int((n+1)/2); mid_tg=int((m+1)/2)
  printf "ember fast median: prefill=%.0f tok/s  decode=%.0f tok/s\n", pp[mid_pp], tg[mid_tg]
}'

echo ""
echo "=== llama.cpp (${REPS} reps, ${TOKENS} tokens) ==="
for i in $(seq 1 $REPS); do
  echo "/exit" | $LLAMA_CLI -m "$MODEL" -p "$PROMPT" \
    -n $TOKENS --temp 0 -ngl 0 --simple-io --log-disable --no-conversation 2>/dev/null \
    | grep -E "Generation:" | while read -r line; do
      echo "rep=$i $line"
    done
  sleep 1
done | awk '
/Generation:/ {
  n=split($0,a," ")
  for(i=1;i<=n;i++) if(a[i]~/t\/s/) {
    val=a[i]; gsub(/[^0-9.]/,"",val)
    tg[++m]=val; tg_sum+=val
  }
}
END {
  asort(tg)
  mid=int((m+1)/2)
  printf "llama.cpp median: decode=%.0f tok/s\n", tg[mid]
}'

echo ""
echo "=== SUMMARY ==="

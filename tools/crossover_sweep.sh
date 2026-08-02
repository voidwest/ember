#!/usr/bin/env bash
# Real-model Q8 gate/up crossover sweep. Results are staged, schema-checked,
# then atomically installed so an interrupted benchmark cannot replace a good run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${1:-}}"
OUTDIR="${OUTDIR:-${2:-$ROOT/artifacts/crossover_sweep}}"
ROWS="${ROWS:-1,2,4,8,16,32,64,128}"
THREADS="${THREADS:-1,2,4,8}"
CACHE_STATES="${CACHE_STATES:-hot,cold}"
WARMUPS="${WARMUPS:-3}"
SAMPLES="${SAMPLES:-9}"
CACHE_MIB="${CACHE_MIB:-64}"

[[ -n "$MODEL" ]] || {
  echo "usage: tools/crossover_sweep.sh MODEL.gguf [OUTDIR] (or set MODEL)" >&2
  exit 2
}
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
for value in "$WARMUPS" "$SAMPLES" "$CACHE_MIB"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "warmups/samples/cache size must be integers" >&2; exit 1; }
done
((SAMPLES > 0)) || { echo "SAMPLES must be greater than zero" >&2; exit 1; }
[[ "$CACHE_STATES" =~ ^(hot|cold)(,(hot|cold))*$ ]] || {
  echo "CACHE_STATES must be a comma-separated list containing hot and/or cold" >&2
  exit 1
}

mkdir -p "$OUTDIR"
RESULT="$OUTDIR/q8_real_model.jsonl"
SUMMARY="$OUTDIR/q8_real_model_summary.json"
TEMP_RESULT="$(mktemp "$OUTDIR/.q8-real.XXXXXX.jsonl")"
TEMP_SUMMARY="$(mktemp "$OUTDIR/.q8-summary.XXXXXX.json")"
TEMP_RAW="$(mktemp "$OUTDIR/.q8-raw.XXXXXX.txt")"
trap 'rm -f "$TEMP_RESULT" "$TEMP_SUMMARY" "$TEMP_RAW"' EXIT

echo "=== real-model Q8 crossover sweep ==="
echo "model=$MODEL rows=$ROWS threads=$THREADS cache=$CACHE_STATES"

cd "$ROOT"
cargo bench --bench q8_matmul -- \
  --model "$MODEL" \
  --rows "$ROWS" \
  --threads "$THREADS" \
  --cache "$CACHE_STATES" \
  --warmups "$WARMUPS" \
  --samples "$SAMPLES" \
  --cache-mib "$CACHE_MIB" \
  > "$TEMP_RAW"
python3 - "$TEMP_RAW" "$TEMP_RESULT" <<'PY'
import json
import os
import sys
from pathlib import Path

source, output = Path(sys.argv[1]), Path(sys.argv[2])
records = []
for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
    text = line.strip()
    if not text.startswith("{"):
        continue
    try:
        record = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value!r}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"invalid benchmark JSON on stdout line {line_number}: {error}")
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("benchmark") != "q8_gate_up"
        or record.get("exact_parity") is not True
    ):
        raise SystemExit(f"unexpected benchmark record on stdout line {line_number}")
    records.append(record)
if not records:
    raise SystemExit("benchmark emitted no validated JSON records")
with output.open("w", encoding="utf-8", newline="\n") as handle:
    for record in records:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

# The plotting parser independently verifies samples, medians, speedups,
# workload identity, and the one-thread baseline for every measured size.
IFS=, read -r -a CACHE_VALUES <<< "$CACHE_STATES"
for cache_state in "${CACHE_VALUES[@]}"; do
  python3 scripts/plot_crossover.py "$TEMP_RESULT" \
    --no-plot --cache-state "$cache_state" --summary-json "$TEMP_SUMMARY"
done

mv -f "$TEMP_RESULT" "$RESULT"
mv -f "$TEMP_SUMMARY" "$SUMMARY"
rm -f "$TEMP_RAW"
trap - EXIT
echo "validated results: $RESULT"
echo "validated summary: $SUMMARY"

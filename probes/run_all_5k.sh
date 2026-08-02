#!/usr/bin/env bash
# Run the full 5k analysis suite. By default this excludes morphology targets
# that are printed verbatim in the historical morph_context prompt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

MODELS=(
  "qwen3:data/arabic_morph_real/probe_baseline_qwen3_5k:qwen3_06b_5k"
  "llama:data/arabic_morph_real/probe_baseline_llama32_5k:llama32_1b_5k"
  "qwen25:data/arabic_morph_real/probe_baseline_qwen25_5k:qwen25_15b_5k"
)

TASKS=(pos features.gender features.number)
PROMPT_FLAGS=()
if [[ "${EMBER_LABEL_REVEALED_POSITIVE_CONTROL:-0}" == "1" ]]; then
  TASKS=(root lemma pos abstract_pattern concrete_pattern features.gender features.number)
  PROMPT_FLAGS=(--allow-label-revealed-prompts)
  echo "WARNING: running a label-revealed positive control; outputs must not be reported as morphology inference." >&2
fi

processed=0
for model_info in "${MODELS[@]}"; do
  IFS=: read -r name dir prefix <<< "$model_info"
  acts="$dir/${prefix}_morph_context_last_activations.npy"
  stimuli="$dir/stimuli.json"

  if [[ ! -f "$acts" ]]; then
    echo "SKIP $name: no activations at $acts"
    continue
  fi
  [[ -f "$stimuli" ]] || { echo "missing stimuli for $name: $stimuli" >&2; exit 1; }
  ((processed += 1))

  echo
  echo "============================================"
  echo "  $name — baseline probes"
  echo "============================================"
  "$PYTHON" -u probes/run_baseline_probes.py \
    --activations "$acts" --stimuli "$stimuli" \
    --output-dir "$dir" --tasks "${TASKS[@]}" --seed 42 \
    --require-activation-provenance "${PROMPT_FLAGS[@]}"

  echo
  echo "  $name — matched controls"
  "$PYTHON" -u probes/run_control_analysis.py \
    --activations "$acts" --stimuli "$stimuli" \
    --output-dir "$dir" --tasks "${TASKS[@]}" --seed 42 \
    --require-activation-provenance "${PROMPT_FLAGS[@]}"

  echo
  echo "  $name — heldout probes"
  "$PYTHON" -u probes/run_heldout_probes.py \
    --activations "$acts" --stimuli "$stimuli" \
    --output-dir "$dir" --tasks "${TASKS[@]}" --seed 42 \
    --require-activation-provenance "${PROMPT_FLAGS[@]}"

  echo
  echo "  $name — grouped split sensitivity"
  "$PYTHON" -u probes/run_group_variance.py \
    --activations "$acts" --stimuli "$stimuli" \
    --heldout-results "$dir/heldout_probe_results.json" \
    --output-dir "$dir" --tasks "${TASKS[@]}" --n-configs 20 --seed 42 \
    --require-activation-provenance "${PROMPT_FLAGS[@]}"

  echo
  echo "  $name — token diagnostics"
  "$PYTHON" -u probes/token_diagnostics.py \
    --activations "$acts" --stimuli "$stimuli" \
    --output-dir "$dir" --tasks "${TASKS[@]}" --seed 42 \
    --require-activation-provenance "${PROMPT_FLAGS[@]}"

  echo
  echo "  $name — leakage audit"
  "$PYTHON" -u probes/audit_probe_leakage.py \
    "$stimuli" "$dir/leakage_audit.json"

  echo "  $name DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

((processed > 0)) || { echo "no activation sets were available" >&2; exit 1; }
echo
echo "ALL PROBE ANALYSES COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/bin/bash
# Auto-run after extractions complete — full probe pipeline for 5k dataset
set -euo pipefail
cd /home/west/ember

MODELS=(
  "qwen3:data/arabic_morph_real/probe_baseline_qwen3_5k:qwen3_06b_5k"
  "llama:data/arabic_morph_real/probe_baseline_llama32_5k:llama32_1b_5k"
  "qwen25:data/arabic_morph_real/probe_baseline_qwen25_5k:qwen25_15b_5k"
)

for model_info in "${MODELS[@]}"; do
  IFS=: read -r name dir prefix <<< "$model_info"
  acts="$dir/${prefix}_morph_context_last_activations.npy"
  
  if [ ! -f "$acts" ]; then
    echo "SKIP $name: no activations at $acts"
    continue
  fi
  
  echo ""
  echo "============================================"
  echo "  $name — baseline probes"
  echo "============================================"
  python -u probes/run_baseline_probes.py \
    --activations "$acts" --stimuli "$dir/stimuli.json" \
    --output-dir "$dir" --seed 42
  
  echo ""
  echo "============================================"
  echo "  $name — control analysis"
  echo "============================================"
  python -u probes/run_control_analysis.py \
    --activations "$acts" --stimuli "$dir/stimuli.json" \
    --output-dir "$dir" --seed 42
  
  echo ""
  echo "============================================"
  echo "  $name — heldout probes"
  echo "============================================"
  python -u probes/run_heldout_probes.py \
    --activations "$acts" --stimuli "$dir/stimuli.json" \
    --output-dir "$dir" --seed 42
  
  echo ""
  echo "============================================"
  echo "  $name — group-variance CI"
  echo "============================================"
  python -u probes/run_group_variance.py \
    --activations "$acts" --stimuli "$dir/stimuli.json" \
    --heldout-results "$dir/heldout_probe_results.json" \
    --output-dir "$dir" --n-configs 20 --seed 42
  
  echo ""
  echo "============================================"
  echo "  $name — token diagnostics"
  echo "============================================"
  python -u probes/token_diagnostics.py \
    --activations "$acts" --stimuli "$dir/stimuli.json" \
    --output-dir "$dir" --seed 42
  
  echo ""
  echo "============================================"
  echo "  $name — leakage audit"
  echo "============================================"
  python -u probes/audit_probe_leakage.py \
    "$dir/stimuli.json" "$dir/leakage_audit.json"
  
  echo ""
  echo "  $name DONE at $(date)"
done

echo ""
echo "============================================"
echo "  ALL PROBE ANALYSES COMPLETE"
echo "============================================"
date

#!/usr/bin/env bash
# Download the quantized GGUF models used by Ember's docs and benchmark
# fixtures. Model files are gitignored (see .gitignore); this script is the
# canonical way to fetch them.
#
# Usage:
#   scripts/download_models.sh             # quickstart set (laptop-sized models)
#   scripts/download_models.sh all         # quickstart + larger matrix models
#   MODEL_DIR=/path/to/models scripts/download_models.sh all
#
# Sources were verified against the Hugging Face Hub on 2026-08-03. Entries
# are "URL|LOCAL_FILENAME" pairs. Files that already exist with a non-zero
# size are skipped; interrupted downloads resume via curl -C -.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT}"

# QUICKSTART: models small enough for a laptop (0.6B-3B, Q8_0).
QUICKSTART=(
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B-Q8_0.gguf"
  "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct-Q8_0.gguf"
  "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf|Llama-3.2-3B-Instruct-Q8_0.gguf"
)

# FULL: larger models used by the extended benchmark matrix
# (probes/benchmarks/*.json). The 8B-class Q8_0 files are ~8-9 GB each.
FULL=(
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf|Qwen3-8B-Q8_0.gguf"
)

# Candidate sources not yet verified against the Hub; fixture filenames may
# be local renames of the upstream files:
#   qwen2.5-1.5b-instruct-q8_0.gguf    <- unsloth/Qwen2.5-1.5B-Instruct-GGUF (Qwen2.5-1.5B-Instruct-Q8_0.gguf)
#   meta-llama-3.1-8b-instruct.Q8_0.gguf <- unsloth/Llama-3.1-8B-Instruct-GGUF (Llama-3.1-8B-Instruct-Q8_0.gguf)
#   gemma-4-E2B-it.Q8_0.gguf           <- source repo TBD; see docs/models.md

download() {
  local url="$1" name="$2"
  local target="$MODEL_DIR/$name"
  if [[ -f "$target" && -s "$target" ]]; then
    echo "skip  $name (exists, $(du -h "$target" | cut -f1))"
    return 0
  fi
  echo "fetch $name"
  curl -fL --retry 3 -C - -o "$target" "$url"
  if [[ ! -s "$target" ]]; then
    rm -f "$target"
    echo "error: download of $name produced an empty file" >&2
    return 1
  fi
  echo "  -> $(du -h "$target" | cut -f1)"
}

mode="${1:-quickstart}"
case "$mode" in
  quickstart) entries=("${QUICKSTART[@]}" ) ;;
  all)        entries=("${QUICKSTART[@]}" "${FULL[@]}") ;;
  *)
    echo "usage: $0 [quickstart|all]" >&2
    exit 2
    ;;
esac

mkdir -p "$MODEL_DIR"
for entry in "${entries[@]}"; do
  download "${entry%%|*}" "${entry#*|}"
done
echo "done. models are in $MODEL_DIR (gitignored)."

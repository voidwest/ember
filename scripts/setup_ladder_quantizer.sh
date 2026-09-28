#!/usr/bin/env bash
# Build the frozen ladder quantizer with explicit float contraction disabled.
# The default M1 compiler contracts multiply/add and changes Q4/Q6 bytes.
# Keep this separate from the optimized inference reference build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.cache/ember/llama.cpp}"
PYTHON_BIN="${PYTHON:-python3}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
PIN=47c786924ad1ab7e91da2cdc72fcdb563780c2bd
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "jobs must be a positive integer" >&2; exit 1; }
[[ "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" == "$PIN" ]] || {
  echo "run scripts/setup_llama_cpp.sh to obtain the pinned checkout first" >&2; exit 1;
}
git -C "$LLAMA_CPP_DIR" diff --quiet HEAD -- || { echo "reference source has tracked changes" >&2; exit 1; }
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build-quantize-portable" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-ffp-contract=off \
  -DCMAKE_CXX_FLAGS=-ffp-contract=off -DGGML_NATIVE=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build "$LLAMA_CPP_DIR/build-quantize-portable" --target llama-quantize -j "$JOBS"
"$PYTHON_BIN" "$ROOT/scripts/reference_build.py" record "$LLAMA_CPP_DIR" "$PIN" \
  --build-dir build-quantize-portable --quantizer-only

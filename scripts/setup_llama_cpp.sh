#!/usr/bin/env bash
# Build the pinned llama.cpp CLI tooling (llama-cli + llama-bench) used as
# the external baseline for v0.3 benchmarks and golden-logit validation.
#
# The pin is frozen in the v0.3 contract doc (decision D3). Every
# benchmark manifest records the binary commit.
#
# Usage:
#   scripts/setup_llama_cpp.sh [--commit SHA] [--jobs N]
#
# Env:
#   LLAMA_CPP_DIR   clone + build location (default ~/.cache/ember/llama.cpp)
#
# Outputs: $LLAMA_CPP_DIR/build/bin/llama-cli, llama-bench, and
# llama-quantize (pinned tag b9999), plus $LLAMA_CPP_DIR/COMMIT
# recording the pinned SHA.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.cache/ember/llama.cpp}"
COMMIT="47c786924ad1ab7e91da2cdc72fcdb563780c2bd"
JOBS="${JOBS:-$(nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LLAMA_CPP_DIR"
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  echo "== cloning llama.cpp (shallow, then fetching the pin) =="
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

echo "== checking out pinned commit $COMMIT =="
git -C "$LLAMA_CPP_DIR" fetch origin "$COMMIT"
git -C "$LLAMA_CPP_DIR" checkout -q "$COMMIT"
git -C "$LLAMA_CPP_DIR" submodule update --init --recursive
printf '%s\n' "$COMMIT" > "$LLAMA_CPP_DIR/COMMIT"

# The b9999-era CMake consolidated llama-cli into the unified `llama`
# executable (app/), and tools/ (which holds the cli/server/bench/quantize
# implementation libraries) is gated behind LLAMA_BUILD_TOOLS. The
# include-dir flags work around app/ missing its transitive include path.
echo "== configuring build (release, llama + llama-bench + llama-quantize) =="
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_APP=ON \
  -DCMAKE_CXX_FLAGS="-I$LLAMA_CPP_DIR/common -I$LLAMA_CPP_DIR/include \
                     -I$LLAMA_CPP_DIR/ggml/include \
                     -I$LLAMA_CPP_DIR/build/ggml/include \
                     -I$LLAMA_CPP_DIR/build/common"

echo "== building with $JOBS jobs =="
cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama llama-bench llama-quantize -j "$JOBS"

echo "== done =="
"$LLAMA_CPP_DIR/build/bin/llama" --version 2>&1 | head -1 || true
"$LLAMA_CPP_DIR/build/bin/llama-bench" --help 2>&1 | head -3

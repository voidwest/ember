#!/usr/bin/env bash
# Pinned external known-answer gate for the canonical Q4_K/Q6_K x Q8_K path.
set -euo pipefail

EXPECTED_REMOTE=https://github.com/ggml-org/llama.cpp.git
EXPECTED_COMMIT=47c786924ad1ab7e91da2cdc72fcdb563780c2bd
EXPECTED_Q8_SHA256=d1fbb94a39f658c146b9a6e797cb8849fa3d8dc39fd412b113b5b0415c1d9bde
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to the pinned llama.cpp checkout}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOBS=${CARGO_BUILD_JOBS:-2}

echo "reference: $EXPECTED_REMOTE @ $EXPECTED_COMMIT" >&2
actual_commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "llama.cpp commit $actual_commit != pinned $EXPECTED_COMMIT" >&2
    exit 1
fi

tmp=$(mktemp -d)
cleanup() {
    git -C "$LLAMA_CPP_DIR" worktree remove --force "$tmp/src" >/dev/null 2>&1 || true
    rm -rf "$tmp"
}
trap cleanup EXIT
git -C "$LLAMA_CPP_DIR" worktree add --detach "$tmp/src" "$EXPECTED_COMMIT" >/dev/null
cmake -S "$tmp/src" -B "$tmp/build" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$tmp/build" --target ggml --parallel "$JOBS" >/dev/null
BUILD_DIR="$tmp/build"
LLAMA_SOURCE="$tmp/src"
for archive in libggml-cpu.a libggml.a libggml-base.a; do
    if [[ ! -f "$BUILD_DIR/ggml/src/$archive" ]]; then
        echo "fresh pinned build did not produce $archive" >&2
        exit 1
    fi
done

"${CC:-cc}" -O2 -fopenmp \
    -I"$LLAMA_SOURCE/ggml/include" -I"$LLAMA_SOURCE/ggml/src" \
    -I"$LLAMA_SOURCE/ggml/src/ggml-cpu" \
    "$ROOT/tools/verify_k_quant_llamacpp.c" \
    "$BUILD_DIR/ggml/src/libggml-cpu.a" \
    "$BUILD_DIR/ggml/src/libggml.a" \
    "$BUILD_DIR/ggml/src/libggml-base.a" \
    -lstdc++ -lm -lpthread -ldl -o "$tmp/verify"

output=$("$tmp/verify" "$tmp/q8-k.bin")
echo "$output"
grep -qx 'q4_generic=c5eb1fdf' <<<"$output"
grep -qx 'q6_generic=c7012543' <<<"$output"
grep -qx 'q4_dispatch=c5eb1fde' <<<"$output"
grep -qx 'q6_dispatch=c7012543' <<<"$output"
[[ $(stat -c %s "$tmp/q8-k.bin") -eq 584 ]]
actual_sha=$(sha256sum "$tmp/q8-k.bin" | cut -d' ' -f1)
if [[ "$actual_sha" != "$EXPECTED_Q8_SHA256" ]]; then
    echo "Q8_K fixture SHA-256 $actual_sha != $EXPECTED_Q8_SHA256" >&2
    exit 1
fi

cd "$ROOT"
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} cargo test --locked --release --lib \
    k_quant_matmul::tests::pinned_llama_cpp_known_answer_vector

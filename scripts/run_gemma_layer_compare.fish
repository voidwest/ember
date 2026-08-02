#!/usr/bin/env fish
# run_gemma_layer_compare.fish
# Build and run Ember + llama.cpp layer comparison for Gemma 4.
#
# Environment variables:
#   GEMMA_MODEL   path to gemma-4-E2B-it-Q8_0.gguf (required)
#   LLAMACPP_DIR  path to llama.cpp checkout (required)
#   LLAMACPP_BUILD path to llama.cpp build dir (default: $LLAMACPP_DIR/build)
#   GEMMA_LAYERS / GEMMA_HIDDEN_SIZE comparison dimensions (defaults: 35 / 1536)
#   LLAMA_PATCHED_DUMP filename written by the patched decode path
#                      (default: llama_35layers.bin)
#
# Output:
#   artifacts/layer_compare_gemma/ember_layers.bin
#   artifacts/layer_compare_gemma/llama_layers.bin
#   artifacts/layer_compare_gemma/report.md
#   artifacts/layer_compare_gemma/report.json

set -q GEMMA_MODEL; or begin
    echo "GEMMA_MODEL is not set" >&2
    exit 1
end
set -q LLAMACPP_DIR; or begin
    echo "LLAMACPP_DIR is not set" >&2
    exit 1
end

test -f "$GEMMA_MODEL"; or begin
    echo "GEMMA_MODEL not found: $GEMMA_MODEL" >&2
    exit 1
end
test -d "$LLAMACPP_DIR"; or begin
    echo "LLAMACPP_DIR not found: $LLAMACPP_DIR" >&2
    exit 1
end

set -q LLAMACPP_BUILD; or set LLAMACPP_BUILD "$LLAMACPP_DIR/build"
set -q GEMMA_LAYERS; or set GEMMA_LAYERS 35
set -q GEMMA_HIDDEN_SIZE; or set GEMMA_HIDDEN_SIZE 1536
set -q LLAMA_PATCHED_DUMP; or set LLAMA_PATCHED_DUMP llama_35layers.bin

string match -qr '^[1-9][0-9]*$' -- "$GEMMA_LAYERS"; or begin
    echo "GEMMA_LAYERS must be a positive integer" >&2
    exit 1
end
string match -qr '^[1-9][0-9]*$' -- "$GEMMA_HIDDEN_SIZE"; or begin
    echo "GEMMA_HIDDEN_SIZE must be a positive integer" >&2
    exit 1
end
string match -qr '^[^/]+$' -- "$LLAMA_PATCHED_DUMP"; and test "$LLAMA_PATCHED_DUMP" != .; and test "$LLAMA_PATCHED_DUMP" != ..; or begin
    echo "LLAMA_PATCHED_DUMP must be a filename, not a path" >&2
    exit 1
end

for tool in cargo cmake g++ mktemp
    command -sq "$tool"; or begin
        echo "required command not found: $tool" >&2
        exit 1
    end
end

set SCRIPT_DIR (realpath (dirname (status --current-filename)))
set REPO_ROOT (realpath "$SCRIPT_DIR/..")
set GEMMA_MODEL (realpath "$GEMMA_MODEL")
set LLAMACPP_DIR (realpath "$LLAMACPP_DIR")
test -d "$LLAMACPP_BUILD"; or begin
    echo "LLAMACPP_BUILD not found: $LLAMACPP_BUILD" >&2
    exit 1
end
set LLAMACPP_BUILD (realpath "$LLAMACPP_BUILD")
set ARTIFACT_DIR "$REPO_ROOT/artifacts/layer_compare_gemma"
set EMBER_BIN "$REPO_ROOT/target/release/ember"

# Ensure output directory exists
mkdir -p "$ARTIFACT_DIR"

echo "=== Building Ember (release) ==="
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml"
or exit 1

echo "=== Building llama.cpp layer dump tool ==="
set JOBS 1
command -sq nproc; and set JOBS (nproc)
cmake --build "$LLAMACPP_BUILD" --target llama --parallel "$JOBS"
or exit 1

for library in \
    "$LLAMACPP_BUILD/src/libllama.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml-base.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml-cpu.a"
    test -f "$library"; or begin
        echo "required llama.cpp static library not found: $library" >&2
        exit 1
    end
end

set LLAMA_DUMP_BIN "$ARTIFACT_DIR/dump_llamacpp_layers"
set LLAMA_DUMP_TMP (mktemp "$ARTIFACT_DIR/.dump_llamacpp_layers.XXXXXX")
or exit 1
g++ -std=c++17 \
    -I"$LLAMACPP_DIR/include" \
    -I"$LLAMACPP_DIR/ggml/include" \
    -I"$LLAMACPP_DIR/src" \
    "$REPO_ROOT/tools/dump_llamacpp_layers.cpp" \
    "$LLAMACPP_BUILD/src/libllama.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml-base.a" \
    "$LLAMACPP_BUILD/ggml/src/libggml-cpu.a" \
    -lpthread -ldl -lm \
    -o "$LLAMA_DUMP_TMP"
or begin
    rm -f -- "$LLAMA_DUMP_TMP"
    exit 1
end
mv -f -- "$LLAMA_DUMP_TMP" "$LLAMA_DUMP_BIN"
or begin
    rm -f -- "$LLAMA_DUMP_TMP"
    exit 1
end

echo "=== Running llama.cpp layer dump (BOS) ==="
pushd "$ARTIFACT_DIR" >/dev/null
or exit 1
"$LLAMA_DUMP_BIN" "$GEMMA_MODEL" "" "$ARTIFACT_DIR/llama_layers.bin" 16 "$LLAMA_PATCHED_DUMP"
set LLAMA_STATUS $status
popd >/dev/null
test "$LLAMA_STATUS" -eq 0
or exit 1

echo "=== Running Ember layer dump (BOS) ==="
"$EMBER_BIN" \
    --model "$GEMMA_MODEL" \
    --arch gemma4 \
    --prompt "" \
    --max-seq-len 16 \
    --temperature 0 \
    --dump-layers "$ARTIFACT_DIR/ember_layers.bin"
or exit 1

echo "=== Comparing layer dumps ==="
set PYTHON_BIN python3
if test -x "$REPO_ROOT/.venv/bin/python"
    set PYTHON_BIN "$REPO_ROOT/.venv/bin/python"
end
"$PYTHON_BIN" "$SCRIPT_DIR/compare_layer_dumps.py" \
    --ember "$ARTIFACT_DIR/ember_layers.bin" \
    --reference "$ARTIFACT_DIR/llama_layers.bin" \
    --layers "$GEMMA_LAYERS" \
    --hidden-size "$GEMMA_HIDDEN_SIZE" \
    --out-md "$ARTIFACT_DIR/report.md" \
    --out-json "$ARTIFACT_DIR/report.json"
or exit 1

echo "=== Complete ==="
echo "Reports: $ARTIFACT_DIR/report.md  $ARTIFACT_DIR/report.json"

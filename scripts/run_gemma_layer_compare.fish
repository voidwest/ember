#!/usr/bin/env fish
# run_gemma_layer_compare.fish
# Build and run Ember + llama.cpp layer comparison for Gemma 4.
#
# Environment variables:
#   GEMMA_MODEL   path to gemma-4-E2B-it-Q8_0.gguf (required)
#   LLAMACPP_DIR  path to llama.cpp checkout (required)
#   LLAMACPP_BUILD path to llama.cpp build dir (default: $LLAMACPP_DIR/build)
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

set ARTIFACT_DIR (dirname (status --current-filename))/../artifacts/layer_compare_gemma
set SCRIPT_DIR  (dirname (status --current-filename))
set EMBER_BIN    (dirname (status --current-filename))/../target/release/ember

# Ensure output directory exists
mkdir -p "$ARTIFACT_DIR"

echo "=== Building Ember (release) ==="
cargo build --release --manifest-path (dirname (status --current-filename))/../Cargo.toml
or exit 1

echo "=== Building llama.cpp layer dump tool ==="
cmake --build "$LLAMACPP_BUILD" --target llama -j(nproc) -C "$LLAMACPP_DIR"
or exit 1

set LLAMA_DUMP_BIN "$LLAMACPP_DIR/tools/dump_llamacpp_layers"
if not test -f "$LLAMA_DUMP_BIN"
    echo "Building dump_llamacpp_layers..."
    g++ -std=c++17 \
        -I"$LLAMACPP_DIR/include" \
        -I"$LLAMACPP_DIR/ggml/include" \
        -I"$LLAMACPP_DIR/src" \
        "$SCRIPT_DIR/../tools/dump_llamacpp_layers.cpp" \
        "$LLAMACPP_BUILD/src/libllama.a" \
        "$LLAMACPP_BUILD/ggml/src/libggml.a" \
        "$LLAMACPP_BUILD/ggml/src/libggml-base.a" \
        "$LLAMACPP_BUILD/ggml/src/libggml-cpu.a" \
        -lpthread -ldl -lm \
        -o "$LLAMA_DUMP_BIN"
    or exit 1
end

echo "=== Running llama.cpp layer dump (BOS) ==="
"$LLAMA_DUMP_BIN" "$GEMMA_MODEL" "" "$ARTIFACT_DIR/llama_layers.bin" 16
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
python3 "$SCRIPT_DIR/compare_layer_dumps.py" \
    --ember "$ARTIFACT_DIR/ember_layers.bin" \
    --reference "$ARTIFACT_DIR/llama_layers.bin" \
    --layers 35 \
    --hidden-size 1536 \
    --out-md "$ARTIFACT_DIR/report.md" \
    --out-json "$ARTIFACT_DIR/report.json"
or exit 1

echo "=== Complete ==="
echo "Reports: $ARTIFACT_DIR/report.md  $ARTIFACT_DIR/report.json"

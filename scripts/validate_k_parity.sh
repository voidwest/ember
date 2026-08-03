#!/usr/bin/env bash
# Gate B: model-level parity (compressed vs eager-f32) on real models.
#
# Runs the env-gated k_parity integration test against every locally
# available llama-family K-quant artifact. The qwen2.5 artifacts
# currently fail a pre-existing qwen2.vocab_size metadata check on both
# paths; the fresh quantized ladder (pinned llama.cpp, commit 11) will
# enable them.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release

run_parity() {
    local model=$1
    local tokenizer=$2
    echo "== parity: ${model} (${tokenizer})"
    EMBER_PARITY_MODEL="$model" \
    EMBER_PARITY_TOKENIZER="$tokenizer" \
    EMBER_PARITY_TOKENS="${EMBER_PARITY_TOKENS:-12}" \
        cargo test --release --test k_parity -- --nocapture
}

run_parity Llama-3.2-1B-Instruct.Q4_K_M.gguf tokenizer.json
run_parity Llama-3.2-1B-Instruct.Q6_K.gguf tokenizer.json

echo "== all Gate B parity checks passed"

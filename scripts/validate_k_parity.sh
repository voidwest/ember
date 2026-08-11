#!/usr/bin/env bash
# Fail-closed real-model matrix for canonical Q4_K/Q6_K x Q8_K execution.
# Only the pinned 1B ladder is admitted; missing files, wrong hashes, fallback,
# missing x86 features, or a non-parallel scheduler are hard failures.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODEL_ROOT=${EMBER_PARITY_MODEL_ROOT:-models/v03-ladder}
TOKENIZER=${EMBER_PARITY_TOKENIZER:-tokenizer.json}
TOKENIZER_SHA256=${EMBER_PARITY_TOKENIZER_SHA256:-6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b}
TOKENS=${EMBER_PARITY_TOKENS:-12}

[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "EMBER_PARITY_TOKENS must be positive" >&2; exit 1; }
[[ -f "$TOKENIZER" ]] || { echo "missing parity tokenizer: $TOKENIZER" >&2; exit 1; }
tokenizer_bytes=$(stat -c %s "$TOKENIZER")
(( tokenizer_bytes <= 67108864 )) || { echo "tokenizer exceeds 64 MiB cap" >&2; exit 1; }
[[ "$TOKENIZER_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "invalid tokenizer SHA-256 pin" >&2; exit 1; }
actual_tokenizer_sha=$(sha256sum "$TOKENIZER" | cut -d' ' -f1)
[[ "$actual_tokenizer_sha" == "$TOKENIZER_SHA256" ]] || {
    echo "tokenizer SHA-256 $actual_tokenizer_sha != $TOKENIZER_SHA256" >&2
    exit 1
}

run_parity() {
    local model=$1
    local sha256=$2
    local dtype=$3
    local layers=$4
    [[ -f "$model" ]] || { echo "missing pinned parity model: $model" >&2; exit 1; }
    local bytes
    bytes=$(stat -c %s "$model")
    (( bytes <= 2500000000 )) || { echo "model exceeds 2.5 GB ladder cap: $model" >&2; exit 1; }

    echo "== parity: $model ($dtype, sha256=$sha256)"
    EMBER_PARITY_REQUIRED=1 \
    EMBER_PARITY_MODEL="$model" \
    EMBER_PARITY_TOKENIZER="$TOKENIZER" \
    EMBER_PARITY_EXPECT_SHA256="$sha256" \
    EMBER_PARITY_EXPECT_DTYPE="$dtype" \
    EMBER_PARITY_EXPECT_LAYERS="$layers" \
    EMBER_PARITY_TIER=x86 \
    EMBER_PARITY_REQUIRE_X86=1 \
    EMBER_PARITY_REQUIRE_PARALLEL=1 \
    EMBER_PARITY_TOKENS="$TOKENS" \
    RAYON_NUM_THREADS=4 \
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} \
        cargo test --locked --release --test k_parity -- --nocapture --test-threads=1
}

run_parity \
    "$MODEL_ROOT/llama-3.2-1b-q4_k_m.gguf" \
    26bac8efd811cb41a80db4393dbe5c8360abd54b98954ec766aa4ba7dacc0bc5 \
    q4_k 16
run_parity \
    "$MODEL_ROOT/llama-3.2-1b-q6_k.gguf" \
    4bf385159856b7c50a938b1228112318d9f99238a76880ea0f6381ab879982b3 \
    q6_k 16

echo "== fail-closed Q4_K/Q6_K scalar+x86, planned/fused, hooks, and allocation gates passed"

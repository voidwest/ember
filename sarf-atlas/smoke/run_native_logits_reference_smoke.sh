#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_DIR="${ROOT_DIR}/sarf-atlas/smoke"
PROMPTS_PATH="${SMOKE_DIR}/prompts.small.jsonl"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 2
  fi
}

toml_string() {
  local value="$1"
  if [[ "${value}" == *$'\n'* || "${value}" == *'"'* || "${value}" == *'\'* ]]; then
    echo "cannot safely write TOML string containing newline, quote, or backslash: ${value}" >&2
    exit 2
  fi
  printf '"%s"' "${value}"
}

require_env MODEL_PATH
require_env TOKENIZER_JSON
require_env OUT_ROOT

require_file "GGUF model" "${MODEL_PATH}"
require_file "tokenizer JSON" "${TOKENIZER_JSON}"
require_file "prompt JSONL" "${PROMPTS_PATH}"

mkdir -p "${OUT_ROOT}/runs"

RUN_ID="native-logits-reference-smoke"
MODEL_ARCH="${MODEL_ARCH:-qwen3}"
CONFIG_PATH="${OUT_ROOT}/native_logits_reference_smoke.local.toml"
RUN_DIR="${OUT_ROOT}/runs/${RUN_ID}"

cat > "${CONFIG_PATH}" <<EOF
# Generated local native logits reference smoke config.
# Do not commit this file. It is not research output.

run_id = "${RUN_ID}"
model_path = $(toml_string "${MODEL_PATH}")
architecture = $(toml_string "${MODEL_ARCH}")
tokenizer_path = $(toml_string "${TOKENIZER_JSON}")
backend = "native"
prompt_template = "{prompt}"
input_jsonl_path = $(toml_string "${PROMPTS_PATH}")
output_dir = $(toml_string "${OUT_ROOT}/runs")
layers = []
token_position = "prompt_final"
sample_id_field = "id"
word_field = "word"
batch_size = 1
dtype = "f32"
output_format = "npy"
prompt_hashes_only = false
write_logits = true
resume = false
record_model_sha256 = false

[run_metadata]
workspace = "sarf-atlas"
purpose = "native logits reference smoke test"
real_logits = true
no_generation = true
no_hidden_states = true
not_research_output = true
EOF

echo "Running native logits reference smoke. This is not research output."
echo "No generation or hidden-state extraction will be requested."
echo "Config: ${CONFIG_PATH}"

(
  cd "${ROOT_DIR}"
  cargo run -- native-logits-reference --config "${CONFIG_PATH}"
  cargo run -- validate-run "${RUN_DIR}"
)

echo "Native logits reference smoke complete."
echo "Run directory: ${RUN_DIR}"
echo "Logits artifact: ${RUN_DIR}/logits.npy"
echo "This is not research output."

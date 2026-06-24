#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_DIR="${ROOT_DIR}/sarf-atlas/smoke"
PROMPTS_PATH="${SMOKE_DIR}/prompts.small.jsonl"
HELPER_PATH="${ROOT_DIR}/scripts/llama_cpp_tokenize_extract.py"
EXPORTER_PATH="${SMOKE_DIR}/export_tokenizer_json_metadata.py"

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

require_executable() {
  local label="$1"
  local path="$2"
  if [[ ! -x "${path}" ]]; then
    echo "missing executable ${label}: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "missing directory ${label}: ${path}" >&2
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
require_env LLAMA_TOKENIZE_BIN
require_env GGUF_PARITY_TOOLS_PATH
require_env OUT_ROOT

require_file "GGUF model" "${MODEL_PATH}"
require_file "tokenizer JSON" "${TOKENIZER_JSON}"
require_executable "llama-tokenize" "${LLAMA_TOKENIZE_BIN}"
require_dir "gguf-parity-tools path" "${GGUF_PARITY_TOOLS_PATH}"
require_file "prompt JSONL" "${PROMPTS_PATH}"
require_executable "Ember external helper" "${HELPER_PATH}"
require_file "tokenizer metadata exporter" "${EXPORTER_PATH}"

mkdir -p "${OUT_ROOT}/runs"

RUN_ID="tokenization-parity-smoke"
MODEL_ARCH="${MODEL_ARCH:-qwen3}"
CONFIG_PATH="${OUT_ROOT}/llama_cpp_tokenization_smoke.local.toml"
LLAMA_RUN_DIR="${OUT_ROOT}/runs/${RUN_ID}"
TOKENIZER_OUT_DIR="${OUT_ROOT}/runs/tokenizer-json"
TOKENIZER_METADATA="${TOKENIZER_OUT_DIR}/metadata.tokenizer-json.json"
LLAMACPP_METADATA="${LLAMA_RUN_DIR}/metadata.llamacpp.json"
AUDIT_REPORT="${LLAMA_RUN_DIR}/token-audit.json"

cat > "${CONFIG_PATH}" <<EOF
# Generated local tokenization-only smoke config.
# Do not commit this file. It is not research output.

run_id = "${RUN_ID}"
model_path = $(toml_string "${MODEL_PATH}")
architecture = $(toml_string "${MODEL_ARCH}")
backend = "llama-cpp-external"
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
write_logits = false
resume = false
record_model_sha256 = false
llama_cpp_binary = $(toml_string "${HELPER_PATH}")

[run_metadata]
workspace = "sarf-atlas"
purpose = "real llama.cpp tokenization smoke test for Ember external backend plumbing"
llama_tokenize_bin = $(toml_string "${LLAMA_TOKENIZE_BIN}")
model_arch = $(toml_string "${MODEL_ARCH}")
real_llama_cpp = true
real_tokenization = true
no_generation = true
no_logits = true
no_hidden_states = true
not_research_output = true
EOF

echo "Running tokenization-only smoke. This is not research output."
echo "No generation, logits, or hidden-state extraction will be requested."
echo "Config: ${CONFIG_PATH}"

(
  cd "${ROOT_DIR}"
  cargo run -- extract --config "${CONFIG_PATH}"
)

if [[ ! -f "${LLAMACPP_METADATA}" ]]; then
  echo "missing llama.cpp tokenization metadata: ${LLAMACPP_METADATA}" >&2
  exit 2
fi

python3 "${EXPORTER_PATH}" \
  --tokenizer "${TOKENIZER_JSON}" \
  --prompts "${PROMPTS_PATH}" \
  --out "${TOKENIZER_METADATA}"

PYTHONPATH="${GGUF_PARITY_TOOLS_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -m parity_tools token-audit \
  --candidate-metadata "${TOKENIZER_METADATA}" \
  --reference-metadata "${LLAMACPP_METADATA}" \
  --out "${AUDIT_REPORT}"

echo "Tokenization parity smoke complete."
echo "Audit report: ${AUDIT_REPORT}"
echo "This is not research output."

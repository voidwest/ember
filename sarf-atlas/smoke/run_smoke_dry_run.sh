#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-sarf-atlas/smoke/llama_cpp_smoke.example.toml}"

cat <<EOF
This helper does not run model inference.

It only prints the Ember command for a local llama.cpp external backend smoke
test. Copy the example config first, replace placeholder paths with local
private paths, then run the command manually when ready.

Config:
  ${CONFIG_PATH}

Command:
  cargo run --release -- extract --config "${CONFIG_PATH}"

After a successful run, compare or validate artifacts with existing Ember paths,
for example:
  cargo run --release -- validate-backends --native-run runs/native-smoke --external-run runs/paper1-llama-cpp-smoke
EOF

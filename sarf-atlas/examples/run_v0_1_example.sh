#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/sarf-atlas-v0.1-example}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHONPATH="${ROOT_DIR}/sarf-atlas/src" \
python3 -m sarf example-workflow --out-dir "${OUT_DIR}"

cat <<EOF
Wrote Sarf v0.1 toy workflow scaffold:
  ${OUT_DIR}

This is not research output.
Replace placeholder model/tokenizer paths in:
  ${OUT_DIR}/ember_native_logits.placeholder.toml

Then run Ember extraction/validation manually when local model files are ready.
EOF

#!/usr/bin/env bash
# Restore the large research data files listed in
# scripts/research_data_manifest.json from the research-data release asset,
# verifying every file against the SHA-256 recorded in that (committed)
# manifest. Files already present and matching are left untouched.
#
# Today the data is still tracked in git, so this is a no-op on a normal
# checkout ("all files already present"). It exists so the data can move to
# a release asset later without losing a verified way back; see
# docs/research-data.md for the plan. Nothing here deletes or untracks files.
#
# Usage:
#   scripts/fetch_research_data.sh [--asset URL|FILE] [--asset-sha256 HEX] [--root DIR]
#   scripts/fetch_research_data.sh --verify [--root DIR]
#
# --asset defaults to the manifest's asset.url (unset until published).
# Env: PYTHON (default python3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODE=fetch
GLOBAL=()
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify) MODE=verify; shift ;;
    --root) GLOBAL+=(--root "${2:?--root needs a directory}"); shift 2 ;;
    --manifest) GLOBAL+=(--manifest "${2:?--manifest needs a file}"); shift 2 ;;
    --asset) ARGS+=(--asset "${2:?--asset needs a URL or file}"); shift 2 ;;
    --asset-sha256) ARGS+=(--asset-sha256 "${2:?--asset-sha256 needs a hash}"); shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == verify ]]; then
  exec "$PYTHON_BIN" "$REPO_ROOT/scripts/research_data.py" "${GLOBAL[@]+"${GLOBAL[@]}"}" verify
fi
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/research_data.py" "${GLOBAL[@]+"${GLOBAL[@]}"}" fetch \
  "${ARGS[@]+"${ARGS[@]}"}"

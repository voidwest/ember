#!/usr/bin/env bash
# Download and verify a golden-ladder release asset prepared by
# scripts/package_ladder_asset.sh, extracting the six rungs into DEST.
#
# Trust flows from one pinned value: the SHA-256 of the index. The index
# pins every part (and the GGUF inside it), and scripts/validate_ladder_inputs.py
# then checks the extracted rungs against the frozen manifest committed in
# this repository. Any mismatch fails closed.
#
# Parts are fetched one at a time and deleted after extraction, so peak disk
# use is the ladder plus one part (not the ladder twice).
#
# Usage: scripts/fetch_ladder_asset.sh --index-url URL --index-sha256 HEX --dest DIR
#                                      [--manifest FILE]
# --manifest defaults to the frozen models/v03-ladder/ladder-manifest.json;
# override it only to test this script with a synthetic ladder.
# Env: PYTHON (default python3), CURL_OPTS (extra curl options)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
INDEX_URL=""
INDEX_SHA=""
DEST=""
MANIFEST="$REPO_ROOT/models/v03-ladder/ladder-manifest.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --index-url) INDEX_URL="${2:?}"; shift 2 ;;
    --index-sha256) INDEX_SHA="${2:?}"; shift 2 ;;
    --dest) DEST="${2:?}"; shift 2 ;;
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$INDEX_URL" && -n "$DEST" ]] || { echo "--index-url and --dest are required" >&2; exit 2; }
[[ "$INDEX_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "--index-sha256 must be 64 hex characters" >&2; exit 2; }
INDEX_SHA="$(printf '%s' "$INDEX_SHA" | tr 'A-F' 'a-f')"

sha256_of() { "$PYTHON_BIN" -c 'import hashlib,sys
with open(sys.argv[1],"rb") as h: print(hashlib.file_digest(h,"sha256").hexdigest())' "$1"; }
fetch() { # url out
  # shellcheck disable=SC2086
  curl --fail --location --silent --show-error --retry 3 --retry-delay 5 ${CURL_OPTS:-} \
    --output "$2" "$1"
}

mkdir -p "$DEST"
STAGE="$DEST/.fetch"
rm -rf "$STAGE"; mkdir -p "$STAGE"
fetch "$INDEX_URL" "$STAGE/index.json"
actual="$(sha256_of "$STAGE/index.json")"
[[ "$actual" == "$INDEX_SHA" ]] || {
  echo "ladder index SHA-256 mismatch: expected $INDEX_SHA, got $actual" >&2; exit 1; }

BASE_URL="${INDEX_URL%/*}"
# name sha256 kind (kind: manifest|part), one per line
"$PYTHON_BIN" - "$STAGE/index.json" > "$STAGE/entries" <<'PYEOF'
import json, re, sys
index = json.load(open(sys.argv[1]))
if index.get("schema") != "ember.ladder-asset.v1":
    raise SystemExit(f"unsupported ladder index schema: {index.get('schema')!r}")
safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
entries = [(index["manifest"], "manifest")] + [(part, "part") for part in index["parts"]]
for entry, kind in entries:
    if not safe.match(entry["name"]) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise SystemExit(f"malformed ladder index entry: {entry!r}")
    print(entry["name"], entry["sha256"], kind)
PYEOF

while read -r name sha kind; do
  echo "fetching $name"
  fetch "$BASE_URL/$name" "$STAGE/$name"
  got="$(sha256_of "$STAGE/$name")"
  [[ "$got" == "$sha" ]] || { echo "SHA-256 mismatch for $name: expected $sha, got $got" >&2; exit 1; }
  if [[ "$kind" == manifest ]]; then
    # The asset must carry exactly the manifest this commit pins.
    cmp -s "$STAGE/$name" "$MANIFEST" || {
      echo "asset manifest differs from the pinned $MANIFEST" >&2; exit 1; }
    cp "$STAGE/$name" "$DEST/ladder-manifest.json"
  else
    members="$(tar -tf "$STAGE/$name")"
    [[ "$members" =~ ^[A-Za-z0-9._-]+\.gguf$ ]] || {
      echo "unexpected contents in $name: $members" >&2; exit 1; }
    tar -xf "$STAGE/$name" -C "$DEST"
  fi
  rm -f "$STAGE/$name"
done < "$STAGE/entries"
rm -rf "$STAGE"

# Final authority: the frozen manifest committed in this repository.
"$PYTHON_BIN" "$REPO_ROOT/scripts/validate_ladder_inputs.py" --models-dir "$DEST" \
  --manifest "$MANIFEST" >/dev/null
echo "ladder verified in $DEST"

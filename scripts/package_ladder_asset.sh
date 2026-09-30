#!/usr/bin/env bash
# Pack the pinned v0.3 golden ladder (six GGUF rungs + ladder-manifest.json)
# into a versioned, SHA-256-pinned release asset for Gate C
# (.github/workflows/golden-ladder.yml).
#
# This script only PREPARES the asset locally. Publishing it (uploading to a
# GitHub release) is a manual maintainer action; see docs/validation.md,
# "Golden ladder as a release asset".
#
# Layout. GitHub release assets are capped at 2 GiB per file and the ladder
# is ~7.6 GB, so the asset is a set of standalone tarballs, one per rung
# (each extracts on its own, so a consumer never needs the whole ladder on
# disk twice), plus a small index that pins them:
#
#   <out>/ember-ladder-<version>.index.json         the pin; its SHA-256 is
#                                                   what the workflow takes
#   <out>/ember-ladder-<version>-<family>-<rung>.tar   one GGUF each
#   <out>/ember-ladder-<version>-ladder-manifest.json  the frozen manifest
#   <out>/SHA256SUMS                                   every file above
#
# Tarballs are deterministic (sorted, mtime 0, uid/gid 0, symlinks
# dereferenced): the same rungs always produce the same bytes and hashes.
#
# The rungs are checked against the frozen manifest with
# scripts/validate_ladder_inputs.py before anything is written; a missing
# or mismatched rung fails closed.
#
# Usage:
#   scripts/package_ladder_asset.sh --version v1 [--ladder DIR] [--out DIR]
#                                   [--manifest FILE]
# Env: PYTHON (default python3; needs 3.11+ for hashlib.file_digest)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
LADDER="$REPO_ROOT/models/v03-ladder"
MANIFEST=""
OUT="$REPO_ROOT/dist/ladder"
VERSION=""

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    --ladder) LADDER="${2:?--ladder needs a directory}"; shift 2 ;;
    --manifest) MANIFEST="${2:?--manifest needs a file}"; shift 2 ;;
    --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$VERSION" ]] || { echo "--version is required (e.g. v1)" >&2; exit 2; }
[[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || { echo "--version must be [A-Za-z0-9._-] (got '$VERSION')" >&2; exit 2; }
MANIFEST="${MANIFEST:-$LADDER/ladder-manifest.json}"
[[ -d "$LADDER" ]] || { echo "ladder directory not found: $LADDER" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "ladder manifest not found: $MANIFEST" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python not found: $PYTHON_BIN" >&2; exit 1; }

# Fail closed on a missing or mismatched rung before writing anything.
"$PYTHON_BIN" "$REPO_ROOT/scripts/validate_ladder_inputs.py" \
  --models-dir "$LADDER" --manifest "$MANIFEST" >/dev/null

mkdir -p "$OUT"
INDEX="$OUT/ember-ladder-$VERSION.index.json"
[[ ! -e "$INDEX" ]] || { echo "refusing to overwrite existing asset: $INDEX" >&2; exit 1; }

"$PYTHON_BIN" - "$REPO_ROOT" "$LADDER" "$MANIFEST" "$OUT" "$VERSION" <<'PYEOF'
import hashlib, io, json, pathlib, sys, tarfile
repo, ladder, manifest, out, version = sys.argv[1:]
ladder, manifest, out = map(pathlib.Path, (ladder, manifest, out))
sys.path.insert(0, str(pathlib.Path(repo) / "scripts"))
from validate_ladder_inputs import FAMILIES, QUANTIZER_COMMIT, RUNGS

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

records = {(r["family"], r["rung"]): r for r in json.loads(manifest.read_text())}
prefix = f"ember-ladder-{version}"
parts = []
for family in FAMILIES:
    for rung in RUNGS:
        name = f"{family}-{rung}.gguf"
        source = (ladder / name).resolve()  # dereference symlinks
        tar_path = out / f"{prefix}-{family}-{rung}.tar"
        if tar_path.exists():
            raise SystemExit(f"refusing to overwrite existing asset part: {tar_path}")
        info = tarfile.TarInfo(name)
        info.size = source.stat().st_size
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar, source.open("rb") as handle:
            tar.addfile(info, handle)
        target = records[(family, rung)]["target"]
        parts.append({
            "name": tar_path.name,
            "sha256": digest(tar_path),
            "bytes": tar_path.stat().st_size,
            "contains": {"file": name, "sha256": target["sha256"], "bytes": target["bytes"]},
        })
        print(f"packed {tar_path.name} ({tar_path.stat().st_size} bytes)", file=sys.stderr)

manifest_copy = out / f"{prefix}-ladder-manifest.json"
if manifest_copy.exists():
    raise SystemExit(f"refusing to overwrite existing asset part: {manifest_copy}")
manifest_copy.write_bytes(manifest.read_bytes())
index = {
    "schema": "ember.ladder-asset.v1",
    "version": version,
    "quantizer_commit": QUANTIZER_COMMIT,
    "manifest": {"name": manifest_copy.name, "sha256": digest(manifest_copy),
                 "bytes": manifest_copy.stat().st_size},
    "parts": parts,
    "note": "Parts are resolved relative to this index's URL. Each .tar holds one GGUF.",
}
index_path = out / f"{prefix}.index.json"
index_path.write_text(json.dumps(index, indent=2) + "\n")
files = [index_path, manifest_copy] + [out / part["name"] for part in parts]
(out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in files))
PYEOF

INDEX_SHA="$("$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$INDEX")"
cat <<EOF
ladder asset $VERSION prepared in $OUT (not published).

  index:        $(basename "$INDEX")
  index sha256: $INDEX_SHA

To publish (maintainer, manual): upload every file in $OUT to one GitHub
release, then run the golden-ladder workflow with
  index_url    = https://github.com/<owner>/<repo>/releases/download/<tag>/$(basename "$INDEX")
  index_sha256 = $INDEX_SHA
(or store them as the repository variables EMBER_LADDER_INDEX_URL and
EMBER_LADDER_INDEX_SHA256). See docs/validation.md.
EOF

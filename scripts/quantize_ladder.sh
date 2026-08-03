#!/usr/bin/env bash
# Build the v0.3 matched quantization ladder from fp16 GGUF sources using
# the pinned llama.cpp quantizer (decision D2): Q8_0, Q6_K, Q4_K_M for
# both validation families. The fp16 sources are NOT requantized ladder
# rungs — every rung is quantized directly from fp16, and every command
# and artifact hash is recorded in a machine-readable manifest.
#
# Usage:
#   scripts/quantize_ladder.sh --model-llama FP16_GGUF --model-qwen FP16_GGUF
#     [--out DIR] [--jobs N]
#
# Env:
#   LLAMA_CPP_DIR   llama.cpp build dir (default ~/.cache/ember/llama.cpp)
#                   — must contain the built llama-quantize binary
#
# Outputs (all under $OUT, default models/v03-ladder):
#   llama-3.2-1b-{q8_0,q6_k,q4_k_m}.gguf
#   qwen2.5-1.5b-{q8_0,q6_k,q4_k_m}.gguf
#   ladder-manifest.json   — per-rung: source, command, sha256, bytes

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.cache/ember/llama.cpp}"
LLAMA_BIN="$LLAMA_CPP_DIR/build/bin/llama"
QUANTIZE_CMD="$LLAMA_BIN quantize"
OUT="${OUT:-$REPO_ROOT/models/v03-ladder}"
JOBS="${JOBS:-$(nproc)}"

LLAMA_FP16=""
QWEN_FP16=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-llama) LLAMA_FP16="$2"; shift 2 ;;
    --model-qwen) QWEN_FP16="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -x "$LLAMA_BIN" ]] || {
  echo "llama binary not found at $LLAMA_BIN — run scripts/setup_llama_cpp.sh first" >&2
  exit 1
}
[[ -n "$LLAMA_FP16" || -n "$QWEN_FP16" ]] || {
  echo "provide at least one of --model-llama / --model-qwen (fp16 GGUF)" >&2
  exit 1
}

mkdir -p "$OUT"
COMMIT="$(cat "$LLAMA_CPP_DIR/COMMIT" 2>/dev/null || echo unknown)"
MANIFEST="$OUT/ladder-manifest.json"
# merge: the ladder may be built in multiple invocations (per family);
# never reset an existing manifest
if [[ -f "$MANIFEST" ]]; then
  echo "merging into existing manifest"
else
  echo "[]" > "$MANIFEST"
fi

quantize_rungs() {
  local family="$1" source="$2"
  [[ -f "$source" ]] || { echo "source not found: $source" >&2; exit 1; }
  local source_sha source_bytes
  source_sha="$(sha256sum "$source" | cut -d' ' -f1)"
  source_bytes="$(stat -c %s "$source")"
  for rung in q8_0 q6_k q4_k_m; do
    local target="$OUT/$family-$rung.gguf"
    echo "== $family: $source -> $rung =="
    rm -f "$target"
    "$LLAMA_BIN" quantize "$source" "$target" "$rung" "$JOBS"
    [[ -f "$target" ]] || { echo "quantize produced no output: $target" >&2; exit 1; }
    local target_sha target_bytes
    target_sha="$(sha256sum "$target" | cut -d' ' -f1)"
    target_bytes="$(stat -c %s "$target")"
    "$REPO_ROOT/.venv/bin/python" - "$MANIFEST" "$family" "$rung" \
      "$source" "$source_sha" "$source_bytes" "$target" "$target_sha" "$target_bytes" "$COMMIT" <<'PYEOF'
import json, sys
manifest_path, family, rung, source, source_sha, source_bytes, target, target_sha, target_bytes, commit = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest.append({
    "family": family,
    "rung": rung,
    "quantizer_commit": commit,
    "command": f"llama quantize {source} {target} {rung}",
    "source": {"path": source, "sha256": source_sha, "bytes": int(source_bytes)},
    "target": {"path": target, "sha256": target_sha, "bytes": int(target_bytes)},
})
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PYEOF
  done
}

[[ -n "$QWEN_FP16" ]] && quantize_rungs "qwen2.5-1.5b" "$QWEN_FP16"
[[ -n "$LLAMA_FP16" ]] && quantize_rungs "llama-3.2-1b" "$LLAMA_FP16"

echo "== ladder complete =="
"$REPO_ROOT/.venv/bin/python" - "$MANIFEST" <<'PYEOF'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for entry in manifest:
    print(f"{entry['family']} {entry['rung']:6s} {entry['target']['bytes']/1e6:9.1f} MB  sha256 {entry['target']['sha256'][:12]}")
print(f"manifest: {sys.argv[1]}")
PYEOF

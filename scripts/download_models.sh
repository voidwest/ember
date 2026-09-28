#!/usr/bin/env bash
# Download the quantized GGUF models used by Ember's docs and benchmark
# fixtures. Model files are gitignored (see .gitignore); this script is the
# canonical way to fetch them.
#
# Usage:
#   scripts/download_models.sh             # quickstart set (laptop-sized models)
#   scripts/download_models.sh all         # quickstart + larger matrix models
#   scripts/download_models.sh research-example  # only the pinned Llama 1B
#   scripts/download_models.sh tokenizer   # only the pinned tokenizer.json
#   MODEL_DIR=/path/to/models scripts/download_models.sh all
#
# Source revisions, SHA-256 values and sizes were verified on 2026-09-27.
# Entries are URL|LOCAL_FILENAME|SHA256|BYTES. Downloads resume in .part
# files and are published only after their complete identities match.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT}"

# QUICKSTART: models small enough for a laptop (0.6B-3B, Q8_0).
# Llama 1B is pinned to the morphology example hash (432f310a...), verified
# 2026-09-27; the moving unsloth/main file has a different identity.
QUICKSTART=(
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf|Qwen3-0.6B-Q8_0.gguf|9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031|639446688"
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q8_0.gguf|Llama-3.2-1B-Instruct-Q8_0.gguf|432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3|1321083008"
  "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/e7d0997e49c9cb00d88b4c1a6a16aa894b0bbc31/Llama-3.2-3B-Instruct-Q8_0.gguf|Llama-3.2-3B-Instruct-Q8_0.gguf|f34112a11b7dad74ab517dedf6dcf00d624c9adac2dc0c72c719ca0478554ef2|3421898816"
)

# FULL: larger models used by the extended benchmark matrix
# (probes/benchmarks/*.json). The 8B-class Q8_0 files are ~8-9 GB each.
FULL=(
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf|Qwen3-8B-Q8_0.gguf|408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6|8709518112"
)

# TOKENIZER: the matching Llama-3.2 tokenizer.json is already tracked at the
# repository root, so installs do not need it. This public, ungated source is
# a documented, byte-identical fallback (verified 2026-09-28: same size and
# SHA-256) for a clean machine that wants to re-fetch or verify it without
# Meta's gated meta-llama repository.
TOKENIZER=(
  "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/5a8abab4a5d6f164389b1079fb721cfab8d7126c/tokenizer.json|tokenizer.json|6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b|17209920"
)

# Candidate sources not yet verified against the Hub; fixture filenames may
# be local renames of the upstream files:
#   qwen2.5-1.5b-instruct-q8_0.gguf    <- unsloth/Qwen2.5-1.5B-Instruct-GGUF (Qwen2.5-1.5B-Instruct-Q8_0.gguf)
#   meta-llama-3.1-8b-instruct.Q8_0.gguf <- unsloth/Llama-3.1-8B-Instruct-GGUF (Llama-3.1-8B-Instruct-Q8_0.gguf)
#   gemma-4-E2B-it.Q8_0.gguf           <- source repo TBD; see docs/models.md

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

matches_pin() {
  [[ -f "$1" ]] && [[ "$(wc -c < "$1" | tr -d '[:space:]')" == "$3" ]] &&
    [[ "$(file_sha256 "$1")" == "$2" ]]
}

download() (
  # A subshell keeps its cleanup trap local to this one download.
  local url="$1" name="$2" sha="$3" bytes="$4"
  local target="$MODEL_DIR/$name" partial="$MODEL_DIR/$name.part"
  download_lock="$MODEL_DIR/$name.download.lock"
  mkdir "$download_lock" 2>/dev/null || {
    echo "download lock exists: $download_lock (check for another download or an interrupted process)" >&2
    exit 1
  }
  trap 'rmdir "$download_lock" 2>/dev/null || true' EXIT
  if [[ -e "$target" || -L "$target" ]]; then
    matches_pin "$target" "$sha" "$bytes" || {
      echo "existing model differs from its pin; preserved: $target (use a new MODEL_DIR)" >&2
      exit 1
    }
    echo "skip  $name (verified)"
    exit 0
  fi
  if [[ -L "$partial" || ( -e "$partial" && ! -f "$partial" ) ]]; then
    echo "partial download is not a regular file: $partial" >&2
    exit 1
  fi
  if ! matches_pin "$partial" "$sha" "$bytes"; then
    if [[ -f "$partial" && "$(wc -c < "$partial" | tr -d '[:space:]')" -ge "$bytes" ]]; then
      echo "partial download differs from its pin; preserved: $partial (use a new MODEL_DIR)" >&2
      exit 1
    fi
    echo "fetch $name"
    curl -fL --retry 3 -C - -o "$partial" "$url"
    matches_pin "$partial" "$sha" "$bytes" || {
      echo "download failed size/SHA-256 verification; retained: $partial" >&2
      exit 1
    }
  fi
  # Hard-link publication refuses a destination created since the first check.
  ln "$partial" "$target"
  rm "$partial"
  echo "  -> $name (verified $bytes bytes)"
)

main() {
  local mode="${1:-quickstart}" entry url name sha bytes
  case "$mode" in
    research-example) entries=("${QUICKSTART[1]}" "${TOKENIZER[@]}") ;;
    quickstart) entries=("${QUICKSTART[@]}" "${TOKENIZER[@]}") ;;
    all)        entries=("${QUICKSTART[@]}" "${TOKENIZER[@]}" "${FULL[@]}") ;;
    tokenizer)  entries=("${TOKENIZER[@]}") ;;
    *) echo "usage: $0 [research-example|quickstart|tokenizer|all]" >&2; return 2 ;;
  esac
  mkdir -p "$MODEL_DIR"
  for entry in "${entries[@]}"; do
    IFS='|' read -r url name sha bytes <<< "$entry"
    download "$url" "$name" "$sha" "$bytes"
  done
  echo "done. verified models are in $MODEL_DIR (gitignored)."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi

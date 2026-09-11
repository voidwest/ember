#!/usr/bin/env bash
# Build the pinned cargo-fuzz instrumentation once, then run all eight campaigns.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${FUZZ_TOOLCHAIN:?set the pinned fuzz toolchain}"
: "${FUZZ_TARGET_TRIPLE:?set the fuzz target triple}"

targets=(gguf_loader gguf_to_llama npy_bytes kv_snapshot_manifest tokenizer_json
         wav_bytes image_bytes image_preprocess)
lockfiles_before=$(sha256sum Cargo.lock fuzz/Cargo.lock)
check_lockfiles() {
  if [[ "$(sha256sum Cargo.lock fuzz/Cargo.lock)" != "$lockfiles_before" ]]; then
    echo "fuzz build changed a locked dependency graph" >&2
    exit 1
  fi
}

# cargo-fuzz 0.13.2 does not forward CARGOFLAGS or offer --locked. Resolve with
# Cargo's actual --locked flag, then fail before campaigns if either lock changes.
cargo "+$FUZZ_TOOLCHAIN" fetch --locked --manifest-path fuzz/Cargo.toml
check_lockfiles
cargo "+$FUZZ_TOOLCHAIN" fuzz check --target "$FUZZ_TARGET_TRIPLE" \
  --target-dir fuzz/target --sanitizer address
check_lockfiles
cargo "+$FUZZ_TOOLCHAIN" fuzz build --target "$FUZZ_TARGET_TRIPLE" \
  --target-dir fuzz/target --sanitizer address
check_lockfiles

# cargo-fuzz's AddressSanitizer runtime default (project.rs in version 0.13.2).
# Preserve ambient options, with the same final override as cargo-fuzz run.
export ASAN_OPTIONS="${ASAN_OPTIONS:+${ASAN_OPTIONS}:}detect_odr_violation=0"
for target in "${targets[@]}"; do
  mkdir -p "fuzz/corpus_work/$target" "fuzz/artifacts/$target"
  binary="fuzz/target/$FUZZ_TARGET_TRIPLE/release/$target"
  if [[ ! -x "$binary" ]]; then
    echo "missing built fuzz target: $binary" >&2
    exit 1
  fi
  "$binary" "-artifact_prefix=$PWD/fuzz/artifacts/$target/" \
    -max_total_time=30 -max_len=262144 -timeout=10 -rss_limit_mb=2048 \
    "fuzz/corpus_work/$target"
done

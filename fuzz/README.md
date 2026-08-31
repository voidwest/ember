# Ember parser fuzzing

This is an isolated [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
package for hostile-input boundaries. It does not require model files. All
targets accept only bounded byte slices; parser-only targets stop before model
construction, while `gguf_to_llama` also exercises the bounded construction
validation path:

- `gguf_loader`: GGUF metadata, tensor descriptors, ranges, and payload
  decoding through eager, scalar, x86, and auto K-strategy paths.
- `gguf_to_llama`: GGUF parsing followed by bounded Llama model-construction
  validation through eager, scalar, x86, and auto K-strategy paths.
- `npy_bytes`: strict 2-D little-endian f32 NPY parsing through
  `read_npy_2d_bytes`.
- `kv_snapshot_manifest`: KV snapshot JSON deserialization and structural
  validation through `KvSnapshotManifest::from_json_bytes`.
- `tokenizer_json`: bounded tokenizer JSON loading, validation, and wrapper
  encode/decode calls through `EmberTokenizer::from_bytes`.

Install the pinned cargo-fuzz release and pinned `nightly-2026-05-26` toolchain, then run one target:

```bash
rustup toolchain install nightly-2026-05-26 --profile minimal
cargo +nightly-2026-05-26 install cargo-fuzz@0.13.2 --locked
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz list
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run gguf_loader -- -max_total_time=60 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run gguf_to_llama -- -max_total_time=60 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run npy_bytes -- -max_total_time=60 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run kv_snapshot_manifest -- -max_total_time=60 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run tokenizer_json -- -max_total_time=60 -timeout=10
```

The repository intentionally keeps generated `fuzz/target/`, crash
artifacts, and working corpora untracked. Seed a target with a small synthetic
input when useful; no production GGUF, activation, or snapshot payloads belong
in this directory. CI uses short smoke fuzzing runs and uploads crash artifacts
for triage.

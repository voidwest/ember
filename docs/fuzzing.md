# Parser fuzzing

Ember includes an isolated `cargo-fuzz` package for hostile-input parser
boundaries. The targets do not load a model or require any model artifact:

| target | boundary |
| --- | --- |
| `gguf_loader` | GGUF v3 header, metadata, tensor descriptors, ranges, supported payloads, and K-strategy dispatch |
| `gguf_to_llama` | GGUF parsing followed by bounded Llama model-construction validation (eager/scalar/x86/auto K strategies) |
| `npy_bytes` | strict 2-D little-endian f32 NPY header and payload parsing |
| `kv_snapshot_manifest` | KV snapshot manifest JSON deserialization and metadata validation |
| `tokenizer_json` | bounded tokenizer JSON validation and tokenizers-wrapper parsing |

Each target rejects inputs larger than 256 KiB before entering the parser. The
path and in-memory NPY readers additionally enforce a 256 MiB production
boundary, while the loader and snapshot code retain their independent
allocation and shape limits. `read_npy_2d_bytes` and
`KvSnapshotManifest::from_json_bytes` are the in-memory entry points used both
by the fuzzers and by callers that need to validate bytes before touching the
filesystem.

## Local runs

Install cargo-fuzz **0.13.2** and the pinned `nightly-2026-05-26` Rust toolchain, then run from the repository
root:

```bash
rustup toolchain install nightly-2026-05-26 --profile minimal
cargo +nightly-2026-05-26 install cargo-fuzz@0.13.2 --locked
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz check
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run gguf_loader -- -max_total_time=300 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run gguf_to_llama -- -max_total_time=300 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run npy_bytes -- -max_total_time=300 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run kv_snapshot_manifest -- -max_total_time=300 -timeout=10
CARGOFLAGS=--locked cargo +nightly-2026-05-26 fuzz run tokenizer_json -- -max_total_time=300 -timeout=10
```

Use a separate corpus directory when preserving seed files. Generated target
builds, crash artifacts, and corpora are ignored by git; production GGUFs and
research payloads must not be copied into the fuzz tree. The CI workflow runs
short bounded smoke campaigns for all five targets and uploads crash artifacts
when a campaign fails.

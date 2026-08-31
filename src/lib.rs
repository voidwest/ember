#![doc = include_str!("../docs/api-stability.md")]
// Keep rustdoc links honest: broken links and public-doc links to private
// items fail the docs job in CI (see .github/workflows/ci.yml).
#![warn(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
// Unsafe-hygiene contract: every unsafe operation inside an `unsafe fn`
// must sit in its own explicit `unsafe { .. }` block (edition-2024 lint,
// enforced early). The reference path is 100% safe Rust; all unsafe is
// contained in the kernel modules (simd.rs, k_quant_matmul.rs) and the
// counting allocator, each with `// SAFETY:` annotations.
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

// NOTE: the counting allocator is deliberately installed by the *library*,
// not by the binary. Gate E (`tests/k_parity.rs`, v04-execution-contract.md)
// measures per-token allocations under `cargo test --all-targets`, which runs
// test binaries linked against this lib — an allocator installed only in
// `main.rs` would make those measurements silently meaningless (always 0).
// The cost is one relaxed atomic pair per allocation, and steady-state
// planned decode performs no allocations at all (see `alloc_counter` module
// docs), so the hot path pays nothing. The ember lib is not consumed as an
// external dependency today; revisit if that changes.
#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

pub mod agent;
// These modules are kept source-visible for the package's separate binary
// target and existing integrations, but are implementation details rather than
// supported API. `doc(hidden)` preserves current paths without advertising
// them in generated documentation; see docs/api-stability.md.
#[doc(hidden)]
pub mod alloc_counter;
pub mod artifact;
#[doc(hidden)]
pub mod atomic_file;
pub mod backend;
pub mod compare;
#[doc(hidden)]
pub mod decode_profile;
pub mod duplex;
// (device bindings live in duplex::device behind the "audio" feature)
pub mod embedding;
pub mod experiments;
pub mod extraction;
pub mod gemma4;
#[doc(hidden)]
pub mod k_matmul;
#[doc(hidden)]
pub mod k_quant_matmul;
pub mod kv_cache;
pub mod kv_compare;
pub mod kv_diagnostics;
pub mod kv_snapshot;
pub mod kv_transfer;
pub mod llama;
pub mod loader;
pub mod model;
#[doc(hidden)]
pub mod model_backend;
pub mod multimodal;
#[doc(hidden)]
pub mod npy;
pub mod plan;
mod plan_build;
#[doc(hidden)]
pub mod planned_decode;
pub mod quant;
#[doc(hidden)]
pub mod quant_k;
#[doc(hidden)]
pub mod residency;
pub mod sampler;
#[doc(hidden)]
pub mod simd;
pub mod smolvlm;
pub mod smolvlm_video;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
pub mod tts;
pub mod ultravox;
pub mod v05;
#[doc(hidden)]
pub mod workspace;

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

// The counting allocator is *not* installed by the library: a library
// `#[global_allocator]` is forced on every consumer (the Python binding,
// downstream crates with their own allocator) and taxes every allocation.
// It is registered here only for the lib's own unit tests; the `ember`
// binary (`src/main.rs`) and the integration tests/examples that measure
// allocations (`tests/k_parity.rs`, `examples/rayon_alloc_test.rs`) register
// it themselves. Zero-allocation assertions check
// `alloc_counter::counting_active()` so a missing registration cannot make
// them pass vacuously.
#[cfg(test)]
#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

pub mod agent;
#[doc(hidden)]
pub mod app_store;
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
pub mod cancel;
pub mod compare;
#[doc(hidden)]
pub mod decode_profile;
pub mod diff_corpus;
pub mod diff_outcome;
pub mod duplex;
// (device bindings live in duplex::device behind the "audio" feature)
pub mod embedding;
pub mod experiments;
pub mod extraction;
pub mod gemma4;
pub mod inspect;
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
pub mod packed_cache;
pub mod plan;
mod plan_build;
#[doc(hidden)]
pub mod planned_decode;
pub mod quant;
#[doc(hidden)]
pub mod quant_fault;
pub mod quant_k;
#[doc(hidden)]
pub mod residency;
pub mod residual_patch;
pub mod runtime_schedule;
pub mod sampler;
#[doc(hidden)]
pub mod simd;
pub mod smolvlm;
pub mod smolvlm_video;
pub mod subprocess;
pub mod support;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
pub mod tts;
pub mod ultravox;
pub mod v05;
#[doc(hidden)]
pub mod workspace;

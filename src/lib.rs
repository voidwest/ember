#![doc = include_str!("../docs/api-stability.md")]
// Keep rustdoc links honest: broken links and public-doc links to private
// items fail the docs job in CI (see .github/workflows/ci.yml).
#![warn(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
// Unsafe-hygiene contract: every unsafe operation inside an `unsafe fn`
// must sit in its own explicit `unsafe { .. }` block (edition-2024 lint,
// enforced early), and every `unsafe` block or impl carries a `// SAFETY:`
// comment (`clippy::undocumented_unsafe_blocks`, enabled in Cargo.toml and
// enforced by CI's `-D warnings`). Unsafe code is confined to the SIMD
// kernels (simd.rs, q8_gemm.rs, attention_kernels.rs, k_quant_matmul*),
// file mappings and page eviction (loader.rs, quant.rs, model.rs,
// packed_cache.rs), the decode arena's f32 views (plan.rs), the `sgemm`
// call in tensor.rs and the counting allocator.
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

// Kernels, models, plans and the hook framework live in the `ember-core`
// crate (so research, CLI and GUI changes do not recompile them); they are
// re-exported here at their established paths.
//
// These modules are kept source-visible for the package's separate binary
// target and existing integrations, but are implementation details rather than
// supported API. `doc(hidden)` preserves current paths without advertising
// them in generated documentation; see docs/api-stability.md.
#[doc(hidden)]
pub use ember_core::{
    alloc_counter, atomic_file, bounded_read, decode_profile, k_matmul, k_quant_matmul,
    planned_decode, quant_fault, residency, simd, workspace,
};
pub use ember_core::{
    backend, cancel, gemma4, half_weight, kv_cache, llama, loader, model, packed_cache, plan,
    quant, quant_k, runtime_schedule, sampler, support, tensor, tokenizer, trace,
};

pub mod agent;
#[doc(hidden)]
pub mod app_store;
pub mod artifact;
pub mod compare;
pub mod diff_corpus;
pub mod diff_outcome;
pub mod duplex;
// (device bindings live in duplex::device behind the "audio" feature)
pub mod embedding;
pub mod experiments;
pub mod extraction;
pub mod inspect;
pub mod kv_compare;
pub mod kv_diagnostics;
pub mod kv_snapshot;
pub mod kv_transfer;
#[doc(hidden)]
pub mod model_backend;
pub mod multimodal;
#[doc(hidden)]
pub mod npy;
pub mod smolvlm;
pub mod smolvlm_video;
pub mod subprocess;
pub mod tts;
pub mod ultravox;
pub mod v05;

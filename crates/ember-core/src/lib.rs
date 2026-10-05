//! Kernels, tensors, quantized weights, GGUF loading, the Llama/Qwen and
//! Gemma 4 models, execution plans, the KV cache, sampling, tokenization,
//! tracing, and the hook framework those models fire.
//!
//! This is an implementation crate of [`ember`](https://github.com/voidwest/ember):
//! `ember` re-exports every public module here at its established path
//! (`ember::llama`, `ember::backend`, ...) and adds the research layers
//! (experiments, bundles, snapshots, multimodal, agent). Depend on `ember`;
//! the API stability policy (`docs/api-stability.md`) is stated there and
//! covers these modules through those paths only.
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

// Registered only for this crate's own unit tests (zero-allocation
// assertions check `alloc_counter::counting_active()`); see the note in the
// `ember` crate root for why a library never installs it for consumers.
#[cfg(test)]
#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

// `doc(hidden)` modules are implementation details kept source-visible for
// the `ember` crate's binary and existing integrations; see
// docs/api-stability.md.
#[doc(hidden)]
pub mod alloc_counter;
#[doc(hidden)]
pub mod atomic_file;
mod attention_kernels;
pub mod backend;
#[doc(hidden)]
pub mod bounded_read;
pub mod cancel;
mod decode_pool;
#[doc(hidden)]
pub mod decode_profile;
pub mod execution_inventory;
pub mod experiments;
pub mod gemma4;
pub mod half_weight;
pub mod hook_types;
#[doc(hidden)]
pub mod k_matmul;
#[doc(hidden)]
pub mod k_quant_matmul;
pub mod kv_cache;
pub mod llama;
pub mod loader;
pub mod model;
pub mod packed_cache;
pub mod plan;
mod plan_build;
#[doc(hidden)]
pub mod planned_decode;
#[cfg(target_arch = "aarch64")]
mod q8_gemm;
pub mod quant;
#[doc(hidden)]
pub mod quant_fault;
pub mod quant_k;
#[doc(hidden)]
pub mod residency;
pub mod runtime_schedule;
pub mod sampler;
#[doc(hidden)]
pub mod simd;
pub mod support;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
#[doc(hidden)]
pub mod workspace;

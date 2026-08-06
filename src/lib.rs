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

pub mod alloc_counter;
pub mod artifact;
pub mod atomic_file;
pub mod backend;
pub mod compare;
pub mod decode_profile;
pub mod experiments;
pub mod extraction;
pub mod gemma4;
pub mod k_matmul;
pub mod k_matmul_x86;
pub mod kv_cache;
pub mod llama;
pub mod loader;
pub mod model;
pub mod model_backend;
pub mod npy;
pub mod plan;
pub mod planned_decode;
pub mod quant;
pub mod quant_k;
pub mod residency;
pub mod sampler;
pub mod simd;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
pub mod v05;
pub mod workspace;

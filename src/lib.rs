extern crate alloc;

/// v0.4 process-wide allocation counter (Gate E measurement).
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

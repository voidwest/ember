//! # Ember
//!
//! a lightweight, cpu-first llm inference engine.
//!
//! ## Architecture
//!
//! the core abstraction is the [`Backend`] trait. model code is generic over
//! the backend, so you can start with [`CpuBackend`] and swap in Candle/GPU
//! later without rewriting your model.
//!
//! ## Design Philosophy
//!
//! - **Explicit memory**: No hidden allocations during inference.
//! - **alloc-first**: Core tensor types avoid `std` where practical.
//! - **Quantization first**: Design for Q4_0/Q8_0 from day one.

// research: `no_std` environments. we're not fully no_std yet (we use vec),
// but keeping `alloc` separate from `std` makes a future port easier.
// try to avoid `std::` types in core modules; use `alloc::` instead.
extern crate alloc;

pub mod backend;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

//! # Ember
//!
//! A lightweight, CPU-first LLM inference engine.
//!
//! ## Architecture
//!
//! The core abstraction is the [`Backend`] trait. Model code is generic over
//! the backend, so you can start with [`CpuBackend`] and swap in Candle/GPU
//! later without rewriting your model.
//!
//! ## Design Philosophy
//!
//! - **Explicit memory**: No hidden allocations during inference.
//! - **no_std friendly**: Core types avoid `std` where possible.
//! - **Quantization first**: Design for Q4_0/Q8_0 from day one.

// RESEARCH: `no_std` environments. We're not fully no_std yet (we use Vec),
// but keeping `alloc` separate from `std` makes a future port easier.
// Try to avoid `std::` types in core modules; use `alloc::` instead.
extern crate alloc;

pub mod backend;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod tensor;
pub mod tokenizer;

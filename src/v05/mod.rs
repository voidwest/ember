//! Ember v0.5: declarative experiment specifications, deterministic
//! experiment bundles, and the reproducible research-workflow CLI.
//!
//! The v0.5 thesis (docs/v05-research-contract.md): tracing, capture,
//! token-selection, intervention, and provenance primitives become a
//! stable experiment interface that researchers can use without writing
//! Rust.

pub mod capture;
pub mod hook;
pub mod intervention;
pub mod spec;
pub mod token_select;

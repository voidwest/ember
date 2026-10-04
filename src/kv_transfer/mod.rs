//! Experimental seam for future KV representation transforms.
//!
//! No learned mapper is implemented here. The stable runtime boundary is a
//! verified [`crate::kv_snapshot::KvSnapshot`] on each side.

pub mod rope;

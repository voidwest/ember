//! Experimental seam for future KV representation transforms.
//!
//! No learned mapper is implemented here. The stable runtime boundary is a
//! verified [`crate::kv_snapshot::KvSnapshot`] on each side.

pub mod rope;

/// Key coordinate spaces used by future external transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKeySpace {
    /// The runtime cache representation (after the model's RoPE stage).
    StoredPostRope,
    /// Position-independent keys after removing RoPE. When K norm is before
    /// RoPE, this remains the normalized representation.
    Content,
}

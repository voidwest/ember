//! Pre-allocated scratch buffers for inference.
//!
//! Every tensor operation in `CpuTensor` allocates a new `Vec<f32>`. For a
//! 16-layer Llama decode step that means ~800+ heap allocations. This module
//! provides a `Workspace` with reusable buffers so the decode hot path can
//! write intermediate results into pre-allocated slices, eliminating almost
//! all per-token allocations.
//!
//! ## Buffer sizing
//!
//! The production Llama path uses this workspace only for single-token decode,
//! with `max_rows = 1`. The type can hold more rows for focused experiments,
//! but prefill continues through the generic tensor path.

use alloc::vec::Vec;

/// Reusable scratch buffers for one transformer forward pass.
///
/// Each buffer is pre-allocated to `max_rows * cols` elements. Callers
/// write into `&mut out[..rows * cols]` where `rows <= max_rows`.
///
/// Cache-line aligned to prevent false sharing when Rayon threads access
/// different fields of the same workspace during parallel matmuls.
#[derive(Debug)]
#[repr(align(64))]
pub struct Workspace {
    /// max row count these buffers were allocated for
    max_rows: usize,

    // -- per-layer intermediates (reused across layers) --
    /// RMS norm output shape [rows, embed_dim]
    pub(crate) norm_out: Vec<f32>,
    /// residual add output shape [rows, embed_dim]
    pub(crate) residual_out: Vec<f32>,
    /// Q projection output shape [rows, n_heads * head_dim]
    pub(crate) q_out: Vec<f32>,
    /// K projection output shape [rows, n_kv_heads * head_dim]
    pub(crate) k_out: Vec<f32>,
    /// V projection output shape [rows, n_kv_heads * head_dim]
    pub(crate) v_out: Vec<f32>,
    /// Attention output projection shape [rows, embed_dim]
    pub(crate) attn_out: Vec<f32>,
    /// Gate projection output shape [rows, inter_dim]
    pub(crate) gate_out: Vec<f32>,
    /// Up projection output shape [rows, inter_dim]
    pub(crate) up_out: Vec<f32>,
    /// Silu(gate) * up (gated activation) shape [rows, inter_dim]
    pub(crate) gated_out: Vec<f32>,
    /// Down projection (MLP output) shape [rows, embed_dim]
    pub(crate) mlp_out: Vec<f32>,

    // -- config for bounds checking --
    embed_dim: usize,
    inter_dim: usize,
    q_dim: usize,
    kv_dim: usize,
}

impl Workspace {
    /// Allocate a workspace sized for `max_rows` tokens.
    ///
    /// Production decode passes `max_rows = 1`.
    pub fn new(
        max_rows: usize,
        embed_dim: usize,
        inter_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(max_rows > 0, "workspace requires at least one row");
        assert!(embed_dim > 0, "workspace embedding width must be non-zero");
        assert!(inter_dim > 0, "workspace MLP width must be non-zero");
        assert!(n_heads > 0, "workspace query-head count must be non-zero");
        assert!(n_kv_heads > 0, "workspace KV-head count must be non-zero");
        assert!(head_dim > 0, "workspace head width must be non-zero");
        let q_dim = n_heads
            .checked_mul(head_dim)
            .expect("workspace query width overflow");
        let kv_dim = n_kv_heads
            .checked_mul(head_dim)
            .expect("workspace KV width overflow");
        // Attention output is q_dim (n_heads * head_dim), which may differ
        // from embed_dim in some architectures. Allocate the larger.
        let attn_dim = q_dim.max(embed_dim);

        let cap = |cols: usize| {
            max_rows
                .checked_mul(cols)
                .expect("workspace buffer size overflow")
        };

        Self {
            max_rows,
            norm_out: vec![0.0; cap(embed_dim)],
            residual_out: vec![0.0; cap(embed_dim)],
            q_out: vec![0.0; cap(q_dim)],
            k_out: vec![0.0; cap(kv_dim)],
            v_out: vec![0.0; cap(kv_dim)],
            attn_out: vec![0.0; cap(attn_dim)],
            gate_out: vec![0.0; cap(inter_dim)],
            up_out: vec![0.0; cap(inter_dim)],
            gated_out: vec![0.0; cap(inter_dim)],
            mlp_out: vec![0.0; cap(embed_dim)],
            embed_dim,
            inter_dim,
            q_dim,
            kv_dim,
        }
    }

    // -- accessors that return correctly-sized slices for `rows` tokens --

    #[inline]
    fn slice_len(&self, rows: usize, columns: usize) -> usize {
        assert!(
            rows <= self.max_rows,
            "workspace request for {rows} rows exceeds capacity {}",
            self.max_rows
        );
        rows.checked_mul(columns)
            .expect("workspace slice size overflow")
    }

    #[inline]
    pub fn norm_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.embed_dim);
        &mut self.norm_out[..len]
    }

    #[inline]
    pub fn residual_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.embed_dim);
        &mut self.residual_out[..len]
    }

    #[inline]
    pub fn q_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.q_dim);
        &mut self.q_out[..len]
    }

    #[inline]
    pub fn k_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.kv_dim);
        &mut self.k_out[..len]
    }

    #[inline]
    pub fn v_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.kv_dim);
        &mut self.v_out[..len]
    }

    #[inline]
    pub fn attn_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.embed_dim);
        &mut self.attn_out[..len]
    }

    #[inline]
    pub fn gate_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.inter_dim);
        &mut self.gate_out[..len]
    }

    #[inline]
    pub fn up_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.inter_dim);
        &mut self.up_out[..len]
    }

    #[inline]
    pub fn gated_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.inter_dim);
        &mut self.gated_out[..len]
    }

    #[inline]
    pub fn mlp_slice(&mut self, rows: usize) -> &mut [f32] {
        let len = self.slice_len(rows, self.embed_dim);
        &mut self.mlp_out[..len]
    }

    #[inline]
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    #[inline]
    pub fn inter_dim(&self) -> usize {
        self.inter_dim
    }

    #[inline]
    pub fn q_dim(&self) -> usize {
        self.q_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

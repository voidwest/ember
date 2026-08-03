use crate::quant::{QuantizedWeight, QuantizedWeightVnni};
use crate::tensor::{CpuTensor, TensorError};
use half::f16;
use rayon::prelude::*;
use std::cell::RefCell;

const PARALLEL_ATTENTION_MIN_HEADS: usize = 4;
const PARALLEL_ATTENTION_MIN_WORK: usize = 32_768;

thread_local! {
    static Q8_0_DECODE_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ATTENTION_SCORE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Three tensors produced by projections that share one input.
pub type TensorTriple<T> = (T, T, T);

/// shape metadata for standard causal self-attention.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSpec<'a> {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// optional block boundaries for batched independent sequences.
    /// when set, token i can only attend to positions in the same block.
    /// boundaries[i] is the first token index of the i-th block.
    /// the last implicit boundary is seq_len.
    pub block_boundaries: Option<&'a [usize]>,
}

/// shape metadata for cached causal self-attention.
#[derive(Debug, Clone, Copy)]
pub struct CachedAttentionSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub total_seq_len: usize,
}

/// the core abstraction over compute hardware.
///
/// model code is generic over the backend, so the same transformer
/// implementation works on `CpuBackend`, or any future gpu/accelerator
/// backend, without modification.
///
/// ## scope
///
/// the trait currently abstracts element-wise ops (`add`, `gelu`, `softmax`),
/// linear algebra (`matmul`, `matmul_q8_0`, `add_broadcast`), attention,
/// normalisation (`layer_norm`), shape manipulation (`slice_cols`,
/// `index_select`, `reshape`), and tensor lifecycle (`zeroes`,
/// `load_from_cpu`, `data`, `shape`).
pub trait Backend {
    type Tensor: Clone + Send + Sync;
    type Error: core::error::Error;

    fn zeroes(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// matrix multiply with an on-the-fly dequantized q8_0 weight.
    ///
    /// `x` is a standard f32 tensor `[seq_len, in_features]`; `w` is a
    /// raw q8_0 block-compressed weight with logical shape
    /// `[out_features, in_features]` (reversed from the gguf native order
    /// so q8_0 blocks are contiguous per output feature).  the weight is
    /// never stored as f32. Activations are quantized once per row and all
    /// prompt/decode projections use packed integer dots.
    fn matmul_q8_0(
        &self,
        x: &Self::Tensor,
        w: &QuantizedWeight,
    ) -> Result<Self::Tensor, Self::Error>;

    /// matrix multiply with an on-the-fly dequantized q4_k/q6_k weight.
    ///
    /// `x` is a standard f32 tensor `[seq_len, in_features]`; `w` is a
    /// raw super-block-compressed weight with logical shape
    /// `[out_features, in_features]` (GGUF dims reversed, blocks
    /// contiguous per output feature). The weight is never stored as f32;
    /// dequantization happens at block granularity inside the kernel.
    fn matmul_k(
        &self,
        x: &Self::Tensor,
        w: &crate::quant_k::KQuantWeight,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Apply two Q8_0 projections to the same activations.
    ///
    /// Backends may override this to share activation packing and scheduling.
    /// The default preserves compatibility for backends without a fused path.
    fn matmul_q8_0_pair(
        &self,
        x: &Self::Tensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        Ok((self.matmul_q8_0(x, first)?, self.matmul_q8_0(x, second)?))
    }

    /// Apply two projections using an optional packed Q8_0 representation.
    ///
    /// `None` asks the caller to use its generic Q8_0 path. This keeps packed
    /// scheduling an optional CPU optimization without making alternate
    /// backends implement or understand the CPU-specific layout.
    #[allow(clippy::type_complexity)]
    fn matmul_q8_0_packed_pair(
        &self,
        _x: &Self::Tensor,
        _first: &QuantizedWeightVnni,
        _second: &QuantizedWeightVnni,
    ) -> Result<Option<(Self::Tensor, Self::Tensor)>, Self::Error> {
        Ok(None)
    }

    /// Apply three Q8_0 projections to the same activations.
    ///
    /// This is primarily used by separate Q/K/V projection weights.
    fn matmul_q8_0_triple(
        &self,
        x: &Self::Tensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
        third: &QuantizedWeight,
    ) -> Result<TensorTriple<Self::Tensor>, Self::Error> {
        Ok((
            self.matmul_q8_0(x, first)?,
            self.matmul_q8_0(x, second)?,
            self.matmul_q8_0(x, third)?,
        ))
    }

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn softmax(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn layer_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, Self::Error>;
    fn index_select(
        &self,
        tensor: &Self::Tensor,
        index: usize,
    ) -> Result<Self::Tensor, Self::Error>;
    /// select one row from a 2D tensor while preserving a 2D `[1, cols]` shape.
    fn row_as_2d(&self, tensor: &Self::Tensor, index: usize) -> Result<Self::Tensor, Self::Error>;
    fn assign_row(&self, dst: &mut Self::Tensor, index: usize, src: &Self::Tensor);
    fn assign_row_from_table(
        &self,
        dst: &mut Self::Tensor,
        dst_index: usize,
        table: &Self::Tensor,
        table_index: usize,
    ) -> Result<(), Self::Error>;
    /// Dequantize one row from a Q8_0 table directly into `dst`.
    fn assign_row_from_q8_0(
        &self,
        dst: &mut Self::Tensor,
        dst_index: usize,
        table: &QuantizedWeight,
        table_index: usize,
    ) -> Result<(), Self::Error>;
    /// Dequantize one row from a K-quant table directly into `dst`.
    fn assign_row_from_k(
        &self,
        dst: &mut Self::Tensor,
        dst_index: usize,
        table: &crate::quant_k::KQuantWeight,
        table_index: usize,
    ) -> Result<(), Self::Error>;
    fn assign_row_sum_from_tables(
        &self,
        dst: &mut Self::Tensor,
        dst_index: usize,
        lhs_table: &Self::Tensor,
        lhs_index: usize,
        rhs_table: &Self::Tensor,
        rhs_index: usize,
    ) -> Result<(), Self::Error>;
    fn slice_cols(&self, x: &Self::Tensor, start: usize, end: usize) -> Self::Tensor;
    fn shape<'a>(&self, x: &'a Self::Tensor) -> &'a [usize];
    fn data<'a>(&self, x: &'a Self::Tensor) -> &'a [f32];
    fn scale_in_place(&self, x: &mut Self::Tensor, scale: f32);
    /// Apply Gemma-style final-logit softcapping.
    ///
    /// The default preserves compatibility for non-CPU backends. Backends
    /// with mutable tensor storage can override this to reuse `x`.
    fn softcap_in_place(&self, x: &mut Self::Tensor, cap: f32) -> Result<(), Self::Error> {
        let shape = self.shape(x).to_vec();
        let data = self
            .data(x)
            .iter()
            .map(|&value| (value / cap).tanh() * cap)
            .collect();
        *x = self.load_from_cpu(data, &shape)?;
        Ok(())
    }
    /// load host-side f32 data into a backend tensor.
    fn load_from_cpu(&self, data: Vec<f32>, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn add_broadcast(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error>;
    fn causal_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        spec: AttentionSpec,
    ) -> Result<Self::Tensor, Self::Error>;
    fn cached_causal_attention(
        &self,
        q: &Self::Tensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
    ) -> Result<Self::Tensor, Self::Error>;
    fn cached_causal_attention_with_scratch(
        &self,
        q: &Self::Tensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
    ) -> Result<Self::Tensor, Self::Error>;

    // -- llama-family primitives ---------------------------------
    // rms norm and silu are needed by llama model code.
    // `CpuTensor` already implements both; these trait methods
    // expose them through the abstraction so `Llama<CpuBackend>`
    // works today, and a future gpu backend must provide them too.

    /// rms normalization: `x * weight / sqrt(mean(x^2) + eps)`.
    /// llama-family models use this instead of layer norm (no mean subtraction, no bias).
    fn rms_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, Self::Error>;

    /// sigmoid linear unit: `x * sigmoid(x)` = `x / (1 + exp(-x))`.
    /// used in llama's swiglu mlp gate: `silu(gate_proj(x)) * up_proj(x)`.
    fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// element-wise multiplication. both tensors must have the same shape.
    /// used in llama's swiglu gate: `silu(gate) * up`.
    fn elemul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// apply rotary position embeddings to a q or k tensor.
    /// `cos` and `sin` are precomputed tables of shape `[max_seq_len, head_dim]`.
    /// `start_pos` is the absolute position of the first token in this batch.
    fn apply_rotary_emb(
        &self,
        x: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        start_pos: usize,
    ) -> Result<Self::Tensor, Self::Error>;
}

/// a composable unit that runs a forward pass.
///
/// see `Block`, `Mlp`, `Attention`, `LayerNorm` for gpt-2 implementations.
pub trait Module<B: Backend> {
    fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error>;
}

/// the default cpu backend. a zero-sized struct that delegates
/// every operation to `CpuTensor` methods.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CpuError {
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("kernel error: {0}")]
    Kernel(String),
    #[error(transparent)]
    Experiment(#[from] crate::experiments::ExperimentFailure),
}

fn q8_matmul_output_len(x: &CpuTensor, w: &QuantizedWeight) -> Result<(usize, usize), CpuError> {
    if x.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "matmul_q8_0: input must be 2D, got shape {:?}",
            x.shape()
        )));
    }

    let (seq_len, in_features) = (x.shape[0], x.shape[1]);
    if in_features != w.in_features() {
        return Err(CpuError::ShapeMismatch(format!(
            "matmul_q8_0: inner dims must match (got {} vs {})",
            in_features,
            w.in_features()
        )));
    }
    let output_len = seq_len.checked_mul(w.out_features()).ok_or_else(|| {
        CpuError::ShapeMismatch("matmul_q8_0: output shape product overflow".into())
    })?;
    Ok((seq_len, output_len))
}

fn k_matmul_output_len(
    x: &CpuTensor,
    w: &crate::quant_k::KQuantWeight,
) -> Result<(usize, usize), CpuError> {
    if x.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "matmul_k: input must be 2D, got shape {:?}",
            x.shape()
        )));
    }
    let (seq_len, in_features) = (x.shape[0], x.shape[1]);
    if in_features != w.in_features() {
        return Err(CpuError::ShapeMismatch(format!(
            "matmul_k: inner dims must match (got {} vs {})",
            in_features,
            w.in_features()
        )));
    }
    let output_len = seq_len
        .checked_mul(w.out_features())
        .ok_or_else(|| CpuError::ShapeMismatch("matmul_k: output shape product overflow".into()))?;
    Ok((seq_len, output_len))
}

fn assert_q8_projection_layout(
    src: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    dst: &[f32],
    operation: &str,
) {
    let expected_src = rows
        .checked_mul(in_features)
        .unwrap_or_else(|| panic!("{operation}: input shape product overflow"));
    let expected_dst = rows
        .checked_mul(out_features)
        .unwrap_or_else(|| panic!("{operation}: output shape product overflow"));
    assert_eq!(
        src.len(),
        expected_src,
        "{operation}: source length does not match rows * in_features"
    );
    assert_eq!(
        dst.len(),
        expected_dst,
        "{operation}: destination length does not match rows * out_features"
    );
}

impl CpuBackend {
    /// Quantize flat f32 activations `src` (shape `[rows, in_features]`) and
    /// compute `dst = src × w` using packed Q8_0 integer dots. Writes into the
    /// pre-allocated `dst` slice, which must have length `rows * w.out_features()`.
    ///
    /// This is the zero-alloc variant of `matmul_q8_0` — it reuses the
    /// thread-local quantized-activation buffer and writes directly into the
    /// caller's output slice instead of wrapping a new `Vec`.
    pub fn matmul_q8_0_into(&self, src: &[f32], rows: usize, w: &QuantizedWeight, dst: &mut [f32]) {
        assert_q8_projection_layout(
            src,
            rows,
            w.in_features(),
            w.out_features(),
            dst,
            "matmul_q8_0_into",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            if rows == 1 {
                crate::simd::matmul_q8_0_decode(&input, w, dst);
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w, dst);
            }
        });
    }

    /// Instrumented single-row projection used only by operator profiling.
    ///
    /// The returned duration covers the matrix kernel, not activation
    /// quantization, so it is directly comparable with the pair/triple timings.
    pub fn matmul_q8_0_into_timed(
        &self,
        src: &[f32],
        w: &QuantizedWeight,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert_q8_projection_layout(
            src,
            1,
            w.in_features(),
            w.out_features(),
            dst,
            "matmul_q8_0_into_timed",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let start = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w, dst);
            start.elapsed()
        })
    }

    /// Q8_0 decode projection using the cache-friendly interleaved layout.
    ///
    /// The quantized activation stays borrowed inside the thread-local closure,
    /// so no fabricated lifetime or raw slice construction is required.
    pub fn matmul_q8_0_interleaved_into(
        &self,
        src: &[f32],
        w: &crate::quant::QuantizedWeightInterleaved,
        dst: &mut [f32],
    ) {
        assert!(crate::simd::interleaved_q8_0_supported());
        assert_eq!(src.len(), w.in_features());
        assert_eq!(dst.len(), w.out_features());
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            crate::simd::matmul_q8_0_decode_interleaved_parallel(&input, w, dst);
        });
    }

    /// Instrumented interleaved projection used only by operator profiling.
    pub fn matmul_q8_0_interleaved_into_timed(
        &self,
        src: &[f32],
        w: &crate::quant::QuantizedWeightInterleaved,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert!(crate::simd::interleaved_q8_0_supported());
        assert_eq!(src.len(), w.in_features());
        assert_eq!(dst.len(), w.out_features());
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let start = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_interleaved_parallel(&input, w, dst);
            start.elapsed()
        })
    }

    /// Batch-1 projection over a 16-output packed Q8_0 weight.
    pub fn matmul_q8_0_packed_into(
        &self,
        src: &[f32],
        weight: &QuantizedWeightVnni,
        dst: &mut [f32],
    ) {
        assert_q8_projection_layout(
            src,
            1,
            weight.in_features(),
            weight.out_features(),
            dst,
            "matmul_q8_0_packed_into",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, weight, dst);
        });
    }

    /// Instrumented packed projection used by the optional operator profiler.
    pub fn matmul_q8_0_packed_into_timed(
        &self,
        src: &[f32],
        weight: &QuantizedWeightVnni,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert_q8_projection_layout(
            src,
            1,
            weight.in_features(),
            weight.out_features(),
            dst,
            "matmul_q8_0_packed_into_timed",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, weight, dst);
            started.elapsed()
        })
    }

    /// Two packed projections sharing one activation quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_packed_pair_into(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
    ) {
        assert_q8_projection_layout(
            src,
            1,
            first.in_features(),
            first.out_features(),
            first_dst,
            "matmul_q8_0_packed_pair_into(first)",
        );
        assert_q8_projection_layout(
            src,
            1,
            second.in_features(),
            second.out_features(),
            second_dst,
            "matmul_q8_0_packed_pair_into(second)",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst);
        });
    }

    /// Instrumented packed pair sharing one activation quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_packed_pair_into_timed(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
    ) -> [std::time::Duration; 2] {
        assert_q8_projection_layout(
            src,
            1,
            first.in_features(),
            first.out_features(),
            first_dst,
            "matmul_q8_0_packed_pair_into_timed(first)",
        );
        assert_q8_projection_layout(
            src,
            1,
            second.in_features(),
            second.out_features(),
            second_dst,
            "matmul_q8_0_packed_pair_into_timed(second)",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let first_started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst);
            let first_elapsed = first_started.elapsed();
            let second_started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst);
            [first_elapsed, second_started.elapsed()]
        })
    }

    /// Three packed projections sharing one activation quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_packed_triple_into(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        third: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
        third_dst: &mut [f32],
    ) {
        for (weight, dst, operation) in [
            (first, &*first_dst, "matmul_q8_0_packed_triple_into(first)"),
            (
                second,
                &*second_dst,
                "matmul_q8_0_packed_triple_into(second)",
            ),
            (third, &*third_dst, "matmul_q8_0_packed_triple_into(third)"),
        ] {
            assert_q8_projection_layout(
                src,
                1,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst);
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, third, third_dst);
        });
    }

    /// Instrumented packed triple sharing one activation quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_packed_triple_into_timed(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        third: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
        third_dst: &mut [f32],
    ) -> [std::time::Duration; 3] {
        for (weight, dst, operation) in [
            (
                first,
                &*first_dst,
                "matmul_q8_0_packed_triple_into_timed(first)",
            ),
            (
                second,
                &*second_dst,
                "matmul_q8_0_packed_triple_into_timed(second)",
            ),
            (
                third,
                &*third_dst,
                "matmul_q8_0_packed_triple_into_timed(third)",
            ),
        ] {
            assert_q8_projection_layout(
                src,
                1,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let first_started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst);
            let first_elapsed = first_started.elapsed();
            let second_started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst);
            let second_elapsed = second_started.elapsed();
            let third_started = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode_packed16_parallel(&input, third, third_dst);
            [first_elapsed, second_elapsed, third_started.elapsed()]
        })
    }

    /// Fused dual Q8_0 projection (gate + up): quantize `src` once, compute
    /// both projections in one pass.
    pub fn matmul_q8_0_pair_into(
        &self,
        src: &[f32],
        rows: usize,
        w_a: &QuantizedWeight,
        w_b: &QuantizedWeight,
        dst_a: &mut [f32],
        dst_b: &mut [f32],
    ) {
        assert_q8_projection_layout(
            src,
            rows,
            w_a.in_features(),
            w_a.out_features(),
            dst_a,
            "matmul_q8_0_pair_into(first)",
        );
        assert_q8_projection_layout(
            src,
            rows,
            w_b.in_features(),
            w_b.out_features(),
            dst_b,
            "matmul_q8_0_pair_into(second)",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            if rows == 1 {
                crate::simd::matmul_q8_0_decode(&input, w_a, dst_a);
                crate::simd::matmul_q8_0_decode(&input, w_b, dst_b);
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w_a, dst_a);
                crate::simd::matmul_q8_0_batch(&input, rows, w_b, dst_b);
            }
        });
    }

    /// Instrumented fused-input pair projection used only by operator profiling.
    pub fn matmul_q8_0_pair_into_timed(
        &self,
        src: &[f32],
        w_a: &QuantizedWeight,
        w_b: &QuantizedWeight,
        dst_a: &mut [f32],
        dst_b: &mut [f32],
    ) -> [std::time::Duration; 2] {
        assert_q8_projection_layout(
            src,
            1,
            w_a.in_features(),
            w_a.out_features(),
            dst_a,
            "matmul_q8_0_pair_into_timed(first)",
        );
        assert_q8_projection_layout(
            src,
            1,
            w_b.in_features(),
            w_b.out_features(),
            dst_b,
            "matmul_q8_0_pair_into_timed(second)",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let start_a = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w_a, dst_a);
            let elapsed_a = start_a.elapsed();
            let start_b = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w_b, dst_b);
            [elapsed_a, start_b.elapsed()]
        })
    }

    /// Fused Q/K/V decode projection: quantize `src` once, compute all three
    /// Q8_0 projections in one pass. For seq_len=1 decode this saves two
    /// activation quantization passes and their scheduling overhead.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_triple_into(
        &self,
        src: &[f32],
        rows: usize,
        w_q: &QuantizedWeight,
        w_k: &QuantizedWeight,
        w_v: &QuantizedWeight,
        dst_q: &mut [f32],
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) {
        for (weight, dst, operation) in [
            (w_q, &*dst_q, "matmul_q8_0_triple_into(q)"),
            (w_k, &*dst_k, "matmul_q8_0_triple_into(k)"),
            (w_v, &*dst_v, "matmul_q8_0_triple_into(v)"),
        ] {
            assert_q8_projection_layout(
                src,
                rows,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            if rows == 1 {
                crate::simd::matmul_q8_0_decode(&input, w_q, dst_q);
                crate::simd::matmul_q8_0_decode(&input, w_k, dst_k);
                crate::simd::matmul_q8_0_decode(&input, w_v, dst_v);
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w_q, dst_q);
                crate::simd::matmul_q8_0_batch(&input, rows, w_k, dst_k);
                crate::simd::matmul_q8_0_batch(&input, rows, w_v, dst_v);
            }
        });
    }

    /// Explicit serial Q/K/V projection with one shared activation packing.
    #[allow(clippy::too_many_arguments)]
    /// Instrumented fused-input triple projection used only by operator
    /// profiling.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_triple_into_timed(
        &self,
        src: &[f32],
        w_q: &QuantizedWeight,
        w_k: &QuantizedWeight,
        w_v: &QuantizedWeight,
        dst_q: &mut [f32],
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> [std::time::Duration; 3] {
        for (weight, dst, operation) in [
            (w_q, &*dst_q, "matmul_q8_0_triple_into_timed(q)"),
            (w_k, &*dst_k, "matmul_q8_0_triple_into_timed(k)"),
            (w_v, &*dst_v, "matmul_q8_0_triple_into_timed(v)"),
        ] {
            assert_q8_projection_layout(
                src,
                1,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(src, &mut input);
            let start_q = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w_q, dst_q);
            let elapsed_q = start_q.elapsed();
            let start_k = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w_k, dst_k);
            let elapsed_k = start_k.elapsed();
            let start_v = std::time::Instant::now();
            crate::simd::matmul_q8_0_decode(&input, w_v, dst_v);
            [elapsed_q, elapsed_k, start_v.elapsed()]
        })
    }

    /// Cached causal attention into caller-owned storage.
    ///
    /// This is the allocation-free form used by the single-token Llama decode
    /// path. `q` and `out` are flat row-major buffers with width
    /// `n_heads * head_dim`.
    pub fn cached_causal_attention_into(
        &self,
        q: &[f32],
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
        out: &mut [f32],
    ) -> Result<(), CpuError> {
        let embed_dim = spec
            .n_heads
            .checked_mul(spec.head_dim)
            .ok_or_else(|| CpuError::ShapeMismatch("attention width overflow".into()))?;
        if embed_dim == 0 || !q.len().is_multiple_of(embed_dim) {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: q len {} is not divisible by width {}",
                q.len(),
                embed_dim
            )));
        }
        let seq_len = q.len() / embed_dim;
        if out.len() != q.len() {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: output len {} != q len {}",
                out.len(),
                q.len()
            )));
        }
        if spec.total_seq_len < seq_len || spec.total_seq_len > spec.max_seq_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: total_seq_len {} invalid for seq_len {} and max_seq_len {}",
                spec.total_seq_len, seq_len, spec.max_seq_len
            )));
        }
        let cache_len = spec
            .n_kv_heads
            .checked_mul(spec.max_seq_len)
            .and_then(|len| len.checked_mul(spec.head_dim))
            .ok_or_else(|| CpuError::ShapeMismatch("attention cache length overflow".into()))?;
        if cached_k.len() != cache_len || cached_v.len() != cache_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: cache len mismatch, got k={} v={}, expected {}",
                cached_k.len(),
                cached_v.len(),
                cache_len
            )));
        }

        let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
        let scale = (spec.head_dim as f32).sqrt().recip();
        let cache_head_stride = spec.max_seq_len * spec.head_dim;
        let parallel_attention =
            should_parallel_attention(spec.n_heads, seq_len, spec.total_seq_len, spec.head_dim);

        out.fill(0.0);
        if parallel_attention && seq_len > 1 {
            out.par_chunks_mut(embed_dim)
                .enumerate()
                .for_each(|(i, out_row)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        let max_j = spec.total_seq_len - seq_len + i;
                        qk_row.resize(max_j + 1, 0.0);
                        for h in 0..spec.n_heads {
                            let q_head_offset = h * spec.head_dim;
                            let kv_h = h / n_repeat;
                            let q_idx = i * embed_dim + q_head_offset;

                            for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                                let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                                *slot = crate::simd::dot_product_f16(
                                    &q[q_idx..q_idx + spec.head_dim],
                                    &cached_k[k_offset..k_offset + spec.head_dim],
                                ) * scale;
                            }

                            softmax_prefix(qk_row.as_mut_slice(), max_j + 1);
                            let head_out =
                                &mut out_row[q_head_offset..q_head_offset + spec.head_dim];
                            for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                                if weight == 0.0 {
                                    continue;
                                }
                                let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                                crate::simd::weighted_add_f16(
                                    head_out,
                                    &cached_v[v_offset..v_offset + spec.head_dim],
                                    weight,
                                );
                            }
                        }
                    });
                });
            return Ok(());
        }

        if parallel_attention {
            debug_assert_eq!(seq_len, 1);
            out.par_chunks_mut(spec.head_dim)
                .enumerate()
                .for_each(|(h, head_out)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        let q_head_offset = h * spec.head_dim;
                        let kv_h = h / n_repeat;
                        let max_j = spec.total_seq_len - 1;
                        qk_row.resize(max_j + 1, 0.0);

                        for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                            let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            *slot = crate::simd::dot_product_f16(
                                &q[q_head_offset..q_head_offset + spec.head_dim],
                                &cached_k[k_offset..k_offset + spec.head_dim],
                            ) * scale;
                        }

                        softmax_prefix(qk_row.as_mut_slice(), max_j + 1);
                        for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                            if weight == 0.0 {
                                continue;
                            }
                            let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            crate::simd::weighted_add_f16(
                                head_out,
                                &cached_v[v_offset..v_offset + spec.head_dim],
                                weight,
                            );
                        }
                    });
                });
            return Ok(());
        }

        if qk_row.capacity() < spec.max_seq_len {
            qk_row.reserve(spec.max_seq_len - qk_row.capacity());
        }
        for h in 0..spec.n_heads {
            let q_head_offset = h * spec.head_dim;
            let kv_h = h / n_repeat;
            for i in 0..seq_len {
                let max_j = spec.total_seq_len - seq_len + i;
                qk_row.resize(max_j + 1, 0.0);
                let q_idx = i * embed_dim + q_head_offset;

                for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                    let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                    *slot = crate::simd::dot_product_f16(
                        &q[q_idx..q_idx + spec.head_dim],
                        &cached_k[k_offset..k_offset + spec.head_dim],
                    ) * scale;
                }

                softmax_prefix(qk_row.as_mut_slice(), max_j + 1);
                let out_offset = i * embed_dim + q_head_offset;
                for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                    if weight == 0.0 {
                        continue;
                    }
                    let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                    crate::simd::weighted_add_f16(
                        &mut out[out_offset..out_offset + spec.head_dim],
                        &cached_v[v_offset..v_offset + spec.head_dim],
                        weight,
                    );
                }
            }
        }
        Ok(())
    }
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    type Error = CpuError;

    fn zeroes(&self, shape: &[usize]) -> Result<CpuTensor, CpuError> {
        Ok(CpuTensor::zeroes(shape))
    }

    fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(a.matmul(b))
    }

    fn matmul_q8_0(&self, x: &CpuTensor, w: &QuantizedWeight) -> Result<CpuTensor, CpuError> {
        let (seq_len, output_len) = q8_matmul_output_len(x, w)?;
        let mut out = vec![0.0f32; output_len];
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            if seq_len == 1 {
                crate::simd::matmul_q8_0_decode(&input, w, &mut out);
            } else {
                crate::simd::matmul_q8_0_batch(&input, seq_len, w, &mut out);
            }
        });
        Ok(CpuTensor::from_data(vec![seq_len, w.out_features()], out))
    }

    fn matmul_k(
        &self,
        x: &CpuTensor,
        w: &crate::quant_k::KQuantWeight,
    ) -> Result<CpuTensor, CpuError> {
        let (seq_len, output_len) = k_matmul_output_len(x, w)?;
        let mut out = vec![0.0f32; output_len];
        crate::k_matmul::matmul_k_into(x.data(), seq_len, w, &mut out).map_err(CpuError::Kernel)?;
        Ok(CpuTensor::from_data(vec![seq_len, w.out_features()], out))
    }

    fn matmul_q8_0_pair(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
    ) -> Result<(CpuTensor, CpuTensor), CpuError> {
        let (seq_len, first_len) = q8_matmul_output_len(x, first)?;
        let (_, second_len) = q8_matmul_output_len(x, second)?;
        let mut first_out = vec![0.0f32; first_len];
        let mut second_out = vec![0.0f32; second_len];
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            if seq_len == 1 {
                crate::simd::matmul_q8_0_decode(&input, first, &mut first_out);
                crate::simd::matmul_q8_0_decode(&input, second, &mut second_out);
            } else {
                crate::simd::matmul_q8_0_batch(&input, seq_len, first, &mut first_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, second, &mut second_out);
            }
        });
        Ok((
            CpuTensor::from_data(vec![seq_len, first.out_features()], first_out),
            CpuTensor::from_data(vec![seq_len, second.out_features()], second_out),
        ))
    }

    fn matmul_q8_0_packed_pair(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
    ) -> Result<Option<(CpuTensor, CpuTensor)>, CpuError> {
        if x.ndim() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: input must be 2D, got shape {:?}",
                x.shape()
            )));
        }
        let (rows, in_features) = (x.shape()[0], x.shape()[1]);
        if in_features != first.in_features() || in_features != second.in_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: inner dims must match (got {}, {} and {})",
                in_features,
                first.in_features(),
                second.in_features()
            )));
        }
        if first.out_features() != second.out_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: output dims must match (got {} and {})",
                first.out_features(),
                second.out_features()
            )));
        }

        // Real-shape A/B measurements show the packed matrix-vector schedule
        // wins for decode and very short prompts, while the generic tiled
        // batch kernel is faster from eight rows onward. Multi-row packed
        // dispatch also needs enough workers to offset its per-row scheduling.
        if rows > 6 || (rows > 1 && rayon::current_num_threads() < 4) {
            return Ok(None);
        }

        let output_features = first.out_features();
        let output_len = rows.checked_mul(output_features).ok_or_else(|| {
            CpuError::ShapeMismatch("matmul_q8_0_packed_pair: output size overflow".into())
        })?;
        let mut first_out = vec![0.0; output_len];
        let mut second_out = vec![0.0; output_len];
        let encoded_row_len = crate::quant::q8_0_encoded_len(in_features);
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            for row in 0..rows {
                let input_row = &input[row * encoded_row_len..(row + 1) * encoded_row_len];
                let output_range = row * output_features..(row + 1) * output_features;
                crate::simd::matmul_q8_0_decode_packed16_parallel(
                    input_row,
                    first,
                    &mut first_out[output_range.clone()],
                );
                crate::simd::matmul_q8_0_decode_packed16_parallel(
                    input_row,
                    second,
                    &mut second_out[output_range],
                );
            }
        });
        Ok(Some((
            CpuTensor::from_data(vec![rows, output_features], first_out),
            CpuTensor::from_data(vec![rows, output_features], second_out),
        )))
    }

    fn matmul_q8_0_triple(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
        third: &QuantizedWeight,
    ) -> Result<(CpuTensor, CpuTensor, CpuTensor), CpuError> {
        let (seq_len, first_len) = q8_matmul_output_len(x, first)?;
        let (_, second_len) = q8_matmul_output_len(x, second)?;
        let (_, third_len) = q8_matmul_output_len(x, third)?;
        let mut first_out = vec![0.0f32; first_len];
        let mut second_out = vec![0.0f32; second_len];
        let mut third_out = vec![0.0f32; third_len];
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            if seq_len == 1 {
                crate::simd::matmul_q8_0_decode(&input, first, &mut first_out);
                crate::simd::matmul_q8_0_decode(&input, second, &mut second_out);
                crate::simd::matmul_q8_0_decode(&input, third, &mut third_out);
            } else {
                crate::simd::matmul_q8_0_batch(&input, seq_len, first, &mut first_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, second, &mut second_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, third, &mut third_out);
            }
        });
        Ok((
            CpuTensor::from_data(vec![seq_len, first.out_features()], first_out),
            CpuTensor::from_data(vec![seq_len, second.out_features()], second_out),
            CpuTensor::from_data(vec![seq_len, third.out_features()], third_out),
        ))
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(a.add(b))
    }

    fn softmax(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.softmax())
    }
    fn gelu(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.gelu())
    }

    fn layer_norm(
        &self,
        x: &CpuTensor,
        weight: &CpuTensor,
        bias: &CpuTensor,
        eps: f32,
    ) -> Result<CpuTensor, CpuError> {
        Ok(x.layer_norm(weight, bias, eps))
    }
    fn index_select(&self, x: &CpuTensor, index: usize) -> Result<CpuTensor, Self::Error> {
        Ok(x.index_select(index)?)
    }
    fn row_as_2d(&self, x: &CpuTensor, index: usize) -> Result<CpuTensor, Self::Error> {
        Ok(x.row_as_2d(index)?)
    }
    fn assign_row(&self, dst: &mut CpuTensor, index: usize, src: &CpuTensor) {
        dst.assign_row(index, src);
    }
    fn assign_row_from_table(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &CpuTensor,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        assign_row_from_table_cpu(dst, dst_index, table, table_index)
    }
    fn assign_row_from_q8_0(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &QuantizedWeight,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        if dst.ndim() != 2
            || dst_index >= dst.shape()[0]
            || table_index >= table.out_features()
            || dst.shape()[1] != table.in_features()
        {
            return Err(CpuError::ShapeMismatch(format!(
                "assign_row_from_q8_0: dst={:?}, dst_index={}, table=[{}, {}], table_index={}",
                dst.shape(),
                dst_index,
                table.out_features(),
                table.in_features(),
                table_index
            )));
        }
        let cols = table.in_features();
        let start = dst_index * cols;
        table.dequantize_row(table_index, &mut dst.data_mut()[start..start + cols]);
        Ok(())
    }
    fn assign_row_from_k(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &crate::quant_k::KQuantWeight,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        if dst.ndim() != 2
            || dst_index >= dst.shape()[0]
            || table_index >= table.out_features()
            || dst.shape()[1] != table.in_features()
        {
            return Err(CpuError::ShapeMismatch(format!(
                "assign_row_from_k: dst={:?}, dst_index={}, table=[{}, {}], table_index={}",
                dst.shape(),
                dst_index,
                table.out_features(),
                table.in_features(),
                table_index
            )));
        }
        let cols = table.in_features();
        let start = dst_index * cols;
        table.dequantize_row(table_index, &mut dst.data_mut()[start..start + cols]);
        Ok(())
    }
    fn assign_row_sum_from_tables(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        lhs_table: &CpuTensor,
        lhs_index: usize,
        rhs_table: &CpuTensor,
        rhs_index: usize,
    ) -> Result<(), Self::Error> {
        assign_row_sum_from_tables_cpu(dst, dst_index, lhs_table, lhs_index, rhs_table, rhs_index)
    }
    fn slice_cols(&self, x: &Self::Tensor, start: usize, end: usize) -> Self::Tensor {
        x.slice_cols(start, end)
    }
    fn shape<'a>(&self, x: &'a CpuTensor) -> &'a [usize] {
        x.shape()
    }
    fn data<'a>(&self, x: &'a Self::Tensor) -> &'a [f32] {
        x.data()
    }
    fn scale_in_place(&self, x: &mut CpuTensor, scale: f32) {
        for value in x.data_mut() {
            *value *= scale;
        }
    }
    fn softcap_in_place(&self, x: &mut CpuTensor, cap: f32) -> Result<(), CpuError> {
        for value in x.data_mut() {
            *value = (*value / cap).tanh() * cap;
        }
        Ok(())
    }
    fn load_from_cpu(&self, data: Vec<f32>, shape: &[usize]) -> Result<CpuTensor, Self::Error> {
        Ok(CpuTensor::from_data(shape.to_vec(), data))
    }
    fn add_broadcast(&self, x: &CpuTensor, bias: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.add_broadcast(bias))
    }

    fn causal_attention(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        spec: AttentionSpec,
    ) -> Result<CpuTensor, CpuError> {
        let seq_len = validate_attention_inputs(q, k, v, spec)?;
        let embed_dim = spec.n_heads * spec.head_dim;
        let kv_dim = spec.n_kv_heads * spec.head_dim;
        let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
        let scale = (spec.head_dim as f32).sqrt().recip();

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // precompute per-position block start for block-diagonal masking
        let block_start: Vec<usize> = if let Some(boundaries) = spec.block_boundaries {
            let mut starts = vec![0usize; seq_len];
            let mut current_start = 0usize;
            let mut bi = 0usize;
            for (i, s) in starts.iter_mut().enumerate() {
                while bi < boundaries.len() && boundaries[bi] <= i {
                    current_start = boundaries[bi];
                    bi += 1;
                }
                *s = current_start;
            }
            starts
        } else {
            vec![0usize; seq_len]
        };
        let use_blocks = spec.block_boundaries.is_some();

        let parallel_attention =
            should_parallel_attention(spec.n_heads, seq_len, seq_len, spec.head_dim);
        if parallel_attention && seq_len > 1 {
            // Prefill rows are independent once the causal range is known.
            // Writing one complete output row per Rayon job avoids the old
            // per-head output buffers and the full-size scatter copy.
            let mut out = vec![0.0f32; seq_len * embed_dim];
            out.par_chunks_mut(embed_dim)
                .enumerate()
                .for_each(|(i, out_row)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        qk_row.resize(seq_len, 0.0);
                        let start = if use_blocks { block_start[i] } else { 0 };
                        let ctx_len = i - start + 1;
                        for h in 0..spec.n_heads {
                            let q_head_offset = h * spec.head_dim;
                            let kv_h = h / n_repeat;
                            let kv_head_offset = kv_h * spec.head_dim;
                            let q_idx = i * embed_dim + q_head_offset;

                            for j in start..=i {
                                let k_idx = j * kv_dim + kv_head_offset;
                                qk_row[j - start] = crate::simd::dot_product(
                                    &q_data[q_idx..q_idx + spec.head_dim],
                                    &k_data[k_idx..k_idx + spec.head_dim],
                                ) * scale;
                            }

                            softmax_prefix(qk_row.as_mut_slice(), ctx_len);
                            let head_out =
                                &mut out_row[q_head_offset..q_head_offset + spec.head_dim];
                            for j in start..=i {
                                let weight = qk_row[j - start];
                                if weight == 0.0 {
                                    continue;
                                }
                                let v_offset = j * kv_dim + kv_head_offset;
                                crate::simd::weighted_add(
                                    head_out,
                                    &v_data[v_offset..v_offset + spec.head_dim],
                                    weight,
                                );
                            }
                        }
                    });
                });
            return Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out));
        }

        if parallel_attention {
            let heads = (0..spec.n_heads)
                .into_par_iter()
                .map(|h| {
                    let q_head_offset = h * spec.head_dim;
                    let kv_h = h / n_repeat;
                    let kv_head_offset = kv_h * spec.head_dim;
                    let mut head_out = vec![0.0f32; seq_len * spec.head_dim];
                    let mut qk_row = vec![0.0f32; seq_len];

                    for (i, &start) in block_start.iter().enumerate() {
                        let q_idx = i * embed_dim + q_head_offset;
                        let start = if use_blocks { start } else { 0 };
                        let ctx_len = i - start + 1;

                        for j in start..=i {
                            let k_idx = j * kv_dim + kv_head_offset;
                            let dot = crate::simd::dot_product(
                                &q_data[q_idx..q_idx + spec.head_dim],
                                &k_data[k_idx..k_idx + spec.head_dim],
                            );
                            qk_row[j - start] = dot * scale;
                        }

                        softmax_prefix(&mut qk_row, ctx_len);

                        let head_offset = i * spec.head_dim;
                        for j in start..=i {
                            let weight = qk_row[j - start];
                            if weight == 0.0 {
                                continue;
                            }
                            let v_offset = j * kv_dim + kv_head_offset;
                            crate::simd::weighted_add(
                                &mut head_out[head_offset..head_offset + spec.head_dim],
                                &v_data[v_offset..v_offset + spec.head_dim],
                                weight,
                            );
                        }
                    }
                    (h, head_out)
                })
                .collect::<Vec<_>>();
            let mut out = vec![0.0f32; seq_len * embed_dim];
            scatter_attention_heads(&heads, seq_len, embed_dim, spec.head_dim, &mut out);
            return Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out));
        }

        let mut out = vec![0.0f32; seq_len * embed_dim];
        let mut qk_row = vec![0.0f32; seq_len];

        for h in 0..spec.n_heads {
            let q_head_offset = h * spec.head_dim;
            let kv_h = h / n_repeat;
            let kv_head_offset = kv_h * spec.head_dim;

            for (i, &start) in block_start.iter().enumerate() {
                let q_idx = i * embed_dim + q_head_offset;
                let start = if use_blocks { start } else { 0 };
                let ctx_len = i - start + 1;

                for j in start..=i {
                    let k_idx = j * kv_dim + kv_head_offset;
                    let dot = crate::simd::dot_product(
                        &q_data[q_idx..q_idx + spec.head_dim],
                        &k_data[k_idx..k_idx + spec.head_dim],
                    );
                    qk_row[j - start] = dot * scale;
                }

                softmax_prefix(&mut qk_row, ctx_len);

                let out_offset = i * embed_dim + q_head_offset;
                for j in start..=i {
                    let weight = qk_row[j - start];
                    if weight == 0.0 {
                        continue;
                    }
                    let v_offset = j * kv_dim + kv_head_offset;
                    crate::simd::weighted_add(
                        &mut out[out_offset..out_offset + spec.head_dim],
                        &v_data[v_offset..v_offset + spec.head_dim],
                        weight,
                    );
                }
            }
        }

        Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out))
    }

    fn cached_causal_attention(
        &self,
        q: &CpuTensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
    ) -> Result<CpuTensor, CpuError> {
        let mut qk_row = Vec::with_capacity(spec.max_seq_len);
        self.cached_causal_attention_with_scratch(q, cached_k, cached_v, spec, &mut qk_row)
    }

    fn cached_causal_attention_with_scratch(
        &self,
        q: &CpuTensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
    ) -> Result<CpuTensor, CpuError> {
        if q.ndim() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: q must be 2D, got {:?}",
                q.shape()
            )));
        }
        let seq_len = q.shape()[0];
        let embed_dim = spec.n_heads * spec.head_dim;
        if q.shape()[1] != embed_dim {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: q width {} != expected {}",
                q.shape()[1],
                embed_dim
            )));
        }
        if spec.total_seq_len < seq_len || spec.total_seq_len > spec.max_seq_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: total_seq_len {} invalid for seq_len {} and max_seq_len {}",
                spec.total_seq_len,
                seq_len,
                spec.max_seq_len
            )));
        }
        let cache_len = spec.n_kv_heads * spec.max_seq_len * spec.head_dim;
        if cached_k.len() != cache_len || cached_v.len() != cache_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: cache len mismatch, got k={} v={}, expected {}",
                cached_k.len(),
                cached_v.len(),
                cache_len
            )));
        }

        let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
        let scale = (spec.head_dim as f32).sqrt().recip();
        let q_data = q.data();
        let cache_head_stride = spec.max_seq_len * spec.head_dim;

        let parallel_attention =
            should_parallel_attention(spec.n_heads, seq_len, spec.total_seq_len, spec.head_dim);
        if parallel_attention && seq_len > 1 {
            let mut out = vec![0.0f32; seq_len * embed_dim];
            out.par_chunks_mut(embed_dim)
                .enumerate()
                .for_each(|(i, out_row)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        let max_j = spec.total_seq_len - seq_len + i;
                        qk_row.resize(max_j + 1, 0.0);
                        for h in 0..spec.n_heads {
                            let q_head_offset = h * spec.head_dim;
                            let kv_h = h / n_repeat;
                            let q_idx = i * embed_dim + q_head_offset;

                            for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                                let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                                *slot = crate::simd::dot_product_f16(
                                    &q_data[q_idx..q_idx + spec.head_dim],
                                    &cached_k[k_offset..k_offset + spec.head_dim],
                                ) * scale;
                            }

                            softmax_prefix(qk_row.as_mut_slice(), max_j + 1);
                            let head_out =
                                &mut out_row[q_head_offset..q_head_offset + spec.head_dim];
                            for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                                if weight == 0.0 {
                                    continue;
                                }
                                let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                                crate::simd::weighted_add_f16(
                                    head_out,
                                    &cached_v[v_offset..v_offset + spec.head_dim],
                                    weight,
                                );
                            }
                        }
                    });
                });
            return Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out));
        }

        if parallel_attention {
            debug_assert_eq!(seq_len, 1);
            let mut out = vec![0.0f32; embed_dim];
            out.par_chunks_mut(spec.head_dim)
                .enumerate()
                .for_each(|(h, head_out)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        let q_head_offset = h * spec.head_dim;
                        let kv_h = h / n_repeat;
                        let max_j = spec.total_seq_len - 1;
                        qk_row.resize(max_j + 1, 0.0);

                        for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                            let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            let dot = crate::simd::dot_product_f16(
                                &q_data[q_head_offset..q_head_offset + spec.head_dim],
                                &cached_k[k_offset..k_offset + spec.head_dim],
                            );
                            *slot = dot * scale;
                        }

                        softmax_prefix(qk_row.as_mut_slice(), max_j + 1);

                        for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                            if weight == 0.0 {
                                continue;
                            }
                            let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            crate::simd::weighted_add_f16(
                                head_out,
                                &cached_v[v_offset..v_offset + spec.head_dim],
                                weight,
                            );
                        }
                    });
                });
            return Ok(CpuTensor::from_data(vec![1, embed_dim], out));
        }

        let mut out = vec![0.0f32; seq_len * embed_dim];
        if qk_row.capacity() < spec.max_seq_len {
            qk_row.reserve(spec.max_seq_len - qk_row.capacity());
        }

        for h in 0..spec.n_heads {
            let q_head_offset = h * spec.head_dim;
            let kv_h = h / n_repeat;

            for i in 0..seq_len {
                let max_j = spec.total_seq_len - seq_len + i;
                qk_row.resize(max_j + 1, 0.0);
                let q_idx = i * embed_dim + q_head_offset;

                for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                    let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                    let dot = crate::simd::dot_product_f16(
                        &q_data[q_idx..q_idx + spec.head_dim],
                        &cached_k[k_offset..k_offset + spec.head_dim],
                    );
                    *slot = dot * scale;
                }

                softmax_prefix(qk_row.as_mut_slice(), max_j + 1);

                let out_offset = i * embed_dim + q_head_offset;
                for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                    if weight == 0.0 {
                        continue;
                    }
                    let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                    crate::simd::weighted_add_f16(
                        &mut out[out_offset..out_offset + spec.head_dim],
                        &cached_v[v_offset..v_offset + spec.head_dim],
                        weight,
                    );
                }
            }
        }

        Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out))
    }

    fn rms_norm(&self, x: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor, CpuError> {
        Ok(x.rms_norm(weight, eps))
    }

    fn silu(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.silu())
    }

    fn elemul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(a.elemul(b))
    }

    fn apply_rotary_emb(
        &self,
        x: &CpuTensor,
        cos: &CpuTensor,
        sin: &CpuTensor,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        Ok(x.apply_rotary_emb(cos, sin, start_pos))
    }
}

fn validate_row_copy_shapes(
    op: &str,
    dst: &CpuTensor,
    dst_index: usize,
    table: &CpuTensor,
    table_index: usize,
) -> Result<usize, CpuError> {
    if dst.ndim() != 2 || table.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: expected 2D dst/table, got dst={:?} table={:?}",
            dst.shape(),
            table.shape()
        )));
    }
    let cols = dst.shape()[1];
    if table.shape()[1] != cols {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: row width mismatch, dst cols {} != table cols {}",
            cols,
            table.shape()[1]
        )));
    }
    if dst_index >= dst.shape()[0] || table_index >= table.shape()[0] {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: row index out of bounds, dst_index={} dst_rows={} table_index={} table_rows={}",
            dst_index,
            dst.shape()[0],
            table_index,
            table.shape()[0]
        )));
    }
    Ok(cols)
}

fn assign_row_from_table_cpu(
    dst: &mut CpuTensor,
    dst_index: usize,
    table: &CpuTensor,
    table_index: usize,
) -> Result<(), CpuError> {
    let cols =
        validate_row_copy_shapes("assign_row_from_table", dst, dst_index, table, table_index)?;
    let dst_start = dst_index * cols;
    let table_start = table_index * cols;
    dst.data_mut()[dst_start..dst_start + cols]
        .copy_from_slice(&table.data()[table_start..table_start + cols]);
    Ok(())
}

fn assign_row_sum_from_tables_cpu(
    dst: &mut CpuTensor,
    dst_index: usize,
    lhs_table: &CpuTensor,
    lhs_index: usize,
    rhs_table: &CpuTensor,
    rhs_index: usize,
) -> Result<(), CpuError> {
    let cols = validate_row_copy_shapes(
        "assign_row_sum_from_tables",
        dst,
        dst_index,
        lhs_table,
        lhs_index,
    )?;
    validate_row_copy_shapes(
        "assign_row_sum_from_tables",
        dst,
        dst_index,
        rhs_table,
        rhs_index,
    )?;
    let dst_start = dst_index * cols;
    let lhs_start = lhs_index * cols;
    let rhs_start = rhs_index * cols;
    let dst_row = &mut dst.data_mut()[dst_start..dst_start + cols];
    let lhs_row = &lhs_table.data()[lhs_start..lhs_start + cols];
    let rhs_row = &rhs_table.data()[rhs_start..rhs_start + cols];
    crate::simd::add(lhs_row, rhs_row, dst_row);
    Ok(())
}

fn validate_attention_inputs(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    spec: AttentionSpec,
) -> Result<usize, CpuError> {
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention expects 2D q/k/v, got q={:?} k={:?} v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let seq_len = q.shape()[0];
    let embed_dim = spec.n_heads * spec.head_dim;
    let kv_dim = spec.n_kv_heads * spec.head_dim;
    if q.shape() != [seq_len, embed_dim] {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention: q shape {:?} != [{}, {}]",
            q.shape(),
            seq_len,
            embed_dim
        )));
    }
    if k.shape() != [seq_len, kv_dim] || v.shape() != [seq_len, kv_dim] {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention: k/v shape mismatch, got k={:?} v={:?}, expected [{}, {}]",
            k.shape(),
            v.shape(),
            seq_len,
            kv_dim
        )));
    }
    validate_gqa(spec.n_heads, spec.n_kv_heads)?;
    Ok(seq_len)
}

fn validate_gqa(n_heads: usize, n_kv_heads: usize) -> Result<usize, CpuError> {
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(CpuError::ShapeMismatch(format!(
            "attention heads must satisfy n_heads % n_kv_heads == 0, got {} and {}",
            n_heads, n_kv_heads
        )));
    }
    Ok(n_heads / n_kv_heads)
}

fn softmax_prefix(row: &mut [f32], len: usize) {
    assert!(
        len > 0 && len <= row.len(),
        "softmax prefix is out of bounds"
    );
    let positive_infinities = row[..len]
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    if positive_infinities > 0 {
        let probability = 1.0 / positive_infinities as f32;
        for slot in row.iter_mut().take(len) {
            *slot = if *slot == f32::INFINITY {
                probability
            } else {
                0.0
            };
        }
        return;
    }
    let max_val = row[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        let uniform = 1.0 / (len as f32);
        for slot in row.iter_mut().take(len) {
            *slot = uniform;
        }
        return;
    }
    let mut sum = 0.0;
    for slot in row.iter_mut().take(len) {
        *slot = (*slot - max_val).exp();
        sum += *slot;
    }
    let inv_sum = sum.recip();
    for slot in row.iter_mut().take(len) {
        *slot *= inv_sum;
    }
}

fn should_parallel_attention(
    n_heads: usize,
    seq_len: usize,
    total_seq_len: usize,
    head_dim: usize,
) -> bool {
    n_heads >= PARALLEL_ATTENTION_MIN_HEADS
        && rayon::current_num_threads() > 1
        && n_heads
            .saturating_mul(seq_len)
            .saturating_mul(total_seq_len)
            .saturating_mul(head_dim)
            >= PARALLEL_ATTENTION_MIN_WORK
}

fn scatter_attention_heads(
    heads: &[(usize, Vec<f32>)],
    seq_len: usize,
    embed_dim: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    for (h, head_out) in heads {
        let q_head_offset = h * head_dim;
        for i in 0..seq_len {
            let dst = i * embed_dim + q_head_offset;
            let src = i * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_q8_weight(out_features: usize, in_features: usize, phase: f32) -> QuantizedWeight {
        let values = (0..out_features * in_features)
            .map(|index| {
                let value = index as f32 * 0.03125 + phase;
                value.sin() * 0.75 + value.cos() * 0.125
            })
            .collect::<Vec<_>>();
        let mut data = Vec::new();
        crate::quant::quantize_q8_0_into(&values, &mut data);
        QuantizedWeight::new(data, vec![out_features, in_features])
    }

    #[test]
    fn packed_pair_matches_generic_for_measured_short_prompt_regime() {
        if !crate::simd::packed_q8_0_vnni_supported() {
            return;
        }

        let backend = CpuBackend;
        let rows = 6;
        let in_features = 64;
        let first = test_q8_weight(64, in_features, 0.25);
        let second = test_q8_weight(64, in_features, 1.5);
        let packed_first = QuantizedWeightVnni::from_quantized(&first);
        let packed_second = QuantizedWeightVnni::from_quantized(&second);
        let input = CpuTensor::from_data(
            vec![rows, in_features],
            (0..rows * in_features)
                .map(|index| (index as f32 * 0.017).sin())
                .collect(),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let expected = pool
            .install(|| backend.matmul_q8_0_pair(&input, &first, &second))
            .unwrap();
        let actual = pool
            .install(|| backend.matmul_q8_0_packed_pair(&input, &packed_first, &packed_second))
            .unwrap()
            .expect("four-thread, six-row input should use packed pair");

        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_pair_defers_to_generic_batch_kernel_after_six_rows() {
        if !crate::simd::packed_q8_0_vnni_supported() {
            return;
        }

        let backend = CpuBackend;
        let first = test_q8_weight(16, 32, 0.25);
        let second = test_q8_weight(16, 32, 1.5);
        let input = CpuTensor::zeroes(&[8, 32]);
        let actual = backend
            .matmul_q8_0_packed_pair(
                &input,
                &QuantizedWeightVnni::from_quantized(&first),
                &QuantizedWeightVnni::from_quantized(&second),
            )
            .unwrap();

        assert!(actual.is_none());
    }

    #[test]
    fn cached_attention_into_matches_tensor_api() {
        let backend = CpuBackend;
        let query = CpuTensor::from_data(vec![1, 4], vec![0.25, -0.5, 0.75, 1.0]);
        let cached_k = [0.5, 0.25, -0.75, 1.0, 1.0, 0.0, 0.5, -0.5].map(f16::from_f32);
        let cached_v = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0].map(f16::from_f32);
        let spec = CachedAttentionSpec {
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 2,
            max_seq_len: 2,
            total_seq_len: 2,
        };

        let expected = backend
            .cached_causal_attention(&query, &cached_k, &cached_v, spec)
            .unwrap();
        let mut scratch = Vec::new();
        let mut actual = vec![0.0; query.len()];
        backend
            .cached_causal_attention_into(
                query.data(),
                &cached_k,
                &cached_v,
                spec,
                &mut scratch,
                &mut actual,
            )
            .unwrap();

        assert_eq!(actual, expected.data());
    }
}

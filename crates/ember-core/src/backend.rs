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
    /// `boundaries[i]` is the first token index of the i-th block.
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

    /// matrix multiply with an f16/bf16 weight kept in its GGUF encoding.
    ///
    /// `x` is `[seq_len, in_features]`; `w` has GGUF dims
    /// `[in_features, out_features]`. Bit-identical to [`Backend::matmul`]
    /// with the f32 weight the eager loader used to build (the widened
    /// values, transposed to `[in_features, out_features]` row-major).
    fn matmul_half(
        &self,
        x: &Self::Tensor,
        w: &crate::half_weight::HalfWeight,
    ) -> Result<Self::Tensor, Self::Error>;

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
    /// activations are packed once per row as Q8_K and consumed by integer dots.
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
    #[allow(
        clippy::type_complexity,
        reason = "the signature spells out each returned tensor; an alias would hide them"
    )]
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

    /// GELU with the tanh approximation (`gelu_pytorch_tanh`) used by
    /// CLIP/SigLIP-family vision towers.
    fn gelu_tanh(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
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
    /// Widen one row of an f16/bf16 table (GGUF dims `[cols, rows]`)
    /// into `dst`; bit-identical to [`Backend::assign_row_from_table`] on
    /// the widened table.
    fn assign_row_from_half(
        &self,
        dst: &mut Self::Tensor,
        dst_index: usize,
        table: &crate::half_weight::HalfWeight,
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
    /// Consume a tensor's host payload. CPU storage moves without a copy;
    /// the default keeps alternate backends source-compatible.
    #[allow(
        clippy::wrong_self_convention,
        reason = "`into_` refers to the tensor argument that the method consumes"
    )]
    fn into_cpu_data(&self, x: Self::Tensor) -> Vec<f32> {
        self.data(&x).to_vec()
    }
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

    // -- buffer-reusing variants ----------------------------------
    // The eager (hooked/generic) prefill keeps one set of scratch tensors
    // per forward call and writes every layer's intermediates into them
    // through these methods. Each default delegates to the allocating
    // operation, so alternate backends stay source-compatible; an override
    // must produce bit-identical values (same kernels, same operation order),
    // only without allocating when `out` already has the capacity.

    /// [`Backend::matmul_q8_0`] written into `out` (resized as needed).
    fn matmul_q8_0_reusing(
        &self,
        x: &Self::Tensor,
        w: &QuantizedWeight,
        out: &mut Self::Tensor,
    ) -> Result<(), Self::Error> {
        *out = self.matmul_q8_0(x, w)?;
        Ok(())
    }

    /// [`Backend::matmul_k`] written into `out` (resized as needed).
    fn matmul_k_reusing(
        &self,
        x: &Self::Tensor,
        w: &crate::quant_k::KQuantWeight,
        out: &mut Self::Tensor,
    ) -> Result<(), Self::Error> {
        *out = self.matmul_k(x, w)?;
        Ok(())
    }

    /// [`Backend::add_broadcast`] applied to `x` in place.
    fn add_broadcast_in_place(
        &self,
        x: &mut Self::Tensor,
        bias: &Self::Tensor,
    ) -> Result<(), Self::Error> {
        *x = self.add_broadcast(x, bias)?;
        Ok(())
    }

    /// [`Backend::add`] accumulated into `x`: `x = x + other`.
    fn add_in_place(&self, x: &mut Self::Tensor, other: &Self::Tensor) -> Result<(), Self::Error> {
        *x = self.add(x, other)?;
        Ok(())
    }

    /// [`Backend::silu`] applied to `x` in place.
    fn silu_in_place(&self, x: &mut Self::Tensor) -> Result<(), Self::Error> {
        *x = self.silu(x)?;
        Ok(())
    }

    /// [`Backend::elemul`] accumulated into `x`: `x = x * other`.
    fn elemul_in_place(
        &self,
        x: &mut Self::Tensor,
        other: &Self::Tensor,
    ) -> Result<(), Self::Error> {
        *x = self.elemul(x, other)?;
        Ok(())
    }

    /// [`Backend::rms_norm`] written into `out` (resized as needed).
    fn rms_norm_reusing(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &mut Self::Tensor,
    ) -> Result<(), Self::Error> {
        *out = self.rms_norm(x, weight, eps)?;
        Ok(())
    }

    /// [`Backend::cached_causal_attention_with_scratch`] written into `out`.
    fn cached_causal_attention_reusing(
        &self,
        q: &Self::Tensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
        out: &mut Self::Tensor,
    ) -> Result<(), Self::Error> {
        *out = self.cached_causal_attention_with_scratch(q, cached_k, cached_v, spec, qk_row)?;
        Ok(())
    }

    /// Run `update` over the tensor's host values and keep the result in `x`.
    /// The default round-trips through host memory; CPU storage is mutated
    /// directly.
    fn update_in_place(
        &self,
        x: &mut Self::Tensor,
        update: impl FnOnce(&mut [f32]),
    ) -> Result<(), Self::Error> {
        let shape = self.shape(x).to_vec();
        let mut data = self.data(x).to_vec();
        update(&mut data);
        *x = self.load_from_cpu(data, &shape)?;
        Ok(())
    }
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

/// Quantize projection input rows; single-row decode inputs are split across
/// the decode team (byte-identical to the serial encoder).
fn quantize_projection_input(src: &[f32], rows: usize, input: &mut Vec<u8>) {
    if rows == 1 {
        crate::simd::quantize_q8_0_decode_into(src, input);
    } else {
        crate::quant::quantize_q8_0_into(src, input);
    }
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

mod attention;
mod cpu;
mod ops;

#[cfg(target_arch = "aarch64")]
pub(crate) use attention::softmax_range;
pub(crate) use attention::{
    cached_attention_dispatch, cached_attention_row_head, prefill_attention_row_head,
    should_parallel_attention, softmax_prefix, validate_gqa,
};
use attention::{
    cached_attention_into, q8_0_matmul_rows, validate_attention_inputs,
    validate_cached_attention_inputs, validate_quant_row_shapes, validate_row_copy_shapes,
};

/// One sequence of [`CpuBackend::cached_decode_attention_batch_into`]: a
/// single-token query against its own KV cache layer.
pub(crate) struct DecodeAttention<'a> {
    pub q: &'a [f32],
    pub cached_k: &'a [f16],
    pub cached_v: &'a [f16],
    pub spec: CachedAttentionSpec,
    pub out: &'a mut [f32],
}

#[cfg(test)]
mod tests;

use crate::quant::QuantizedWeight;
use crate::tensor::{CpuTensor, TensorError};
use rayon::prelude::*;
use std::cell::RefCell;

const Q8_0_PREFILL_BLOCK_SIZE: usize = 256;
const PARALLEL_ATTENTION_MIN_HEADS: usize = 4;
const PARALLEL_ATTENTION_MIN_WORK: usize = 32_768;

thread_local! {
    static Q8_0_PREFILL_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// shape metadata for standard causal self-attention.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
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
    /// never stored as f32 - columns are dequantized in blocks and
    /// multiplied with `sgemm`.
    fn matmul_q8_0(
        &self,
        x: &Self::Tensor,
        w: &QuantizedWeight,
    ) -> Result<Self::Tensor, Self::Error>;
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
        cached_k: &[f32],
        cached_v: &[f32],
        spec: CachedAttentionSpec,
    ) -> Result<Self::Tensor, Self::Error>;
    fn cached_causal_attention_with_scratch(
        &self,
        q: &Self::Tensor,
        cached_k: &[f32],
        cached_v: &[f32],
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
        if x.ndim() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0: input must be 2D, got shape {:?}",
                x.shape()
            )));
        }
        let (seq_len, in_features) = (x.shape[0], x.shape[1]);
        let out_features = w.out_features();
        if in_features != w.in_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0: inner dims must match (got {} vs {})",
                in_features,
                w.in_features()
            )));
        }

        let x_data = x.data();
        let mut out = vec![0.0f32; seq_len * out_features];

        // decode path: single input row, no column reuse.
        // skip the dense w_block buffer and sgemm dispatch overhead;
        // compute the dot product directly from compressed Q8_0 data.
        if seq_len == 1 {
            crate::simd::matmul_q8_0_decode(x_data, w, &mut out);
            return Ok(CpuTensor::from_data(vec![seq_len, out_features], out));
        }

        Q8_0_PREFILL_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            matmul_q8_0_prefill(
                x_data,
                w,
                seq_len,
                in_features,
                out_features,
                &mut out,
                &mut scratch,
            );
        });

        Ok(CpuTensor::from_data(vec![seq_len, out_features], out))
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

        if should_parallel_attention(spec.n_heads, seq_len, seq_len, spec.head_dim) {
            let heads = (0..spec.n_heads)
                .into_par_iter()
                .map(|h| {
                    let q_head_offset = h * spec.head_dim;
                    let kv_h = h / n_repeat;
                    let kv_head_offset = kv_h * spec.head_dim;
                    let mut head_out = vec![0.0f32; seq_len * spec.head_dim];
                    let mut qk_row = vec![0.0f32; seq_len];

                    for i in 0..seq_len {
                        let q_idx = i * embed_dim + q_head_offset;

                        for (j, slot) in qk_row.iter_mut().enumerate().take(i + 1) {
                            let k_idx = j * kv_dim + kv_head_offset;
                            let dot = crate::simd::dot_product(
                                &q_data[q_idx..q_idx + spec.head_dim],
                                &k_data[k_idx..k_idx + spec.head_dim],
                            );
                            *slot = dot * scale;
                        }

                        softmax_prefix(&mut qk_row, i + 1);

                        let head_offset = i * spec.head_dim;
                        for (j, &weight) in qk_row.iter().enumerate().take(i + 1) {
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

            for i in 0..seq_len {
                let q_idx = i * embed_dim + q_head_offset;

                for (j, slot) in qk_row.iter_mut().enumerate().take(i + 1) {
                    let k_idx = j * kv_dim + kv_head_offset;
                    let dot = crate::simd::dot_product(
                        &q_data[q_idx..q_idx + spec.head_dim],
                        &k_data[k_idx..k_idx + spec.head_dim],
                    );
                    *slot = dot * scale;
                }

                softmax_prefix(&mut qk_row, i + 1);

                let out_offset = i * embed_dim + q_head_offset;
                for (j, &weight) in qk_row.iter().enumerate().take(i + 1) {
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
        cached_k: &[f32],
        cached_v: &[f32],
        spec: CachedAttentionSpec,
    ) -> Result<CpuTensor, CpuError> {
        let mut qk_row = Vec::with_capacity(spec.max_seq_len);
        self.cached_causal_attention_with_scratch(q, cached_k, cached_v, spec, &mut qk_row)
    }

    fn cached_causal_attention_with_scratch(
        &self,
        q: &CpuTensor,
        cached_k: &[f32],
        cached_v: &[f32],
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

        if should_parallel_attention(spec.n_heads, seq_len, spec.total_seq_len, spec.head_dim) {
            let heads = (0..spec.n_heads)
                .into_par_iter()
                .map(|h| {
                    let q_head_offset = h * spec.head_dim;
                    let kv_h = h / n_repeat;
                    let mut head_out = vec![0.0f32; seq_len * spec.head_dim];
                    let mut qk_row = Vec::with_capacity(spec.total_seq_len);

                    for i in 0..seq_len {
                        let max_j = spec.total_seq_len - seq_len + i;
                        qk_row.resize(max_j + 1, 0.0);
                        let q_idx = i * embed_dim + q_head_offset;

                        for (j, slot) in qk_row.iter_mut().enumerate().take(max_j + 1) {
                            let k_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            let dot = crate::simd::dot_product(
                                &q_data[q_idx..q_idx + spec.head_dim],
                                &cached_k[k_offset..k_offset + spec.head_dim],
                            );
                            *slot = dot * scale;
                        }

                        softmax_prefix(qk_row.as_mut_slice(), max_j + 1);

                        let head_offset = i * spec.head_dim;
                        for (j, &weight) in qk_row.iter().enumerate().take(max_j + 1) {
                            if weight == 0.0 {
                                continue;
                            }
                            let v_offset = kv_h * cache_head_stride + j * spec.head_dim;
                            crate::simd::weighted_add(
                                &mut head_out[head_offset..head_offset + spec.head_dim],
                                &cached_v[v_offset..v_offset + spec.head_dim],
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
                    let dot = crate::simd::dot_product(
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
                    crate::simd::weighted_add(
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

fn matmul_q8_0_prefill(
    x_data: &[f32],
    w: &QuantizedWeight,
    seq_len: usize,
    in_features: usize,
    out_features: usize,
    out: &mut [f32],
    w_block: &mut Vec<f32>,
) {
    // w_block is column-major [in_features, block_len]:
    //   w_block[j * in_features + i] = weight[i, j_block + j]
    let required = in_features * Q8_0_PREFILL_BLOCK_SIZE;
    if w_block.len() < required {
        w_block.resize(required, 0.0);
    }

    let mut j = 0;
    while j < out_features {
        let block_len = (out_features - j).min(Q8_0_PREFILL_BLOCK_SIZE);

        for b in 0..block_len {
            let dst = &mut w_block[b * in_features..(b + 1) * in_features];
            w.dequantize_row(j + b, dst);
        }

        // x [seq_len, in_features] @ w_block [in_features, block_len]
        // -> write to out[:, j..j+block_len]
        //
        // sgemm(m, k, n, alpha, A, rsa, csa, B, rsb, csb, beta, C, rsc, csc)
        //   A: row-major [m, k] -> rsa=k, csa=1
        //   B: column-major [k, n] -> rsb=1, csb=k
        //   C: row-major [m, n] -> rsc=n_full, csc=1
        unsafe {
            matrixmultiply::sgemm(
                seq_len,
                in_features,
                block_len,
                1.0,
                x_data.as_ptr(),
                in_features as isize,
                1,
                w_block.as_ptr(),
                1,
                in_features as isize,
                0.0,
                out.as_mut_ptr().add(j),
                out_features as isize,
                1,
            );
        }
        j += Q8_0_PREFILL_BLOCK_SIZE;
    }
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

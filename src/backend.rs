use crate::quant::QuantizedWeight;
use crate::tensor::{CpuTensor, TensorError};

/// the core abstraction over compute hardware.
///
/// model code is generic over the backend, so the same transformer
/// implementation works on `CpuBackend`, or any future gpu/accelerator
/// backend, without modification.
///
/// ## scope
///
/// the trait currently abstracts element-wise ops (`add`, `gelu`, `softmax`),
/// linear algebra (`matmul`, `matmul_q8_0`, `add_broadcast`), normalisation
/// (`layer_norm`), shape manipulation (`slice_cols`, `index_select`, `reshape`),
/// and tensor lifecycle (`zeroes`, `load_from_cpu`, `data`, `shape`).
///
/// **attention is not yet abstracted** - the model's `Attention::forward*`
/// methods call `data()` to extract raw f32 slices and run the attention
/// math in scalar cpu loops. a gpu backend would still execute attention
/// on the cpu through this path. adding `fn attention(...)` to the trait
/// is the next step; for now the abstraction is honest about what it covers.
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
    fn assign_row(&self, dst: &mut Self::Tensor, index: usize, src: &Self::Tensor);
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

        // dequantize columns in blocks, multiply with sgemm.
        // w_block is column-major [in_features, block_len]:
        //   w_block[j * in_features + i] = weight[i, j_block + j]
        const BLOCK_SIZE: usize = 256;
        let mut w_block = vec![0.0f32; in_features * BLOCK_SIZE];

        let mut j = 0;
        while j < out_features {
            let block_len = (out_features - j).min(BLOCK_SIZE);

            // dequantize this block of columns into w_block
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
                    in_features as isize, // rsa
                    1,                    // csa
                    w_block.as_ptr(),
                    1,                    // rsb (column-major)
                    in_features as isize, // csb
                    0.0,
                    out.as_mut_ptr().add(j),
                    out_features as isize, // rsc
                    1,                     // csc
                );
            }
            j += BLOCK_SIZE;
        }

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
    fn assign_row(&self, dst: &mut CpuTensor, index: usize, src: &CpuTensor) {
        dst.assign_row(index, src);
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

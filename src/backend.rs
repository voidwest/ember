use crate::tensor::{CpuTensor, TensorError};

/// the core abstraction over compute hardware.
///
/// model code is generic over the backend, so the same transformer
/// implementation works on `CpuBackend`, or any future gpu/accelerator
/// backend, without modification.
pub trait Backend {
    type Tensor: Clone + Send + Sync;
    type Error: core::error::Error;

    fn zeroes(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
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
    #[allow(clippy::wrong_self_convention)]
    fn from_cpu(&self, data: Vec<f32>, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn add_broadcast(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
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
    fn from_cpu(&self, data: Vec<f32>, shape: &[usize]) -> Result<CpuTensor, Self::Error> {
        Ok(CpuTensor::from_data(shape.to_vec(), data))
    }
    fn add_broadcast(&self, x: &CpuTensor, bias: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.add_broadcast(bias))
    }
}

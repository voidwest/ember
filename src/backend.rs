
use crate::tensor::CpuTensor;

pub trait Backend{
    type Tensor: Clone + Send + Sync;
    type Error: core::error::Error;
}


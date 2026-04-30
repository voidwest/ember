use crate::backend::{Backend, Module};
use alloc::vec::Vec;

pub struct Linear<B: Backend> {
    weight: B::Tensor,
    bias: Option<B::Tensor>,
}

impl<B: Backend> Linear<B> {
    pub fn new(weight: B::Tensor, bias: Option<B::Tensor>) -> Self {
        Self { weight, bias }
    }
    pub fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        
    }
}

use crate::backend::{Backend, Module};
use alloc::vec::Vec;

pub struct Linear<B: Backend> {
    weight: B::Tensor,
    bias: Option<B::Tensor>,
}

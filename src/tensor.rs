use std::num::NonZeroUsize;

use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl CpuTensor {
    pub fn zeroes(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.into(),
            strides,
            data: vec![0.0; len],
        }
    }
}

pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
    let expected = shape.iter().product::<usize>();
    assert_eq!(
        expected,
        data.len(),
        "shape product ({}) != data len ({})",
        expected,
        data.len()
    );
    let strides = Self::compute_strides(&shape);
    Self {
        shape,
        strides,
        data,
    }
}

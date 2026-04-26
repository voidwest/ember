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

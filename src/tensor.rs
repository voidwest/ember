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

pub fn shape(&self) -> &[usize] {
    &self.shape
}

pub fn data(&self) -> &[f32] {
    &self.data
}

pub fn data_mut(&mut self) -> &mut [f32] {
    &mut self.data
}

pub fn ndim(&self) -> usize {
    self.shape.len()
}
pub fn len(&self) -> usize {
    self.data.len()
}

pub fn is_empty(&self) -> bool {
    self.data.is_empty()
}

// get 1 element w/ n-dim indices
// very slow, in the SIMD kernel i'll itr through flat data slice directly
pub fn get(&self, indices: &[usize]) -> f32 {
    assert_eq!(indices.len(), self.shape.len());

    let mut idx = 0;
    for (i, &dim_idx) in indices.iter().enumerate() {
        assert!(dim_idx < self.shape[i]);
        idx += dim_idx * self.strides[i];
    }
    self.data[idx]
}

// reshape w/o data changing
pub fn reshape(&self, new_shape: &[usize]) -> Self {
    let new_len: usize = new_shape.iter().product();
    assert_eq!(new_len, self.len(), "reshape: total elements gotta match");
    Self::from_data(new_shape.into(), self.data.clone())
}

pub fn add(&self, other: &Self) -> Self {
    assert_eq!(
        self.shape, other.shape,
        "addition: shapes must match (for now)"
    );

    let data: Vec<f32> = self
        .data
        .iter()
        .zip(&other.data)
        .map(|(a, b)| a + b)
        .collect();
    Self::from_data(self.shape.clone(), data)
}

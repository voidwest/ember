use alloc::vec::Vec;
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<f32>,
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

    // matrix mult, 2d tensors

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2d");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2d");
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k1, k2, "matmul: inner dims must match");

        let mut out = Self::zeroes(&[m, n]);

        // bad impl, replacing later
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self.get(&[i, k]) * other.get(&[k, j]);
                }
                out.data[i * n + j] = sum;
            }
        }
        out
    }

    //softmax along the last dimension

    pub fn softmax(&self) -> Self {
        assert!(!self.shape.is_empty(), "softmax needs 1 dim min");
        let last_dim = self.shape[self.shape.len() - 1];
        let batch: usize = self.shape[..self.shape.len() - 1].iter().product();

        let mut out_data = vec![0.0f32; self.len()];

        for b in 0..batch {
            let offset = b * last_dim;
            let slice = &self.data[offset..offset + last_dim];

            //stable softmax: stubtract max
            let max = slice.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
            let mut sum = 0.0;
            for i in 0..last_dim {
                let e = (slice[i] - max).exp();
                out_data[offset + i] = e;
                sum += e;
            }
            for i in 0..last_dim {
                out_data[offset + i] /= sum;
            }
        }
        Self::from_data(self.shape.clone(), out_data)
    }

    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| 0.5 * x * (1.0 + libm::erff(x / f32::sqrt(2.0))))
            .collect();
        Self::from_data(self.shape.clone(), data)
    }

    pub fn layer_norm(&self, weight: &Self, bias: &Self, eps: f32) -> Self {
        assert_eq!(self.ndim(), 2, "layer_norm expects 2d [batch, features]");
        let (batch, features) = (self.shape[0], self.shape[1]);
        assert_eq!(weight.len(), features);
        assert_eq!(bias.len(), features);

        let mut out = vec![0.0f32; self.len()];
        for b in 0..batch {
            let offset = b * features;
            let slice = &self.data[offset..offset + features];

            let mean = slice.iter().sum::<f32>() / features as f32;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features as f32;
            let std = (var + eps).sqrt();

            for i in 0..features {
                let normalized = (slice[i] - mean) / std;
                out[offset + i] = normalized * weight.data[i] + bias.data[i];
            }
        }
        Self::from_data(self.shape.clone(), out)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    pub fn index_select(&self, index: usize) -> Self {
        if self.shape.len() < 2 {
            eprintln!("cannot index_select a tensor with less than 2 dimensions")
        }

        let row_size = self.shape[1];
        let start = index * row_size;
        let end = start + row_size;

        if end > self.data.len() {
            eprintln!(
                "index {} out of bounds (max index: {})",
                index,
                self.data.len() / row_size - 1
            );
        }

        let row_data = self.data[start..end].to_vec();

        let new_shape = vec![row_size];
        let new_strides = vec![1];

        CpuTensor {
            shape: new_shape,
            data: row_data,
            strides: new_strides,
        }
    }
    pub fn assign_row(&mut self, index: usize, src: &CpuTensor) {
        let row_size = self.shape[1];
        let start = index * row_size;
        let end = start + row_size;

        if src.data.len() != row_size {
            panic!(
                "source tensor size {} does not match destination row size {}",
                src.data.len(),
                row_size
            );
        }

        self.data[start..end].copy_from_slice(&src.data);
    }
    pub fn slice_cols(&self, start_col: usize, end_col: usize) -> Self {
        if self.shape.len() < 2 {
            panic!("slice_cols requires a 2D tensor [rows, cols]");
        }

        let rows = self.shape[0];
        let old_cols = self.shape[1];
        let new_cols = end_col - start_col;

        if end_col > old_cols {
            panic!("column slice out of bounds: {} > {}", end_col, old_cols);
        }

        let mut new_data = Vec::with_capacity(rows * new_cols);

        for r in 0..rows {
            let row_start = r * old_cols;
            let slice_start = row_start + start_col;
            let slice_end = row_start + end_col;

            new_data.extend_from_slice(&self.data[slice_start..slice_end]);
        }

        CpuTensor {
            shape: vec![rows, new_cols],
            data: new_data,
            strides: vec![new_cols, 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = CpuTensor::zeroes(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_add() {
        let a = CpuTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = CpuTensor::from_data(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let c = a.add(&b);
        assert_eq!(c.data(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_matmul() {
        let a = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = CpuTensor::from_data(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        // r0: [1+0+3, 0+2+0] = [4, 2]
        // r1: [4+0+6, 0+5+0] = [10, 5]
        assert_eq!(c.data(), &[4.0, 2.0, 10.0, 5.0]);
    }
    #[test]
    fn test_softmax() {
        let t = CpuTensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let s = t.softmax();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

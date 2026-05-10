use alloc::string::String;
use alloc::vec::Vec;

/// a row-major f32 tensor backed by a flat vec.
///
/// shape is [d0, d1, d2, ...] with strides computed for efficient
/// indexing. the data is always contiguous — strides are used only
/// for bounds-aware access, not for views into other storage.
/// all pure operations return a new allocation; nothing mutates in place.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    /// dimensions of the tensor
    pub shape: Vec<usize>,
    /// strides for each dimension (contiguous row-major)
    pub strides: Vec<usize>,
    /// flat f32 data buffer
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorError {
    #[error("index {index} out of bounds for shape {shape:?}")]
    IndexOutOfBounds { index: usize, shape: Vec<usize> },
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
}

impl CpuTensor {
    /// allocate a zero-filled tensor with the given shape
    #[must_use]
    #[inline]
    pub fn zeroes(shape: &[usize]) -> Self {
        let len = shape.iter().product();
        let strides = Self::compute_strides(shape);
        Self {
            shape: shape.into(),
            strides,
            data: vec![0.0; len],
        }
    }
    /// add a 1d bias to every row of a 2d tensor (broadcast).
    ///
    /// ## panics
    /// - if `self` is not 2d.
    /// - if `bias` is not 1d.
    /// - if `bias.shape[0]` does not match `self.shape[1]`.
    #[must_use]
    #[inline]
    pub fn add_broadcast(&self, bias: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "add_broadcast: lhs must be 2D");
        assert_eq!(bias.ndim(), 1, "add_broadcast: rhs must be 1D");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        assert_eq!(
            bias.shape[0], cols,
            "add_broadcast: bias size must match cols"
        );
        let mut new_data = self.data.clone();
        for r in 0..rows {
            for c in 0..cols {
                new_data[r * cols + c] += bias.data[c];
            }
        }
        CpuTensor::from_data(self.shape.clone(), new_data)
    }
    /// 2d matrix transpose. panics if not 2d.
    #[must_use]
    #[inline]
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose only supports 2D tensors");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                new_data[c * rows + r] = self.data[r * cols + c];
            }
        }
        CpuTensor::from_data(vec![cols, rows], new_data)
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

    #[inline]
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
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn get(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.shape.len());

        let mut idx = 0;
        for (i, &dim_idx) in indices.iter().enumerate() {
            assert!(dim_idx < self.shape[i]);
            idx += dim_idx * self.strides[i];
        }
        self.data[idx]
    }

    /// reshape a tensor without copying data.
    /// panics if the new shape has a different total element count.
    #[must_use]
    #[inline]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_len: usize = new_shape.iter().product();
        assert_eq!(new_len, self.len(), "reshape: total elements gotta match");
        Self::from_data(new_shape.into(), self.data.clone())
    }

    /// element-wise addition. panics if shapes differ.
    #[must_use]
    #[inline]
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

    /// matrix multiplication via `matrixmultiply::sgemm`.
    /// both tensors must be 2d with matching inner dimensions.
    #[must_use]
    #[inline]
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2d");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2d");
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k1, k2, "matmul: inner dims must match");

        let mut out = vec![0.0f32; m * n];

        unsafe {
            matrixmultiply::sgemm(
                m,
                k1,
                n,
                1.0,
                self.data.as_ptr(),
                k1 as isize,
                1,
                other.data.as_ptr(),
                n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        Self::from_data(vec![m, n], out)
    }

    /// softmax along the last dimension, numerically stable with max
    /// subtraction. if every logit in a row is -infinity (fully masked),
    /// returns a uniform distribution over that row.
    #[must_use]
    #[inline]
    pub fn softmax(&self) -> Self {
        assert!(!self.shape.is_empty(), "softmax needs 1 dim min");
        let last_dim = self.shape[self.shape.len() - 1];
        let batch: usize = self.shape[..self.shape.len() - 1].iter().product();

        let mut out_data = vec![0.0f32; self.len()];

        for b in 0..batch {
            let offset = b * last_dim;
            let slice = &self.data[offset..offset + last_dim];

            let max = slice.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
            if max == f32::NEG_INFINITY {
                let uniform = 1.0 / (last_dim as f32);
                for i in 0..last_dim {
                    out_data[offset + i] = uniform;
                }
                continue;
            }
            let mut sum = 0.0;
            for i in 0..last_dim {
                let e = (slice[i] - max).exp();
                out_data[offset + i] = e;
                sum += e;
            }
            let inv_sum = sum.recip();
            for i in 0..last_dim {
                out_data[offset + i] *= inv_sum;
            }
        }
        Self::from_data(self.shape.clone(), out_data)
    }

    /// gaussian error linear unit: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    /// uses `libm::erff` for portable float math.
    #[must_use]
    #[inline]
    pub fn gelu(&self) -> Self {
        let inv_sqrt_2 = 0.707_106_77_f32;
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                let z = x * inv_sqrt_2;
                0.5 * x * (1.0 + libm::erff(z))
            })
            .collect();
        Self::from_data(self.shape.clone(), data)
    }

    /// rms normalization over the last dimension of a 2d `[batch, features]`
    /// tensor. normalizes each row independently: `x * weight / sqrt(mean(x²) + eps)`.
    /// lLaMA-family models use this instead of layer_norm — no mean subtraction, no bias.
    #[must_use]
    #[inline]
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Self {
        assert_eq!(self.ndim(), 2, "rms_norm expects 2d [batch, features]");
        let (batch, features) = (self.shape[0], self.shape[1]);
        assert_eq!(weight.len(), features);

        let mut out = vec![0.0f32; self.len()];
        for b in 0..batch {
            let offset = b * features;
            let slice = &self.data[offset..offset + features];

            let mean_sq = slice.iter().map(|x| x * x).sum::<f32>() / features as f32;
            let rstd = (mean_sq + eps).sqrt().recip();

            for i in 0..features {
                out[offset + i] = slice[i] * rstd * weight.data[i];
            }
        }
        Self::from_data(self.shape.clone(), out)
    }

    /// sigmoid linear unit: `x * sigmoid(x)` = `x / (1 + exp(-x))`.
    /// lLaMA-family mlp uses this in the swiGLU gate path.
    #[must_use]
    #[inline]
    pub fn silu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        Self::from_data(self.shape.clone(), data)
    }

    /// layer normalization over the last dimension of a 2d `[batch, features]`
    /// tensor. normalizes each row independently: `(x - mean) / sqrt(var + eps)
    /// * weight + bias`.
    #[must_use]
    #[inline]
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

    #[inline]
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// select the `index`-th row from a 2d tensor.
    ///
    /// returns a **1d** tensor of shape `[row_size]` — the row is flattened.
    /// if you need a 2d `[1, row_size]` result, reshape the output.
    pub fn index_select(&self, index: usize) -> Result<Self, TensorError> {
        if self.shape.len() < 2 {
            return Err(TensorError::ShapeMismatch(
                "cannot index_select a tensor with less than 2 dimensions".into(),
            ));
        }

        let row_size = self.shape[1];
        let max_index = self.data.len() / row_size;
        if index >= max_index {
            return Err(TensorError::IndexOutOfBounds {
                index,
                shape: self.shape.clone(),
            });
        }

        let start = index * row_size;
        let end = start + row_size;

        let row_data = self.data[start..end].to_vec();

        Ok(CpuTensor {
            shape: vec![row_size],
            data: row_data,
            strides: vec![1],
        })
    }

    #[inline]
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
    #[must_use]
    #[inline]
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

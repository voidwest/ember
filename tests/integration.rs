use ember::backend::{Backend, CpuBackend};
use ember::tensor::CpuTensor;

#[test]
fn test_matrix_multiplication_accuracy() {
    // 2x3 matrix
    let a = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // 3x2 matrix
    let b = CpuTensor::from_data(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let c = a.matmul(&b);

    // expected: [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ]
    //           [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ]
    let expected = vec![58.0, 64.0, 139.0, 154.0];

    assert_eq!(c.shape(), &[2, 2]);
    for (i, &val) in c.data().iter().enumerate() {
        assert!((val - expected[i]).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_logic() {
    let t = CpuTensor::from_data(vec![1, 3], vec![1.0, 2.0, 3.0]);
    let s = t.softmax();

    let sum: f32 = s.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax rows must sum to 1.0");

    // values should be in increasing order because inputs were [1, 2, 3]
    assert!(s.data()[0] < s.data()[1]);
    assert!(s.data()[1] < s.data()[2]);
}

#[test]
fn test_layer_norm_stability() {
    let t = CpuTensor::from_data(vec![1, 4], vec![10.0, 10.0, 10.0, 10.0]);
    let weight = CpuTensor::from_data(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    let bias = CpuTensor::from_data(vec![4], vec![0.0, 0.0, 0.0, 0.0]);

    let normed = t.layer_norm(&weight, &bias, 1e-5);

    // if all inputs are the same, the mean is 10 and variance is 0.
    // normalized values should be 0.
    for &val in normed.data() {
        assert!((val - 0.0).abs() < 1e-5);
    }
}

#[test]
#[should_panic(expected = "inner dims must match")]
fn test_matmul_dimension_mismatch() {
    let a = CpuTensor::zeroes(&[2, 2]);
    let b = CpuTensor::zeroes(&[3, 2]); // Invalid: 2 != 3
    let _ = a.matmul(&b);
}

#[test]
fn test_cpu_backend_abstraction() {
    let backend = CpuBackend;
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = [2, 2];

    let tensor = backend.from_cpu(data.clone(), &shape).unwrap();

    assert_eq!(backend.shape(&tensor), &shape);
    assert_eq!(backend.data(&tensor), &data);
}

#[test]
fn test_empty_tensor() {
    let t = CpuTensor::zeroes(&[0]);
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert_eq!(t.ndim(), 1);
}

#[test]
fn test_zero_tensor() {
    let t = CpuTensor::zeroes(&[3, 4]);
    assert_eq!(t.shape(), &[3, 4]);
    assert!(t.data().iter().all(|&x| x == 0.0));
}

#[test]
fn test_extreme_values() {
    let t = CpuTensor::from_data(
        vec![1, 3],
        vec![1e10, -1e10, 0.0],
    );
    let s = t.softmax();
    let sum: f32 = s.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should handle extreme values");
}

#[test]
fn test_all_masked() {
    let t = CpuTensor::from_data(
        vec![1, 4],
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
    );
    let s = t.softmax();
    let sum: f32 = s.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "all masked should sum to 1");
    for v in s.data().iter() {
        assert!((v - 0.25).abs() < 1e-5, "all masked should be uniform distribution");
    }
}

#[test]
fn test_transpose() {
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tt = t.transpose();
    assert_eq!(tt.shape(), &[3, 2]);
    assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_reshape() {
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = t.reshape(&[3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
}

#[test]
#[should_panic(expected = "total elements gotta match")]
fn test_reshape_invalid() {
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let _ = t.reshape(&[2, 4]);
}

#[test]
fn test_slice_cols() {
    let t = CpuTensor::from_data(
        vec![2, 4],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let sliced = t.slice_cols(1, 3);
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.data(), &[2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn test_gelu() {
    let t = CpuTensor::from_data(vec![1, 4], vec![0.0, 1.0, -1.0, 2.0]);
    let g = t.gelu();
    assert!(g.data()[0].abs() < 1e-5);
    assert!((g.data()[1] - 0.841192).abs() < 1e-3);
    assert!((g.data()[2] - (-0.158808)).abs() < 1e-3);
    assert!((g.data()[3] - 1.954030).abs() < 1e-3);
}

#[test]
fn test_add_broadcast() {
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bias = CpuTensor::from_data(vec![3], vec![0.1, 0.2, 0.3]);
    let result = t.add_broadcast(&bias);
    let expected = vec![1.1, 2.2, 3.3, 4.1, 5.2, 6.3];
    assert_eq!(result.data(), &expected);
}

#[test]
#[should_panic(expected = "bias size must match cols")]
fn test_add_broadcast_mismatch() {
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bias = CpuTensor::from_data(vec![2], vec![0.1, 0.2]);
    let _ = t.add_broadcast(&bias);
}

#[test]
fn test_softmax_batch() {
    let t = CpuTensor::from_data(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
    );
    let s = t.softmax();
    let sum1: f32 = s.data()[0..3].iter().sum();
    let sum2: f32 = s.data()[3..6].iter().sum();
    assert!((sum1 - 1.0).abs() < 1e-5, "batch 1 softmax sum");
    assert!((sum2 - 1.0).abs() < 1e-5, "batch 2 softmax sum");
}

#[test]
fn test_softmax_random_values() {
    for _ in 0..5 {
        let t = CpuTensor::from_data(
            vec![1, 4],
            vec![rand_f32(), rand_f32(), rand_f32(), rand_f32()],
        );
        let s = t.softmax();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1 for random values");
        assert!(s.data()[0] >= 0.0 && s.data()[3] <= 1.0);
    }
}

fn rand_f32() -> f32 {
    static COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let idx = n as usize % FIXED_VALUES.len();
    FIXED_VALUES[idx]
}

static FIXED_VALUES: &[f32] = &[
    0.5, -0.3, 1.2, -0.7, 0.1, -1.5, 2.0, -0.2, 0.8, -0.9,
    1.1, -0.4, 0.3, -1.8, 2.5, -0.1, 0.6, -1.2, 1.9, -0.5,
];

#[test]
fn test_matmul_identity() {
    let a = CpuTensor::from_data(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let b = CpuTensor::from_data(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]);
    let c = a.matmul(&b);
    assert_eq!(c.data(), b.data());
}

#[test]
fn test_layer_norm() {
    let t = CpuTensor::from_data(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let weight = CpuTensor::from_data(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    let bias = CpuTensor::from_data(vec![4], vec![0.0, 0.0, 0.0, 0.0]);
    let normed = t.layer_norm(&weight, &bias, 1e-5);
    assert_eq!(normed.shape(), &[2, 4]);
}

#[test]
fn test_index_select() {
    let t = CpuTensor::from_data(vec![3, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let row1 = t.index_select(1).unwrap();
    assert_eq!(row1.shape(), &[4]);
    assert_eq!(row1.data(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_assign_row() {
    let mut t = CpuTensor::zeroes(&[2, 4]);
    let src = CpuTensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    t.assign_row(1, &src);
    assert_eq!(&t.data()[4..8], src.data());
}

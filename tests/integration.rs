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

use ember::tensor::CpuTensor;

#[test]
fn test_tensor_pipeline() {
    let a = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = CpuTensor::from_data(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2]);
}

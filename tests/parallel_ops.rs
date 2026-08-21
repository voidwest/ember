//! Bit-exactness contracts for the multi-threaded tensor ops.
//!
//! The parallel variants must produce *identical bits* to their serial
//! counterparts: matmul splits rows (each output element is one dot product
//! whose accumulation order never depends on the partitioning), softmax
//! splits rows (row-independent math), elementwise ops split flat chunks
//! (each output depends on exactly one input). These tests pin that claim
//! so threading changes can never silently alter numerics.

use ember::tensor::CpuTensor;

fn deterministic(rows: usize, cols: usize) -> CpuTensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    CpuTensor::from_data(vec![rows, cols], data)
}

#[test]
fn par_matmul_is_bit_identical_to_serial() {
    // large enough to trigger the parallel path (>= 64M MACs, >= 64 rows)
    let a = deterministic(333, 1021);
    let b = deterministic(1021, 331);
    let serial = a.matmul(&b);
    let par = a.par_matmul(&b);
    assert_eq!(serial.shape(), par.shape());
    assert_eq!(
        serial.data(),
        par.data(),
        "par_matmul must be bit-identical to matmul"
    );

    // non-power-of-two row counts force ragged final chunks
    let a = deterministic(257, 967);
    let b = deterministic(967, 257);
    let serial = a.matmul(&b);
    let par = a.par_matmul(&b);
    assert_eq!(serial.data(), par.data());
}

#[test]
fn par_matmul_small_shapes_fall_back_and_match() {
    let a = deterministic(8, 16);
    let b = deterministic(16, 8);
    let serial = a.matmul(&b);
    let par = a.par_matmul(&b);
    assert_eq!(serial.data(), par.data());
}

#[test]
fn par_softmax_is_bit_identical_to_serial() {
    let rows = 4096;
    let cols = 1024;
    let x = deterministic(rows, cols);
    let serial = x.softmax();
    let par = x.par_softmax();
    assert_eq!(
        serial.data(),
        par.data(),
        "par_softmax must be bit-identical to softmax"
    );

    // special cases: a +inf row and an all -inf row must match too
    let mut data = deterministic(4, 8).data().to_vec();
    data[0] = f32::INFINITY;
    for v in data[8..16].iter_mut() {
        *v = f32::NEG_INFINITY;
    }
    let x = CpuTensor::from_data(vec![4, 8], data);
    assert_eq!(x.softmax().data(), x.par_softmax().data());
}

#[test]
fn par_gelu_tanh_is_bit_identical_to_serial() {
    let n = 2 << 20; // above the elementwise parallel threshold
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 104_729) % 4000) as f32 / 100.0 - 20.0)
        .collect();
    let x = CpuTensor::from_data(vec![n], data);
    let serial = x.gelu_tanh();
    let par = x.par_gelu_tanh();
    assert_eq!(
        serial.data(),
        par.data(),
        "par_gelu_tanh must be bit-identical to gelu_tanh"
    );
}

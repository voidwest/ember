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
    let expected = [58.0, 64.0, 139.0, 154.0];

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
    let b = CpuTensor::zeroes(&[3, 2]); // invalid: 2 != 3
    let _ = a.matmul(&b);
}

#[test]
fn test_cpu_backend_abstraction() {
    let backend = CpuBackend;
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = [2, 2];

    let tensor = backend.load_from_cpu(data.clone(), &shape).unwrap();

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
    let t = CpuTensor::from_data(vec![1, 3], vec![1e10, -1e10, 0.0]);
    let s = t.softmax();
    let sum: f32 = s.data().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should handle extreme values"
    );
}

#[test]
fn test_all_masked() {
    let t = CpuTensor::from_data(
        vec![1, 4],
        vec![
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    );
    let s = t.softmax();
    let sum: f32 = s.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "all masked should sum to 1");
    for v in s.data().iter() {
        assert!(
            (v - 0.25).abs() < 1e-5,
            "all masked should be uniform distribution"
        );
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
    let t = CpuTensor::from_data(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
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
    assert!((g.data()[3] - 1.954_03).abs() < 1e-3);
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
    let t = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
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
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1 for random values"
        );
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
    0.5, -0.3, 1.2, -0.7, 0.1, -1.5, 2.0, -0.2, 0.8, -0.9, 1.1, -0.4, 0.3, -1.8, 2.5, -0.1, 0.6,
    -1.2, 1.9, -0.5,
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
    let t = CpuTensor::from_data(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let row1 = t.index_select(1).unwrap();
    assert_eq!(row1.shape(), &[4]);
    assert_eq!(row1.data(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_backend_row_as_2d() {
    let backend = CpuBackend;
    let t = CpuTensor::from_data(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let row1 = backend.row_as_2d(&t, 1).expect("row_as_2d should work");

    assert_eq!(row1.shape(), &[1, 4]);
    assert_eq!(row1.data(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_assign_row() {
    let mut t = CpuTensor::zeroes(&[2, 4]);
    let src = CpuTensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    t.assign_row(1, &src);
    assert_eq!(&t.data()[4..8], src.data());
}

#[test]
fn test_model_forward_pass() {
    if !std::path::Path::new("gpt2.Q8_0.gguf").exists() {
        eprintln!("skipping model test: no gguf file found");
        return;
    }

    use ember::backend::{Backend, CpuBackend};
    use ember::loader::load_gguf;
    use ember::model::Gpt2;

    let loader = load_gguf("gpt2.Q8_0.gguf").expect("failed to load model");
    let model = Gpt2::from_loader(loader).expect("failed to build model");
    let backend = CpuBackend;

    let logits = model
        .forward(&backend, &[15496])
        .expect("forward pass failed");

    let shape = backend.shape(&logits);
    assert_eq!(shape.len(), 2, "logits should be 2D [seq_len, vocab]");
    assert_eq!(shape[0], 1, "single token input -> single output row");

    let vocab_size = shape[1];
    assert!(
        vocab_size > 50000,
        "gpt2 vocab should be ~50257, got {}",
        vocab_size
    );

    let data = backend.data(&logits);
    assert!(!data.iter().any(|x| x.is_nan()), "logits contain NaN");
    assert!(
        data.iter().any(|x| *x != 0.0),
        "logits are all zeros - suspicious"
    );

    let top = data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(top < vocab_size, "predicted token out of vocab range");
}

#[test]
fn test_tokenizer_roundtrip() {
    if !std::path::Path::new("tokenizer.json").exists() {
        eprintln!("skipping tokenizer test: no tokenizer.json");
        return;
    }

    use ember::tokenizer::EmberTokenizer;
    let tok = EmberTokenizer::from_file("tokenizer.json").expect("failed to load tokenizer");

    let input = "hello world";
    let ids = tok.encode(input).expect("encode failed");
    let decoded = tok.decode(&ids).expect("decode failed");

    let decoded_lower = decoded.trim().to_lowercase();
    assert!(
        decoded_lower.contains("hello"),
        "roundtrip lost 'hello': got '{}'",
        decoded
    );
    assert!(
        decoded_lower.contains("world"),
        "roundtrip lost 'world': got '{}'",
        decoded
    );
}

#[test]
fn test_tokenizer_vocab_size() {
    if !std::path::Path::new("tokenizer.json").exists() {
        return;
    }
    let tok = ember::tokenizer::EmberTokenizer::from_file("tokenizer.json")
        .expect("failed to load tokenizer");
    // architecture-agnostic: just verify non-zero vocab
    assert!(tok.vocab_size() > 0, "vocab size should be positive");
}

#[test]
fn test_tokenizer_empty_string() {
    if !std::path::Path::new("tokenizer.json").exists() {
        return;
    }
    let tok = ember::tokenizer::EmberTokenizer::from_file("tokenizer.json")
        .expect("failed to load tokenizer");
    let ids = tok.encode("").expect("encode empty string failed");
    // some tokenizers add BOS/control tokens — just verify decode roundtrips
    if !ids.is_empty() {
        let decoded = tok.decode(&ids).expect("decode empty string failed");
        // non-empty decode of empty string is OK (BOS token etc.)
        assert!(!decoded.is_empty() || ids.len() <= 1);
    }
}

/// build a minimal valid GGUF v3 file in memory, suitable for testing the parser.
/// contains one metadata key-value and one f32 tensor.
fn build_minimal_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    // magic "GGUF" as little-endian u32: 0x46554747
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    // version 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor count = 1
    buf.extend_from_slice(&1u64.to_le_bytes());
    // metadata kv count = 1
    buf.extend_from_slice(&1u64.to_le_bytes());

    // metadata: key = "general.name", value type = 8 (string), value = "test"
    let key = b"general.name";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&8u32.to_le_bytes()); // string type
    let val = b"test";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    // tensor info: name = "test.weight", 2 dims [2, 4], dtype f32 (0), offset 0
    let tname = b"test.weight";
    buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    buf.extend_from_slice(tname);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&2u64.to_le_bytes()); // dim 0
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim 1
    buf.extend_from_slice(&0u32.to_le_bytes()); // dtype: f32
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // compute padding to 32-byte alignment
    let current_pos = buf.len() as u64;
    let alignment = 32u64;
    let data_start = (current_pos + alignment - 1) & !(alignment - 1);
    let padding = (data_start - current_pos) as usize;
    buf.resize(buf.len() + padding, 0);

    // tensor data: 8 f32 values (2 * 4), all 1.0
    for _ in 0..8 {
        buf.extend_from_slice(&1.0f32.to_le_bytes());
    }

    buf
}

/// build a minimal GGUF v3 file containing one tensor with caller-provided
/// dtype, dims, and raw tensor payload bytes.
fn build_single_tensor_gguf(dtype: u32, dims: &[u64], tensor_bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());

    let key = b"general.name";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&8u32.to_le_bytes());
    let val = b"layout-test";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    let tname = b"test.weight";
    buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    buf.extend_from_slice(tname);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dim in dims {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let current_pos = buf.len() as u64;
    let alignment = 32u64;
    let data_start = (current_pos + alignment - 1) & !(alignment - 1);
    let padding = (data_start - current_pos) as usize;
    buf.resize(buf.len() + padding, 0);

    buf.extend_from_slice(tensor_bytes);
    buf
}

#[test]
fn test_load_minimal_gguf() {
    use ember::loader::load_gguf_from_reader;
    use std::io::Cursor;

    let gguf_bytes = build_minimal_gguf();
    let mut cursor = Cursor::new(&gguf_bytes);
    let loader = load_gguf_from_reader(&mut cursor).expect("should parse minimal gguf");

    assert_eq!(loader.tensors.len(), 1, "expected one tensor");

    let name = loader
        .metadata
        .get("general.name")
        .expect("metadata should contain 'general.name'");
    match name {
        ember::loader::GgufValue::Str(s) => assert_eq!(s, "test"),
        _ => panic!("expected Str value"),
    }

    use ember::loader::LoadedTensor;
    let tensor = loader
        .tensors
        .get("test.weight")
        .expect("tensor 'test.weight' not found");
    let f32_tensor = match tensor {
        LoadedTensor::F32(t) => t,
        _ => panic!("expected F32 tensor, got Q8_0"),
    };
    assert_eq!(f32_tensor.shape(), &[2, 4]);
    assert!(f32_tensor.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
}

#[test]
fn test_load_f16_keeps_logical_shape() {
    use ember::loader::{load_gguf_from_reader, LoadedTensor};
    use half::f16;
    use std::io::Cursor;

    let mut tensor_bytes = Vec::new();
    for value in 0..8 {
        tensor_bytes.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes());
    }
    let gguf_bytes = build_single_tensor_gguf(1, &[2, 4], &tensor_bytes);
    let mut cursor = Cursor::new(&gguf_bytes);
    let loader = load_gguf_from_reader(&mut cursor).expect("should parse f16 gguf");

    let tensor = loader
        .tensors
        .get("test.weight")
        .expect("tensor 'test.weight' not found");
    let f16_tensor = match tensor {
        LoadedTensor::F32(t) => t,
        _ => panic!("expected f16 tensor to load as F32"),
    };
    assert_eq!(f16_tensor.shape(), &[2, 4]);
    assert_eq!(f16_tensor.data()[7], 7.0);
}

#[test]
fn test_load_q8_0_reverses_to_quantized_matmul_shape() {
    use ember::loader::{load_gguf_from_reader, LoadedTensor};
    use ember::quant::Q8_0_TYPE_SIZE;
    use std::io::Cursor;

    let tensor_bytes = vec![0u8; 2 * Q8_0_TYPE_SIZE];
    let gguf_bytes = build_single_tensor_gguf(8, &[32, 2], &tensor_bytes);
    let mut cursor = Cursor::new(&gguf_bytes);
    let loader = load_gguf_from_reader(&mut cursor).expect("should parse q8_0 gguf");

    let tensor = loader
        .tensors
        .get("test.weight")
        .expect("tensor 'test.weight' not found");
    let q8 = match tensor {
        LoadedTensor::Q8_0(qw) => qw,
        _ => panic!("expected Q8_0 tensor"),
    };
    assert_eq!(q8.out_features(), 2);
    assert_eq!(q8.in_features(), 32);
}

#[test]
fn test_load_minimal_gguf_bad_magic() {
    use ember::loader::load_gguf_from_reader;
    use std::io::Cursor;

    let bad_bytes = vec![0u8; 16]; // all zeros, not "GGUF"
    let mut cursor = Cursor::new(&bad_bytes);
    let result = load_gguf_from_reader(&mut cursor);
    assert!(result.is_err(), "should fail on bad magic");
}

#[test]
fn test_quantized_weight_try_new_rejects_bad_shape() {
    use ember::quant::QuantizedWeight;

    let raw = vec![0u8; 34];
    let result = QuantizedWeight::try_new(raw, vec![1, 31]);

    assert!(result.is_err(), "q8_0 in_features must align to block size");
}

#[test]
fn test_matmul_q8_0_dimension_mismatch_returns_error() {
    use ember::backend::{Backend, CpuBackend};
    use ember::quant::QuantizedWeight;

    let backend = CpuBackend;
    let x = CpuTensor::zeroes(&[1, 16]);
    let weight = QuantizedWeight::try_new(vec![0u8; 34], vec![1, 32]).unwrap();

    let result = backend.matmul_q8_0(&x, &weight);

    assert!(result.is_err(), "q8_0 matmul mismatch should not panic");
}

#[test]
fn test_backend_causal_attention_shapes() {
    use ember::backend::{AttentionSpec, Backend, CpuBackend};

    let backend = CpuBackend;
    let q = CpuTensor::from_data(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = q.clone();
    let v = CpuTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let out = backend
        .causal_attention(
            &q,
            &k,
            &v,
            AttentionSpec {
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
            },
        )
        .expect("attention should run");

    assert_eq!(out.shape(), &[2, 2]);
    assert!((out.data()[0] - 1.0).abs() < 1e-5);
    assert!((out.data()[1] - 2.0).abs() < 1e-5);
    assert!(out.data()[2] > 1.0 && out.data()[2] < 3.0);
    assert!(out.data()[3] > 2.0 && out.data()[3] < 4.0);
}

#[test]
fn test_backend_cached_causal_attention_shapes() {
    use ember::backend::{Backend, CachedAttentionSpec, CpuBackend};

    let backend = CpuBackend;
    let q = CpuTensor::from_data(vec![1, 2], vec![0.0, 1.0]);
    let cached_k = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let cached_v = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];

    let out = backend
        .cached_causal_attention(
            &q,
            &cached_k,
            &cached_v,
            CachedAttentionSpec {
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                max_seq_len: 3,
                total_seq_len: 2,
            },
        )
        .expect("cached attention should run");

    assert_eq!(out.shape(), &[1, 2]);
    assert!(out.data()[0] > 1.0 && out.data()[0] < 3.0);
    assert!(out.data()[1] > 2.0 && out.data()[1] < 4.0);

    let mut scratch = Vec::with_capacity(3);
    let scratch_out = backend
        .cached_causal_attention_with_scratch(
            &q,
            &cached_k,
            &cached_v,
            CachedAttentionSpec {
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                max_seq_len: 3,
                total_seq_len: 2,
            },
            &mut scratch,
        )
        .expect("scratch-backed cached attention should run");

    assert_eq!(scratch_out, out);
    assert!(scratch.capacity() >= 3);
}

#[test]
fn test_sampler_temperature_zero() {
    use ember::sampler::sample_token;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let logits = vec![-1.0, 2.0, 0.5, 1.0];
    let mut rng = StdRng::seed_from_u64(42);
    let token = sample_token(&logits, 0.0, None, None, &mut rng);
    assert_eq!(
        token, 1,
        "temperature 0 should always pick argmax (index 1, value 2.0)"
    );
}

#[test]
fn test_sampler_temperature_nonzero() {
    use ember::sampler::sample_token;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // one token dominates heavily - sampling should still sometimes pick it
    let logits = vec![100.0, 0.0, 0.0, 0.0];
    let mut rng = StdRng::seed_from_u64(12345);
    let mut counts = [0usize; 4];
    for _ in 0..100 {
        let token = sample_token(&logits, 1.0, None, None, &mut rng);
        counts[token] += 1;
    }
    // first token should be picked most of the time
    assert!(
        counts[0] > 50,
        "dominant logit should be sampled most often, got {}",
        counts[0]
    );
}

#[test]
fn test_sampler_top_k() {
    use ember::sampler::sample_token;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // top_k=2 should only ever pick from the top 2 tokens,
    // even over many samples
    let logits = vec![3.0, 2.0, 1.0, 0.0];
    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..50 {
        let token = sample_token(&logits, 1.0, Some(2), None, &mut rng);
        assert!(
            token == 0 || token == 1,
            "top_k=2 should only allow tokens 0 or 1, got {}",
            token
        );
    }
}

#[test]
fn test_softmax_1d_all_masked() {
    use ember::sampler::softmax_1d;
    let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let probs = softmax_1d(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for &p in &probs {
        assert!((p - 1.0 / 3.0).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_1d_normal() {
    use ember::sampler::softmax_1d;
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax_1d(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(probs[0] < probs[1]);
    assert!(probs[1] < probs[2]);
}

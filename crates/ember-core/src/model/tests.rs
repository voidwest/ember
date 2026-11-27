use super::*;
use crate::backend::CpuBackend;
use crate::kv_cache::KVCache;
use crate::loader::{GgufLoader, GgufValue, LoadedTensor};
use crate::tensor::CpuTensor;
use std::collections::HashMap;

#[test]
fn gpt2_rejects_hostile_metadata_and_non_2d_linear_weights() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::Str("gpt2".to_string()),
    );
    metadata.insert("gpt2.block_count".to_string(), GgufValue::U32(1_000_000));
    let loader = GgufLoader::for_test(metadata, HashMap::new());
    let count_err = Gpt2::from_loader(loader)
        .err()
        .expect("hostile block_count must be rejected before allocation");
    assert!(count_err.to_string().contains("block_count"), "{count_err}");

    let mut loader = GgufLoader::for_test(
        HashMap::new(),
        HashMap::from([(
            "bad.weight".to_string(),
            LoadedTensor::F32(CpuTensor::from_data(vec![4], vec![0.0; 4])),
        )]),
    );
    let linear_err = take_gpt2_linear(&mut loader, "bad.weight", None)
        .err()
        .expect("non-2D linear weights must be rejected");
    assert!(
        linear_err.to_string().contains("2D linear weight"),
        "{linear_err}"
    );
}

fn backend() -> CpuBackend {
    CpuBackend
}

/// Build a Linear with weight `[in_f, out_f]` (Linear's layout) and
/// bias `[out_f]`, deterministic values.
fn linear(in_f: usize, out_f: usize, scale: f32) -> Linear<CpuBackend> {
    let data: Vec<f32> = (0..in_f * out_f)
        .map(|i| (i as f32) * scale + 0.5)
        .collect();
    let bias: Vec<f32> = (0..out_f).map(|i| -1.0 - i as f32).collect();
    Linear::new(
        CpuTensor::from_data(vec![in_f, out_f], data),
        Some(CpuTensor::from_data(vec![out_f], bias)),
    )
}

fn manual_matmul_bias(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // x [m, k] row-major, w [k, n] row-major (Linear stores [in, out]),
    // b [n]; out[i][j] = sum_l x[i][l] * w[l][j] + b[j]
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = b[j];
            for l in 0..k {
                acc += x[i * k + l] * w[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn linear_forward_matches_manual_matmul() {
    let backend = backend();
    let w = linear(4, 3, 0.25); // [in 4, out 3]
    let x = CpuTensor::from_data(vec![2, 4], vec![1.0, -2.0, 3.0, 0.5, 0.0, 1.0, 1.0, 1.0]);
    let out = w.forward(&backend, &x).expect("forward");
    // hand reference
    let wdata: Vec<f32> = (0..12).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let bdata: Vec<f32> = vec![-1.0, -2.0, -3.0];
    let expected = manual_matmul_bias(
        &[1.0, -2.0, 3.0, 0.5, 0.0, 1.0, 1.0, 1.0],
        &wdata,
        &bdata,
        2,
        4,
        3,
    );
    for (i, (got, want)) in out.data().iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "element {i}: got {got} want {want}"
        );
    }
}

#[test]
fn linear_forward_pair_matches_separate_forwards() {
    let backend = backend();
    let a = linear(3, 2, 0.5);
    let b = linear(3, 2, 1.0);
    let x = CpuTensor::from_data(vec![2, 3], vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5]);
    let (pa, pb) = a.forward_pair(&backend, &x, &b).expect("pair");
    let (sa, sb) = (
        a.forward(&backend, &x).expect("a"),
        b.forward(&backend, &x).expect("b"),
    );
    for (got, want) in pa.data().iter().zip(sa.data()) {
        assert!((got - want).abs() < 1e-5);
    }
    for (got, want) in pb.data().iter().zip(sb.data()) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn layer_norm_matches_manual_formula() {
    let backend = backend();
    // 2 rows x 3 cols
    let weight = CpuTensor::from_data(vec![3], vec![1.0, 2.0, 0.5]);
    let bias = CpuTensor::from_data(vec![3], vec![0.1, -0.2, 0.3]);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    let x = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = ln.forward(&backend, &x).expect("forward");
    for row in 0..2 {
        let mean = (0..3).map(|c| x.data()[row * 3 + c]).sum::<f32>() / 3.0;
        let var = (0..3)
            .map(|c| (x.data()[row * 3 + c] - mean).powi(2))
            .sum::<f32>()
            / 3.0;
        let rstd = (var + 1e-5).sqrt().recip();
        for c in 0..3 {
            let want =
                (x.data()[row * 3 + c] - mean) * rstd * ln.weight.data()[c] + ln.bias.data()[c];
            assert!(
                (out.data()[row * 3 + c] - want).abs() < 1e-4,
                "row {row} col {c}: got {} want {want}",
                out.data()[row * 3 + c]
            );
        }
    }
}

#[test]
fn mlp_matches_manual_gelu() {
    let backend = backend();
    let mlp = Mlp::new(linear(2, 4, 0.3), linear(4, 2, 0.7));
    let x = CpuTensor::from_data(vec![1, 2], vec![0.5, -1.5]);
    let out = mlp.forward(&backend, &x).expect("forward");
    // c_fc: [4, 2] * [2] -> hidden [4]; gelu; c_proj: [2, 4]
    let fc_w: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 + 0.5).collect();
    let fc_b: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
    let hidden = manual_matmul_bias(&[0.5, -1.5], &fc_w, &fc_b, 1, 2, 4);
    let gelu = |v: f32| 0.5 * v * (1.0 + libm::erff(v / std::f32::consts::SQRT_2));
    let hidden_g = hidden.iter().map(|&v| gelu(v)).collect::<Vec<_>>();
    let proj_w: Vec<f32> = (0..8).map(|i| (i as f32) * 0.7 + 0.5).collect();
    let proj_b: Vec<f32> = vec![-1.0, -2.0];
    let expected = manual_matmul_bias(&hidden_g, &proj_w, &proj_b, 1, 4, 2);
    for (i, (got, want)) in out.data().iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "element {i}: got {got} want {want}"
        );
    }
}

#[test]
fn attention_matches_manual_causal_scaled_dot_product() {
    let backend = backend();
    // 1 head, embed 4, seq 2.
    let c_attn = linear(4, 12, 0.5); // qkv: [in 4, out 12]
    let c_proj = linear(4, 4, 0.25);
    let attn = Attention::new(c_attn, c_proj, 1);
    let mut cache = KVCache::new(1, 1, 4, 8);
    let x = CpuTensor::from_data(vec![2, 4], vec![1.0, 0.0, 0.5, -0.5, 0.0, 1.0, 0.0, 0.25]);
    let out = attn
        .forward_with_cache(&backend, &x, &mut cache, 0)
        .expect("forward");

    // Manual reference.
    let w: Vec<f32> = (0..12 * 4).map(|i| (i as f32) * 0.5 + 0.5).collect();
    let b: Vec<f32> = (0..12).map(|i| -1.0 - i as f32).collect();
    let qkv = manual_matmul_bias(
        &[1.0, 0.0, 0.5, -0.5, 0.0, 1.0, 0.0, 0.25],
        &w,
        &b,
        2,
        4,
        12,
    );
    // qkv is column-interleaved per row: [q | k | v] with 4 columns each.
    let row_at = |row: usize, offset: usize| -> Vec<f32> {
        qkv[row * 12 + offset..row * 12 + offset + 4].to_vec()
    };
    let q: Vec<f32> = (0..2).flat_map(|r| row_at(r, 0)).collect();
    let k: Vec<f32> = (0..2).flat_map(|r| row_at(r, 4)).collect();
    let v: Vec<f32> = (0..2).flat_map(|r| row_at(r, 8)).collect();
    let d = 4.0f32.sqrt();
    // scores[seq][seq] = q[s] . k[t] / sqrt(d), causal mask, softmax
    let mut sm = [[0.0f32; 2]; 2];
    for s in 0..2 {
        let mut row = [f32::NEG_INFINITY; 2];
        for t in 0..=s {
            row[t] = (0..4).map(|l| q[s * 4 + l] * k[t * 4 + l]).sum::<f32>() / d;
        }
        let maxv = row[0].max(row[1]);
        let e0 = (row[0] - maxv).exp();
        let e1 = (row[1] - maxv).exp();
        sm[s][0] = e0 / (e0 + e1);
        sm[s][1] = e1 / (e0 + e1);
    }
    let attn_out: Vec<f32> = (0..2)
        .flat_map(|s| {
            (0..4)
                .map(|l| sm[s][0] * v[l] + sm[s][1] * v[4 + l])
                .collect::<Vec<_>>()
        })
        .collect();
    let proj_w: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let proj_b: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
    let expected = manual_matmul_bias(&attn_out, &proj_w, &proj_b, 2, 4, 4);
    for (i, (got, want)) in out.data().iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "element {i}: got {got} want {want}"
        );
    }
}

#[test]
fn block_forward_with_cache_preserves_shape_and_residual() {
    let backend = backend();
    let ln1 = LayerNorm::new(
        CpuTensor::from_data(vec![4], vec![1.0; 4]),
        CpuTensor::from_data(vec![4], vec![0.0; 4]),
        1e-5,
    );
    let attn = Attention::new(linear(4, 12, 0.2), linear(4, 4, 0.3), 1);
    let ln2 = LayerNorm::new(
        CpuTensor::from_data(vec![4], vec![1.0; 4]),
        CpuTensor::from_data(vec![4], vec![0.0; 4]),
        1e-5,
    );
    let mlp = Mlp::new(linear(4, 16, 0.1), linear(16, 4, 0.2));
    let block = Block::new(ln1, attn, ln2, mlp);
    let mut cache = KVCache::new(1, 1, 4, 8);
    let x = CpuTensor::from_data(vec![2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let out = block
        .forward_with_cache(&backend, &x, &mut cache, 0)
        .expect("forward");
    assert_eq!(out.shape(), &[2, 4], "block preserves shape");
    assert!(out.data().iter().all(|v| v.is_finite()));
}

#[cfg(test)]
mod cpu_session_tests {
    use super::with_cpu_session;

    #[test]
    fn session_enters_worker_and_preserves_nested_custom_pool() {
        std::thread::spawn(|| {
            assert!(rayon::current_thread_index().is_none());
            with_cpu_session(|| assert!(rayon::current_thread_index().is_some()));
        })
        .join()
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let index = rayon::current_thread_index();
            with_cpu_session(|| {
                assert_eq!(rayon::current_thread_index(), index);
                assert_eq!(rayon::current_num_threads(), 2);
            });
        });
    }

    #[test]
    fn session_propagates_panics_and_accepts_concurrent_callers() {
        let threads: Vec<_> = (0..4)
            .map(|index| {
                std::thread::spawn(move || {
                    assert!(std::panic::catch_unwind(|| with_cpu_session(|| panic!(
                        "session test"
                    )))
                    .is_err());
                    with_cpu_session(|| index * 2)
                })
            })
            .collect();
        for (index, thread) in threads.into_iter().enumerate() {
            assert_eq!(thread.join().unwrap(), index * 2);
        }
    }
}

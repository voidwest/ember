//! A/B harness: compare k_gemv column-blocking widths (EMBER_KGEMV_CB)
//! on the real projection shapes, interleaved to cancel thermal drift.
//! Usage: ember_ab_cb <model.gguf> <threads> <samples> [cbs...]
use ember::k_gemv::{matmul_k_gemv_parallel, matmul_k_gemv_serial};
use ember::loader::{load_gguf_with_k_strategy, LoadedTensor};
use ember::quant_k::KQuantWeight;
use rayon::ThreadPoolBuilder;
use std::time::Instant;

struct Proj {
    name: &'static str,
    weight: KQuantWeight,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let threads: usize = args[2].parse().unwrap();
    let samples: usize = args[3].parse().unwrap();
    let cbs: Vec<u8> = args[4..].iter().map(|s| s.parse().unwrap()).collect();
    assert!(!cbs.is_empty(), "need at least one cb");

    let loader = load_gguf_with_k_strategy(model_path, ember::quant_k::KStrategy::Auto, false)?;
    let mut weights: std::collections::HashMap<String, KQuantWeight> =
        std::collections::HashMap::new();
    for (name, t) in &loader.tensors {
        if let LoadedTensor::KQuant(w) = t {
            let parts: Vec<&str> = name.split('.').collect();
            let short = if parts.len() >= 3 && parts[0] == "blk" && parts[1] == "0" {
                let base = parts[2]
                    .trim_start_matches("attn_")
                    .trim_start_matches("ffn_");
                match base {
                    "q" => "q",
                    "k" => "k",
                    "v" => "v",
                    "output" => "o",
                    "gate" => "gate",
                    "up" => "up",
                    "down" => "down",
                    _ => continue,
                }
                .to_string()
            } else if name == "output.weight" {
                "lm_head".to_string()
            } else {
                continue;
            };
            weights.entry(short).or_insert_with(|| w.clone());
        }
    }
    let mut projs: Vec<Proj> = Vec::new();
    for short in ["q", "k", "v", "o", "gate", "up", "down", "lm_head"] {
        if let Some(w) = weights.get(short) {
            projs.push(Proj {
                name: short,
                weight: w.clone(),
            });
        }
    }
    if projs.is_empty() {
        anyhow::bail!("no K-quant projections");
    }

    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    println!("model={model_path} threads={threads} samples={samples} cbs={cbs:?}");

    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let next_f = |state: &mut u64| -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 33) as u32 as i32) as f32) * (1.0 / 1073741824.0)
    };

    // 3 interleaved rounds
    for round in 0..3 {
        for &cb in &cbs {
            std::env::set_var("EMBER_KGEMV_CB", cb.to_string());
            for p in &projs {
                let w = &p.weight;
                let in_f = w.in_features();
                let out_f = w.out_features();
                let src: Vec<f32> = (0..in_f).map(|_| next_f(&mut rng_state)).collect();
                let mut dst = vec![0.0f32; out_f];
                let mut ser = vec![0.0f32; out_f];
                let mut par = vec![0.0f32; out_f];
                // warmup
                matmul_k_gemv_serial(&src, w, &mut ser).unwrap();
                pool.install(|| matmul_k_gemv_parallel(&src, w, &mut par).unwrap());
                let mut ser_t = Vec::new();
                let mut par_t = Vec::new();
                for _ in 0..samples {
                    let t = Instant::now();
                    matmul_k_gemv_serial(&src, w, &mut dst).unwrap();
                    ser_t.push(t.elapsed().as_nanos() as f64);
                    let t = Instant::now();
                    pool.install(|| matmul_k_gemv_parallel(&src, w, &mut dst).unwrap());
                    par_t.push(t.elapsed().as_nanos() as f64);
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                let sm = med(&mut ser_t);
                let pm = med(&mut par_t);
                println!("cb={cb} round={round} {:<8} {:<5} {:>6}x{:<6} ser={:>9.1}us par={:>9.1}us  ok={}",
                    p.name, format!("{:?}", w.dtype()).to_lowercase(), in_f, out_f,
                    sm/1000.0, pm/1000.0, ser == par);
            }
        }
    }
    Ok(())
}

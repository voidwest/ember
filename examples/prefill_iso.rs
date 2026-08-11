//! Isolated prefill kernel timing: rows=26 on the real projection shapes.
//! Usage: prefill_iso <model.gguf>
use ember::loader::{load_gguf_with_k_strategy, LoadedTensor};
use ember::quant_k::KQuantWeight;
use std::time::Instant;

struct Proj {
    name: &'static str,
    weight: KQuantWeight,
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args().nth(1).unwrap();
    let loader = load_gguf_with_k_strategy(&model_path, ember::quant_k::KStrategy::Auto, false)?;
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
            } else {
                continue;
            };
            weights.entry(short).or_insert_with(|| w.clone());
        }
    }
    let mut projs: Vec<Proj> = Vec::new();
    for short in ["q", "k", "v", "o", "gate", "up", "down"] {
        if let Some(w) = weights.get(short) {
            projs.push(Proj {
                name: short,
                weight: w.clone(),
            });
        }
    }
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((rng >> 33) as u32 as i32) as f32) * (1.0 / 1073741824.0)
    };
    let rows = 26usize;
    for p in &projs {
        let w = &p.weight;
        let in_f = w.in_features();
        let out_f = w.out_features();
        let src: Vec<f32> = (0..rows * in_f).map(|_| next()).collect();
        let mut dst = vec![0.0f32; rows * out_f];
        // warmup
        ember::k_matmul::matmul_k_into(&src, rows, w, &mut dst).unwrap();
        dst.fill(0.0);
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            ember::k_matmul::matmul_k_into(&src, rows, w, &mut dst).unwrap();
            times.push(t.elapsed().as_nanos() as f64);
            dst.fill(0.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[3];
        let mut times_p = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            ember::k_matmul::matmul_k_into_parallel(&src, rows, w, &mut dst).unwrap();
            times_p.push(t.elapsed().as_nanos() as f64);
            dst.fill(0.0);
        }
        times_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_p = times_p[3];
        let macs = rows as f64 * in_f as f64 * out_f as f64;
        println!(
            "{:<7} {:>5}x{:<6} {:5} rows={:<3} ser={:>8.2} ms ({:>5.1} G) par={:>8.2} ms ({:>5.1} G, {:>4.1}x)",
            p.name, in_f, out_f, format!("{:?}", w.dtype()).to_lowercase(), rows,
            med / 1e6, 2.0 * macs / med,
            med_p / 1e6, 2.0 * macs / med_p, med / med_p
        );
    }
    Ok(())
}

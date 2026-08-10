//! K-quant batch-1 GEMV microbenchmark over real transformer projection
//! shapes (Llama-3.2-1B Q4_K_M / Q6_K), comparing:
//!   1. existing (legacy) Ember scalar row-1 kernel
//!   2. existing (legacy) Ember AVX2 row-1 kernel
//!   3. new serial GEMV (k_gemv)
//!   4. new parallel GEMV (k_gemv, coarse join split)
//!   5. Q8_0 packed VNNI decode (roofline/context baseline)
//!
//! Reports wall time, effective GB/s (weight bytes read), outputs/s,
//! speedup over legacy, and fraction of the host copy bandwidth.
//!
//! Usage: cargo bench --bench k_gemv -- --model Llama-3.2-1B-Instruct.Q4_K_M.gguf

use anyhow::bail;
use clap::Parser;
use ember::loader::{load_gguf_with_k_strategy, LoadedTensor};
use ember::quant_k::{KQuantDtype, KQuantWeight};
use serde::Serialize;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(about = "Benchmark Ember K-quant batch-1 GEMV paths on real projection shapes")]
struct Args {
    #[arg(long, hide = true)]
    bench: bool,
    /// GGUF model containing the projection weights.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Thread counts to test.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
    threads: Vec<usize>,
    /// Untimed calls of each mode before sampling.
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    /// Timed calls collected for each mode and configuration.
    #[arg(long, default_value_t = 9)]
    samples: usize,
    /// Deterministic activation vector (no model forward needed).
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Serialize)]
struct Row {
    schema_version: u32,
    benchmark: &'static str,
    model: String,
    projection: &'static str,
    in_features: usize,
    out_features: usize,
    dtype: &'static str,
    bytes: usize,
    macs: u64,
    path: &'static str,
    threads: usize,
    samples: usize,
    median_ns: u64,
    min_ns: u64,
    p95_ns: u64,
    gb_per_s: f64,
    outputs_per_s: f64,
    speedup_over_legacy_scalar: f64,
    fraction_of_copy_bw: f64,
}

fn median(mut v: Vec<u64>) -> (u64, u64, u64) {
    v.sort_unstable();
    let n = v.len();
    let p95 = v[((n as f64 * 0.95) as usize).min(n - 1)];
    (v[n / 2], v[0], p95)
}

#[allow(clippy::type_complexity)] // bench-harness callback signature
fn bench_one(
    src: &[f32],
    w: &KQuantWeight,
    dst: &mut [f32],
    warmups: usize,
    samples: usize,
    run: &dyn Fn(&[f32], &KQuantWeight, &mut [f32]), // clippy: type-complexity allowed below
) -> (u64, u64, u64) {
    for _ in 0..warmups {
        run(src, w, dst);
    }
    dst.fill(0.0);
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        run(src, w, dst);
        times.push(t0.elapsed().as_nanos() as u64);
        black_box(dst[0]);
        dst.fill(0.0);
    }
    median(times)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let Some(model_path) = args.model.as_ref() else {
        return Ok(());
    };
    if args.samples == 0 || args.threads.is_empty() {
        bail!("--samples and --threads must be non-empty");
    }
    let loader = load_gguf_with_k_strategy(model_path, ember::quant_k::KStrategy::Auto, false)?;
    // Collect the 8 representative projections from layer 0 (+ lm_head).
    struct Proj {
        name: &'static str,
        weight: KQuantWeight,
    }
    let mut projs: Vec<Proj> = Vec::new();
    // Build a short-name -> KQuantWeight map from the loaded tensor map
    // (layer-0 projections: blk.0.<proj>.weight; head: output.weight).
    let mut weights: std::collections::HashMap<String, KQuantWeight> =
        std::collections::HashMap::new();
    for (name, t) in &loader.tensors {
        if let LoadedTensor::KQuant(w) = t {
            // blk.0.attn_q.weight -> "q"; output.weight -> "lm_head"
            let parts: Vec<&str> = name.split('.').collect();
            let short = if parts.len() >= 3 && parts[0] == "blk" && parts[1] == "0" {
                let base = parts[2].trim_start_matches("attn_");
                let base = base.trim_start_matches("ffn_");
                match base {
                    "q" => "q",
                    "k" => "k",
                    "v" => "v",
                    "output" => "o",
                    "gate" => "gate",
                    "up" => "up",
                    "down" => "down",
                    other => {
                        eprintln!("skip {name}: {other}");
                        continue;
                    }
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
    for short in ["q", "k", "v", "o", "gate", "up", "down"] {
        if let Some(w) = weights.get(short) {
            projs.push(Proj {
                name: short,
                weight: w.clone(),
            });
        }
    }
    if let Some(w) = weights.get("output") {
        projs.push(Proj {
            name: "lm_head",
            weight: w.clone(),
        });
    }
    if projs.is_empty() {
        bail!("model has no K-quant projections (Q8_0/F32 only); nothing to benchmark");
    }

    // Q8_0 roofline: the LM head weight if present, else the first K weight
    let mut q8_baseline: Option<(String, Vec<f32>, ember::quant::QuantizedWeight)> = None;
    {
        let q8_head = loader.tensors.get("output.weight").and_then(|t| match t {
            LoadedTensor::Q8_0(w) => Some(w.clone()),
            _ => None,
        });
        if let Some(w) = q8_head {
            q8_baseline = Some((
                "lm_head".to_string(),
                (0..w.in_features())
                    .map(|i| ((i * 7919) % 100) as f32 / 50.0 - 1.0)
                    .collect(),
                w,
            ));
        }
    }
    let _model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048))?;

    let copy_bw_gib = 28.0; // measured multithreaded copy ceiling on the dossier host

    for &threads in &args.threads {
        for p in &projs {
            let w = &p.weight;
            let in_f = w.in_features();
            let out_f = w.out_features();
            let bytes = w.byte_len();
            let macs = in_f as u64 * out_f as u64;
            // deterministic activation in [-1, 1)
            let src: Vec<f32> = (0..in_f)
                .map(|i| ((i.wrapping_mul(7919) ^ args.seed as usize) % 100) as f32 / 50.0 - 1.0)
                .collect();
            let mut dst = vec![0.0f32; out_f];
            let dtype = match w.dtype() {
                KQuantDtype::Q4K => "q4_k",
                KQuantDtype::Q6K => "q6_k",
            };
            let model_name = model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let is_q4 = matches!(w.dtype(), KQuantDtype::Q4K);
            // 1. legacy scalar
            let (med, mn, p95) =
                bench_one(&src, w, &mut dst, args.warmups, args.samples, &|s, w, d| {
                    ember::k_matmul::bench_legacy_row1_scalar(s, w, d);
                });
            let legacy_ns = med as f64;
            // 2. legacy avx2
            let (med_a, mn_a, p95_a) = bench_one(
                &src,
                w,
                &mut dst,
                args.warmups,
                args.samples,
                &|s, w, d| unsafe {
                    if is_q4 {
                        ember::k_matmul_x86::bench_legacy_q4k_row1_avx2(s, w, 0, d);
                    } else {
                        ember::k_matmul_x86::bench_legacy_q6k_row1_avx2(s, w, 0, d);
                    }
                },
            );
            // 3. new serial
            let (med_s, mn_s, p95_s) =
                bench_one(&src, w, &mut dst, args.warmups, args.samples, &|s, w, d| {
                    ember::k_gemv::matmul_k_gemv_serial(s, w, d).unwrap();
                });
            // 4. new parallel (only meaningful when threads > 1)
            let (med_p, mn_p, p95_p) = if threads > 1 {
                bench_one(&src, w, &mut dst, args.warmups, args.samples, &|s, w, d| {
                    ember::k_gemv::matmul_k_gemv_parallel(s, w, d).unwrap();
                })
            } else {
                (med_s, mn_s, p95_s)
            };

            for (path, ns, mn_ns, p95_ns) in [
                ("legacy_scalar", med, mn, p95),
                ("legacy_avx2", med_a, mn_a, p95_a),
                ("new_serial", med_s, mn_s, p95_s),
                ("new_parallel", med_p, mn_p, p95_p),
            ] {
                let row = Row {
                    schema_version: 1,
                    benchmark: "k_gemv",
                    model: model_name.clone(),
                    projection: p.name,
                    in_features: in_f,
                    out_features: out_f,
                    dtype,
                    bytes,
                    macs,
                    path,
                    threads,
                    samples: args.samples,
                    median_ns: ns,
                    min_ns: mn_ns,
                    p95_ns,
                    gb_per_s: bytes as f64 / ns as f64,
                    outputs_per_s: out_f as f64 * 1e9 / ns as f64,
                    speedup_over_legacy_scalar: legacy_ns / ns as f64,
                    fraction_of_copy_bw: (bytes as f64 / ns as f64) / (copy_bw_gib * 1.073741824e9),
                };
                println!("{}", serde_json::to_string(&row)?);
            }
        }
        // Q8 roofline for the lm_head (same workload, Q8_0 path)
        if let Some((_name, src, w)) = &q8_baseline {
            let mut dst = vec![0.0f32; w.out_features()];
            let run = |s: &[f32], w: &ember::quant::QuantizedWeight, d: &mut [f32]| {
                let mut encoded = Vec::new();
                ember::quant::quantize_q8_0_into(s, &mut encoded);
                ember::backend::CpuBackend.matmul_q8_0_into(s, 1, w, d);
            };
            for _ in 0..args.warmups {
                run(src, w, &mut dst);
            }
            dst.fill(0.0);
            let mut times = Vec::with_capacity(args.samples);
            for _ in 0..args.samples {
                let t0 = Instant::now();
                run(src, w, &mut dst);
                times.push(t0.elapsed().as_nanos() as u64);
                black_box(dst[0]);
                dst.fill(0.0);
            }
            let (med, _, _) = median(times);
            println!(
                "{}",
                serde_json::to_string(&Row {
                    schema_version: 1,
                    benchmark: "k_gemv",
                    model: model_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    projection: "lm_head",
                    in_features: w.in_features(),
                    out_features: w.out_features(),
                    dtype: "q8_0",
                    bytes: w.byte_len(),
                    macs: w.in_features() as u64 * w.out_features() as u64,
                    path: "q8_roofline",
                    threads,
                    samples: args.samples,
                    median_ns: med,
                    min_ns: 0,
                    p95_ns: 0,
                    gb_per_s: w.byte_len() as f64 / med as f64,
                    outputs_per_s: w.out_features() as f64 * 1e9 / med as f64,
                    speedup_over_legacy_scalar: 0.0,
                    fraction_of_copy_bw: (w.byte_len() as f64 / med as f64)
                        / (copy_bw_gib * 1.073741824e9),
                })?
            );
        }
    }
    Ok(())
}

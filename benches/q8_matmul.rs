use anyhow::{bail, Context};
use clap::{Parser, ValueEnum};
use ember::backend::CpuBackend;
use ember::loader::{load_gguf, LoadedTensor};
use ember::quant::{QuantizedWeight, QuantizedWeightVnni};
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum CacheState {
    Hot,
    Cold,
}

#[derive(Clone, Copy)]
enum ProjectionMode {
    Separate,
    Paired,
    PackedPaired,
}

#[derive(Debug, Parser)]
#[command(about = "Benchmark Ember Q8 matmuls using real model projection shapes")]
struct Args {
    /// Cargo passes this marker to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,

    /// GGUF model containing the projection weights.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Transformer layer whose gate/up projections are measured.
    #[arg(long, default_value_t = 0)]
    layer: usize,

    /// Activation row counts, covering decode and prompt regimes.
    #[arg(long, value_delimiter = ',', default_value = "1,6,32,128")]
    rows: Vec<usize>,

    /// Rayon thread counts to test.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    threads: Vec<usize>,

    /// CPU-cache states. Cold runs sweep a separate buffer before timing.
    #[arg(long, value_delimiter = ',', default_value = "hot,cold")]
    cache: Vec<CacheState>,

    /// Untimed calls of each mode before sampling.
    #[arg(long, default_value_t = 3)]
    warmups: usize,

    /// Timed calls collected for each mode and configuration.
    #[arg(long, default_value_t = 9)]
    samples: usize,

    /// Size of the cache-thrashing buffer used by cold runs.
    #[arg(long, default_value_t = 64)]
    cache_mib: usize,
}

#[derive(Serialize)]
struct TimingStats {
    samples_ns: Vec<u64>,
    min_ns: u64,
    median_ns: u64,
    p95_ns: u64,
    mean_ns: f64,
}

#[derive(Serialize)]
struct ResultRow {
    schema_version: u32,
    benchmark: &'static str,
    model: String,
    layer: usize,
    first_tensor: String,
    second_tensor: String,
    rows: usize,
    input_features: usize,
    output_features: usize,
    threads: usize,
    cache_state: CacheState,
    warmups: usize,
    samples_per_mode: usize,
    separate: TimingStats,
    paired: TimingStats,
    paired_speedup: f64,
    paired_gflops: f64,
    packed_paired: Option<TimingStats>,
    packed_speedup_over_paired: Option<f64>,
    packed_weight_bytes: Option<usize>,
    exact_parity: bool,
}

fn take_q8(loader: &mut ember::loader::GgufLoader, name: &str) -> anyhow::Result<QuantizedWeight> {
    match loader
        .tensors
        .remove(name)
        .with_context(|| format!("missing tensor '{name}'"))?
    {
        LoadedTensor::Q8_0(weight) => Ok(weight),
        LoadedTensor::F32(_) => bail!("tensor '{name}' is not Q8_0"),
    }
}

fn activation_data(rows: usize, features: usize) -> Vec<f32> {
    (0..rows * features)
        .map(|index| {
            let phase = (index % features) as f32 * 0.013 + (index / features) as f32 * 0.17;
            phase.sin() * 0.75 + phase.cos() * 0.125
        })
        .collect()
}

fn sweep_cache(buffer: &mut [u64]) {
    for value in buffer.iter_mut().step_by(8) {
        *value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    }
    black_box(buffer);
}

#[allow(clippy::too_many_arguments)]
fn invoke(
    mode: ProjectionMode,
    backend: &CpuBackend,
    input: &[f32],
    rows: usize,
    first: &QuantizedWeight,
    second: &QuantizedWeight,
    packed: Option<(&QuantizedWeightVnni, &QuantizedWeightVnni)>,
    first_output: &mut [f32],
    second_output: &mut [f32],
) {
    match mode {
        ProjectionMode::Separate => {
            backend.matmul_q8_0_into(input, rows, first, first_output);
            backend.matmul_q8_0_into(input, rows, second, second_output);
        }
        ProjectionMode::Paired => {
            backend.matmul_q8_0_pair_into(input, rows, first, second, first_output, second_output)
        }
        ProjectionMode::PackedPaired => {
            let (packed_first, packed_second) =
                packed.expect("packed mode requires packed weights");
            let input_features = packed_first.in_features();
            let output_features = packed_first.out_features();
            for row in 0..rows {
                backend.matmul_q8_0_packed_pair_into(
                    &input[row * input_features..(row + 1) * input_features],
                    packed_first,
                    packed_second,
                    &mut first_output[row * output_features..(row + 1) * output_features],
                    &mut second_output[row * output_features..(row + 1) * output_features],
                );
            }
        }
    }
    black_box((&*first_output, &*second_output));
}

fn packed_vnni_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn stats(mut samples: Vec<u64>) -> TimingStats {
    samples.sort_unstable();
    let count = samples.len();
    let p95_index = count.saturating_mul(95).div_ceil(100).saturating_sub(1);
    TimingStats {
        min_ns: samples[0],
        median_ns: samples[count / 2],
        p95_ns: samples[p95_index.min(count - 1)],
        mean_ns: samples.iter().map(|&value| value as f64).sum::<f64>() / count as f64,
        samples_ns: samples,
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let Some(model) = args.model.as_ref() else {
        return Ok(());
    };
    if args.samples == 0 {
        bail!("--samples must be greater than zero");
    }
    if args.rows.contains(&0) || args.threads.contains(&0) {
        bail!("row and thread counts must be greater than zero");
    }

    let model_path = model.to_string_lossy().into_owned();
    let mut loader = load_gguf(&model_path)?;
    let first_name = format!("blk.{}.ffn_gate.weight", args.layer);
    let second_name = format!("blk.{}.ffn_up.weight", args.layer);
    let first = take_q8(&mut loader, &first_name)?;
    let second = take_q8(&mut loader, &second_name)?;
    if first.in_features() != second.in_features() || first.out_features() != second.out_features()
    {
        bail!(
            "gate/up shapes differ: {}x{} versus {}x{}",
            first.in_features(),
            first.out_features(),
            second.in_features(),
            second.out_features()
        );
    }
    let packed = packed_vnni_supported().then(|| {
        (
            QuantizedWeightVnni::from_quantized(&first),
            QuantizedWeightVnni::from_quantized(&second),
        )
    });

    let backend = CpuBackend;
    let cache_words = args
        .cache_mib
        .checked_mul(1024 * 1024)
        .context("cache buffer size overflow")?
        / std::mem::size_of::<u64>();
    let mut cache_buffer = vec![0u64; cache_words];

    for &threads in &args.threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("build benchmark Rayon pool")?;
        for &rows in &args.rows {
            let input = activation_data(rows, first.in_features());
            let output_len = rows
                .checked_mul(first.out_features())
                .context("benchmark output size overflow")?;
            let mut first_output = vec![0.0; output_len];
            let mut second_output = vec![0.0; output_len];
            let mut oracle_first = vec![0.0; output_len];
            let mut oracle_second = vec![0.0; output_len];

            pool.install(|| {
                invoke(
                    ProjectionMode::Separate,
                    &backend,
                    &input,
                    rows,
                    &first,
                    &second,
                    None,
                    &mut oracle_first,
                    &mut oracle_second,
                );
                invoke(
                    ProjectionMode::Paired,
                    &backend,
                    &input,
                    rows,
                    &first,
                    &second,
                    None,
                    &mut first_output,
                    &mut second_output,
                );
            });
            assert_eq!(oracle_first, first_output, "gate output parity");
            assert_eq!(oracle_second, second_output, "up output parity");
            if let Some((packed_first, packed_second)) = &packed {
                pool.install(|| {
                    invoke(
                        ProjectionMode::PackedPaired,
                        &backend,
                        &input,
                        rows,
                        &first,
                        &second,
                        Some((packed_first, packed_second)),
                        &mut first_output,
                        &mut second_output,
                    )
                });
                assert_eq!(oracle_first, first_output, "packed gate output parity");
                assert_eq!(oracle_second, second_output, "packed up output parity");
            }

            for &cache_state in &args.cache {
                for _ in 0..args.warmups {
                    let modes = if packed.is_some() {
                        &[
                            ProjectionMode::Separate,
                            ProjectionMode::Paired,
                            ProjectionMode::PackedPaired,
                        ][..]
                    } else {
                        &[ProjectionMode::Separate, ProjectionMode::Paired][..]
                    };
                    for &mode in modes {
                        if matches!(cache_state, CacheState::Cold) {
                            sweep_cache(&mut cache_buffer);
                        }
                        pool.install(|| {
                            invoke(
                                mode,
                                &backend,
                                &input,
                                rows,
                                &first,
                                &second,
                                packed.as_ref().map(|(first, second)| (first, second)),
                                &mut first_output,
                                &mut second_output,
                            )
                        });
                    }
                }

                let mut separate_samples = Vec::with_capacity(args.samples);
                let mut paired_samples = Vec::with_capacity(args.samples);
                let mut packed_samples = Vec::with_capacity(args.samples);
                let order = if packed.is_some() {
                    &[
                        ProjectionMode::Separate,
                        ProjectionMode::Paired,
                        ProjectionMode::PackedPaired,
                        ProjectionMode::PackedPaired,
                        ProjectionMode::Paired,
                        ProjectionMode::Separate,
                    ][..]
                } else {
                    &[
                        ProjectionMode::Separate,
                        ProjectionMode::Paired,
                        ProjectionMode::Paired,
                        ProjectionMode::Separate,
                    ][..]
                };
                let mut iteration = 0;
                while separate_samples.len() < args.samples
                    || paired_samples.len() < args.samples
                    || (packed.is_some() && packed_samples.len() < args.samples)
                {
                    let mode = order[iteration % order.len()];
                    iteration += 1;
                    let samples = match mode {
                        ProjectionMode::Separate => &mut separate_samples,
                        ProjectionMode::Paired => &mut paired_samples,
                        ProjectionMode::PackedPaired => &mut packed_samples,
                    };
                    if samples.len() == args.samples {
                        continue;
                    }
                    if matches!(cache_state, CacheState::Cold) {
                        sweep_cache(&mut cache_buffer);
                    }
                    let started = Instant::now();
                    pool.install(|| {
                        invoke(
                            mode,
                            &backend,
                            &input,
                            rows,
                            &first,
                            &second,
                            packed.as_ref().map(|(first, second)| (first, second)),
                            &mut first_output,
                            &mut second_output,
                        )
                    });
                    samples.push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                }

                let separate = stats(separate_samples);
                let paired = stats(paired_samples);
                let packed_paired = packed.is_some().then(|| stats(packed_samples));
                let operations = 2.0
                    * 2.0
                    * rows as f64
                    * first.in_features() as f64
                    * first.out_features() as f64;
                let result = ResultRow {
                    schema_version: 1,
                    benchmark: "q8_gate_up",
                    model: model_path.clone(),
                    layer: args.layer,
                    first_tensor: first_name.clone(),
                    second_tensor: second_name.clone(),
                    rows,
                    input_features: first.in_features(),
                    output_features: first.out_features(),
                    threads,
                    cache_state,
                    warmups: args.warmups,
                    samples_per_mode: args.samples,
                    paired_speedup: separate.median_ns as f64 / paired.median_ns as f64,
                    paired_gflops: operations / paired.median_ns as f64,
                    packed_speedup_over_paired: packed_paired
                        .as_ref()
                        .map(|packed| paired.median_ns as f64 / packed.median_ns as f64),
                    packed_weight_bytes: packed
                        .as_ref()
                        .map(|(first, second)| first.byte_len().saturating_add(second.byte_len())),
                    packed_paired,
                    separate,
                    paired,
                    exact_parity: true,
                };
                println!("{}", serde_json::to_string(&result)?);
            }
        }
    }
    Ok(())
}

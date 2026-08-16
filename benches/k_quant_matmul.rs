//! Auditable benchmark for the canonical Q4_K/Q6_K × Q8_K matmul.
//!
//! Samples are interleaved by path. Every JSON record includes raw timings,
//! checksums, actual dispatch, the pinned model hash, and build/host provenance.
//!
//! Usage:
//! `cargo bench --bench k_quant_matmul -- --model model.gguf --k-strategy x86 --expected-model-sha256 SHA256`

use anyhow::{bail, Context};
use clap::Parser;
use ember::loader::{load_gguf_with_k_strategy, LoadedTensor};
use ember::plan::{resolve_kernel, PLAN_KERNEL_REVISION};
use ember::quant_k::{KExecution, KQuantDtype, KQuantWeight, KStrategy};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, hide = true)]
    bench: bool,
    #[arg(long)]
    model: Option<PathBuf>,
    /// Fail closed unless the model file has this SHA-256.
    #[arg(long)]
    expected_model_sha256: Option<String>,
    /// Development-only opt-out; release evidence must never set this.
    #[arg(long)]
    allow_unpinned_model: bool,
    /// Exact GGUF tensor name; omit to sample up to eight sorted K tensors.
    #[arg(long)]
    tensor: Option<String>,
    #[arg(long, value_delimiter = ',', default_value = "1,17")]
    rows: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,4,8")]
    threads: Vec<usize>,
    /// Explicit loader tier. `x86` and `scalar` fail rather than falling back.
    #[arg(long, default_value = "x86")]
    k_strategy: String,
    /// Omit the slow exact-f32 oracle from timed samples (not the preflight).
    #[arg(long)]
    skip_exact: bool,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    #[arg(long, default_value_t = 9)]
    samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchPath {
    ExactF32,
    Q8KSerial,
    Q8KParallel,
}

impl BenchPath {
    fn name(self) -> &'static str {
        match self {
            Self::ExactF32 => "exact_f32_oracle",
            Self::Q8KSerial => "q8_k_serial",
            Self::Q8KParallel => "q8_k_parallel_requested",
        }
    }

    fn parallel_requested(self) -> bool {
        matches!(self, Self::Q8KParallel)
    }
}

#[derive(Serialize)]
struct Preflight {
    exact_f32_checksum: String,
    q8_k_serial_checksum: String,
    q8_k_parallel_checksum: String,
    serial_dispatch: &'static str,
    parallel_dispatch: &'static str,
    serial_parallel_bit_identical: bool,
    all_outputs_finite: bool,
    q8_k_vs_exact_normalized_rmse: f64,
}

#[derive(Serialize)]
struct RawSample {
    round: usize,
    order_in_round: usize,
    path: &'static str,
    actual_scheduler: &'static str,
    elapsed_ns: u64,
    output_checksum: String,
}

#[derive(Serialize)]
struct PathSummary {
    path: &'static str,
    actual_scheduler: &'static str,
    samples_ns: Vec<u64>,
    median_ns: u64,
    output_checksum: String,
    physical_weight_gb_per_s: f64,
    logical_row_weight_gb_per_s: f64,
}

#[derive(Serialize)]
struct Provenance {
    ember_version: &'static str,
    build_git_commit: &'static str,
    build_git_dirty: &'static str,
    runtime_git_commit: String,
    runtime_git_dirty: String,
    rustc_version: &'static str,
    target: &'static str,
    os: &'static str,
    cpu_model: String,
    unix_time_seconds: u64,
    command: Vec<String>,
    benchmark_executable_sha256: String,
    cpu_features: Vec<&'static str>,
    available_parallelism: usize,
}

#[derive(Serialize)]
struct Case<'a> {
    schema_version: u32,
    benchmark: &'static str,
    model_path: String,
    model_sha256: &'a str,
    model_bytes: u64,
    model_hash_pinned: bool,
    tensor: &'a str,
    dtype: &'static str,
    input_features: usize,
    output_features: usize,
    rows: usize,
    rayon_threads: usize,
    requested_k_strategy: &'a str,
    recorded_execution: &'static str,
    kernel: &'static str,
    kernel_revision: u32,
    required_cpu_features: Option<&'static str>,
    transient_q8_k_workspace_bytes: usize,
    workspace_scope: &'static str,
    warmups: usize,
    samples_per_path: usize,
    preflight: Preflight,
    raw_interleaved_samples: Vec<RawSample>,
    summaries: Vec<PathSummary>,
    provenance: &'a Provenance,
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn checksum(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn run_path(
    path: BenchPath,
    src: &[f32],
    rows: usize,
    weight: &KQuantWeight,
    dst: &mut [f32],
) -> anyhow::Result<&'static str> {
    dst.fill(0.0);
    match path {
        BenchPath::ExactF32 => {
            ember::k_matmul::bench_exact_f32(src, weight, dst);
            Ok("serial-oracle")
        }
        BenchPath::Q8KSerial => {
            ember::k_quant_matmul::matmul_k_q8_into_with_dispatch(src, rows, weight, dst, false)
                .map_err(anyhow::Error::msg)
        }
        BenchPath::Q8KParallel => {
            ember::k_quant_matmul::matmul_k_q8_into_with_dispatch(src, rows, weight, dst, true)
                .map_err(anyhow::Error::msg)
        }
    }
}

fn scheduler(path: BenchPath, rows: usize, weight: &KQuantWeight) -> &'static str {
    match path {
        BenchPath::ExactF32 => "serial-oracle",
        _ => ember::k_quant_matmul::scheduler_name(rows, weight, path.parallel_requested()),
    }
}

fn preflight(src: &[f32], rows: usize, weight: &KQuantWeight) -> anyhow::Result<Preflight> {
    let output_len = rows
        .checked_mul(weight.out_features())
        .context("benchmark output shape overflow")?;
    let mut exact = vec![0.0; output_len];
    let mut serial = vec![0.0; output_len];
    let mut parallel = vec![0.0; output_len];
    let _ = run_path(BenchPath::ExactF32, src, rows, weight, &mut exact)?;
    let serial_dispatch = run_path(BenchPath::Q8KSerial, src, rows, weight, &mut serial)?;
    let parallel_dispatch = run_path(BenchPath::Q8KParallel, src, rows, weight, &mut parallel)?;

    let bit_identical = serial
        .iter()
        .zip(&parallel)
        .all(|(left, right)| left.to_bits() == right.to_bits());
    if !bit_identical {
        bail!("correctness preflight: serial and requested-parallel outputs differ");
    }
    let all_finite = exact
        .iter()
        .chain(&serial)
        .chain(&parallel)
        .all(|value| value.is_finite());
    if !all_finite {
        bail!("correctness preflight: non-finite output");
    }
    let squared_error: f64 = exact
        .iter()
        .zip(&serial)
        .map(|(&reference, &actual)| f64::from(actual - reference).powi(2))
        .sum();
    let squared_reference: f64 = exact
        .iter()
        .map(|&reference| f64::from(reference).powi(2))
        .sum();
    let normalized_rmse = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    if normalized_rmse > 0.05 {
        bail!("correctness preflight: Q8_K normalized RMSE {normalized_rmse:.6} exceeds 0.05");
    }

    Ok(Preflight {
        exact_f32_checksum: checksum(&exact),
        q8_k_serial_checksum: checksum(&serial),
        q8_k_parallel_checksum: checksum(&parallel),
        serial_dispatch,
        parallel_dispatch,
        serial_parallel_bit_identical: bit_identical,
        all_outputs_finite: all_finite,
        q8_k_vs_exact_normalized_rmse: normalized_rmse,
    })
}

fn detected_cpu_features() -> Vec<&'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        for (name, present) in [
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("ssse3", std::is_x86_feature_detected!("ssse3")),
        ] {
            if present {
                features.push(name);
            }
        }
        features
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

fn runtime_git_state() -> (String, String) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| (!output.stdout.is_empty()).to_string())
        .unwrap_or_else(|| "unknown".to_string());
    (commit, dirty)
}

fn cpu_model() -> String {
    #[cfg(target_os = "linux")]
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(model) = cpuinfo.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|value| value.split_once(':'))
                .map(|(_, value)| value.trim())
        })
    {
        return model.to_string();
    }
    "unknown".to_string()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let Some(model) = args.model.as_ref() else {
        return Ok(());
    };
    if args.samples == 0 || args.threads.is_empty() || args.rows.is_empty() {
        bail!("--samples, --rows, and --threads must be non-empty");
    }
    if args.rows.contains(&0) || args.threads.contains(&0) {
        bail!("--rows and --threads values must be nonzero");
    }
    let strategy = KStrategy::from_cli(&args.k_strategy).map_err(anyhow::Error::msg)?;
    if !matches!(strategy, KStrategy::Scalar | KStrategy::X86) {
        bail!("audited benchmark requires explicit --k-strategy scalar or x86");
    }
    let path_count = if args.skip_exact { 2 } else { 3 };
    if args.warmups < path_count
        || args.samples < path_count
        || !args.warmups.is_multiple_of(path_count)
        || !args.samples.is_multiple_of(path_count)
    {
        bail!("--warmups and --samples must be positive multiples of the {path_count} timed paths");
    }
    if args.expected_model_sha256.is_none() && !args.allow_unpinned_model {
        bail!(
            "--expected-model-sha256 is required with --model (or explicitly opt out with --allow-unpinned-model)"
        );
    }
    const MAX_MODEL_BYTES: u64 = 2_000_000_000;
    let metadata = std::fs::metadata(model)?;
    if !metadata.is_file() {
        bail!("model path must be a regular file");
    }
    let model_bytes = metadata.len();
    if model_bytes > MAX_MODEL_BYTES {
        bail!(
            "model is {model_bytes} bytes; this benchmark's audited 1B/1.5B ladder caps files at {MAX_MODEL_BYTES} bytes"
        );
    }
    if let Some(expected) = args.expected_model_sha256.as_ref()
        && (expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        bail!("--expected-model-sha256 must be exactly 64 hexadecimal characters");
    }
    let model_sha256 = sha256_file(model)?;
    if let Some(expected) = args.expected_model_sha256.as_ref()
        && !model_sha256.eq_ignore_ascii_case(expected)
    {
        bail!("model SHA-256 {model_sha256} != expected {expected}");
    }
    let executable = std::env::current_exe()?;
    let (runtime_git_commit, runtime_git_dirty) = runtime_git_state();
    let provenance = Provenance {
        ember_version: env!("CARGO_PKG_VERSION"),
        build_git_commit: option_env!("EMBER_GIT_COMMIT").unwrap_or("unknown"),
        build_git_dirty: option_env!("EMBER_GIT_DIRTY").unwrap_or("unknown"),
        runtime_git_commit,
        runtime_git_dirty,
        rustc_version: option_env!("EMBER_RUSTC_VERSION").unwrap_or("unknown"),
        target: option_env!("EMBER_TARGET").unwrap_or(std::env::consts::ARCH),
        os: std::env::consts::OS,
        cpu_model: cpu_model(),
        unix_time_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        command: std::env::args().collect(),
        benchmark_executable_sha256: sha256_file(&executable)?,
        cpu_features: detected_cpu_features(),
        available_parallelism: std::thread::available_parallelism()?.get(),
    };

    // `allow_fallback=false`: explicit scalar/x86 requests are fail-closed.
    let loader = load_gguf_with_k_strategy(model, strategy, false)?;
    let mut weights: Vec<_> = loader
        .tensors
        .iter()
        .filter_map(|(name, loaded)| match loaded {
            LoadedTensor::KQuant(weight)
                if args.tensor.as_ref().is_none_or(|wanted| wanted == name) =>
            {
                Some((name.as_str(), weight))
            }
            _ => None,
        })
        .collect();
    weights.sort_unstable_by_key(|(name, _)| *name);
    weights.truncate(if args.tensor.is_some() { 1 } else { 8 });
    if weights.is_empty() {
        bail!("model has no matching compressed Q4_K/Q6_K tensor");
    }

    for &rows in &args.rows {
        for &threads in &args.threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?;
            for &(name, weight) in &weights {
                let input_len = rows
                    .checked_mul(weight.in_features())
                    .context("benchmark input shape overflow")?;
                let src: Vec<f32> = (0..input_len)
                    .map(|index| ((index * 7919) % 100) as f32 / 50.0 - 1.0)
                    .collect();
                let output_len = rows
                    .checked_mul(weight.out_features())
                    .context("benchmark output shape overflow")?;
                let workspace_bytes = rows
                    .checked_mul(weight.in_features() / ember::quant_k::QK_K)
                    .and_then(|blocks| blocks.checked_mul(ember::k_quant_matmul::Q8_K_BLOCK_BYTES))
                    .context("Q8_K workspace size overflow")?;
                let dtype = match weight.dtype() {
                    KQuantDtype::Q4K => "q4_k",
                    KQuantDtype::Q6K => "q6_k",
                };
                let execution = match weight.execution() {
                    KExecution::EagerF32 => bail!("packed benchmark received eager-f32 weight"),
                    KExecution::CompressedScalar => "compressed_scalar",
                    KExecution::CompressedX86 => "compressed_x86",
                };
                let kernel = resolve_kernel(dtype, execution);
                let timed_paths: Vec<_> = if args.skip_exact {
                    vec![BenchPath::Q8KSerial, BenchPath::Q8KParallel]
                } else {
                    vec![
                        BenchPath::ExactF32,
                        BenchPath::Q8KSerial,
                        BenchPath::Q8KParallel,
                    ]
                };

                let (preflight, raw_samples, summaries) = pool.install(|| {
                    let preflight = preflight(&src, rows, weight)?;
                    let mut destinations = vec![vec![0.0f32; output_len]; timed_paths.len()];
                    // Warm paths in rotating order as well; no path receives a
                    // systematic thermal/cache position.
                    for round in 0..args.warmups {
                        for offset in 0..timed_paths.len() {
                            let index = (round + offset) % timed_paths.len();
                            run_path(
                                timed_paths[index],
                                &src,
                                rows,
                                weight,
                                &mut destinations[index],
                            )?;
                        }
                    }

                    let mut raw = Vec::with_capacity(args.samples * timed_paths.len());
                    let mut samples_by_path =
                        vec![Vec::with_capacity(args.samples); timed_paths.len()];
                    let mut checksums_by_path = vec![String::new(); timed_paths.len()];
                    for round in 0..args.samples {
                        for offset in 0..timed_paths.len() {
                            let index = (round + offset) % timed_paths.len();
                            let path = timed_paths[index];
                            let started = Instant::now();
                            let actual_scheduler =
                                run_path(path, &src, rows, weight, &mut destinations[index])?;
                            let elapsed_ns = started.elapsed().as_nanos() as u64;
                            black_box(destinations[index].as_slice());
                            let output_checksum = checksum(&destinations[index]);
                            if checksums_by_path[index].is_empty() {
                                checksums_by_path[index] = output_checksum.clone();
                            } else if checksums_by_path[index] != output_checksum {
                                bail!("non-deterministic output checksum for {}", path.name());
                            }
                            samples_by_path[index].push(elapsed_ns);
                            raw.push(RawSample {
                                round,
                                order_in_round: offset,
                                path: path.name(),
                                actual_scheduler,
                                elapsed_ns,
                                output_checksum,
                            });
                        }
                    }
                    let summaries = timed_paths
                        .iter()
                        .enumerate()
                        .map(|(index, &path)| {
                            let samples_ns = samples_by_path[index].clone();
                            let mut sorted = samples_ns.clone();
                            sorted.sort_unstable();
                            let middle = sorted.len() / 2;
                            let median_ns = if sorted.len().is_multiple_of(2) {
                                ((u128::from(sorted[middle - 1]) + u128::from(sorted[middle])) / 2)
                                    as u64
                            } else {
                                sorted[middle]
                            };
                            PathSummary {
                                path: path.name(),
                                actual_scheduler: scheduler(path, rows, weight),
                                samples_ns,
                                median_ns,
                                output_checksum: checksums_by_path[index].clone(),
                                physical_weight_gb_per_s: weight.byte_len() as f64
                                    / median_ns as f64,
                                logical_row_weight_gb_per_s: weight.byte_len() as f64 * rows as f64
                                    / median_ns as f64,
                            }
                        })
                        .collect();
                    Ok::<_, anyhow::Error>((preflight, raw, summaries))
                })?;

                println!(
                    "{}",
                    serde_json::to_string(&Case {
                        schema_version: 4,
                        benchmark: "k_q8_k_matmul",
                        model_path: model.display().to_string(),
                        model_sha256: &model_sha256,
                        model_bytes,
                        model_hash_pinned: args.expected_model_sha256.is_some(),
                        tensor: name,
                        dtype,
                        input_features: weight.in_features(),
                        output_features: weight.out_features(),
                        rows,
                        rayon_threads: threads,
                        requested_k_strategy: &args.k_strategy,
                        recorded_execution: execution,
                        kernel: kernel.name(),
                        kernel_revision: PLAN_KERNEL_REVISION,
                        required_cpu_features: kernel.cpu_feature(),
                        transient_q8_k_workspace_bytes: workspace_bytes,
                        workspace_scope: "per invoking OS thread; retained to peak TLS capacity after warmup; nested calls take independent storage",
                        warmups: args.warmups,
                        samples_per_path: args.samples,
                        preflight,
                        raw_interleaved_samples: raw_samples,
                        summaries,
                        provenance: &provenance,
                    })?
                );
            }
        }
    }
    Ok(())
}

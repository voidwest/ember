//! Lightweight execution tracer for Ember's CPU inference path.
//!
//! Uses thread-local storage so the tracing API is a near-zero-overhead no-op when
//! disabled.  Instrumentation hooks are placed at semantic transformer boundaries:
//! embedding, RMSNorm, Q/K/V projections, RoPE, attention-score computation,
//! attention-output projection, MLP gate/up/down projections, final norm, and LM head.
//!
//! ## Usage
//!
//! ```ignore
//! use ember::trace::{Tracer, enable_tracing, disable_tracing};
//!
//! enable_tracing("decode", 0);
//! // ... run inference ...
//! let report = disable_tracing().unwrap();
//! println!("{}", report.summary());
//! ```

use crate::tensor::CpuTensor;
use std::cell::RefCell;
use std::time::Instant;

// ---------------------------------------------------------------------------
// thread-local tracer singleton
// ---------------------------------------------------------------------------

thread_local! {
    static TRACER: RefCell<Option<Tracer>> = const { RefCell::new(None) };
    static CURRENT_LAYER: RefCell<usize> = const { RefCell::new(0) };
    static VALUES_LEVEL: RefCell<TraceValuesLevel> = const { RefCell::new(TraceValuesLevel::None) };
}

/// Set the trace-values collection level.
#[inline]
pub fn set_values_level(level: TraceValuesLevel) {
    VALUES_LEVEL.with(|v| *v.borrow_mut() = level);
}

/// Check whether output value collection is enabled.
#[inline]
pub fn values_enabled() -> bool {
    VALUES_LEVEL.with(|v| *v.borrow() != TraceValuesLevel::None)
}

/// Compute L2 norm, abs_max, and a lightweight fingerprint from tensor data.
///
/// The fingerprint is a 64-bit hash built from every 64th element, mixed with
/// a multiplicative constant and the element index.  It is stable across runs
/// with the same tensor contents but not collision-resistant — use it to
/// detect numerical divergence, not for cryptographic integrity.
pub fn compute_tensor_values(data: &[f32]) -> TraceValues {
    let mut sum_sq: f64 = 0.0;
    let mut abs_max: f32 = 0.0;
    let mut fp: u64 = 0x9e3779b97f4a7c15; // golden-ratio start
    for (i, &val) in data.iter().enumerate() {
        let v = val as f64;
        sum_sq += v * v;
        let a = val.abs();
        if a > abs_max {
            abs_max = a;
        }
        // Mix every 64th element into the fingerprint
        if i % 64 == 0 {
            fp = fp
                .wrapping_mul(6364136223846793005)
                .wrapping_add(val.to_bits() as u64);
            fp ^= (i as u64).rotate_left(17);
        }
    }
    TraceValues {
        output_l2_norm: sum_sq.sqrt(),
        output_abs_max: abs_max,
        output_fingerprint: fp,
    }
}

/// Set the current layer for the next `record` calls.
/// Called by `LlamaBlock::forward_with_cache` before entering sub-operations.
#[inline]
pub fn set_current_layer(layer: usize) {
    CURRENT_LAYER.with(|l| *l.borrow_mut() = layer);
}

/// Get the current layer set by `set_current_layer`.
#[inline]
pub fn current_layer() -> usize {
    CURRENT_LAYER.with(|l| *l.borrow())
}

/// Enable tracing for a given phase and token index.
///
/// Only one tracer can be active per thread at a time.  If a tracer is already
/// running this is a no-op and returns `false`.
pub fn enable_tracing(phase: &'static str, token_index: usize) -> bool {
    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        if t.is_some() {
            return false;
        }
        *t = Some(Tracer::new(phase, token_index));
        true
    })
}

/// Disable tracing and return the accumulated report.
pub fn disable_tracing() -> Option<TraceReport> {
    TRACER.with(|t| t.borrow_mut().take().map(Tracer::finish))
}

/// Returns `true` when tracing is currently active on this thread.
#[inline]
pub fn is_tracing() -> bool {
    TRACER.with(|t| t.borrow().is_some())
}

/// Record a trace event directly (without the RAII span pattern).
/// Used when timing is done manually or when output shapes are only known after the operation.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn record(
    name: &str,
    layer: usize,
    op_kind: OpKind,
    input_shape: Vec<usize>,
    input_bytes: usize,
    output_shape: Vec<usize>,
    output_bytes: usize,
    estimated_flops: u64,
    duration_ns: u64,
    values: Option<TraceValues>,
) {
    TRACER.with(|t| {
        if let Some(ref mut tracer) = *t.borrow_mut() {
            tracer.push(OpTrace {
                name: name.to_string(),
                layer,
                op_kind,
                phase: tracer.phase.to_string(),
                token_index: tracer.token_index,
                duration_ns,
                input_shape,
                output_shape,
                input_bytes,
                output_bytes,
                estimated_flops,
                output_l2_norm: values.map(|v| v.output_l2_norm),
                output_abs_max: values.map(|v| v.output_abs_max),
                output_fingerprint: values.map(|v| v.output_fingerprint),
            });
        }
    });
}

// ---------------------------------------------------------------------------
// public data types
// ---------------------------------------------------------------------------

/// Granularity level for trace events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TraceLevel {
    /// Top-level operations only (embed, per-layer blocks, final norm, lm_head).
    Coarse,
    /// Per-operation within each layer (attention sub-ops, MLP sub-ops).
    Fine,
}

/// Category of a traced operation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum OpKind {
    Embedding,
    RmsNorm,
    MatMul,
    MatMulQ8_0,
    RoPE,
    AttentionScore,
    Silu,
    Elemul,
    ResidualAdd,
    Other,
}

/// Trace-values collection level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TraceValuesLevel {
    /// Do not collect output norms or fingerprints.
    None,
    /// Collect L2 norm, abs_max, and a lightweight fingerprint.
    Summary,
}

/// Per-operation value summary (computed from output tensor).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TraceValues {
    pub output_l2_norm: f64,
    pub output_abs_max: f32,
    pub output_fingerprint: u64,
}

/// A single traced operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpTrace {
    /// Human-readable operation name (e.g. "attn_rms_norm", "q_proj").
    pub name: String,
    /// Layer index (0-based), or `usize::MAX` for global operations.
    pub layer: usize,
    /// Operation category.
    pub op_kind: OpKind,
    /// Phase string: "prefill" or "decode".
    pub phase: String,
    /// Token index within the phase (0 for prefill, step number for decode).
    pub token_index: usize,
    /// Wall-clock duration in nanoseconds.
    pub duration_ns: u64,
    /// Input tensor shape.
    pub input_shape: Vec<usize>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Bytes read (input tensors + weight bytes for matmuls).
    pub input_bytes: usize,
    /// Bytes written (output tensor).
    pub output_bytes: usize,
    /// Estimated floating-point operations.
    pub estimated_flops: u64,
    /// Output L2 norm (optional, collected when --trace-values summary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_l2_norm: Option<f64>,
    /// Output absolute max (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_abs_max: Option<f32>,
    /// Lightweight output fingerprint (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_fingerprint: Option<u64>,
}

impl OpKind {
    /// Return a short label string for this kind.
    pub fn label(&self) -> &'static str {
        match self {
            OpKind::Embedding => "embedding",
            OpKind::RmsNorm => "rms_norm",
            OpKind::MatMul => "matmul",
            OpKind::MatMulQ8_0 => "matmul_q8_0",
            OpKind::RoPE => "rope",
            OpKind::AttentionScore => "attention",
            OpKind::Silu => "silu",
            OpKind::Elemul => "elemul",
            OpKind::ResidualAdd => "residual_add",
            OpKind::Other => "other",
        }
    }
}

/// Run-level metadata about the execution environment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunMetadata {
    pub cpu_model: String,
    pub cpu_cores_physical: usize,
    pub cpu_cores_logical: usize,
    pub frequency_governor: String,
    pub thread_count: usize,
    pub build_mode: String,
    pub commit_hash: String,
    pub rust_version: String,
    pub kernel_version: String,
}

/// Aggregated trace report produced after disabling tracing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TraceReport {
    pub phase: String,
    pub token_index: usize,
    pub events: Vec<OpTrace>,
    pub total_duration_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_metadata: Option<RunMetadata>,
}

/// Collect system-level metadata for a trace run.
pub fn collect_run_metadata(thread_count: usize) -> RunMetadata {
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.trim().starts_with("model name"))
        .and_then(|l| l.split(':').nth(1).map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_cores_logical = cpuinfo
        .lines()
        .filter(|line| line.starts_with("processor"))
        .count();
    let physical_pairs = cpuinfo
        .split("\n\n")
        .filter_map(|record| {
            let mut package = None;
            let mut core = None;
            for line in record.lines() {
                let (key, value) = line.split_once(':')?;
                match key.trim() {
                    "physical id" => package = Some(value.trim().to_string()),
                    "core id" => core = Some(value.trim().to_string()),
                    _ => {}
                }
            }
            Some((package?, core?))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let cpu_cores_physical = if physical_pairs.is_empty() {
        cpuinfo
            .lines()
            .find(|line| line.starts_with("cpu cores"))
            .and_then(|line| line.split_once(':'))
            .and_then(|(_, value)| value.trim().parse().ok())
            .unwrap_or(cpu_cores_logical)
    } else {
        physical_pairs.len()
    };

    let frequency_governor =
        read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .unwrap_or_else(|| "unknown".to_string());

    let kernel_version = std::fs::read_to_string("/proc/version")
        .unwrap_or_default()
        .lines()
        .next()
        .map(|l| l.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // build mode: detect from binary path or env
    let build_mode = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
    .to_string();

    let commit_hash = option_env!("EMBER_GIT_COMMIT")
        .unwrap_or("unknown")
        .to_string();

    let rust_version = option_env!("EMBER_RUST_VERSION")
        .unwrap_or(env!("CARGO_PKG_RUST_VERSION"))
        .to_string();

    RunMetadata {
        cpu_model,
        cpu_cores_physical,
        cpu_cores_logical,
        frequency_governor,
        thread_count,
        build_mode,
        commit_hash,
        rust_version,
        kernel_version,
    }
}

fn read_first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
}

// ---------------------------------------------------------------------------
// internal tracer
// ---------------------------------------------------------------------------

struct Tracer {
    phase: &'static str,
    token_index: usize,
    events: Vec<OpTrace>,
    start: Instant,
}

impl Tracer {
    fn new(phase: &'static str, token_index: usize) -> Self {
        Self {
            phase,
            token_index,
            events: Vec::with_capacity(512),
            start: Instant::now(),
        }
    }

    fn push(&mut self, event: OpTrace) {
        self.events.push(event);
    }

    fn finish(self) -> TraceReport {
        TraceReport {
            phase: self.phase.to_string(),
            token_index: self.token_index,
            events: self.events,
            total_duration_ns: self.start.elapsed().as_nanos() as u64,
            run_metadata: None,
        }
    }
}

// ---------------------------------------------------------------------------
// RAII span — the primary instrumentation primitive
// ---------------------------------------------------------------------------

/// A timing span that records a trace event on drop.
///
/// Create one with [`TraceSpan::begin`], then call `.end()` explicitly *or*
/// let it drop — both paths record the event with the same fields.  Explicit
/// `.end()` is preferred because it lets you provide the output shape; the
/// Drop impl fills in `output_shape = vec![]` as a sentinel.
pub struct TraceSpan {
    name: String,
    layer: usize,
    op_kind: OpKind,
    start: Instant,
    input_shape: Vec<usize>,
    input_bytes: usize,
    estimated_flops: u64,
    recorded: bool,
}

impl TraceSpan {
    /// Begin timing an operation.
    ///
    /// `input_shape` and `input_bytes` describe the input tensor(s).
    /// `estimated_flops` should be computed for the operation being wrapped.
    #[inline]
    pub fn begin(
        name: &str,
        layer: usize,
        op_kind: OpKind,
        input_shape: Vec<usize>,
        input_bytes: usize,
        estimated_flops: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            layer,
            op_kind,
            start: Instant::now(),
            input_shape,
            input_bytes,
            estimated_flops,
            recorded: false,
        }
    }

    /// End timing and record the event with the given output tensor.
    #[inline]
    pub fn end(self, output_shape: Vec<usize>, output_bytes: usize) {
        self.end_with_values(output_shape, output_bytes, None);
    }

    /// End timing and record the event with output shape and optional value summary.
    #[inline]
    pub fn end_with_values(
        mut self,
        output_shape: Vec<usize>,
        output_bytes: usize,
        values: Option<TraceValues>,
    ) {
        self.recorded = true;
        let duration_ns = self.start.elapsed().as_nanos() as u64;
        let name = std::mem::take(&mut self.name);
        let input_shape = std::mem::take(&mut self.input_shape);
        TRACER.with(|t| {
            if let Some(ref mut tracer) = *t.borrow_mut() {
                tracer.push(OpTrace {
                    name,
                    layer: self.layer,
                    op_kind: self.op_kind,
                    phase: tracer.phase.to_string(),
                    token_index: tracer.token_index,
                    duration_ns,
                    input_shape,
                    output_shape,
                    input_bytes: self.input_bytes,
                    output_bytes,
                    estimated_flops: self.estimated_flops,
                    output_l2_norm: values.map(|v| v.output_l2_norm),
                    output_abs_max: values.map(|v| v.output_abs_max),
                    output_fingerprint: values.map(|v| v.output_fingerprint),
                });
            }
        });
    }
}

impl Drop for TraceSpan {
    fn drop(&mut self) {
        if !self.recorded {
            let duration_ns = self.start.elapsed().as_nanos() as u64;
            TRACER.with(|t| {
                if let Some(ref mut tracer) = *t.borrow_mut() {
                    tracer.push(OpTrace {
                        name: self.name.clone(),
                        layer: self.layer,
                        op_kind: self.op_kind,
                        phase: tracer.phase.to_string(),
                        token_index: tracer.token_index,
                        duration_ns,
                        input_shape: self.input_shape.clone(),
                        output_shape: vec![],
                        input_bytes: self.input_bytes,
                        output_bytes: 0,
                        estimated_flops: self.estimated_flops,
                        output_l2_norm: None,
                        output_abs_max: None,
                        output_fingerprint: None,
                    });
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// convenience constructor — returns None when tracing is off
// ---------------------------------------------------------------------------

/// Create a trace span only when tracing is active.
///
/// Returns `None` when tracing is disabled, making this a compile-time-friendly
/// no-op in the common path.
#[inline]
pub fn span(
    name: &str,
    layer: usize,
    op_kind: OpKind,
    input_shape: Vec<usize>,
    input_bytes: usize,
    estimated_flops: u64,
) -> Option<TraceSpan> {
    if is_tracing() {
        Some(TraceSpan::begin(
            name,
            layer,
            op_kind,
            input_shape,
            input_bytes,
            estimated_flops,
        ))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// FLOP estimation helpers
// ---------------------------------------------------------------------------

/// FLOPs for a dense matmul: `[M, K] × [K, N] → [M, N]`.
/// Standard count: 2 * M * N * K (one multiply + one add per output element per
/// inner-product term).
#[inline]
pub fn flops_matmul(m: usize, n: usize, k: usize) -> u64 {
    2u64 * m as u64 * n as u64 * k as u64
}

/// FLOPs for RMS normalization on a 2D tensor `[seq_len, dim]`.
/// For each element: square (1), accumulate in mean (1 div), rsqrt (1),
/// multiply by scale (1).  Approximate as 4 * N.
#[inline]
pub fn flops_rms_norm(seq_len: usize, dim: usize) -> u64 {
    4u64 * seq_len as u64 * dim as u64
}

/// FLOPs for RoPE on `[seq_len, n_heads * head_dim]`.
/// For each element-pair: 2 mults + 2 adds = 4.  Approximate as 4 * N.
#[inline]
pub fn flops_rope(seq_len: usize, width: usize) -> u64 {
    4u64 * seq_len as u64 * width as u64
}

/// FLOPs for multi-head attention (QK^T + softmax + V product).
/// QK^T: [seq_q, heads, head_dim] × [heads, head_dim, seq_k] → 2 * seq_q * heads * head_dim * seq_k
/// softmax: ~5 * seq_q * heads * seq_k  (exp, sum, div per element)
/// V product: 2 * seq_q * heads * seq_k * head_dim
#[inline]
pub fn flops_attention(seq_q: usize, heads: usize, head_dim: usize, seq_k: usize) -> u64 {
    let qk = 2u64 * seq_q as u64 * heads as u64 * head_dim as u64 * seq_k as u64;
    let softmax = 5u64 * seq_q as u64 * heads as u64 * seq_k as u64;
    let attn_v = 2u64 * seq_q as u64 * heads as u64 * seq_k as u64 * head_dim as u64;
    qk + softmax + attn_v
}

/// FLOPs for SiLU: x * sigmoid(x).
/// sigmoid: exp + add + div ≈ 3, multiply by x: 1.  Total ~4 per element.
#[inline]
pub fn flops_silu(n: usize) -> u64 {
    4u64 * n as u64
}

/// FLOPs for element-wise multiplication.
#[inline]
pub fn flops_elemul(n: usize) -> u64 {
    n as u64
}

/// FLOPs for residual addition (element-wise).
#[inline]
pub fn flops_residual_add(n: usize) -> u64 {
    n as u64
}

/// FLOPs for embedding lookup: zero (pure memory).
#[inline]
pub fn flops_embedding() -> u64 {
    0
}

// ---------------------------------------------------------------------------
// byte-count helpers
// ---------------------------------------------------------------------------

/// Bytes in a CpuTensor (f32 elements × 4 bytes).
#[inline]
pub fn bytes_tensor(t: &CpuTensor) -> usize {
    t.data().len() * 4
}

/// Bytes from a shape (number of elements × 4).
#[inline]
pub fn bytes_from_shape(shape: &[usize]) -> usize {
    shape.iter().product::<usize>() * 4
}

/// Bytes read by a matmul: input bytes + weight bytes.
/// For f32 weights: M×K + K×N elements.
/// For q8_0: `weight_bytes` parameter should be `data.len()`.
#[inline]
pub fn bytes_matmul_input(m: usize, k: usize, weight_bytes: usize) -> usize {
    m * k * 4 + weight_bytes
}

/// Bytes written by a matmul: M × N elements × 4.
#[inline]
pub fn bytes_matmul_output(m: usize, n: usize) -> usize {
    m * n * 4
}

/// Dequant FLOPs for q8_0: each output element requires 1 scale lookup
/// (shared across 32 elements) and 1 multiply.  Conservative: 2 per element.
/// These are *in addition* to the matmul FLOPs.
#[inline]
pub fn flops_dequant(n_elements: usize) -> u64 {
    2u64 * n_elements as u64
}

// ---------------------------------------------------------------------------
// report formatting
// ---------------------------------------------------------------------------

impl TraceReport {
    /// Produce a human-readable summary with per-layer breakdown, hot-op
    /// ranking, and arithmetic-intensity estimates.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        let total_ns = self.total_duration_ns.max(1) as f64;

        out.push_str(&format!("\n{} summary\n{}\n", self.phase, "-".repeat(30)));

        let total_ms = total_ns / 1_000_000.0;
        let tok_s = 1.0 / (total_ms / 1000.0);
        out.push_str(&format!(
            "Total duration: {:.2} ms  ({:.2} tok/s)\n",
            total_ms, tok_s
        ));

        // Run metadata
        if let Some(ref meta) = self.run_metadata {
            out.push_str(&format!(
                "CPU: {}  cores: {}P/{}L  governor: {}  threads: {}\n",
                meta.cpu_model,
                meta.cpu_cores_physical,
                meta.cpu_cores_logical,
                meta.frequency_governor,
                meta.thread_count,
            ));
            out.push_str(&format!(
                "build: {}  rust: {}  commit: {}\n",
                meta.build_mode,
                meta.rust_version,
                &meta.commit_hash[..meta.commit_hash.len().min(8)],
            ));
        }

        // Group by (layer, op_name) and aggregate
        use std::collections::BTreeMap;
        let mut by_layer: BTreeMap<usize, Vec<&OpTrace>> = BTreeMap::new();
        for ev in &self.events {
            by_layer.entry(ev.layer).or_default().push(ev);
        }

        // Per-layer breakdown
        for (layer, events) in &by_layer {
            let layer_ns: u64 = events.iter().map(|e| e.duration_ns).sum();
            let layer_pct = layer_ns as f64 / total_ns * 100.0;

            let label = if *layer == usize::MAX {
                "global ".to_string()
            } else {
                format!("layer {:>2} ", layer)
            };

            out.push_str(&format!(
                "\n{} {:.1}%  ({:.2} ms)\n",
                label,
                layer_pct,
                layer_ns as f64 / 1_000_000.0
            ));

            for ev in events {
                let pct = ev.duration_ns as f64 / total_ns * 100.0;
                let ms = ev.duration_ns as f64 / 1_000_000.0;
                // Use 2 decimal places for sub-1% ops so small ops don't show as "0.0%"
                let pct_fmt = if pct < 1.0 {
                    format!("{:>5.2}", pct)
                } else {
                    format!("{:>5.1}", pct)
                };
                let gflops = if ev.duration_ns > 0 {
                    ev.estimated_flops as f64 / ev.duration_ns as f64 // GFLOP/s
                } else {
                    0.0
                };
                let ai = if ev.input_bytes > 0 {
                    ev.estimated_flops as f64 / ev.input_bytes as f64
                } else {
                    0.0
                };

                let shape_in = format_shape(&ev.input_shape);
                let shape_out = format_shape(&ev.output_shape);

                out.push_str(&format!(
                    "  {:24} {}% {:>7.2} ms  {:>12} -> {:>12}  {:>6.1} GFLOPS/s  AI {:.1}",
                    ev.name, pct_fmt, ms, shape_in, shape_out, gflops, ai,
                ));

                // Show dequant bytes for q8_0 matmuls
                if ev.op_kind == OpKind::MatMulQ8_0 {
                    let weight_bytes = ev
                        .input_bytes
                        .saturating_sub(ev.input_shape.iter().product::<usize>() * 4);
                    out.push_str(&format!(
                        "  [q8_0: {} B weight]",
                        format_bytes(weight_bytes)
                    ));
                }

                // Show value summary when collected
                if let Some(l2) = ev.output_l2_norm {
                    out.push_str(&format!("  |L2|={:.2}", l2));
                }
                if let Some(am) = ev.output_abs_max {
                    out.push_str(&format!("  max={:.3}", am));
                }
                if let Some(fp) = ev.output_fingerprint {
                    out.push_str(&format!("  fp={:016x}", fp));
                }

                out.push('\n');
            }
        }

        // Hot operations ranking (aggregated by name)
        out.push_str(&format!("\nHot operations\n{}\n", "-".repeat(14)));
        let mut by_name: BTreeMap<&str, (u64, OpKind)> = BTreeMap::new();
        for ev in &self.events {
            let entry = by_name.entry(&ev.name).or_insert((0, ev.op_kind));
            entry.0 += ev.duration_ns;
        }
        let mut ranked: Vec<_> = by_name.into_iter().collect();
        ranked.sort_by_key(|b| std::cmp::Reverse(b.1 .0));

        for (i, (name, (ns, kind))) in ranked.iter().enumerate() {
            let pct = *ns as f64 / total_ns * 100.0;
            out.push_str(&format!(
                " {:>2}. {:24} {:>5.1}%  ({}) \n",
                i + 1,
                name,
                pct,
                kind.label(),
            ));
        }

        // Category aggregation
        out.push_str(&format!("\nBy category\n{}\n", "-".repeat(11)));
        let mut by_cat: BTreeMap<OpKind, (u64, u64)> = BTreeMap::new(); // (duration, flops)
        for ev in &self.events {
            let entry = by_cat.entry(ev.op_kind).or_insert((0, 0));
            entry.0 += ev.duration_ns;
            entry.1 += ev.estimated_flops;
        }
        let mut cat_ranked: Vec<_> = by_cat.into_iter().collect();
        cat_ranked.sort_by_key(|b| std::cmp::Reverse(b.1 .0));

        for (kind, (ns, total_flops)) in &cat_ranked {
            let pct = *ns as f64 / total_ns * 100.0;
            let gflops_s = if *ns > 0 {
                *total_flops as f64 / *ns as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                " {:20} {:>5.1}%  {:>8.1} GFLOPS/s\n",
                kind.label(),
                pct,
                gflops_s,
            ));
        }

        out
    }

    /// Serialize all events as pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.events).unwrap_or_else(|e| format!("{{error: {e}}}"))
    }

    /// Return events sorted by layer then duration (descending within layer).
    pub fn sorted_events(&self) -> Vec<&OpTrace> {
        let mut events: Vec<&OpTrace> = self.events.iter().collect();
        events.sort_by(|a, b| {
            a.layer
                .cmp(&b.layer)
                .then_with(|| b.duration_ns.cmp(&a.duration_ns))
        });
        events
    }
}

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        "(none)".to_string()
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", parts.join(", "))
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_disabled_is_noop() {
        assert!(!is_tracing());
        let s = span("test", 0, OpKind::Other, vec![1], 4, 0);
        assert!(s.is_none());
    }

    #[test]
    fn trace_lifecycle() {
        assert!(enable_tracing("decode", 0));
        assert!(is_tracing());

        {
            let s = span("rms_norm", 3, OpKind::RmsNorm, vec![1, 4096], 16384, 16384);
            assert!(s.is_some());
            let s = s.unwrap();
            s.end(vec![1, 4096], 16384);
        }

        let report = disable_tracing().unwrap();
        assert_eq!(report.phase, "decode");
        assert_eq!(report.events.len(), 1);
        assert_eq!(report.events[0].name, "rms_norm");
        assert_eq!(report.events[0].layer, 3);
    }

    #[test]
    fn double_enable_is_noop() {
        assert!(enable_tracing("prefill", 0));
        assert!(!enable_tracing("prefill", 1)); // already enabled
        assert!(is_tracing());
        let report = disable_tracing().unwrap();
        assert_eq!(report.token_index, 0); // first enable won
    }

    #[test]
    fn flops_matmul_correct() {
        // [1, 4096] × [4096, 4096] = 2 * 1 * 4096 * 4096 = 33,554,432
        assert_eq!(flops_matmul(1, 4096, 4096), 33_554_432);
    }

    #[test]
    fn json_roundtrip() {
        enable_tracing("decode", 0);
        let s = span(
            "gate_proj",
            5,
            OpKind::MatMulQ8_0,
            vec![1, 4096],
            4096 * 4 + 4096 * 4096 / 32 * 34,
            flops_matmul(1, 4096, 14336),
        );
        s.unwrap().end(vec![1, 14336], 14336 * 4);
        let report = disable_tracing().unwrap();

        let json = report.to_json();
        let parsed: Vec<OpTrace> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "gate_proj");
        assert_eq!(parsed[0].op_kind, OpKind::MatMulQ8_0);
    }
}

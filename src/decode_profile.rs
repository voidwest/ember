//! Optional operator-level profiling for the allocation-free Llama decode path.
//!
//! Profiling is explicitly enabled by the decode benchmark command. The normal
//! inference path only reads one relaxed atomic flag per token and does not
//! allocate, take timestamps, or lock.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static EVENTS: RefCell<Vec<DecodeOpEvent>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeExecutionMode {
    Serial,
    RowParallelRayon,
    PackedRowParallelRayon,
    InterleavedSerial,
    InterleavedRowParallelRayon,
    /// Quantized matvec with the output dimension split across the rayon
    /// pool (decode rows = 1; each output column accumulates identically).
    ColumnParallelRayon,
}

#[derive(Debug, Clone, Copy)]
struct DecodeOpEvent {
    pub layer: usize,
    pub operator: &'static str,
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub macs: u64,
    pub execution_mode: DecodeExecutionMode,
    pub quantization: &'static str,
    pub thread_count: usize,
    pub elapsed_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct DecodeOpSummary {
    pub architecture: String,
    pub layer: usize,
    pub operator: &'static str,
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub approximate_macs: u64,
    pub approximate_flops: u64,
    pub quantization: &'static str,
    pub execution_mode: DecodeExecutionMode,
    pub thread_count: usize,
    pub samples: usize,
    pub total_elapsed_ns: u64,
    pub median_elapsed_ns: u64,
    pub p95_elapsed_ns: u64,
    pub min_elapsed_ns: u64,
    pub max_elapsed_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SummaryKey {
    layer: usize,
    operator: &'static str,
    input_dimension: usize,
    output_dimension: usize,
    macs: u64,
    execution_mode: DecodeExecutionMode,
    quantization: &'static str,
    thread_count: usize,
}

/// Enable profiling and discard any events left by an earlier benchmark.
pub fn start() {
    EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        events.clear();
        let capacity = events.capacity();
        if capacity < 32_768 {
            events.reserve(32_768 - capacity);
        }
    });
    ENABLED.store(true, Ordering::Release);
}

/// Temporarily disable event collection without discarding existing samples.
pub fn pause() {
    ENABLED.store(false, Ordering::Release);
}

/// Resume event collection after [`pause`].
pub fn resume() {
    ENABLED.store(true, Ordering::Release);
}

/// Whether the current decode should take the instrumented branch.
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Legacy Llama/Q8-only recorder. New callers should record the resident weight
/// kind with [`record_quantized`], including `"not_applicable"` for non-weight ops.
#[inline]
pub fn record(
    layer: usize,
    operator: &'static str,
    input_dimension: usize,
    output_dimension: usize,
    execution_mode: DecodeExecutionMode,
    elapsed: Duration,
) {
    record_quantized(
        layer,
        operator,
        input_dimension,
        output_dimension,
        execution_mode,
        "Q8_0",
        elapsed,
    );
}

/// Record one completed operation with its actual resident weight kind.
/// Callers guard this with [`is_enabled`] so normal decode avoids profiling state.
#[inline]
pub fn record_quantized(
    layer: usize,
    operator: &'static str,
    input_dimension: usize,
    output_dimension: usize,
    execution_mode: DecodeExecutionMode,
    quantization: &'static str,
    elapsed: Duration,
) {
    let macs = input_dimension.saturating_mul(output_dimension) as u64;
    EVENTS.with(|events| {
        events.borrow_mut().push(DecodeOpEvent {
            layer,
            operator,
            input_dimension,
            output_dimension,
            macs,
            execution_mode,
            quantization,
            thread_count: rayon::current_num_threads().max(1),
            elapsed_ns: elapsed.as_nanos().min(u64::MAX as u128) as u64,
        });
    });
}

/// Legacy Llama-only summary. Use [`finish_for_architecture`] for other models.
pub fn finish() -> Vec<DecodeOpSummary> {
    finish_for_architecture("llama")
}

/// Stop profiling and aggregate by operator shape and actual weight kind.
pub fn finish_for_architecture(architecture: &str) -> Vec<DecodeOpSummary> {
    ENABLED.store(false, Ordering::Release);
    let events = EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()));
    summarize_events(events, architecture)
}

fn summarize_events(events: Vec<DecodeOpEvent>, architecture: &str) -> Vec<DecodeOpSummary> {
    let mut grouped = BTreeMap::<SummaryKey, Vec<u64>>::new();
    for event in events {
        grouped
            .entry(SummaryKey {
                layer: event.layer,
                operator: event.operator,
                input_dimension: event.input_dimension,
                output_dimension: event.output_dimension,
                macs: event.macs,
                execution_mode: event.execution_mode,
                quantization: event.quantization,
                thread_count: event.thread_count,
            })
            .or_default()
            .push(event.elapsed_ns);
    }

    grouped
        .into_iter()
        .map(|(key, mut elapsed)| {
            elapsed.sort_unstable();
            let samples = elapsed.len();
            let p95_index = (samples.saturating_mul(95).div_ceil(100))
                .saturating_sub(1)
                .min(samples - 1);
            DecodeOpSummary {
                architecture: architecture.to_owned(),
                layer: key.layer,
                operator: key.operator,
                input_dimension: key.input_dimension,
                output_dimension: key.output_dimension,
                approximate_macs: key.macs,
                approximate_flops: key.macs.saturating_mul(2),
                quantization: key.quantization,
                execution_mode: key.execution_mode,
                thread_count: key.thread_count,
                samples,
                total_elapsed_ns: elapsed.iter().sum(),
                median_elapsed_ns: elapsed[samples / 2],
                p95_elapsed_ns: elapsed[p95_index],
                min_elapsed_ns: elapsed[0],
                max_elapsed_ns: elapsed[samples - 1],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summaries_preserve_architecture_and_mixed_operator_weight_kinds() {
        let event = |quantization, elapsed_ns| DecodeOpEvent {
            layer: 2,
            operator: "q",
            input_dimension: 256,
            output_dimension: 64,
            macs: 256 * 64,
            execution_mode: DecodeExecutionMode::ColumnParallelRayon,
            quantization,
            thread_count: 4,
            elapsed_ns,
        };
        let summaries = summarize_events(
            vec![
                event("Q4_K", 10),
                event("Q6_K", 20),
                event("Q4_K", 30),
                event("F32", 5),
            ],
            "qwen3",
        );
        assert_eq!(summaries.len(), 3);
        assert!(summaries
            .iter()
            .all(|summary| summary.architecture == "qwen3"));
        let q4 = summaries
            .iter()
            .find(|summary| summary.quantization == "Q4_K")
            .unwrap();
        assert_eq!(q4.samples, 2);
        assert_eq!(q4.total_elapsed_ns, 40);
        assert_eq!(q4.min_elapsed_ns, 10);
        assert_eq!(q4.max_elapsed_ns, 30);
        let json = serde_json::to_value(&summaries).unwrap();
        assert_eq!(json[0]["quantization"], "F32");
        assert_eq!(json[0]["architecture"], "qwen3");
    }
}

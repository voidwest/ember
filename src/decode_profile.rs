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
pub struct DecodeOpEvent {
    pub layer: usize,
    pub operator: &'static str,
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub macs: u64,
    pub execution_mode: DecodeExecutionMode,
    pub thread_count: usize,
    pub elapsed_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct DecodeOpSummary {
    pub architecture: &'static str,
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

/// Record one completed projection. Callers must guard this with
/// [`is_enabled`] so normal decode never touches thread-local profiling state.
#[inline]
pub fn record(
    layer: usize,
    operator: &'static str,
    input_dimension: usize,
    output_dimension: usize,
    execution_mode: DecodeExecutionMode,
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
            thread_count: rayon::current_num_threads().max(1),
            elapsed_ns: elapsed.as_nanos().min(u64::MAX as u128) as u64,
        });
    });
}

/// Stop profiling and aggregate events by stable operator shape.
pub fn finish() -> Vec<DecodeOpSummary> {
    ENABLED.store(false, Ordering::Release);
    let events = EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()));
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
                architecture: "llama",
                layer: key.layer,
                operator: key.operator,
                input_dimension: key.input_dimension,
                output_dimension: key.output_dimension,
                approximate_macs: key.macs,
                approximate_flops: key.macs.saturating_mul(2),
                quantization: "Q8_0",
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

//! Persistent worker-team experiment for batch-1 quantized decode.
//!
//! A dispatch publishes one immutable job descriptor, wakes workers created at
//! team initialization, and waits until every worker has either completed its
//! static row partition or reported a caught panic. The caller's input,
//! weights, and mutable output remain borrowed until completion.

use crate::quant::{QuantizedWeight, Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerWaitStrategy {
    Sleep,
    Hybrid { spin_iterations: usize },
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum WorkerTeamError {
    #[error("worker team needs at least one worker")]
    NoWorkers,
    #[error("input length {actual} does not match encoded Q8_0 row length {expected}")]
    InputLength { actual: usize, expected: usize },
    #[error("output length {actual} does not match weight rows {expected}")]
    OutputLength { actual: usize, expected: usize },
    #[error("projection-task dispatch needs at least two workers")]
    PairNeedsTwoWorkers,
    #[error("failed to create persistent decode worker: {0}")]
    ThreadSpawn(String),
    #[error("failed to create model-owned Rayon pool: {0}")]
    RayonPoolBuild(String),
    #[error("persistent decode worker panicked: {0}")]
    WorkerPanic(String),
}

#[derive(Clone, Copy)]
struct MatrixJob {
    input: *const u8,
    input_len: usize,
    weight_data: *const u8,
    weight_len: usize,
    blocks_per_row: usize,
    output: *mut f32,
    output_len: usize,
}

impl MatrixJob {
    const EMPTY: Self = Self {
        input: ptr::null(),
        input_len: 0,
        weight_data: ptr::null(),
        weight_len: 0,
        blocks_per_row: 0,
        output: ptr::null_mut(),
        output_len: 0,
    };

    fn new(
        input: &[u8],
        weight: &QuantizedWeight,
        output: &mut [f32],
    ) -> Result<Self, WorkerTeamError> {
        let input_len = weight.in_features() / Q8_0_BLOCK_SIZE * Q8_0_TYPE_SIZE;
        if input.len() != input_len {
            return Err(WorkerTeamError::InputLength {
                actual: input.len(),
                expected: input_len,
            });
        }
        if output.len() != weight.out_features() {
            return Err(WorkerTeamError::OutputLength {
                actual: output.len(),
                expected: weight.out_features(),
            });
        }
        Ok(Self {
            input: input.as_ptr(),
            input_len: input.len(),
            weight_data: weight.data().as_ptr(),
            weight_len: weight.data().len(),
            blocks_per_row: weight.in_features() / Q8_0_BLOCK_SIZE,
            output: output.as_mut_ptr(),
            output_len: output.len(),
        })
    }
}

// SAFETY: MatrixJob is only published by WorkerTeam::dispatch while the
// synchronous caller retains valid borrows of all pointed-to storage. Workers
// cannot retain a descriptor after signalling completion, and dispatch cannot
// return until every worker has signalled.
unsafe impl Send for MatrixJob {}

#[derive(Clone, Copy)]
enum JobKind {
    Noop,
    Matrices {
        jobs: [MatrixJob; 2],
        job_count: usize,
    },
    #[cfg(test)]
    Panic,
}

struct Control {
    generation: u64,
    shutdown: bool,
    job: JobKind,
}

struct Shared {
    control: Mutex<Control>,
    dispatch_lock: Mutex<()>,
    wake: Condvar,
    published_generation: AtomicU64,
    shutdown: AtomicBool,
    completed: AtomicUsize,
    completion_lock: Mutex<()>,
    completion: Condvar,
    worker_panic: Mutex<Option<String>>,
    startup_count: AtomicUsize,
    startup_lock: Mutex<()>,
    startup: Condvar,
    pinned_workers: AtomicUsize,
    worker_count: usize,
    wait_strategy: WorkerWaitStrategy,
}

pub struct WorkerTeam {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    cpu_ids: Vec<usize>,
}

pub struct PinnedRayonPool {
    pool: rayon::ThreadPool,
    worker_count: usize,
    cpu_ids: Vec<usize>,
    pinned_workers: Arc<AtomicUsize>,
}

impl std::fmt::Debug for PinnedRayonPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedRayonPool")
            .field("worker_count", &self.worker_count)
            .field("cpu_ids", &self.cpu_ids)
            .field(
                "pinned_workers",
                &self.pinned_workers.load(Ordering::Acquire),
            )
            .finish()
    }
}

impl PinnedRayonPool {
    pub fn new(worker_count: usize, pin_to_physical_cores: bool) -> Result<Self, WorkerTeamError> {
        if worker_count == 0 {
            return Err(WorkerTeamError::NoWorkers);
        }
        let physical_cpus = physical_cpu_ids();
        let cpu_ids = if pin_to_physical_cores && physical_cpus.len() >= worker_count {
            physical_cpus[..worker_count].to_vec()
        } else {
            Vec::new()
        };
        let handler_cpu_ids = cpu_ids.clone();
        let pinned_workers = Arc::new(AtomicUsize::new(0));
        let handler_pinned_workers = Arc::clone(&pinned_workers);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|worker| format!("ember-rayon-{worker}"))
            .start_handler(move |worker| {
                if handler_cpu_ids
                    .get(worker)
                    .copied()
                    .is_some_and(pin_current_thread)
                {
                    handler_pinned_workers.fetch_add(1, Ordering::Release);
                }
            })
            .build()
            .map_err(|error| WorkerTeamError::RayonPoolBuild(error.to_string()))?;
        Ok(Self {
            pool,
            worker_count,
            cpu_ids,
            pinned_workers,
        })
    }

    pub fn install<Operation, Output>(&self, operation: Operation) -> Output
    where
        Operation: FnOnce() -> Output + Send,
        Output: Send,
    {
        self.pool.install(operation)
    }
}

impl std::fmt::Debug for WorkerTeam {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerTeam")
            .field("worker_count", &self.worker_count())
            .field("cpu_ids", &self.cpu_ids)
            .field("pinned_workers", &self.pinned_workers())
            .field("wait_strategy", &self.shared.wait_strategy)
            .finish()
    }
}

impl WorkerTeam {
    pub fn new(
        worker_count: usize,
        wait_strategy: WorkerWaitStrategy,
        pin_to_physical_cores: bool,
    ) -> Result<Self, WorkerTeamError> {
        if worker_count == 0 {
            return Err(WorkerTeamError::NoWorkers);
        }
        let physical_cpus = physical_cpu_ids();
        let cpu_ids = if pin_to_physical_cores && physical_cpus.len() >= worker_count {
            physical_cpus[..worker_count].to_vec()
        } else {
            Vec::new()
        };
        let shared = Arc::new(Shared {
            control: Mutex::new(Control {
                generation: 0,
                shutdown: false,
                job: JobKind::Noop,
            }),
            dispatch_lock: Mutex::new(()),
            wake: Condvar::new(),
            published_generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            completed: AtomicUsize::new(0),
            completion_lock: Mutex::new(()),
            completion: Condvar::new(),
            worker_panic: Mutex::new(None),
            startup_count: AtomicUsize::new(0),
            startup_lock: Mutex::new(()),
            startup: Condvar::new(),
            pinned_workers: AtomicUsize::new(0),
            worker_count,
            wait_strategy,
        });
        let mut workers = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let worker_shared = Arc::clone(&shared);
            let cpu_id = cpu_ids.get(worker_id).copied();
            match thread::Builder::new()
                .name(format!("ember-decode-{worker_id}"))
                .spawn(move || worker_loop(worker_shared, worker_id, cpu_id))
            {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    shared.shutdown.store(true, Ordering::Release);
                    {
                        let mut control = lock_unpoisoned(&shared.control);
                        control.shutdown = true;
                        control.generation = control.generation.wrapping_add(1);
                        shared
                            .published_generation
                            .store(control.generation, Ordering::Release);
                    }
                    shared.wake.notify_all();
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(WorkerTeamError::ThreadSpawn(error.to_string()));
                }
            }
        }

        let mut startup_guard = lock_unpoisoned(&shared.startup_lock);
        while shared.startup_count.load(Ordering::Acquire) != worker_count {
            startup_guard = wait_unpoisoned(&shared.startup, startup_guard);
        }
        drop(startup_guard);

        Ok(Self {
            shared,
            workers,
            cpu_ids,
        })
    }

    #[inline]
    pub fn worker_count(&self) -> usize {
        self.shared.worker_count
    }

    #[inline]
    pub fn pinned_workers(&self) -> usize {
        self.shared.pinned_workers.load(Ordering::Acquire)
    }

    pub fn matmul(
        &self,
        input: &[u8],
        weight: &QuantizedWeight,
        output: &mut [f32],
    ) -> Result<(), WorkerTeamError> {
        let job = MatrixJob::new(input, weight, output)?;
        self.dispatch(JobKind::Matrices {
            jobs: [job, MatrixJob::EMPTY],
            job_count: 1,
        })
    }

    /// Execute two independent projections concurrently, statically assigning
    /// alternating workers to each projection and row-partitioning within each
    /// worker group.
    pub fn matmul_pair(
        &self,
        input: &[u8],
        first_weight: &QuantizedWeight,
        first_output: &mut [f32],
        second_weight: &QuantizedWeight,
        second_output: &mut [f32],
    ) -> Result<(), WorkerTeamError> {
        if self.worker_count() < 2 {
            return Err(WorkerTeamError::PairNeedsTwoWorkers);
        }
        let first = MatrixJob::new(input, first_weight, first_output)?;
        let second = MatrixJob::new(input, second_weight, second_output)?;
        self.dispatch(JobKind::Matrices {
            jobs: [first, second],
            job_count: 2,
        })
    }

    fn dispatch(&self, job: JobKind) -> Result<(), WorkerTeamError> {
        let _dispatch_guard = lock_unpoisoned(&self.shared.dispatch_lock);
        self.shared.completed.store(0, Ordering::Release);
        *lock_unpoisoned(&self.shared.worker_panic) = None;

        let generation = {
            let mut control = lock_unpoisoned(&self.shared.control);
            control.job = job;
            control.generation = control.generation.wrapping_add(1);
            control.generation
        };
        self.shared
            .published_generation
            .store(generation, Ordering::Release);
        self.shared.wake.notify_all();

        let mut completion_guard = lock_unpoisoned(&self.shared.completion_lock);
        while self.shared.completed.load(Ordering::Acquire) != self.worker_count() {
            completion_guard = wait_unpoisoned(&self.shared.completion, completion_guard);
        }
        drop(completion_guard);

        if let Some(message) = lock_unpoisoned(&self.shared.worker_panic).take() {
            return Err(WorkerTeamError::WorkerPanic(message));
        }
        Ok(())
    }

    #[cfg(test)]
    fn dispatch_noop(&self) -> Result<(), WorkerTeamError> {
        self.dispatch(JobKind::Noop)
    }

    #[cfg(test)]
    fn dispatch_panic(&self) -> Result<(), WorkerTeamError> {
        self.dispatch(JobKind::Panic)
    }
}

impl Drop for WorkerTeam {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        {
            let mut control = lock_unpoisoned(&self.shared.control);
            control.shutdown = true;
            control.generation = control.generation.wrapping_add(1);
            self.shared
                .published_generation
                .store(control.generation, Ordering::Release);
        }
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize, cpu_id: Option<usize>) {
    if cpu_id.is_some_and(pin_current_thread) {
        shared.pinned_workers.fetch_add(1, Ordering::Release);
    }
    let started = shared.startup_count.fetch_add(1, Ordering::AcqRel) + 1;
    if started == shared.worker_count {
        let _startup_guard = lock_unpoisoned(&shared.startup_lock);
        shared.startup.notify_one();
    }

    let mut seen_generation = 0;
    loop {
        if let WorkerWaitStrategy::Hybrid { spin_iterations } = shared.wait_strategy {
            for _ in 0..spin_iterations {
                if shared.shutdown.load(Ordering::Acquire)
                    || shared.published_generation.load(Ordering::Acquire) != seen_generation
                {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        let (generation, job) = {
            let mut control = lock_unpoisoned(&shared.control);
            while !control.shutdown && control.generation == seen_generation {
                control = wait_unpoisoned(&shared.wake, control);
            }
            if control.shutdown {
                return;
            }
            (control.generation, control.job)
        };
        seen_generation = generation;

        let result = catch_unwind(AssertUnwindSafe(|| match job {
            JobKind::Noop => {}
            JobKind::Matrices { jobs, job_count } => {
                let projection = worker_id % job_count;
                let local_worker_id = worker_id / job_count;
                let workers_for_projection = (shared.worker_count - 1 - projection) / job_count + 1;
                // SAFETY: dispatch is synchronous and cannot publish another
                // job until all workers finish. Each projection has a distinct
                // output, and static row ranges within a projection never
                // overlap. Input and weight slices are read-only.
                unsafe {
                    execute_matrix_job(jobs[projection], local_worker_id, workers_for_projection);
                }
            }
            #[cfg(test)]
            JobKind::Panic => panic!("injected persistent-worker panic"),
        }));
        if let Err(payload) = result {
            let message = if let Some(message) = payload.downcast_ref::<&'static str>() {
                (*message).to_string()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "non-string panic payload".to_string()
            };
            let mut worker_panic = lock_unpoisoned(&shared.worker_panic);
            if worker_panic.is_none() {
                *worker_panic = Some(message);
            }
        }

        let completed = shared.completed.fetch_add(1, Ordering::AcqRel) + 1;
        if completed == shared.worker_count {
            let _completion_guard = lock_unpoisoned(&shared.completion_lock);
            shared.completion.notify_one();
        }
    }
}

unsafe fn execute_matrix_job(job: MatrixJob, worker_id: usize, worker_count: usize) {
    let chunk_rows = job
        .output_len
        .div_ceil(worker_count)
        .next_multiple_of(8)
        .max(8);
    let row_start = worker_id.saturating_mul(chunk_rows).min(job.output_len);
    let row_end = row_start.saturating_add(chunk_rows).min(job.output_len);
    if row_start == row_end {
        return;
    }
    let weight_row_bytes = job.blocks_per_row * Q8_0_TYPE_SIZE;
    let weight_start = row_start * weight_row_bytes;
    let weight_bytes = (row_end - row_start) * weight_row_bytes;
    assert!(weight_start + weight_bytes <= job.weight_len);

    // SAFETY: the dispatching caller keeps these allocations borrowed and
    // alive until every worker completes. `row_start..row_end` is a static,
    // non-overlapping partition for this projection.
    let input = unsafe { std::slice::from_raw_parts(job.input, job.input_len) };
    let weight_data =
        unsafe { std::slice::from_raw_parts(job.weight_data.add(weight_start), weight_bytes) };
    let output =
        unsafe { std::slice::from_raw_parts_mut(job.output.add(row_start), row_end - row_start) };
    crate::simd::matmul_q8_0_decode_raw_chunk(input, weight_data, job.blocks_per_row, output);
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wait_unpoisoned<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar
        .wait(guard)
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Return one allowed logical CPU for each physical package/core pair.
///
/// Linux uses the current process affinity mask and sysfs topology. Other
/// platforms gracefully fall back to the available logical CPU range.
pub fn physical_cpu_ids() -> Vec<usize> {
    let allowed = allowed_cpu_ids();
    let mut seen = BTreeSet::new();
    let mut physical = Vec::new();
    for cpu in &allowed {
        let package = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        ))
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
        let core =
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id"))
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok());
        let Some(key) = package.zip(core) else {
            continue;
        };
        if seen.insert(key) {
            physical.push(*cpu);
        }
    }
    if physical.is_empty() {
        allowed
    } else {
        physical
    }
}

#[cfg(target_os = "linux")]
fn allowed_cpu_ids() -> Vec<usize> {
    // SAFETY: cpu_set_t is zero-initialized before sched_getaffinity writes it,
    // and the provided size exactly matches the destination object.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            return (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
                .collect();
        }
    }
    logical_cpu_fallback()
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpu_ids() -> Vec<usize> {
    logical_cpu_fallback()
}

fn logical_cpu_fallback() -> Vec<usize> {
    (0..thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1))
        .collect()
}

#[cfg(target_os = "linux")]
pub(crate) fn pin_current_thread(cpu_id: usize) -> bool {
    // SAFETY: the local cpu_set_t is initialized with libc's CPU_ZERO/CPU_SET,
    // and pthread_setaffinity_np only reads it for the duration of the call.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu_id, &mut set);
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        ) == 0
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn pin_current_thread(_cpu_id: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::hint::black_box;
    use std::time::Instant;

    fn weight(out_features: usize, in_features: usize, seed: usize) -> QuantizedWeight {
        let blocks = out_features * in_features / Q8_0_BLOCK_SIZE;
        let mut data = Vec::with_capacity(blocks * Q8_0_TYPE_SIZE);
        for block in 0..blocks {
            data.extend_from_slice(
                &f16::from_f32(0.005 + (block % 7) as f32 * 0.001)
                    .to_bits()
                    .to_le_bytes(),
            );
            for index in 0..Q8_0_BLOCK_SIZE {
                data.push((((block * 17 + index * 13 + seed) % 31) as i8 - 15) as u8);
            }
        }
        QuantizedWeight::new(data, vec![out_features, in_features])
    }

    fn measure(mut operation: impl FnMut(), warmups: usize, samples: usize) -> (u64, u64) {
        for _ in 0..warmups {
            operation();
        }
        let mut elapsed = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = Instant::now();
            operation();
            elapsed.push(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        elapsed.sort_unstable();
        let p95 = samples.saturating_mul(95).div_ceil(100).saturating_sub(1);
        (elapsed[samples / 2], elapsed[p95.min(samples - 1)])
    }

    #[test]
    fn physical_cpu_selection_uses_one_logical_cpu_per_core() {
        let cpus = physical_cpu_ids();
        assert!(!cpus.is_empty());
        assert!(cpus.len() <= allowed_cpu_ids().len());
    }

    #[test]
    fn worker_panic_is_propagated_and_team_remains_usable() {
        let team = WorkerTeam::new(2, WorkerWaitStrategy::Sleep, false).unwrap();
        assert!(matches!(
            team.dispatch_panic(),
            Err(WorkerTeamError::WorkerPanic(_))
        ));
        team.dispatch_noop().unwrap();
    }

    /// Compare condition-variable and hybrid persistent workers.
    ///
    /// `RAYON_NUM_THREADS=4 cargo test --release -- persistent_worker_sweep --ignored --nocapture`
    #[test]
    #[ignore]
    fn persistent_worker_sweep() {
        let workers = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        let input_dimension = 2_048;
        let output_dimension = 8_192;
        let mut input = Vec::new();
        crate::quant::quantize_q8_0_into(&vec![0.25; input_dimension], &mut input);
        let first = weight(output_dimension, input_dimension, 1);
        let second = weight(output_dimension, input_dimension, 2);
        println!("threads,strategy,operation,median_ns,p95_ns,pinned_workers");

        for strategy in [
            WorkerWaitStrategy::Sleep,
            WorkerWaitStrategy::Hybrid {
                spin_iterations: 10_000,
            },
        ] {
            let team = WorkerTeam::new(workers, strategy, true).unwrap();
            let mut first_output = vec![0.0; output_dimension];
            let mut second_output = vec![0.0; output_dimension];
            let (noop_median, noop_p95) = measure(|| team.dispatch_noop().unwrap(), 50, 500);
            let (single_median, single_p95) = measure(
                || {
                    team.matmul(&input, &first, &mut first_output).unwrap();
                    black_box(&first_output);
                },
                10,
                50,
            );
            let strategy_name = match strategy {
                WorkerWaitStrategy::Sleep => "sleep",
                WorkerWaitStrategy::Hybrid { .. } => "hybrid",
            };
            println!(
                "{workers},{strategy_name},noop,{noop_median},{noop_p95},{}",
                team.pinned_workers()
            );
            println!(
                "{workers},{strategy_name},single,{single_median},{single_p95},{}",
                team.pinned_workers()
            );

            if workers >= 2 {
                let (pair_median, pair_p95) = measure(
                    || {
                        team.matmul_pair(
                            &input,
                            &first,
                            &mut first_output,
                            &second,
                            &mut second_output,
                        )
                        .unwrap();
                        black_box((&first_output, &second_output));
                    },
                    10,
                    50,
                );
                println!(
                    "{workers},{strategy_name},pair,{pair_median},{pair_p95},{}",
                    team.pinned_workers()
                );
            }
        }
    }
}

//! Spin-waiting worker team for single-token decode matvecs.
//!
//! A Llama decode step issues several short, back-to-back parallel regions
//! per layer (Q/K/V, O, gate/up, down) — over a hundred per token. Rayon's
//! fork/join and worker wake-up cost, plus static chunking that leaves the
//! whole region waiting on an efficiency core, dominated those regions. This
//! team keeps `threads - 1` workers spinning briefly between regions and hands
//! out small chunks through one atomic counter, so fast cores simply claim more
//! chunks than slow ones.
//!
//! The team only schedules work: every chunk is computed by the same serial
//! kernel with the same arithmetic regardless of which thread claims it, so
//! results never depend on the schedule.
//!
//! Callers that cannot use the team (a nested call from a Rayon worker, or a
//! concurrent decode already holding it) receive `false` from [`run`] and must
//! execute the chunks some other way.
//!
//! Unsafe code here is limited to erasing the job closure's lifetime while it
//! is shared with workers; the protocol below guarantees the caller's stack
//! frame outlives every dereference.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::time::{Duration, Instant};

/// Chunk-counter value marking "no job published".
const CLOSED: u64 = u32::MAX as u64;
const CHUNK_MASK: u64 = u32::MAX as u64;
/// How long an idle worker spins before parking on the condition variable.
/// Decode regions follow each other within microseconds; between tokens the
/// caller samples and feeds the next token, which is also short.
const SPIN_BUDGET: Duration = Duration::from_millis(2);

type Job<'a> = &'a (dyn Fn(usize) + Sync);

struct Shared {
    /// `epoch << 32 | next_chunk`. Claiming a chunk is a CAS on this word, so
    /// a worker can never claim a chunk of an epoch other than the one whose
    /// job pointer it loaded (see [`Shared::try_run_chunk`]).
    state: AtomicU64,
    chunks: AtomicUsize,
    /// Thin pointer to the caller's `Job` fat reference.
    job: AtomicPtr<()>,
    done: AtomicUsize,
    panicked: AtomicBool,
    sleepers: AtomicUsize,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    fn work_available(&self) -> bool {
        let state = self.state.load(SeqCst);
        let chunk = state & CHUNK_MASK;
        chunk != CLOSED && (chunk as usize) < self.chunks.load(SeqCst)
    }

    /// Claim and execute one chunk. Returns `false` when no chunk is left.
    ///
    /// Protocol: the publisher writes `chunks` and `job` only while the state
    /// is CLOSED, then stores `(epoch, 0)`. After a job completes it stores
    /// `(epoch, CLOSED)` before touching `chunks`/`job` again. A successful
    /// CAS from `s = (epoch, c)` with `c < chunks` therefore proves that the
    /// state was not closed between our loads and the claim, so the loaded
    /// `job` belongs to `epoch`, and the job cannot complete (and the caller
    /// cannot return) until this chunk increments `done`.
    fn try_run_chunk(&self) -> bool {
        loop {
            let state = self.state.load(SeqCst);
            let chunk = state & CHUNK_MASK;
            if chunk == CLOSED {
                return false;
            }
            let chunks = self.chunks.load(SeqCst);
            if chunk as usize >= chunks {
                return false;
            }
            let job = self.job.load(SeqCst);
            if self
                .state
                .compare_exchange(state, state + 1, SeqCst, SeqCst)
                .is_err()
            {
                continue;
            }
            // SAFETY: the CAS above proves `job` points at the live `Job`
            // reference on the publisher's stack (see protocol above); the
            // publisher waits for `done == chunks` before returning.
            let job: Job<'_> = unsafe { *job.cast::<Job<'_>>() };
            if catch_unwind(AssertUnwindSafe(|| job(chunk as usize))).is_err() {
                self.panicked.store(true, SeqCst);
            }
            self.done.fetch_add(1, SeqCst);
            return true;
        }
    }

    fn worker_loop(&self) {
        loop {
            if self.try_run_chunk() {
                continue;
            }
            let idle_start = Instant::now();
            let mut spins = 0u32;
            loop {
                if self.shutdown.load(SeqCst) {
                    return;
                }
                if self.work_available() {
                    break;
                }
                std::hint::spin_loop();
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(256) && idle_start.elapsed() > SPIN_BUDGET {
                    let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
                    self.sleepers.fetch_add(1, SeqCst);
                    while !self.work_available() && !self.shutdown.load(SeqCst) {
                        guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                    }
                    self.sleepers.fetch_sub(1, SeqCst);
                    break;
                }
            }
        }
    }
}

struct Team {
    shared: Arc<Shared>,
    threads: usize,
    epoch: u64,
}

impl Team {
    fn new(threads: usize) -> Option<Self> {
        let shared = Arc::new(Shared {
            state: AtomicU64::new(CLOSED),
            chunks: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        for index in 1..threads {
            let worker = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("ember-decode-{index}"))
                .spawn(move || worker.worker_loop());
            if spawned.is_err() {
                shutdown(&shared);
                return None;
            }
        }
        Some(Self {
            shared,
            threads,
            epoch: 0,
        })
    }

    fn run(&mut self, chunks: usize, job: Job<'_>) {
        let shared = &*self.shared;
        self.epoch = (self.epoch + 1) & CHUNK_MASK;
        if self.epoch == 0 {
            self.epoch = 1;
        }
        // State is CLOSED here: publish the job, then open the epoch.
        shared.chunks.store(chunks, SeqCst);
        shared
            .job
            .store((&raw const job).cast_mut().cast::<()>(), SeqCst);
        shared.done.store(0, SeqCst);
        shared.panicked.store(false, SeqCst);
        shared.state.store(self.epoch << 32, SeqCst);
        if shared.sleepers.load(SeqCst) > 0 {
            let _guard = shared.lock.lock().unwrap_or_else(|e| e.into_inner());
            shared.wake.notify_all();
        }
        while shared.try_run_chunk() {}
        while shared.done.load(SeqCst) < chunks {
            std::hint::spin_loop();
        }
        shared.state.store(self.epoch << 32 | CLOSED, SeqCst);
        if shared.panicked.load(SeqCst) {
            resume_unwind(Box::new("decode worker panicked"));
        }
    }
}

fn shutdown(shared: &Shared) {
    shared.shutdown.store(true, SeqCst);
    let _guard = shared.lock.lock().unwrap_or_else(|e| e.into_inner());
    shared.wake.notify_all();
}

impl Drop for Team {
    fn drop(&mut self) {
        shutdown(&self.shared);
    }
}

static TEAM: Mutex<Option<Team>> = Mutex::new(None);

/// Performance-core count on Apple silicon (`hw.perflevel0.logicalcpu`).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn performance_cores() -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut value: c_int = 0;
    let mut len = std::mem::size_of::<c_int>();
    // SAFETY: the name is NUL-terminated and `oldp`/`oldlenp` describe a
    // writable `c_int`; no new value is set.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && len == std::mem::size_of::<c_int>() && value > 0).then_some(value as usize)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn performance_cores() -> Option<usize> {
    None
}

/// Team members for the current Rayon context:
/// `rayon::current_num_threads()`, capped at the performance-core count on
/// CPUs with efficiency cores.
///
/// Spinning members all wait for a region's slowest chunk, and decode is
/// memory-bound, so efficiency cores only lengthen every region's tail. On
/// M1 Pro (8P + 2E) with `RAYON_NUM_THREADS=10`: 59.8 tok/s uncapped versus
/// 74.4 capped at 8.
fn team_size() -> usize {
    static CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let threads = rayon::current_num_threads();
    CAP.get_or_init(performance_cores)
        .map_or(threads, |cap| threads.min(cap))
}

thread_local! {
    /// Set while this thread runs a CPU inference session
    /// ([`crate::model::with_cpu_session`]): a sequential decode loop that may
    /// use the team even though it executes on a Rayon worker.
    static SESSION_ROOT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set inside Rayon parallel work that issues decode regions per item.
    static SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with `flag` set on this thread, restoring the previous value.
fn with_flag<R>(
    flag: &'static std::thread::LocalKey<std::cell::Cell<bool>>,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore(&'static std::thread::LocalKey<std::cell::Cell<bool>>, bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            self.0.with(|cell| cell.set(self.1));
        }
    }
    let _restore = Restore(flag, flag.with(|cell| cell.replace(true)));
    f()
}

/// Mark the current thread as the root of a sequential inference session so
/// its decode regions use the team even when it is a Rayon worker (the CLI
/// runs every session on one; its sibling workers are idle during decode).
pub(crate) fn session<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&SESSION_ROOT, f)
}

/// Keep decode regions issued by `f` off the team. Used for per-item decode
/// calls inside Rayon parallel iterators, where sibling workers are busy.
pub(crate) fn without_team<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&SUPPRESSED, f)
}

/// Whether a call from the current thread may use the decode team.
///
/// Rayon workers are excluded unless they are a session root: their
/// siblings may be busy, and a nested team region would oversubscribe the
/// cores. Work wrapped in [`without_team`] never uses it.
#[inline]
pub(crate) fn available() -> bool {
    (rayon::current_thread_index().is_none() || SESSION_ROOT.with(std::cell::Cell::get))
        && !SUPPRESSED.with(std::cell::Cell::get)
        && team_size() > 1
}

/// Execute `job(0..chunks)` across the decode team (the caller participates).
///
/// Returns `false` without running anything when the team is unavailable
/// (called from a Rayon worker, single-threaded pool, or another thread is
/// using the team); the caller must then run the chunks itself. The team has
/// `rayon::current_num_threads()` members (capped at the performance cores,
/// see [`team_size`]) so `RAYON_NUM_THREADS` keeps controlling decode
/// parallelism.
pub(crate) fn run(chunks: usize, job: &(dyn Fn(usize) + Sync)) -> bool {
    if chunks == 0 {
        return true;
    }
    if !available() {
        return false;
    }
    let mut guard = match TEAM.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return false,
    };
    let threads = team_size();
    if guard.as_ref().is_none_or(|team| team.threads != threads) {
        *guard = None;
        *guard = Team::new(threads);
    }
    let Some(team) = guard.as_mut() else {
        return false;
    };
    // Chunk panics (including the caller's own) are caught per chunk and
    // re-raised only after every worker is done with the job reference.
    team.run(chunks, job);
    true
}

/// A mutable slice shared with team chunks that each touch a disjoint range.
pub(crate) struct SharedMut<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: `SharedMut` only hands out ranges through the unsafe `range`
// accessor, whose contract requires callers to keep concurrent ranges
// disjoint; `T: Send` makes moving element access across threads sound.
unsafe impl<T: Send> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    pub(crate) fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// # Safety
    /// No other live reference (from this or any other chunk) may overlap
    /// `start..start + len`, and the source slice must outlive the result.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn range(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start <= self.len && len <= self.len - start);
        // SAFETY: in bounds by the assertion; exclusivity is the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

/// Run `job(0..chunks)` on the decode team, or serially on this thread when
/// the team is unavailable or there is only one chunk.
pub(crate) fn run_or_serial(chunks: usize, job: &(dyn Fn(usize) + Sync)) {
    if chunks <= 1 || !run(chunks, job) {
        for chunk in 0..chunks {
            job(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn every_chunk_runs_exactly_once_across_many_regions() {
        if !available() {
            return;
        }
        let counts: Vec<AtomicU32> = (0..257).map(|_| AtomicU32::new(0)).collect();
        for round in 1..=200u32 {
            let n = 1 + (round as usize * 37) % counts.len();
            let ran = run(n, &|chunk| {
                counts[chunk].fetch_add(1, SeqCst);
            });
            if !ran {
                for count in counts.iter().take(n) {
                    count.fetch_add(1, SeqCst);
                }
            }
            for (index, count) in counts.iter().enumerate() {
                let expected = if index < n { 1 } else { 0 };
                assert_eq!(
                    count.swap(0, SeqCst),
                    expected,
                    "round {round} chunk {index}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_reports_performance_cores() {
        let cores = performance_cores().expect("hw.perflevel0.logicalcpu");
        assert!(cores >= 1 && cores <= std::thread::available_parallelism().unwrap().get());
    }

    #[test]
    fn nested_calls_from_rayon_workers_decline() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| assert!(!run(4, &|_| {})));
        assert!(!without_team(|| run(4, &|_| {})));
    }

    #[test]
    fn session_roots_on_rayon_workers_use_the_team() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let expected = team_size() > 1;
        pool.install(|| {
            session(|| {
                let hits = AtomicU32::new(0);
                // The team may be held by a concurrently running test.
                let ran = run(16, &|_| {
                    hits.fetch_add(1, SeqCst);
                });
                if ran {
                    assert_eq!(hits.load(SeqCst), 16);
                }
                assert!(available() == expected);
                assert!(!without_team(available));
            });
            assert!(!available());
        });
    }
}

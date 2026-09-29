//! Allocation counting for Gate E (zero steady-state allocations in the
//! planned decode loop) and benchmark allocation reports.
//!
//! [`CountingAllocator`] wraps `std::alloc::System`. The library does **not**
//! install it: a `#[global_allocator]` in a library is forced on every
//! consumer (including the Python binding) and conflicts with downstream
//! crates that bring their own. Only the targets that read the counts
//! register it: the `ember` binary (`src/main.rs`), the library's unit
//! tests (`#[cfg(test)]` in `lib.rs`), and the integration tests and
//! examples that measure allocations. Without it installed every count
//! reads zero, so zero-allocation assertions must first check
//! [`counting_active`].
//!
//! Per-thread counting is flag-gated: `count_allocations` turns tracking on
//! for the calling thread, runs the closure, and returns how many
//! allocations it performed — independent of other threads' activity.
//! Process-global totals are only maintained inside a [`track_global`]
//! window, so outside one an allocation costs a single relaxed load of a
//! read-mostly flag instead of contended atomic read-modify-writes.
//!
//! Steady-state planned decode performs no allocations, so the hot token
//! loop pays nothing. This is the documented mechanism for
//! `docs/v04-execution-contract.md` Gate E.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counting wrapper around the system allocator. Register it with
/// `#[global_allocator]` in the final binary/test target that reads counts.
pub struct CountingAllocator;

/// Number of live [`track_global`] guards; global totals are updated only
/// while it is non-zero.
static GLOBAL_TRACKING: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_REQUESTED_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn count_one(layout_size: usize) {
    if GLOBAL_TRACKING.load(Ordering::Relaxed) != 0 {
        TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        TOTAL_REQUESTED_BYTES.fetch_add(layout_size, Ordering::Relaxed);
    }
    TRACK_ALLOCATIONS
        .try_with(|tracking| {
            if tracking.get() {
                // saturating: a debug-build usize overflow would panic inside
                // the global allocator, which is catastrophic
                ALLOCATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));
                ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get().saturating_add(layout_size)));
            }
        })
        .ok();
}

// SAFETY: forwards to `System`, then counts; the counting operations are
// panic-free and cannot recurse into the allocator. Only successful calls are
// counted: a null return allocated nothing (and a failed realloc leaves the
// old block live), so counting it would skew the reported figures.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated to the system allocator with the same layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            count_one(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: delegated to the system allocator with the original layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated to the system allocator with the same layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            count_one(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // One allocation event of `new_size` requested bytes.
        // SAFETY: delegated to the system allocator with the original ptr/layout.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            count_one(new_size);
        }
        new_ptr
    }
}

/// Run `run` with allocation tracking enabled on the calling thread and
/// return its result plus the number of allocation events performed.
pub fn count_allocations<T>(run: impl FnOnce() -> T) -> (T, usize) {
    count_allocations_with_bytes(run).map_allocations()
}

/// Run `run` with allocation tracking enabled on the calling thread and
/// return its result plus the number of allocation events and the total
/// requested bytes (layout sizes) of those events.
pub fn count_allocations_with_bytes<T>(run: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATION_COUNT.with(|count| count.set(0));
    ALLOCATED_BYTES.with(|bytes| bytes.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    // The guard clears the flag even if `run` panics, so a panicking
    // measurement cannot silently poison the next one.
    struct ClearOnDrop;
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        }
    }
    let _guard = ClearOnDrop;
    let result = run();
    let allocations = ALLOCATION_COUNT.with(Cell::get);
    let bytes = ALLOCATED_BYTES.with(Cell::get);
    (result, allocations, bytes)
}

/// Adapter turning a `(T, usize, usize)` triple into a `(T, usize)` pair.
trait MapAllocations<T> {
    fn map_allocations(self) -> (T, usize);
}

impl<T> MapAllocations<T> for (T, usize, usize) {
    fn map_allocations(self) -> (T, usize) {
        (self.0, self.1)
    }
}

/// Whether allocations are actually being counted, i.e. whether
/// [`CountingAllocator`] is the registered global allocator of this
/// executable. Without it every count reads zero, which would make a
/// zero-allocation assertion pass vacuously; such tests assert this first.
pub fn counting_active() -> bool {
    let ((), allocations) = count_allocations(|| drop(std::hint::black_box(Box::new(0u64))));
    allocations > 0
}

/// Keeps process-global totals ([`total_allocations`],
/// [`total_requested_bytes`]) updating while alive; see [`track_global`].
#[must_use = "global totals stop updating when the guard is dropped"]
pub struct GlobalTracking(());

impl Drop for GlobalTracking {
    fn drop(&mut self) {
        GLOBAL_TRACKING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Start maintaining the process-global allocation totals until the returned
/// guard is dropped (guards nest). Read the totals before and after the
/// measured region, both inside the guard's lifetime: every allocation on
/// any thread in between is counted.
pub fn track_global() -> GlobalTracking {
    GLOBAL_TRACKING.fetch_add(1, Ordering::SeqCst);
    GlobalTracking(())
}

/// Allocation events counted inside [`track_global`] windows since process
/// start (monotonic; take deltas across a window).
pub fn total_allocations() -> usize {
    TOTAL_ALLOCATIONS.load(Ordering::Relaxed)
}

/// Requested bytes of allocation/reallocation events counted inside
/// [`track_global`] windows since process start. Freeing memory never
/// decreases this counter.
pub fn total_requested_bytes() -> usize {
    TOTAL_REQUESTED_BYTES.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_is_active_in_unit_tests() {
        assert!(
            counting_active(),
            "lib unit tests register CountingAllocator"
        );
    }

    #[test]
    fn counts_allocations_only_while_tracking() {
        let (result, allocations) = count_allocations(|| {
            let buffer = vec![0u8; 1024];
            buffer.len()
        });
        assert_eq!(result, 1024);
        assert_eq!(allocations, 1, "one allocation for the Vec");
        // outside tracking, the count is not bumped
        let (_, quiet) = count_allocations(|| {});
        assert_eq!(quiet, 0);
    }

    #[test]
    fn requested_bytes_include_allocations_freed_inside_measurement() {
        let _global = track_global();
        let before = total_requested_bytes();
        let buffer = std::hint::black_box(vec![0u8; 8192]);
        drop(buffer);
        assert!(total_requested_bytes().wrapping_sub(before) >= 8192);
    }

    #[test]
    fn global_counters_are_monotonic() {
        let _global = track_global();
        let before = total_allocations();
        let _ = std::hint::black_box(String::from("allocation event"));
        assert!(total_allocations() > before);
    }
}

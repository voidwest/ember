//! Process-wide allocation counting for Gate E (zero steady-state
//! allocations in the planned decode loop) and memory reporting.
//!
//! A `#[global_allocator]` wrapping `std::alloc::System` (installed in
//! `lib.rs`). Counting is flag-gated per thread: `count_allocations` turns
//! tracking on for the calling thread, runs the closure, and returns how
//! many allocations it performed — independent of other threads' activity.
//! A global total is always maintained (one relaxed atomic per allocation)
//! for benchmark and residency reports.
//!
//! Steady-state planned decode performs no allocations, so the hot token
//! loop pays nothing. This is the documented mechanism for
//! `docs/v04-execution-contract.md` Gate E.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The installed global allocator (see `lib.rs`).
pub struct CountingAllocator;

static TOTAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn count_one(layout_size: usize) {
    TOTAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    TOTAL_ALLOCATED_BYTES.fetch_add(layout_size, Ordering::Relaxed);
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

// SAFETY: forwards to `System` after counting; the counting operations are
// panic-free and cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_one(layout.size());
        // SAFETY: delegated to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        TOTAL_ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: delegated to the system allocator with the original layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_one(layout.size());
        // SAFETY: delegated to the system allocator with the same layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Event counter increments once; live bytes change by exactly the
        // delta (count_one adds new_size, then the old layout is retired).
        count_one(new_size);
        TOTAL_ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: delegated to the system allocator with the original ptr/layout.
        unsafe { System.realloc(ptr, layout, new_size) }
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

/// Total allocation events since process start.
pub fn total_allocations() -> usize {
    TOTAL_ALLOCATIONS.load(Ordering::Relaxed)
}

/// Bytes currently allocated (approximate; realloc deltas are best-effort).
pub fn total_allocated_bytes() -> usize {
    TOTAL_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn global_counters_are_monotonic() {
        let before = total_allocations();
        let _ = String::from("allocation event");
        assert!(total_allocations() > before);
    }
}

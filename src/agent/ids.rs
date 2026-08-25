//! Deterministic run/step/event identity (Track Y).
//!
//! Identity is content- or entropy-derived (SHA-256), never process-memory
//! or timestamp-only. Run ids mix wall-clock entropy with the process id
//! and a process-lifetime counter so two runs started in the same
//! nanosecond still cannot collide.

use crate::extraction::sha256_bytes;
use std::sync::atomic::{AtomicU64, Ordering};

static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// First 12 hex chars of the SHA-256 of `bytes`.
pub fn short_hash(bytes: &[u8]) -> String {
    sha256_bytes(bytes)[..12].to_string()
}

/// Fresh run id: `run-` + 16 hex chars derived from time, pid, counter.
pub fn new_run_id() -> String {
    let seq = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let material = format!("ember-agent|{now}|{}|{seq}", std::process::id());
    format!("run-{}", &sha256_bytes(material.as_bytes())[..16])
}

/// Monotonic step ids are assigned by the session (`model-0`, `tool-1`,
/// ...); event ordering inside a trace is a u64 sequence number, not an id.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ids_are_unique_and_well_formed() {
        let a = new_run_id();
        let b = new_run_id();
        assert_ne!(a, b);
        assert!(a.starts_with("run-"));
        assert_eq!(a.len(), "run-".len() + 16);
    }

    #[test]
    fn short_hash_is_stable_hex() {
        assert_eq!(short_hash(b"abc"), short_hash(b"abc"));
        assert_eq!(short_hash(b"abc").len(), 12);
    }
}

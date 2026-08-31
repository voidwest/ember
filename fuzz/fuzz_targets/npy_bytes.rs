//! Fuzz the in-memory NPY parser with bounded, attacker-controlled bytes.
//!
//! No filesystem or model artifact is needed; this exercises the same parser
//! used by the path-based NPY loader after its file read completes.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    // Keep standalone runs predictable even when a caller supplies a large
    // custom input. `cargo fuzz` also imposes its own default max length.
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = ember::npy::read_npy_2d_bytes(data);
});

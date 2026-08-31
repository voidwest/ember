//! Fuzz PNG/JPEG decode + RGB tensor construction with bounded bytes.
//!
//! The decoder limits (max dims / alloc budget) are applied inside
//! `decode_rgb_bytes`; the fuzzer verifies that no malformed or
//! decompression-bomb input can panic or exhaust memory in the decode,
//! RGB conversion, or tensor construction steps.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = ember::multimodal::image::decode_rgb_bytes(data);
});

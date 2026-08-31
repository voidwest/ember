//! Fuzz KV snapshot manifest deserialization and structural validation.
//!
//! The target stops before payload files or a live KV cache are opened. This
//! keeps the target model-free while exercising the trust boundary shared by
//! on-disk snapshot loading and in-memory callers.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = ember::kv_snapshot::KvSnapshotManifest::from_json_bytes(data);
});

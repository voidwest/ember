//! Fuzz bounded tokenizer.json loading and the tokenizer wrapper's panic boundary.
//!
//! The target does not need a model or tokenizer fixture. Parsed payloads are
//! exercised through small encode/decode/vocabulary calls as a second hostile
//! input surface.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    if let Ok(tokenizer) = ember::tokenizer::EmberTokenizer::from_bytes(data) {
        let _ = tokenizer.encode("hello world");
        let _ = tokenizer.encode_with_offsets("hello world");
        let _ = tokenizer.decode(&[0, 1, 2]);
        let _ = tokenizer.validate_model_vocab(1 << 20);
    }
});

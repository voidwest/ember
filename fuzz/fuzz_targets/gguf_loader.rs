//! Fuzz the GGUF byte parser and tensor-range validation.
//!
//! This target intentionally stops at `load_gguf_from_reader`: it exercises
//! metadata, tensor descriptors, alignment, overflow checks, and supported
//! payload decoding without requiring a model file or constructing a model.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    // Exercise both the eager reference path and the compressed K-family
    // dispatch used by production model loading. Malformed bytes are expected
    // to return errors under every strategy.
    for strategy in [
        ember::quant_k::KStrategy::EagerF32,
        ember::quant_k::KStrategy::Scalar,
        ember::quant_k::KStrategy::X86,
        ember::quant_k::KStrategy::Auto,
    ] {
        let mut cursor = std::io::Cursor::new(data);
        let _ = ember::loader::load_gguf_from_reader_with_k_strategy(&mut cursor, strategy, true);
    }
});

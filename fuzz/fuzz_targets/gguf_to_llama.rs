//! Fuzz the GGUF parser followed by Llama model-construction validation.
//!
//! Only model-free, bounded byte inputs are accepted. A file that parses may
//! still be rejected by architecture/config/tensor checks; both outcomes are
//! useful, while panics are bugs at the untrusted model boundary.

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    for strategy in [
        ember::quant_k::KStrategy::EagerF32,
        ember::quant_k::KStrategy::Scalar,
        ember::quant_k::KStrategy::Auto,
        ember::quant_k::KStrategy::X86,
    ] {
        let mut cursor = std::io::Cursor::new(data);
        if let Ok(loader) = ember::loader::load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            strategy,
            true,
        ) {
            let _ = ember::llama::Llama::from_loader(loader);
        }
    }
});

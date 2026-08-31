//! Fuzz the in-memory WAV parser (RIFF/WAVE decode) with bounded bytes.
//!
//! Exercises the same parser the path-based and API paths use after their
//! file/byte reads complete: header chunk arithmetic, format tags, sample
//! rates, channel counts, and sample conversion. A panic here is a bug
//! (hostile input must produce `Err`, never a crash).

#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let _ = ember::multimodal::audio::decode_wav_bytes(data);
});

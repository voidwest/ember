//! Fuzz preprocessing geometry over attacker-shaped pixel tensors.
//!
//! Builds `[3, h, w]` tensors from bounded fuzzer bytes (dims capped at 65
//! so the input stays small) and runs the full resize/tile/normalize
//! recipe plus the geometry-only `tile_grid_for` mirror. A panic on any
//! shape combination is a bug: preprocessing must fail closed or succeed.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ember::multimodal::image::{preprocess, tile_grid_for, ImagePreprocessConfig, Resample};
use ember::tensor::CpuTensor;

const MAX_INPUT_BYTES: usize = 256 * 1024;
const MAX_DIM: usize = 65; // covers 64 plus rounding/parity edge cases

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    if data.len() < 2 {
        return;
    }
    let h = data[0] as usize % MAX_DIM;
    let w = data[1] as usize % MAX_DIM;
    let need = 3 * h * w;
    let body = &data[2..];
    if body.len() < need * 4 {
        return;
    }
    let mut px = Vec::with_capacity(need);
    for chunk in body[..need * 4].chunks_exact(4) {
        px.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let img = CpuTensor::from_data(vec![3, h, w], px);
    let cfg = ImagePreprocessConfig {
        resize_longest_edge: Some(2048),
        tile_size: Some(512),
        resample: Resample::Lanczos,
        rescale_factor: 1.0 / 255.0,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let _ = preprocess(&img, &cfg);
    let _ = tile_grid_for((h, w), &cfg);
});

//! Ownership-aware cross-request batching for modality encoder work.
//!
//! Multiple independent requests may contribute compatible media to one
//! encoder execution. Two properties make this safe:
//!
//! 1. **Isolation**: the towers treat the leading tensor dimension as
//!    independent samples — attention never crosses batch entries, so
//!    co-scheduling tiles from different requests cannot mix their content.
//! 2. **Ownership**: every input carries a [`SegmentId`] (request + part);
//!    outputs are split back along recorded row ranges and returned per
//!    owner. Any count mismatch fails closed instead of misattributing
//!    features.
//!
//! Compatible work is grouped by *tile geometry*: each distinct `[c, h, w]`
//! gets its own encoder pass, so heterogeneous requests still batch where
//! they overlap and never force padding on each other.

use crate::backend::CpuBackend;
use crate::multimodal::request::SegmentId;
use crate::tensor::CpuTensor;
use anyhow::{ensure, Result};

/// One image's preprocessed tiles plus its owner.
#[derive(Debug, Clone)]
pub struct BatchedImageInput {
    /// Who this work belongs to (request id + part index).
    pub owner: SegmentId,
    /// Normalized tiles `[n_tiles, c, h, w]`.
    pub tiles: CpuTensor,
}

/// One image's projected features, returned to its owner.
#[derive(Debug, Clone)]
pub struct BatchedImageOutput {
    pub owner: SegmentId,
    /// Projected rows `[n_tiles * tokens_per_tile, llm_width]`.
    pub features: CpuTensor,
}

/// Encode images from many requests in as few passes as possible.
///
/// `patch_size` and `scale_factor` come from the tower configuration
/// (tokens_per_tile = patches / scale²); `project` runs one geometry
/// group's concatenated tiles `[n, c, h, w]` through tower (+ projector)
/// and returns `(projected_rows, trace)` where the trace type is caller
/// chosen (`VisionTrace` for validated runs, `()` otherwise).
///
/// Returns the per-owner outputs in input order, one trace per executed
/// group, and the concatenation of every group's projected rows (single
/// groups are moved, multi-group results are copied once) so callers keep
/// byte-identical dump parity with the historical single-pass path.
pub fn batch_encode_images<T>(
    backend: &CpuBackend,
    inputs: &[BatchedImageInput],
    patch_size: usize,
    scale_factor: usize,
    project: impl Fn(&CpuBackend, &CpuTensor) -> Result<(CpuTensor, T)>,
) -> Result<(Vec<BatchedImageOutput>, Vec<T>, CpuTensor)> {
    ensure!(
        patch_size > 0,
        "batch_encode_images: patch_size must be non-zero"
    );
    ensure!(
        scale_factor > 0,
        "batch_encode_images: scale_factor must be non-zero"
    );
    let scale2 = scale_factor
        .checked_mul(scale_factor)
        .ok_or_else(|| anyhow::anyhow!("batch_encode_images: scale_factor overflow"))?;

    // group inputs by tile geometry, preserving order inside groups
    let mut group_index: HashMap<[usize; 3], usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let shape = [
            input.tiles.shape()[1],
            input.tiles.shape()[2],
            input.tiles.shape()[3],
        ];
        let g = *group_index.entry(shape).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }

    // one encoder pass per geometry group
    let mut chunks: Vec<CpuTensor> = Vec::with_capacity(groups.len());
    let mut traces: Vec<T> = Vec::with_capacity(groups.len());
    let mut chunk_width = 0usize;
    for idxs in &groups {
        let first = &inputs[idxs[0]].tiles;
        let (_n, c, h, w) = (
            first.shape()[0],
            first.shape()[1],
            first.shape()[2],
            first.shape()[3],
        );
        let total_tiles: usize = idxs.iter().map(|&i| inputs[i].tiles.shape()[0]).sum();
        let tile_len = c * h * w;
        let mut pixels = vec![0.0f32; total_tiles * tile_len];
        let mut off = 0usize;
        for &i in idxs {
            let t = &inputs[i].tiles;
            ensure!(
                t.len() == t.shape()[0] * tile_len,
                "geometry group drift at input {i}"
            );
            pixels[off..off + t.len()].copy_from_slice(t.data());
            off += t.len();
        }
        let batch = CpuTensor::from_data(vec![total_tiles, c, h, w], pixels);
        let (projected, trace) = project(backend, &batch)?;
        chunk_width = projected.shape()[1];
        chunks.push(projected);
        traces.push(trace);
    }

    // split back per owner over recorded row ranges (fail closed)
    let mut cursor_per_group = vec![0usize; groups.len()];
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs.iter() {
        let shape = [
            input.tiles.shape()[1],
            input.tiles.shape()[2],
            input.tiles.shape()[3],
        ];
        let g = group_index[&shape];
        let n_tiles = input.tiles.shape()[0];
        let (h, w) = (shape[1], shape[2]);
        let rows = n_tiles * ((h / patch_size) * (w / patch_size)) / scale2;
        let start = cursor_per_group[g];
        let chunk = &chunks[g];
        ensure!(
            start + rows <= chunk.shape()[0],
            "ownership split overflow for {:?}: rows [{start}, {}) exceed group {}",
            input.owner,
            start + rows,
            chunk.shape()[0]
        );
        outputs.push(BatchedImageOutput {
            owner: input.owner,
            features: CpuTensor::from_data(
                vec![rows, chunk_width],
                chunk.data()[start * chunk_width..(start + rows) * chunk_width].to_vec(),
            ),
        });
        cursor_per_group[g] = start + rows;
    }
    for (g, chunk) in chunks.iter().enumerate() {
        ensure!(
            cursor_per_group[g] == chunk.shape()[0],
            "group {g}: {}/{} feature rows unclaimed — ownership accounting drifted",
            chunk.shape()[0] - cursor_per_group[g],
            chunk.shape()[0]
        );
    }
    let projected_all = if chunks.len() == 1 {
        chunks.pop().expect("len checked")
    } else {
        let rows: usize = chunks.iter().map(|t| t.shape()[0]).sum();
        let mut data = Vec::with_capacity(rows * chunk_width.max(1));
        for c in &chunks {
            data.extend_from_slice(c.data());
        }
        CpuTensor::from_data(vec![rows, chunk_width], data)
    };
    Ok((outputs, traces, projected_all))
}

use std::collections::HashMap;

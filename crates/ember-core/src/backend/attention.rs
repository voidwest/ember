use super::*;

pub(super) fn validate_quant_row_shapes(
    operation: &str,
    dst: &CpuTensor,
    dst_index: usize,
    table_out: usize,
    table_in: usize,
    table_index: usize,
) -> Result<(), CpuError> {
    if dst.ndim() != 2
        || dst_index >= dst.shape()[0]
        || table_index >= table_out
        || dst.shape()[1] != table_in
    {
        return Err(CpuError::ShapeMismatch(format!(
            "{operation}: dst={:?}, dst_index={}, table=[{table_out}, {table_in}], table_index={}",
            dst.shape(),
            dst_index,
            table_index
        )));
    }
    Ok(())
}

pub(super) fn validate_row_copy_shapes(
    op: &str,
    dst: &CpuTensor,
    dst_index: usize,
    table: &CpuTensor,
    table_index: usize,
) -> Result<usize, CpuError> {
    if dst.ndim() != 2 || table.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: expected 2D dst/table, got dst={:?} table={:?}",
            dst.shape(),
            table.shape()
        )));
    }
    let cols = dst.shape()[1];
    if table.shape()[1] != cols {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: row width mismatch, dst cols {} != table cols {}",
            cols,
            table.shape()[1]
        )));
    }
    if dst_index >= dst.shape()[0] || table_index >= table.shape()[0] {
        return Err(CpuError::ShapeMismatch(format!(
            "{op}: row index out of bounds, dst_index={} dst_rows={} table_index={} table_rows={}",
            dst_index,
            dst.shape()[0],
            table_index,
            table.shape()[0]
        )));
    }
    Ok(cols)
}

pub(super) fn validate_attention_inputs(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    spec: AttentionSpec,
) -> Result<usize, CpuError> {
    if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention expects 2D q/k/v, got q={:?} k={:?} v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let seq_len = q.shape()[0];
    let embed_dim = spec.n_heads * spec.head_dim;
    let kv_dim = spec.n_kv_heads * spec.head_dim;
    if q.shape() != [seq_len, embed_dim] {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention: q shape {:?} != [{}, {}]",
            q.shape(),
            seq_len,
            embed_dim
        )));
    }
    if k.shape() != [seq_len, kv_dim] || v.shape() != [seq_len, kv_dim] {
        return Err(CpuError::ShapeMismatch(format!(
            "causal_attention: k/v shape mismatch, got k={:?} v={:?}, expected [{}, {}]",
            k.shape(),
            v.shape(),
            seq_len,
            kv_dim
        )));
    }
    validate_gqa(spec.n_heads, spec.n_kv_heads)?;
    Ok(seq_len)
}

/// Quantize `x` (`[rows, in_features]`) once and write `x × w` into `out`:
/// the shared body of the allocating and reusing Q8_0 matmuls.
pub(super) fn q8_0_matmul_rows(x: &[f32], rows: usize, w: &QuantizedWeight, out: &mut [f32]) {
    Q8_0_DECODE_INPUT.with(|input| {
        let mut input = input.borrow_mut();
        crate::quant::quantize_q8_0_into(x, &mut input);
        if rows == 1 {
            crate::simd::matmul_q8_0_decode(&input, w, out);
        } else {
            crate::simd::matmul_q8_0_batch(&input, rows, w, out);
        }
    });
}

/// Shape checks for cached causal attention; returns the query row count.
pub(super) fn validate_cached_attention_inputs(
    q: &CpuTensor,
    cached_k: &[f16],
    cached_v: &[f16],
    spec: &CachedAttentionSpec,
) -> Result<usize, CpuError> {
    if q.ndim() != 2 {
        return Err(CpuError::ShapeMismatch(format!(
            "cached_causal_attention: q must be 2D, got {:?}",
            q.shape()
        )));
    }
    let seq_len = q.shape()[0];
    let embed_dim = spec.n_heads * spec.head_dim;
    if q.shape()[1] != embed_dim {
        return Err(CpuError::ShapeMismatch(format!(
            "cached_causal_attention: q width {} != expected {}",
            q.shape()[1],
            embed_dim
        )));
    }
    if spec.total_seq_len < seq_len || spec.total_seq_len > spec.max_seq_len {
        return Err(CpuError::ShapeMismatch(format!(
            "cached_causal_attention: total_seq_len {} invalid for seq_len {} and max_seq_len {}",
            spec.total_seq_len, seq_len, spec.max_seq_len
        )));
    }
    let cache_len = spec.n_kv_heads * spec.max_seq_len * spec.head_dim;
    if cached_k.len() != cache_len || cached_v.len() != cache_len {
        return Err(CpuError::ShapeMismatch(format!(
            "cached_causal_attention: cache len mismatch, got k={} v={}, expected {}",
            cached_k.len(),
            cached_v.len(),
            cache_len
        )));
    }
    validate_gqa(spec.n_heads, spec.n_kv_heads)?;
    Ok(seq_len)
}

/// Cached causal attention into a zeroed `out` of `[seq_len, n_heads *
/// head_dim]` values (the kernels accumulate into it). Inputs must have
/// passed [`validate_cached_attention_inputs`].
pub(super) fn cached_attention_into(
    q: &CpuTensor,
    cached_k: &[f16],
    cached_v: &[f16],
    spec: &CachedAttentionSpec,
    seq_len: usize,
    qk_row: &mut Vec<f32>,
    out: &mut [f32],
) -> Result<(), CpuError> {
    let embed_dim = spec.n_heads * spec.head_dim;
    let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
    let scale = (spec.head_dim as f32).sqrt().recip();
    let cache_head_stride = spec.max_seq_len * spec.head_dim;
    cached_attention_dispatch(
        q.data(),
        cached_k,
        cached_v,
        spec,
        seq_len,
        embed_dim,
        n_repeat,
        scale,
        cache_head_stride,
        spec.head_dim,
        None,
        false,
        qk_row,
        out,
    );
    Ok(())
}

pub(crate) fn validate_gqa(n_heads: usize, n_kv_heads: usize) -> Result<usize, CpuError> {
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(CpuError::ShapeMismatch(format!(
            "attention heads must satisfy n_heads % n_kv_heads == 0, got {} and {}",
            n_heads, n_kv_heads
        )));
    }
    Ok(n_heads / n_kv_heads)
}

pub(crate) fn softmax_prefix(row: &mut [f32], len: usize) {
    softmax_range(row, 0, len);
}

/// Softmax over `row[start..end]`. One shared implementation for the
/// cached/prefill attention kernels (backend + gemma4): +inf handling,
/// all-−inf uniform fallback, max-subtract, normalize.
pub(crate) fn softmax_range(row: &mut [f32], start: usize, end: usize) {
    assert!(
        start < end && end <= row.len(),
        "softmax range is out of bounds"
    );
    let positive_infinities = row[start..end]
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    if positive_infinities > 0 {
        let probability = 1.0 / positive_infinities as f32;
        for value in &mut row[start..end] {
            *value = if *value == f32::INFINITY {
                probability
            } else {
                0.0
            };
        }
        return;
    }
    let max_val = row[start..end]
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        let uniform = 1.0 / (end - start) as f32;
        row[start..end].fill(uniform);
        return;
    }
    let mut sum = 0.0;
    for value in &mut row[start..end] {
        *value = (*value - max_val).exp();
        sum += *value;
    }
    let inv_sum = sum.recip();
    for value in &mut row[start..end] {
        *value *= inv_sum;
    }
}

/// One (row, head) of cached causal attention: qk dots over `[min_j,
/// max_j]`, softmax, weighted V sum into `out_slice` (the caller-provided
/// head-sized destination). `min_j` implements sliding-window masking
/// (0 = full causal prefix, matching the previous `softmax_prefix` form).
/// Bit-identical to the pre-consolidation serial bodies: same dot order,
/// same softmax, same accumulation.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "kernel signature: buffers, strides and shape stay separate arguments on the hot path; the index is also used in arithmetic or to index several arrays"
)]
pub(crate) fn cached_attention_row_head(
    q: &[f32],
    cached_k: &[f16],
    cached_v: &[f16],
    row: usize,
    head: usize,
    embed_dim: usize,
    head_dim: usize,
    kv_head_dim: usize,
    n_repeat: usize,
    scale: f32,
    cache_head_stride: usize,
    max_j: usize,
    min_j: usize,
    qk_scratch: &mut [f32],
    out_slice: &mut [f32],
) {
    let q_head_offset = head * head_dim;
    let kv_h = head / n_repeat;
    let q_idx = row * embed_dim + q_head_offset;
    for j in min_j..=max_j {
        let k_offset = kv_h * cache_head_stride + j * kv_head_dim;
        qk_scratch[j] = crate::simd::dot_product_f16(
            &q[q_idx..q_idx + head_dim],
            &cached_k[k_offset..k_offset + head_dim],
        ) * scale;
    }
    softmax_range(qk_scratch, min_j, max_j + 1);
    for j in min_j..=max_j {
        let weight = qk_scratch[j];
        if weight == 0.0 {
            continue;
        }
        let v_offset = kv_h * cache_head_stride + j * kv_head_dim;
        crate::simd::weighted_add_f16(out_slice, &cached_v[v_offset..v_offset + head_dim], weight);
    }
}

/// Cached-attention dispatch shared by the ordinary and Gemma paths:
/// parallel over rows (seq > 1), parallel over heads (seq == 1), serial.
/// Each branch computes identical per-(row, head) values, with optional
/// sliding-window masking and an independent physical KV head width.
#[allow(
    clippy::too_many_arguments,
    reason = "kernel signature: buffers, strides and shape stay separate arguments on the hot path"
)]
pub(crate) fn cached_attention_dispatch(
    q: &[f32],
    cached_k: &[f16],
    cached_v: &[f16],
    spec: &CachedAttentionSpec,
    seq_len: usize,
    embed_dim: usize,
    n_repeat: usize,
    scale: f32,
    cache_head_stride: usize,
    cache_head_dim: usize,
    sliding_window: Option<usize>,
    tiled_decode: bool,
    qk_row: &mut Vec<f32>,
    out: &mut [f32],
) {
    let compute_head = |i: usize, h: usize, scores: &mut Vec<f32>, head_out: &mut [f32]| {
        let max_j = spec.total_seq_len - seq_len + i;
        let min_j = sliding_window.map_or(0, |window| (max_j + 1).saturating_sub(window));
        scores.resize(max_j + 1, 0.0);
        // Prefill rows and the Q8 fast decode path take the register-tiled
        // kernel (bit-identical per element). The reference path keeps the
        // generic body for single-row decode, so it stays an independent
        // check of that kernel.
        if seq_len > 1 || tiled_decode {
            let q_idx = i * embed_dim + h * spec.head_dim;
            if crate::attention_kernels::cached_row_head(
                &q[q_idx..q_idx + spec.head_dim],
                cached_k,
                cached_v,
                (h / n_repeat) * cache_head_stride,
                cache_head_dim,
                scale,
                min_j,
                max_j,
                scores,
                head_out,
            ) {
                return;
            }
        }
        cached_attention_row_head(
            q,
            cached_k,
            cached_v,
            i,
            h,
            embed_dim,
            spec.head_dim,
            cache_head_dim,
            n_repeat,
            scale,
            cache_head_stride,
            max_j,
            min_j,
            scores,
            head_out,
        );
    };
    let attended =
        sliding_window.map_or(spec.total_seq_len, |window| window.min(spec.total_seq_len));
    let parallel_attention =
        should_parallel_attention(spec.n_heads, seq_len, attended, spec.head_dim);
    if parallel_attention && seq_len > 1 {
        out.par_chunks_mut(embed_dim)
            .enumerate()
            .for_each(|(i, out_row)| {
                ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                    let mut qk_row = qk_row.borrow_mut();
                    for h in 0..spec.n_heads {
                        let head_out = &mut out_row[h * spec.head_dim..(h + 1) * spec.head_dim];
                        compute_head(i, h, &mut qk_row, head_out);
                    }
                });
            });
        return;
    }
    if parallel_attention {
        debug_assert_eq!(seq_len, 1);
        // Single-token decode: one head per chunk on the decode team, which
        // is already spinning between the surrounding matvec regions. Each
        // head is computed exactly as on the Rayon path below.
        let n_heads = out.len() / spec.head_dim;
        let heads = crate::decode_pool::SharedMut::new(out);
        let ran = crate::decode_pool::run(n_heads, &|h| {
            // SAFETY: each chunk index is a distinct head, so the head
            // slices are disjoint.
            let head_out = unsafe { heads.range(h * spec.head_dim, spec.head_dim) };
            ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                let mut qk_row = qk_row.borrow_mut();
                compute_head(0, h, &mut qk_row, head_out);
            });
        });
        if ran {
            return;
        }
        out.par_chunks_mut(spec.head_dim)
            .enumerate()
            .for_each(|(h, head_out)| {
                ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                    let mut qk_row = qk_row.borrow_mut();
                    compute_head(0, h, &mut qk_row, head_out);
                });
            });
        return;
    }
    if qk_row.capacity() < spec.max_seq_len {
        qk_row.reserve(spec.max_seq_len - qk_row.capacity());
    }
    for h in 0..spec.n_heads {
        for i in 0..seq_len {
            let out_offset = i * embed_dim + h * spec.head_dim;
            compute_head(
                i,
                h,
                qk_row,
                &mut out[out_offset..out_offset + spec.head_dim],
            );
        }
    }
}

/// One (row, head) of prefill (block-diagonal) causal attention: qk dots
/// over `[start, row]`, softmax, weighted V sum. Shared by the parallel
/// and serial prefill branches (bit-identical per (row, head)).
#[allow(
    clippy::too_many_arguments,
    reason = "kernel signature: buffers, strides and shape stay separate arguments on the hot path"
)]
pub(crate) fn prefill_attention_row_head(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    row: usize,
    head: usize,
    embed_dim: usize,
    head_dim: usize,
    kv_dim: usize,
    n_repeat: usize,
    scale: f32,
    start: usize,
    qk_scratch: &mut [f32],
    out_slice: &mut [f32],
) {
    let q_head_offset = head * head_dim;
    let kv_h = head / n_repeat;
    let kv_head_offset = kv_h * head_dim;
    let q_idx = row * embed_dim + q_head_offset;
    let ctx_len = row - start + 1;
    for j in start..=row {
        let k_idx = j * kv_dim + kv_head_offset;
        qk_scratch[j - start] = crate::simd::dot_product(
            &q_data[q_idx..q_idx + head_dim],
            &k_data[k_idx..k_idx + head_dim],
        ) * scale;
    }
    softmax_prefix(qk_scratch, ctx_len);
    for j in start..=row {
        let weight = qk_scratch[j - start];
        if weight == 0.0 {
            continue;
        }
        let v_offset = j * kv_dim + kv_head_offset;
        crate::simd::weighted_add(out_slice, &v_data[v_offset..v_offset + head_dim], weight);
    }
}

pub(crate) fn should_parallel_attention(
    n_heads: usize,
    seq_len: usize,
    total_seq_len: usize,
    head_dim: usize,
) -> bool {
    n_heads >= PARALLEL_ATTENTION_MIN_HEADS
        && rayon::current_num_threads() > 1
        && n_heads
            .saturating_mul(seq_len)
            .saturating_mul(total_seq_len)
            .saturating_mul(head_dim)
            >= PARALLEL_ATTENTION_MIN_WORK
}

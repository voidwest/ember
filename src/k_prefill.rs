//! Register-blocked batch-1 K-quant prefill GEMM (rows > 1), exact f32
//! activations.
//!
//! The v0.3 prefill path (`matmul_k_scalar_with` / the AVX2 batch kernels)
//! is catastrophic on this host (4.7 tok/s vs llama.cpp 131.9 — a 28x gap):
//! for each output column it re-reads every activation from L2, horizontal-
//! reduces every 8 values, and read-modify-writes `dst` per 8-value chunk.
//!
//! This module replaces that with a register-blocked GEMM:
//!   - `RT` rows x `CT` columns of 16-lane (zmm) accumulators carried
//!     across whole super-blocks, reduced once per output element;
//!   - the dequantized weight block (v) is shared across the RT rows;
//!   - the activation chunk (x) is shared across the CT columns;
//!   - the parallel split is over column tiles, so each rayon task reads
//!     only its own columns of weights (total weight traffic ~1x) and x
//!     re-reads hit L2 (x is small: rows x in_features x 4 bytes).
//!
//! Numerics: per (row, column) the accumulation order matches the AVX2
//! batch kernel's (block -> g -> c, two 16-lane accumulator groups), so
//! the eager-vs-compressed delta stays inside the frozen Gate-A envelope
//! (1e-4 relative) — same margin the existing AVX2 kernels hold. Serial
//! and parallel entries are bit-identical by construction (each output
//! element is produced by exactly one task with the same order).

use crate::quant_k::{KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES, QK_K};

/// Column range handed to each rayon task (multiple of the register CT).
const PARALLEL_COL_TILE: usize = 256;

/// Send+Sync wrapper for the disjoint column-range dst pointer handed to
/// rayon tasks. The ranges provably never overlap and all tasks join
/// before `dst` is read again; the wrapper exists only to express that
/// disjointness to rayon.
#[derive(Clone, Copy)]
struct SendDst(*mut f32);
// SAFETY: tasks touch disjoint ranges and join before use.
unsafe impl Send for SendDst {}
unsafe impl Sync for SendDst {}

/// Validate the layout, then dispatch to the AVX-512 register-blocked
/// body when available (else delegate to the existing v0.3 kernels so
/// non-AVX-512 builds keep their current behavior exactly).
pub fn matmul_k_prefill_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let in_features = w.in_features();
    let out_features = w.out_features();
    if src.len() != rows * in_features {
        return Err(format!(
            "k_prefill: src len {} != rows {rows} * in_features {in_features}",
            src.len()
        ));
    }
    if dst.len() != rows * out_features {
        return Err(format!(
            "k_prefill: dst len {} != rows {rows} * out_features {out_features}",
            dst.len()
        ));
    }
    #[cfg(target_arch = "x86_64")]
    {
        let use_avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq");
        let forced_legacy = std::env::var("EMBER_KPREFILL_LEGACY").is_ok_and(|v| v == "1");
        if use_avx512 && !forced_legacy {
            match w.dtype() {
                KQuantDtype::Q4K => {
                    // SAFETY: layout validated above; the AVX-512 bodies
                    // only touch their own disjoint dst range and read
                    // src/w immutably.
                    unsafe { prefill_q4k_avx512(src, rows, w, 0, out_features, dst) }
                    return Ok(());
                }
                KQuantDtype::Q6K => {
                    unsafe { prefill_q6k_avx512(src, rows, w, 0, out_features, dst) }
                    return Ok(());
                }
            }
        }
    }
    // Non-AVX-512: keep the existing v0.3 kernels.
    crate::k_matmul::matmul_k_legacy_prefill_into(src, rows, w, dst)
}

/// Column-parallel prefill entry. Each rayon task computes a disjoint
/// column range for all rows, so results are bit-identical to the serial
/// entry and total weight traffic is ~1x.
pub fn matmul_k_prefill_into_parallel(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let in_features = w.in_features();
    let out_features = w.out_features();
    if src.len() != rows * in_features {
        return Err(format!(
            "k_prefill: src len {} != rows {rows} * in_features {in_features}",
            src.len()
        ));
    }
    if dst.len() != rows * out_features {
        return Err(format!(
            "k_prefill: dst len {} != rows {rows} * out_features {out_features}",
            dst.len()
        ));
    }
    let col_tile = std::env::var("EMBER_KPREFILL_CTILE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PARALLEL_COL_TILE);
    if rayon::current_num_threads() <= 1 || out_features < 2 * col_tile {
        return matmul_k_prefill_into(src, rows, w, dst);
    }
    #[cfg(target_arch = "x86_64")]
    {
        let use_avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq");
        let forced_legacy = std::env::var("EMBER_KPREFILL_LEGACY").is_ok_and(|v| v == "1");
        if use_avx512 && !forced_legacy {
            use rayon::prelude::*;
            let task: fn(usize, usize, SendDst, usize, &[f32], usize, &KQuantWeight) =
                match w.dtype() {
                    KQuantDtype::Q4K => prefill_q4k_task,
                    KQuantDtype::Q6K => prefill_q6k_task,
                };
            let dst_ptr = SendDst(dst.as_mut_ptr());
            let dst_len = dst.len();
            (0..out_features)
                .into_par_iter()
                .step_by(col_tile)
                .for_each(move |c0| {
                    // The wrapper is moved whole (not field-captured), so
                    // the closure stays Send+Sync via the explicit impls.
                    task(c0, col_tile, dst_ptr, dst_len, src, rows, w)
                });
            return Ok(());
        }
    }
    crate::k_matmul::matmul_k_legacy_prefill_into(src, rows, w, dst)
}

/// Q4_K register-blocked body over output columns `c0..c1` (all rows).
///
/// Loop order: row tiles -> column register tiles -> super-blocks ->
/// 32-value groups. Accumulators are carried across super-blocks so each
/// output element is reduced exactly once.
/// One rayon task: compute output columns `c0..c1` for every row into the
/// disjoint dst range. `dst_ptr`/`dst_len` describe the full validated dst
/// buffer; the task only touches columns c0..c1 of every row.
#[cfg(target_arch = "x86_64")]
fn prefill_q4k_task(
    c0: usize,
    col_tile: usize,
    dst_ptr: SendDst,
    dst_len: usize,
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
) {
    let c1 = (c0 + col_tile).min(w.out_features());
    // SAFETY: dst_ptr points at the validated dst buffer; this task only
    // writes columns c0..c1 of every row (disjoint across tasks).
    let dst_cols = unsafe { std::slice::from_raw_parts_mut(dst_ptr.0, dst_len) };
    // SAFETY: as above; layout validated at entry.
    unsafe { prefill_q4k_avx512(src, rows, w, c0, c1, dst_cols) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn prefill_q4k_avx512(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    c0: usize,
    c1: usize,
    dst: &mut [f32],
) {
    x86::prefill_q4k_cols(src, rows, w, c0, c1, dst)
}

/// Q6_K register-blocked body over output columns `c0..c1` (all rows).
/// Tile: 2 rows x 1 column, 4 zmm accumulators per (row, column) matching
/// the Q6_K decode body's lane layout.
#[cfg(target_arch = "x86_64")]
unsafe fn prefill_q6k_avx512(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    c0: usize,
    c1: usize,
    dst: &mut [f32],
) {
    x86::prefill_q6k_cols(src, rows, w, c0, c1, dst)
}

/// One rayon task for the Q6_K body (see [`prefill_q4k_task`]).
#[cfg(target_arch = "x86_64")]
fn prefill_q6k_task(
    c0: usize,
    col_tile: usize,
    dst_ptr: SendDst,
    dst_len: usize,
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
) {
    let c1 = (c0 + col_tile).min(w.out_features());
    // SAFETY: dst_ptr points at the validated dst buffer; this task only
    // writes columns c0..c1 of every row (disjoint across tasks).
    let dst_cols = unsafe { std::slice::from_raw_parts_mut(dst_ptr.0, dst_len) };
    // SAFETY: as above; layout validated at entry.
    unsafe { prefill_q6k_avx512(src, rows, w, c0, c1, dst_cols) }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use core::arch::x86_64::*;

    /// Q4_K register-tile body generated for a compile-time (RT x CT)
    /// tile. All loops over rows/columns are constants, so the compiler
    /// keeps the accumulators (and the shared activation chunks) in zmm
    /// registers across the whole super-block loop — no stack round-trips.
    /// Remainder rows/columns are handled by the smaller tiles below.
    macro_rules! q4k_tile_fn {
        ($name:ident, $rt:expr, $ct:expr) => {
            #[target_feature(enable = "avx512f")]
            #[allow(clippy::too_many_arguments)] // fixed tile-shape kernel signature
            unsafe fn $name(
                src: &[f32],
                data: &[u8],
                blocks_per_row: usize,
                r0: usize,
                c0: usize,
                in_features: usize,
                out_features: usize,
                dst: &mut [f32],
            ) {
                let mask0f = _mm_set1_epi8(0x0F);
                // ONE accumulator per (row, column): the two 16-lane
                // sub-chunks of each 32-value group share a register, so
                // 4x4 uses 16 zmm of accumulators (fits the file without
                // spills; the final horizontal sum is lane-agnostic).
                let mut acc = [[_mm512_setzero_ps(); $ct]; $rt];
                for b in 0..blocks_per_row {
                    let xb = b * QK_K;
                    for g in 0..4 {
                        for c in 0..2 {
                            // activation chunk (32 values) per row
                            let mut xv = [[_mm512_setzero_ps(); 2]; $rt];
                            for r in 0..$rt {
                                let xp = src
                                    .as_ptr()
                                    .add((r0 + r) * in_features + xb + g * 64 + c * 16);
                                xv[r][0] = _mm512_loadu_ps(xp);
                                xv[r][1] = _mm512_loadu_ps(xp.add(32));
                            }
                            for col in 0..$ct {
                                let j = c0 + col;
                                let block = &data[(j * blocks_per_row + b) * Q4_K_BLOCK_BYTES
                                    ..(j * blocks_per_row + b + 1) * Q4_K_BLOCK_BYTES];
                                let d =
                                    half::f16::from_bits(u16::from_le_bytes([block[0], block[1]]))
                                        .to_f32();
                                let min =
                                    half::f16::from_bits(u16::from_le_bytes([block[2], block[3]]))
                                        .to_f32();
                                let (ds, ms) = crate::k_gemv::unpack_k4_scales(&block[4..16]);
                                let d1 = d * f32::from(ds[2 * g]);
                                let m1 = -min * f32::from(ms[2 * g]);
                                let d2 = d * f32::from(ds[2 * g + 1]);
                                let m2 = -min * f32::from(ms[2 * g + 1]);
                                let bd1 = _mm512_set1_ps(d1);
                                let bm1 = _mm512_set1_ps(m1);
                                let bd2 = _mm512_set1_ps(d2);
                                let bm2 = _mm512_set1_ps(m2);
                                let q16 = _mm_loadu_si128(
                                    block.as_ptr().add(16 + g * 32 + c * 16) as *const __m128i
                                );
                                let ql = _mm_and_si128(q16, mask0f);
                                let qh = _mm_and_si128(_mm_srli_epi16(q16, 4), mask0f);
                                let v_low = _mm512_fmadd_ps(
                                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ql)),
                                    bd1,
                                    bm1,
                                );
                                let v_high = _mm512_fmadd_ps(
                                    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qh)),
                                    bd2,
                                    bm2,
                                );
                                for r in 0..$rt {
                                    acc[r][col] = _mm512_fmadd_ps(xv[r][0], v_low, acc[r][col]);
                                    acc[r][col] = _mm512_fmadd_ps(xv[r][1], v_high, acc[r][col]);
                                }
                            }
                        }
                    }
                }
                for r in 0..$rt {
                    for col in 0..$ct {
                        dst[(r0 + r) * out_features + c0 + col] +=
                            _mm512_reduce_add_ps(acc[r][col]);
                    }
                }
            }
        };
    }

    q4k_tile_fn!(q4k_tile_4x2, 4, 2);
    q4k_tile_fn!(q4k_tile_4x1, 4, 1);
    q4k_tile_fn!(q4k_tile_2x4, 2, 4);
    q4k_tile_fn!(q4k_tile_2x2, 2, 2);
    q4k_tile_fn!(q4k_tile_2x1, 2, 1);
    q4k_tile_fn!(q4k_tile_4x4, 4, 4);
    q4k_tile_fn!(q4k_tile_1x4, 1, 4);
    q4k_tile_fn!(q4k_tile_1x2, 1, 2);
    q4k_tile_fn!(q4k_tile_1x1, 1, 1);

    /// Serial column-range body for Q4_K: constant-size tiles with
    /// remainder rows/columns handled by the smaller tile shapes.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn prefill_q4k_cols(
        src: &[f32],
        rows: usize,
        w: &KQuantWeight,
        c0: usize,
        c1: usize,
        dst: &mut [f32],
    ) {
        let in_features = w.in_features();
        let out_features = w.out_features();
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();
        // Tile-shape A/B knob (default 4x2). Only the main-tile loop reads
        // it; remainder tiles are fixed small shapes.
        let tile = std::env::var("EMBER_KPREFILL_TILE").unwrap_or_else(|_| "44".into());
        if tile == "24" {
            let mut rt = 0usize;
            while rt + 2 <= rows {
                let mut ct = c0;
                while ct + 4 <= c1 {
                    q4k_tile_2x4(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 4;
                }
                while ct + 2 <= c1 {
                    q4k_tile_2x2(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 2;
                }
                if ct < c1 {
                    q4k_tile_2x1(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                }
                rt += 2;
            }
            if rt < rows {
                let mut ct = c0;
                while ct + 4 <= c1 {
                    q4k_tile_1x4(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 4;
                }
                while ct + 2 <= c1 {
                    q4k_tile_1x2(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 2;
                }
                if ct < c1 {
                    q4k_tile_1x1(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                }
            }
            return;
        }
        if tile == "44" {
            let mut rt = 0usize;
            while rt + 4 <= rows {
                let mut ct = c0;
                while ct + 4 <= c1 {
                    q4k_tile_4x4(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 4;
                }
                while ct + 2 <= c1 {
                    q4k_tile_4x2(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 2;
                }
                if ct < c1 {
                    q4k_tile_4x1(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                }
                rt += 4;
            }
            if rt + 2 <= rows {
                let mut ct = c0;
                while ct + 4 <= c1 {
                    q4k_tile_2x4(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 4;
                }
                while ct + 2 <= c1 {
                    q4k_tile_2x2(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 2;
                }
                if ct < c1 {
                    q4k_tile_2x1(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                }
                rt += 2;
            }
            if rt < rows {
                let mut ct = c0;
                while ct + 4 <= c1 {
                    q4k_tile_1x4(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 4;
                }
                while ct + 2 <= c1 {
                    q4k_tile_1x2(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                    ct += 2;
                }
                if ct < c1 {
                    q4k_tile_1x1(
                        src,
                        data,
                        blocks_per_row,
                        rt,
                        ct,
                        in_features,
                        out_features,
                        dst,
                    );
                }
            }
            return;
        }
        let mut rt = 0usize;
        while rt + 4 <= rows {
            let mut ct = c0;
            while ct + 2 <= c1 {
                q4k_tile_4x2(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
                ct += 2;
            }
            if ct < c1 {
                q4k_tile_4x1(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
            }
            rt += 4;
        }
        if rt + 2 <= rows {
            let mut ct = c0;
            while ct + 2 <= c1 {
                q4k_tile_2x2(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
                ct += 2;
            }
            if ct < c1 {
                q4k_tile_2x1(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
            }
            rt += 2;
        }
        if rt < rows {
            let mut ct = c0;
            while ct + 2 <= c1 {
                q4k_tile_1x2(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
                ct += 2;
            }
            if ct < c1 {
                q4k_tile_1x1(
                    src,
                    data,
                    blocks_per_row,
                    rt,
                    ct,
                    in_features,
                    out_features,
                    dst,
                );
            }
        }
    }
    /// Q6_K body over columns `c0..c1` for all rows. Tile: 2 rows x 1
    /// column, 4 zmm accumulators per (row, column).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn prefill_q6k_cols(
        src: &[f32],
        rows: usize,
        w: &KQuantWeight,
        c0: usize,
        c1: usize,
        dst: &mut [f32],
    ) {
        const RT: usize = 2;
        let in_features = w.in_features();
        let out_features = w.out_features();
        let blocks_per_row = w.blocks_per_row();
        let data = w.data();
        let mask0f = _mm_set1_epi8(0x0F);
        let mask03 = _mm_set1_epi8(0x03);
        let thirty_two = _mm_set1_epi8(32);
        let mut rt = 0usize;
        while rt < rows {
            let rt_here = (rows - rt).min(RT);
            let mut j = c0;
            while j < c1 {
                let row_bytes = j * blocks_per_row * crate::quant_k::Q6_K_BLOCK_BYTES;
                let mut acc = [[[_mm512_setzero_ps(); 4]; 1]; 2];
                for b in 0..blocks_per_row {
                    let block = &data[row_bytes + b * crate::quant_k::Q6_K_BLOCK_BYTES..];
                    let d =
                        half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
                    let scales = &block[192..208];
                    let ql = &block[0..128];
                    let qh = &block[128..192];
                    let xb = b * QK_K;
                    for half in 0..2 {
                        let q = half * 64;
                        let h = half * 32;
                        let s = half * 8;
                        let y = half * 128;
                        let sc = |i: usize| f32::from(i8::from_le_bytes([scales[i]]));
                        for c16 in 0..2 {
                            let ql_lo =
                                _mm_loadu_si128(ql.as_ptr().add(q + c16 * 16) as *const __m128i);
                            let ql_hi = _mm_loadu_si128(
                                ql.as_ptr().add(q + 32 + c16 * 16) as *const __m128i
                            );
                            let qh16 =
                                _mm_loadu_si128(qh.as_ptr().add(h + c16 * 16) as *const __m128i);
                            let qh_lo = _mm_and_si128(qh16, mask03);
                            let qh_sh2 = _mm_and_si128(_mm_srli_epi16(qh16, 2), mask03);
                            let qh_sh4 = _mm_and_si128(_mm_srli_epi16(qh16, 4), mask03);
                            let qh_sh6 = _mm_and_si128(_mm_srli_epi16(qh16, 6), mask03);
                            let shl4 = |v: __m128i| {
                                _mm_and_si128(_mm_slli_epi16(v, 4), _mm_set1_epi8(-16))
                            };
                            let q1 = _mm_sub_epi8(
                                _mm_or_si128(_mm_and_si128(ql_lo, mask0f), shl4(qh_lo)),
                                thirty_two,
                            );
                            let q2 = _mm_sub_epi8(
                                _mm_or_si128(_mm_and_si128(ql_hi, mask0f), shl4(qh_sh2)),
                                thirty_two,
                            );
                            let q3 = _mm_sub_epi8(
                                _mm_or_si128(
                                    _mm_and_si128(_mm_srli_epi16(ql_lo, 4), mask0f),
                                    shl4(qh_sh4),
                                ),
                                thirty_two,
                            );
                            let q4 = _mm_sub_epi8(
                                _mm_or_si128(
                                    _mm_and_si128(_mm_srli_epi16(ql_hi, 4), mask0f),
                                    shl4(qh_sh6),
                                ),
                                thirty_two,
                            );
                            let d1 = d * sc(s + c16);
                            let d2 = d * sc(s + c16 + 2);
                            let d3 = d * sc(s + c16 + 4);
                            let d4 = d * sc(s + c16 + 6);
                            let v1 = _mm512_mul_ps(
                                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)),
                                _mm512_set1_ps(d1),
                            );
                            let v2 = _mm512_mul_ps(
                                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2)),
                                _mm512_set1_ps(d2),
                            );
                            let v3 = _mm512_mul_ps(
                                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3)),
                                _mm512_set1_ps(d3),
                            );
                            let v4 = _mm512_mul_ps(
                                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q4)),
                                _mm512_set1_ps(d4),
                            );
                            #[allow(clippy::needless_range_loop)] // SIMD accumulator banking
                            for r in 0..rt_here {
                                let x =
                                    src.as_ptr().add((rt + r) * in_features + xb + y + c16 * 16);
                                acc[r][0][0] =
                                    _mm512_fmadd_ps(_mm512_loadu_ps(x), v1, acc[r][0][0]);
                                acc[r][0][1] =
                                    _mm512_fmadd_ps(_mm512_loadu_ps(x.add(32)), v2, acc[r][0][1]);
                                acc[r][0][2] =
                                    _mm512_fmadd_ps(_mm512_loadu_ps(x.add(64)), v3, acc[r][0][2]);
                                acc[r][0][3] =
                                    _mm512_fmadd_ps(_mm512_loadu_ps(x.add(96)), v4, acc[r][0][3]);
                            }
                        }
                    }
                }
                for r in 0..rt_here {
                    let s = _mm512_add_ps(
                        _mm512_add_ps(acc[r][0][0], acc[r][0][1]),
                        _mm512_add_ps(acc[r][0][2], acc[r][0][3]),
                    );
                    dst[(rt + r) * out_features + j] += _mm512_reduce_add_ps(s);
                }
                j += 1;
            }
            rt += RT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k_matmul::tests::{
        eager_reference, seeded_activations, seeded_q4_blocks, seeded_q6_blocks,
    };

    fn assert_gate_a(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs <= 1e-4 * scale,
            "Gate A exceeded: max_abs {max_abs} > {} (scale {scale})",
            1e-4 * scale
        );
    }

    #[test]
    fn q4k_prefill_matches_eager_oracle_across_shapes() {
        for (rows, in_features, out_features) in [
            (2usize, 2048usize, 2048usize),
            (4, 2048, 8192),
            (26, 2048, 2048),
            (26, 2048, 8192),
            (33, 8192, 2048),
            (8, 512, 512),
        ] {
            let blocks = in_features / 256 * out_features;
            let weight = KQuantWeight::new(
                seeded_q4_blocks(blocks, 0x81_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q4K,
            );
            let src = seeded_activations(rows * in_features, 0x82_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let mut actual = vec![0.0f32; rows * out_features];
            matmul_k_prefill_into(&src, rows, &weight, &mut actual).unwrap();
            assert_gate_a(&actual, &expected);
            // serial and parallel must be bit-identical
            let mut parallel = vec![0.0f32; rows * out_features];
            matmul_k_prefill_into_parallel(&src, rows, &weight, &mut parallel).unwrap();
            assert_eq!(actual, parallel, "prefill serial/parallel divergence");
        }
    }

    #[test]
    fn q6k_prefill_matches_eager_oracle_across_shapes() {
        for (rows, in_features, out_features) in [
            (2usize, 2048usize, 2048usize),
            (4, 2048, 8192),
            (26, 2048, 2048),
            (26, 8192, 2048),
            (33, 8192, 2048),
            (8, 512, 512),
        ] {
            let blocks = in_features / 256 * out_features;
            let weight = KQuantWeight::new(
                seeded_q6_blocks(blocks, 0x91_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q6K,
            );
            let src = seeded_activations(rows * in_features, 0x92_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let mut actual = vec![0.0f32; rows * out_features];
            matmul_k_prefill_into(&src, rows, &weight, &mut actual).unwrap();
            assert_gate_a(&actual, &expected);
            let mut parallel = vec![0.0f32; rows * out_features];
            matmul_k_prefill_into_parallel(&src, rows, &weight, &mut parallel).unwrap();
            assert_eq!(actual, parallel, "prefill serial/parallel divergence (Q6K)");
        }
    }

    #[test]
    fn q4k_prefill_length_mismatches_are_rejected() {
        let weight = KQuantWeight::new(seeded_q4_blocks(8, 0x83), [2, 1024], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 1024, 0x84);
        let mut dst = vec![0.0f32; 2 * 2];
        assert!(matmul_k_prefill_into(&src[..2047], 2, &weight, &mut dst).is_err());
        assert!(matmul_k_prefill_into(&src, 2, &weight, &mut dst[..3]).is_err());
    }

    #[test]
    fn q4k_prefill_zero_scale_blocks_contribute_exactly_zero() {
        let mut bytes = seeded_q4_blocks(2, 0x85);
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            block[0..4].fill(0);
        }
        let weight = KQuantWeight::new(bytes, [1, 512], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 512, 0x86);
        let mut actual = vec![1.0f32; 2];
        matmul_k_prefill_into(&src, 2, &weight, &mut actual).unwrap();
        assert_eq!(actual, vec![1.0; 2]);
    }
}

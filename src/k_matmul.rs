//! Scalar compressed-resident K-quant matmul kernels.
//!
//! The v0.3 reference execution path for Q4_K/Q6_K: weights stay packed,
//! each 256-value super-block is dequantized into a thread-local scratch
//! and dotted with the activation slice. There is no persistent f32
//! expansion and no allocation in the hot path. These kernels are the
//! oracle that the optimized x86 paths (v0.3 commits 7/8) are validated
//! against, and the eager-f32 dequant-then-gemm path is *their* oracle.
//!
//! Layout: `KQuantWeight` stores super-blocks contiguous per output
//! feature (`[out, in]`, GGUF dims reversed), matching the Q8_0
//! convention.

use crate::quant_k::{
    dequant_q4_k, dequant_q6_k, KExecution, KQuantDtype, KQuantWeight, Q4_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, QK_K,
};

// One dequantized super-block per thread (256 f32 = 1 KiB).
thread_local! {
    static BLOCK_SCRATCH: std::cell::RefCell<[f32; QK_K]> =
        const { std::cell::RefCell::new([0.0f32; QK_K]) };
}

/// Scalar compressed-resident matmul: `dst = src × w`.
///
/// `src` is `[rows, in_features]` row-major f32; `dst` is
/// `[rows, out_features]` row-major f32 and must be **zero-initialized**
/// by the caller (the kernel accumulates, matching the Q8_0 `_into`
/// convention).
///
/// Returns an explicit error for source/destination length mismatches.
/// Unsupported dtypes are a hard error — never a silent conversion to a
/// persistent f32 representation.
pub fn matmul_k_scalar_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let in_features = w.in_features();
    let out_features = w.out_features();
    let expected_src = rows
        .checked_mul(in_features)
        .ok_or_else(|| "matmul_k_scalar: input shape product overflow".to_string())?;
    if src.len() != expected_src {
        return Err(format!(
            "matmul_k_scalar: src len {} != rows {rows} * in_features {in_features}",
            src.len()
        ));
    }
    let expected_dst = rows
        .checked_mul(out_features)
        .ok_or_else(|| "matmul_k_scalar: output shape product overflow".to_string())?;
    if dst.len() != expected_dst {
        return Err(format!(
            "matmul_k_scalar: dst len {} != rows {rows} * out_features {out_features}",
            dst.len()
        ));
    }
    match w.dtype() {
        KQuantDtype::Q6K => matmul_k_scalar_with(dequant_q6_k, Q6_K_BLOCK_BYTES, src, rows, w, dst),
        KQuantDtype::Q4K => matmul_k_scalar_with(dequant_q4_k, Q4_K_BLOCK_BYTES, src, rows, w, dst),
    }
}

/// Dispatch entry: executes the per-tensor execution decision recorded at
/// load (scalar or AVX2), never silently downgrading. `dst` must be
/// zero-initialized (accumulation semantics, same as the scalar entry).
///
/// A `CompressedX86` decision without AVX2 at runtime is a hard error —
/// the loader only records that decision after checking the feature set,
/// so a mismatch is a bug, not a fallback.
pub fn matmul_k_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let in_features = w.in_features();
    let out_features = w.out_features();
    let expected_src = rows
        .checked_mul(in_features)
        .ok_or_else(|| "matmul_k: input shape product overflow".to_string())?;
    if src.len() != expected_src {
        return Err(format!(
            "matmul_k: src len {} != rows {rows} * in_features {in_features}",
            src.len()
        ));
    }
    let expected_dst = rows
        .checked_mul(out_features)
        .ok_or_else(|| "matmul_k: output shape product overflow".to_string())?;
    if dst.len() != expected_dst {
        return Err(format!(
            "matmul_k: dst len {} != rows {rows} * out_features {out_features}",
            dst.len()
        ));
    }
    // Batch-1 decode: route through the bandwidth-competitive GEMV
    // (exact f32 activations). Prefill (rows > 1) routes through the
    // register-blocked AVX-512 prefill GEMM when available, falling back
    // to the v0.3 kernels otherwise.
    if rows == 1 {
        return crate::k_gemv::matmul_k_gemv_serial(src, w, dst);
    }
    crate::k_prefill::matmul_k_prefill_into(src, rows, w, dst)
}

/// Column-parallel decode matvec: the same math as [`matmul_k_into`] with
/// the output dimension split across the rayon pool. Each output column
/// accumulates identically to the serial kernel, so results are
/// bit-identical. Only single-row (decode) matvecs of sufficient size are
/// parallelized; everything else defers to the serial kernel.
pub fn matmul_k_into_parallel(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    let in_features = w.in_features();
    let out_features = w.out_features();
    let expected_src = rows
        .checked_mul(in_features)
        .ok_or_else(|| "matmul_k_parallel: input shape product overflow".to_string())?;
    if src.len() != expected_src {
        return Err(format!(
            "matmul_k_parallel: src len {} != rows {rows} * in_features {in_features}",
            src.len()
        ));
    }
    let expected_dst = rows
        .checked_mul(out_features)
        .ok_or_else(|| "matmul_k_parallel: output shape product overflow".to_string())?;
    if dst.len() != expected_dst {
        return Err(format!(
            "matmul_k_parallel: dst len {} != rows {rows} * out_features {out_features}",
            dst.len()
        ));
    }
    // Batch-1 decode: route through the bandwidth-competitive GEMV, which
    // applies its own measured shape-dependent parallel threshold (the old
    // 8M-MAC rule serialized q/k/v/o even at high thread counts).
    if rows == 1 {
        return crate::k_gemv::matmul_k_gemv_parallel(src, w, dst);
    }
    // Prefill: the register-blocked GEMM has its own column-tile parallel
    // split (bit-identical to its serial entry).
    crate::k_prefill::matmul_k_prefill_into_parallel(src, rows, w, dst)
}

/// v0.3 prefill entry (rows > 1, no register blocking): the scalar or
/// AVX2 batch kernels by execution decision. Called by [`crate::k_prefill`]
/// when the AVX-512 register-blocked body is unavailable; never routes back
/// through [`matmul_k_into`] (that would recurse).
pub fn matmul_k_legacy_prefill_into(
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    match w.execution() {
        KExecution::CompressedScalar => matmul_k_scalar_into(src, rows, w, dst),
        KExecution::CompressedX86 => {
            if !crate::k_matmul_x86::avx2_supported() {
                return Err(
                    "matmul_k: compressed-x86 recorded at load but AVX2 is unavailable at runtime"
                        .to_string(),
                );
            }
            // Safety: layout validated by the caller; the AVX2 kernels only
            // touch dst rows/cols they own.
            unsafe { crate::k_matmul_x86::matmul_k_avx2_into(src, rows, w, dst)? }
            Ok(())
        }
        KExecution::EagerF32 => {
            Err("matmul_k: eager-f32 tensors are f32 CpuTensors, not KQuantWeight".to_string())
        }
    }
}

/// Shared scalar kernel body. `dequant` must dequantize one
/// `block_bytes`-sized super-block into `[f32; QK_K]`.
///
/// Indexing is in bounds by construction: `KQuantWeight` validation
/// guarantees `data.len() == out_features * blocks_per_row * block_bytes`
/// and every slice below is derived from those checked quantities.
fn matmul_k_scalar_with(
    dequant: fn(&[u8], &mut [f32]),
    block_bytes: usize,
    src: &[f32],
    rows: usize,
    w: &KQuantWeight,
    dst: &mut [f32],
) -> Result<(), String> {
    if rows == 1 {
        // single-row body shared with the column-parallel entry so serial
        // and parallel stay bit-identical
        matmul_k_scalar_row1_chunk_into(dequant, block_bytes, src, w, 0, dst);
        return Ok(());
    }
    let in_features = w.in_features();
    let out_features = w.out_features();
    let blocks_per_row = w.blocks_per_row();
    let data = w.data();
    for j in 0..out_features {
        let row_bytes = j * blocks_per_row * block_bytes;
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * block_bytes..row_bytes + (b + 1) * block_bytes];
            BLOCK_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                dequant(block, &mut scratch[..QK_K]);
                let x_base = b * QK_K;
                for r in 0..rows {
                    let x_row = r * in_features;
                    let mut acc = 0.0f32;
                    for k in 0..QK_K {
                        acc += src[x_row + x_base + k] * scratch[k];
                    }
                    dst[r * out_features + j] += acc;
                }
            });
        }
    }
    Ok(())
}

/// Benchmark-only accessor for the pre-GEMV scalar row-1 kernel.
#[doc(hidden)]
pub fn bench_legacy_row1_scalar(src: &[f32], w: &KQuantWeight, dst: &mut [f32]) {
    let block_bytes = match w.dtype() {
        KQuantDtype::Q4K => Q4_K_BLOCK_BYTES,
        KQuantDtype::Q6K => Q6_K_BLOCK_BYTES,
    };
    let dequant: fn(&[u8], &mut [f32]) = match w.dtype() {
        KQuantDtype::Q4K => dequant_q4_k,
        KQuantDtype::Q6K => dequant_q6_k,
    };
    matmul_k_scalar_row1_chunk_into(dequant, block_bytes, src, w, 0, dst);
}

/// Single-row (decode) scalar body over `dst_chunk` columns starting at
/// `j0`, with one scalar accumulator per output column (bit-identical to
/// the serial accumulation into a zeroed `dst`).
fn matmul_k_scalar_row1_chunk_into(
    dequant: fn(&[u8], &mut [f32]),
    block_bytes: usize,
    src: &[f32],
    w: &KQuantWeight,
    j0: usize,
    dst_chunk: &mut [f32],
) {
    let blocks_per_row = w.blocks_per_row();
    let data = w.data();
    for (i, j) in (j0..j0 + dst_chunk.len()).enumerate() {
        let mut acc_j = 0.0f32;
        let row_bytes = j * blocks_per_row * block_bytes;
        for b in 0..blocks_per_row {
            let block = &data[row_bytes + b * block_bytes..row_bytes + (b + 1) * block_bytes];
            BLOCK_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                dequant(block, &mut scratch[..QK_K]);
                let x_base = b * QK_K;
                let mut acc = 0.0f32;
                for k in 0..QK_K {
                    acc += src[x_base + k] * scratch[k];
                }
                acc_j += acc;
            });
        }
        dst_chunk[i] = acc_j;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tensor::CpuTensor;

    /// Deterministic pseudo-random Q6_K block payload with sanitized f16
    /// scale (offset 208): random bytes must not produce NaN/Inf scales.
    pub(crate) fn seeded_q6_blocks(blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * Q6_K_BLOCK_BYTES];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        for block in bytes.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            let bits = u16::from_le_bytes([block[208], block[209]]);
            let bits = bits & 0x7FFF;
            let bits = if bits >= 0x7C00 { 0x3C00 } else { bits };
            block[208..210].copy_from_slice(&bits.to_le_bytes());
        }
        bytes
    }

    /// Deterministic activations in [-4, 4).
    pub(crate) fn seeded_activations(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (state >> 33) as u32 as i32;
            values.push(v as f32 * (4.0 / 2147483648.0));
        }
        values
    }

    /// Deterministic pseudo-random Q4_K block payload with sanitized f16
    /// scale/min fields (offsets 0 and 2).
    pub(crate) fn seeded_q4_blocks(blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * Q4_K_BLOCK_BYTES];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            for offset in [0usize, 2] {
                let bits = u16::from_le_bytes([block[offset], block[offset + 1]]);
                let bits = bits & 0x7FFF;
                let bits = if bits >= 0x7C00 { 0x3C00 } else { bits };
                block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        bytes
    }

    /// The eager-f32 oracle: `dequantize_all` ([out, in], contiguous per
    /// output row) transposed to the model's `[in, out]` layout, then a
    /// dense `CpuTensor::matmul` — exactly the reference path the scalar
    /// compressed kernels are validated against.
    pub(crate) fn eager_reference(w: &KQuantWeight, src: &[f32], rows: usize) -> Vec<f32> {
        let w_full = w.dequantize_all().transpose();
        let x = CpuTensor::from_data(vec![rows, w.in_features()], src.to_vec());
        x.matmul(&w_full).data().to_vec()
    }

    /// Gate A (contract doc section 9):
    /// `max_abs <= 1e-4 * max(1, max_abs_ref)`.
    fn assert_gate_a(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let mut max_abs = 0.0f32;
        for (a, e) in actual.iter().zip(expected) {
            max_abs = max_abs.max((a - e).abs());
        }
        let gate = 1e-4 * scale;
        assert!(
            max_abs <= gate,
            "Gate A exceeded: max_abs {max_abs} > {gate} (scale {scale})"
        );
    }

    #[test]
    fn q6_k_scalar_matches_eager_oracle_across_shapes() {
        for (rows, in_features, out_features) in [
            (1, 256, 128),
            (2, 512, 512),
            (8, 2048, 1536),
            (32, 256, 512),
            (1, 3584, 256),
            (4, 2048, 2048),
        ] {
            let blocks = in_features / 256 * out_features;
            let weight = KQuantWeight::new(
                seeded_q6_blocks(blocks, 0x51_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q6K,
            );
            let src = seeded_activations(rows * in_features, 0xAC_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let mut actual = vec![0.0f32; rows * out_features];
            matmul_k_scalar_into(&src, rows, &weight, &mut actual).unwrap();
            assert_gate_a(&actual, &expected);
        }
    }

    /// Gate A: the column-parallel decode matvec must be bit-identical to
    /// the serial kernel for both dtypes and both execution paths, across
    /// the decode projection shapes (gate/up, down, head).
    #[test]
    fn parallel_matvec_matches_serial_bit_identical() {
        use crate::quant_k::KQuantDtype;
        let shapes = [(2048usize, 8192usize), (8192, 2048), (1024, 16384)];
        let executions: &[KExecution] = if crate::k_matmul_x86::avx2_supported() {
            &[KExecution::CompressedScalar, KExecution::CompressedX86]
        } else {
            &[KExecution::CompressedScalar]
        };
        for &(in_features, out_features) in &shapes {
            for &dtype in &[KQuantDtype::Q4K, KQuantDtype::Q6K] {
                // Q4_K and Q6_K super-blocks are both QK_K = 256 wide
                let blocks = in_features / 256 * out_features;
                let payload = match dtype {
                    KQuantDtype::Q6K => seeded_q6_blocks(blocks, 0x51_00 + in_features as u64),
                    KQuantDtype::Q4K => seeded_q4_blocks(blocks, 0x41_00 + in_features as u64),
                };
                let src = seeded_activations(in_features, 0xAC_00 + out_features as u64);
                for &execution in executions {
                    let weight =
                        KQuantWeight::new(payload.clone(), [out_features, in_features], dtype)
                            .with_execution(execution);
                    let mut serial = vec![0.0f32; out_features];
                    let mut parallel = vec![0.0f32; out_features];
                    matmul_k_into(&src, 1, &weight, &mut serial).unwrap();
                    matmul_k_into_parallel(&src, 1, &weight, &mut parallel).unwrap();
                    assert_eq!(
                        serial, parallel,
                        "parallel matvec diverged from serial: {in_features}x{out_features} {dtype:?} {execution:?}"
                    );
                }
            }
        }
    }

    /// Gate E instrumentation: quantify what the column-parallel matvec
    /// allocates per call (rayon's per-iterator task structures are a
    /// documented dependency of the parallel path). The result feeds the
    /// steady-state allocation accounting in the execution contract.
    #[test]
    fn parallel_matvec_allocation_count_is_quantified() {
        use crate::quant_k::KQuantDtype;
        if !crate::k_matmul_x86::avx2_supported() {
            return;
        }
        let (in_features, out_features) = (2048usize, 8192usize);
        let blocks = in_features / 256 * out_features;
        let weight = KQuantWeight::new(
            seeded_q6_blocks(blocks, 0x71),
            [out_features, in_features],
            KQuantDtype::Q6K,
        )
        .with_execution(KExecution::CompressedX86);
        let src = seeded_activations(in_features, 0x81);
        // warm the rayon pool + the per-thread dequant scratch
        let mut warmup = vec![0.0f32; out_features];
        matmul_k_into_parallel(&src, 1, &weight, &mut warmup).unwrap();
        let (_, allocations) = crate::alloc_counter::count_allocations(|| {
            let mut dst = vec![0.0f32; out_features];
            matmul_k_into_parallel(&src, 1, &weight, &mut dst).unwrap();
            dst[0]
        });
        eprintln!("parallel matvec allocations per call (incl. the dst Vec): {allocations}");
        // one for the dst Vec; rayon's task structures are the documented
        // remainder and must stay small (a bounded constant, not linear in
        // the output size)
        assert!(
            allocations <= 64,
            "parallel matvec allocated {allocations} times per call; expected a small constant"
        );
    }

    #[test]
    fn q6_k_zero_scale_blocks_contribute_exactly_zero() {
        // d = 0 (f16 zero at offset 208) dequantizes every value to 0.0,
        // so the row must stay exactly zero — not merely small.
        let mut bytes = seeded_q6_blocks(2, 0x0D);
        bytes[208..210].copy_from_slice(&0u16.to_le_bytes());
        bytes[208 + Q6_K_BLOCK_BYTES..210 + Q6_K_BLOCK_BYTES].copy_from_slice(&0u16.to_le_bytes());
        let weight = KQuantWeight::new(bytes, [1, 512], KQuantDtype::Q6K);
        let src = seeded_activations(2 * 512, 0x0E);
        let mut actual = vec![1.0f32; 2]; // nonzero sentinel: accumulation must not disturb
        matmul_k_scalar_into(&src, 2, &weight, &mut actual).unwrap();
        assert_eq!(actual, vec![1.0; 2]);
    }

    #[test]
    fn q6_k_negative_scales_and_extreme_quants_pass_gate() {
        // scales pinned to int8 extremes and ql/qh saturated; exercises
        // sign extension and the 6-bit nibble reconstruction.
        let mut bytes = seeded_q6_blocks(4, 0x1F);
        for block in bytes.chunks_exact_mut(Q6_K_BLOCK_BYTES) {
            block[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            block[192..208].fill(0x80); // all scales = -128
            block[0..128].fill(0xFF); // ql all set
            block[128..192].fill(0xFF); // qh all set
        }
        let weight = KQuantWeight::new(bytes, [1, 1024], KQuantDtype::Q6K);
        let src = seeded_activations(2 * 1024, 0x20);
        let expected = eager_reference(&weight, &src, 2);
        let mut actual = vec![0.0f32; 2];
        matmul_k_scalar_into(&src, 2, &weight, &mut actual).unwrap();
        assert_gate_a(&actual, &expected);
        assert!(actual.iter().all(|v| v.is_finite()));
        // with d = 1 and scale -128 the values are large and negative
        assert!(
            actual[0] < -1e6,
            "expected large negative accumulation, got {}",
            actual[0]
        );
    }

    #[test]
    fn matmul_rejects_length_mismatches() {
        let weight = KQuantWeight::new(seeded_q6_blocks(1, 0x21), [1, 256], KQuantDtype::Q6K);
        let src = seeded_activations(256, 0x22);
        let mut dst = vec![0.0f32; 1];
        // src too short
        assert!(matmul_k_scalar_into(&src[..255], 1, &weight, &mut dst).is_err());
        // dst too short
        assert!(matmul_k_scalar_into(&src, 1, &weight, &mut dst[..0]).is_err());
        // rows inconsistent with src length
        let mut short_dst = vec![0.0f32; 2];
        assert!(matmul_k_scalar_into(&src, 2, &weight, &mut short_dst).is_err());
    }

    #[test]
    fn q4_k_scalar_matches_eager_oracle_across_shapes() {
        for (rows, in_features, out_features) in [
            (1, 256, 128),
            (2, 512, 512),
            (8, 2048, 1536),
            (32, 256, 512),
            (1, 3584, 256),
            (4, 2048, 2048),
        ] {
            let blocks = in_features / 256 * out_features;
            let weight = KQuantWeight::new(
                seeded_q4_blocks(blocks, 0x41_00 + rows as u64 * 101 + in_features as u64),
                [out_features, in_features],
                KQuantDtype::Q4K,
            );
            let src = seeded_activations(rows * in_features, 0x4C_00 + out_features as u64);
            let expected = eager_reference(&weight, &src, rows);
            let mut actual = vec![0.0f32; rows * out_features];
            matmul_k_scalar_into(&src, rows, &weight, &mut actual).unwrap();
            assert_gate_a(&actual, &expected);
        }
    }

    #[test]
    fn q4_k_zero_scale_and_min_contribute_exactly_zero() {
        // d = 0 and min = 0 (f16 zeros at offsets 0 and 2) dequantize
        // every value to 0.0, so the row must stay exactly zero.
        let mut bytes = seeded_q4_blocks(2, 0x4D);
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            block[0..4].fill(0);
        }
        let weight = KQuantWeight::new(bytes, [1, 512], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 512, 0x4E);
        let mut actual = vec![1.0f32; 2]; // nonzero sentinel: accumulation must not disturb
        matmul_k_scalar_into(&src, 2, &weight, &mut actual).unwrap();
        assert_eq!(actual, vec![1.0; 2]);
    }

    #[test]
    fn q4_k_saturated_scale_fields_and_quants_pass_gate() {
        // every 6-bit scale/min field pinned to 63 and every nibble set:
        // exercises the 12-byte K4 scale bit-reshuffle (get_scale_min_k4)
        // and the low/high nibble pairing.
        let mut bytes = seeded_q4_blocks(4, 0x4F);
        for block in bytes.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            block[4..16].fill(0xFF); // all 6-bit scale/min fields = 63
            block[16..144].fill(0xFF); // all quants = 15
        }
        let weight = KQuantWeight::new(bytes, [1, 1024], KQuantDtype::Q4K);
        let src = seeded_activations(2 * 1024, 0x50);
        let expected = eager_reference(&weight, &src, 2);
        let mut actual = vec![0.0f32; 2];
        matmul_k_scalar_into(&src, 2, &weight, &mut actual).unwrap();
        assert_gate_a(&actual, &expected);
        assert!(actual.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q4_k_zero_scale_keeps_min_contribution() {
        // d = 0 with min != 0: every value is -min*m (scale fields 63),
        // so the row equals -(63 * min) * sum(x) — the min path must
        // survive a zero scale.
        let mut bytes = seeded_q4_blocks(1, 0x51);
        bytes[0..2].fill(0); // d = 0
        bytes[2..4].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes()); // min = 2
        bytes[4..16].fill(0xFF); // sc/m = 63
        bytes[16..144].fill(0x00); // all quants = 0
        let weight = KQuantWeight::new(bytes, [1, 256], KQuantDtype::Q4K);
        let src = seeded_activations(256, 0x52);
        let expected = eager_reference(&weight, &src, 1);
        let mut actual = vec![0.0f32; 1];
        matmul_k_scalar_into(&src, 1, &weight, &mut actual).unwrap();
        assert_gate_a(&actual, &expected);
        // value = d*sc*q - min*m = 0 - 2*63 = -126 per element
        let sum: f32 = src.iter().sum();
        assert!((actual[0] + 126.0 * sum).abs() <= 1e-3 * (126.0 * sum).abs().max(1.0));
    }

    #[test]
    fn q4_k_length_mismatches_are_rejected() {
        let weight = KQuantWeight::new(seeded_q4_blocks(1, 0x53), [1, 256], KQuantDtype::Q4K);
        let src = seeded_activations(256, 0x54);
        let mut dst = vec![0.0f32; 1];
        assert!(matmul_k_scalar_into(&src[..255], 1, &weight, &mut dst).is_err());
        assert!(matmul_k_scalar_into(&src, 1, &weight, &mut dst[..0]).is_err());
    }
}

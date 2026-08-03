//! GGML K-family super-block dequantization (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
//!
//! Transcribed from llama.cpp `ggml-quants.c` / `ggml-common.h`
//! (reference: ggml-org/llama.cpp master). Each block dequantizes
//! `QK_K = 256` values. Byte layouts and formulas must match llama.cpp
//! exactly; the end-to-end logits comparison against llama.cpp is the
//! validation gate for any change here.
//!
//! Loaders use these to materialize K-quant tensors as f32 (the
//! dequant-to-f32 path); memory cost is 4 bytes/value, which is acceptable
//! for small-model research loads.

use crate::quant::QuantizedData;
use crate::tensor::CpuTensor;
use anyhow::{bail, Result};
use memmap2::Mmap;
use std::ops::Range;
use std::sync::Arc;

/// Super-block size (values per block) for all K-family types.
pub const QK_K: usize = 256;

pub const Q2_K_BLOCK_BYTES: usize = 84;
pub const Q3_K_BLOCK_BYTES: usize = 110;
pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Per-dtype block byte size; `None` for non-K dtypes.
pub fn k_block_bytes(dtype: u32) -> Option<usize> {
    match dtype {
        DTYPE_Q2_K => Some(Q2_K_BLOCK_BYTES),
        DTYPE_Q3_K => Some(Q3_K_BLOCK_BYTES),
        DTYPE_Q4_K => Some(Q4_K_BLOCK_BYTES),
        DTYPE_Q5_K => Some(Q5_K_BLOCK_BYTES),
        DTYPE_Q6_K => Some(Q6_K_BLOCK_BYTES),
        _ => None,
    }
}

/// GGUF dtype codes for the K family (GGML_TYPE_*, current numbering).
///
/// Note: the K-family codes were renumbered in 2024 (Q2_K=10 ... Q6_K=14,
/// Q8_K=15); the older numbering (Q2_K=11 ... Q6_K=15) is not used by any
/// current writer.
pub const DTYPE_Q2_K: u32 = 10;
pub const DTYPE_Q3_K: u32 = 11;
pub const DTYPE_Q4_K: u32 = 12;
pub const DTYPE_Q5_K: u32 = 13;
pub const DTYPE_Q6_K: u32 = 14;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Dequantize one Q2_K block (256 values, 84 bytes).
///
/// Reference: `dequantize_row_q2_K`. Scales/mins are 4-bit pairs in
/// `scales[16]`; quants are 2-bit, four values per byte.
pub fn dequant_q2_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q2_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let scales = &block[0..16];
    let qs = &block[16..80];
    let mut is = 0usize;
    let mut q = 0usize;
    let mut out_idx = 0usize;
    for _ in 0..2 {
        // two 128-value halves; four 2-bit planes each
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * f32::from(sc & 0x0F);
            let ml = min * f32::from(sc >> 4);
            for l in 0..16 {
                out[out_idx] = dl * f32::from((qs[q + l] >> shift) & 3) - ml;
                out_idx += 1;
            }
            let sc = scales[is];
            is += 1;
            let dl = d * f32::from(sc & 0x0F);
            let ml = min * f32::from(sc >> 4);
            for l in 0..16 {
                out[out_idx] = dl * f32::from((qs[q + 16 + l] >> shift) & 3) - ml;
                out_idx += 1;
            }
            shift += 2;
        }
        q += 32;
    }
}

/// Dequantize one Q3_K block (256 values, 110 bytes).
///
/// Reference: `dequantize_row_q3_K` (including the 12-byte scale
/// bit-reshuffle into 16 int8 sub-block scales).
pub fn dequant_q3_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q3_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let hmask = &block[0..32];
    let qs = &block[32..96];

    // reshape the 12 scale bytes into 16 int8 sub-block scales
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    let s0 = u32::from_le_bytes(block[96..100].try_into().unwrap());
    let s1 = u32::from_le_bytes(block[100..104].try_into().unwrap());
    let tmp = u32::from_le_bytes(block[104..108].try_into().unwrap());
    let a0 = (s0 & K2) | ((tmp & K1) << 4);
    let a1 = (s1 & K2) | (((tmp >> 2) & K1) << 4);
    let a2 = ((s0 >> 4) & K2) | (((tmp >> 4) & K1) << 4);
    let a3 = ((s1 >> 4) & K2) | (((tmp >> 6) & K1) << 4);
    let mut scales = [0u8; 16];
    scales[0..4].copy_from_slice(&a0.to_le_bytes());
    scales[4..8].copy_from_slice(&a1.to_le_bytes());
    scales[8..12].copy_from_slice(&a2.to_le_bytes());
    scales[12..16].copy_from_slice(&a3.to_le_bytes());
    // the C source holds these in int8_t; bytes >= 0x80 are negative
    let signed = |b: u8| i32::from(i8::from_le_bytes([b]));

    // The reference advances `q` by 32 per 128-value chunk and never
    // advances `hm`; `m` doubles across all eight iterations (1..=128)
    // so each hmask byte covers 8 values.
    let mut is = 0usize;
    let mut m = 1u8;
    let mut out_idx = 0usize;
    for chunk in 0..2 {
        let q = chunk * 32;
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (signed(scales[is]) - 32) as f32;
            is += 1;
            for l in 0..16 {
                let qv = f32::from((qs[q + l] >> shift) & 3);
                let sub = if hmask[l] & m != 0 { 0.0 } else { 4.0 };
                out[out_idx] = dl * (qv - sub);
                out_idx += 1;
            }
            let dl = d_all * (signed(scales[is]) - 32) as f32;
            is += 1;
            for l in 0..16 {
                let qv = f32::from((qs[q + 16 + l] >> shift) & 3);
                let sub = if hmask[16 + l] & m != 0 { 0.0 } else { 4.0 };
                out[out_idx] = dl * (qv - sub);
                out_idx += 1;
            }
            shift += 2;
            m <<= 1;
        }
    }
}

/// `get_scale_min_k4`: unpack one 6-bit (scale, min) pair from the 12-byte
/// K4-style scale array. `j` is the sub-block index (0..8).
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one Q4_K block (256 values, 144 bytes).
///
/// Reference: `dequantize_row_q4_K` + `get_scale_min_k4`.
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q4_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut is = 0usize;
    let mut q = 0usize;
    let mut out_idx = 0usize;
    for _ in 0..4 {
        // 64 values per sub-block pair: two 32-value halves (low/high nibble)
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = min * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = min * f32::from(m);
        for l in 0..32 {
            out[out_idx] = d1 * f32::from(qs[q + l] & 0x0F) - m1;
            out_idx += 1;
        }
        for l in 0..32 {
            out[out_idx] = d2 * f32::from(qs[q + l] >> 4) - m2;
            out_idx += 1;
        }
        q += 32;
        is += 2;
    }
}

/// Dequantize one Q5_K block (256 values, 176 bytes).
///
/// Reference: `dequantize_row_q5_K` (K4-style scales + 5th-bit `qh`).
pub fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q5_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut is = 0usize;
    let mut ql = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    let mut out_idx = 0usize;
    for _ in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = min * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = min * f32::from(m);
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out[out_idx] = d1 * f32::from((qs[ql + l] & 0x0F) + hi) - m1;
            out_idx += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[out_idx] = d2 * f32::from((qs[ql + l] >> 4) + hi) - m2;
            out_idx += 1;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantize one Q6_K block (256 values, 210 bytes).
///
/// Reference: `dequantize_row_q6_K` (6-bit quants via `ql` + `qh`,
/// int8 per-16 scales).
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q6_K_BLOCK_BYTES);
    assert!(out.len() >= QK_K);
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    // scales are int8_t in the reference; bytes >= 0x80 are negative
    let signed_scale = |b: u8| i32::from(i8::from_le_bytes([b]));
    let mut y = 0usize;
    let mut q = 0usize;
    let mut h = 0usize;
    let mut s = 0usize;
    for _ in 0..2 {
        // 128 values per half, four 32-value groups
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from((ql[q + l] & 0x0F) | ((qh[h + l] & 3) << 4)) - 32;
            let q2 = i32::from((ql[q + l + 32] & 0x0F) | (((qh[h + l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[q + l] >> 4) | (((qh[h + l] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[q + l + 32] >> 4) | (((qh[h + l] >> 6) & 3) << 4)) - 32;
            out[y + l] = d * (signed_scale(scales[s + is]) as f32) * (q1 as f32);
            out[y + l + 32] = d * (signed_scale(scales[s + is + 2]) as f32) * (q2 as f32);
            out[y + l + 64] = d * (signed_scale(scales[s + is + 4]) as f32) * (q3 as f32);
            out[y + l + 96] = d * (signed_scale(scales[s + is + 6]) as f32) * (q4 as f32);
        }
        y += 128;
        q += 64;
        h += 32;
        s += 8;
    }
}

/// Dequantize a full K-quant tensor payload into `out`.
///
/// `bytes` must contain whole blocks of the given dtype; `out` must have
/// exactly `element_count` slots. Returns an error on length mismatch.
pub fn dequant_tensor(dtype: u32, bytes: &[u8], out: &mut [f32]) -> Result<(), String> {
    let (block_bytes, block_count) = match dtype {
        DTYPE_Q2_K => (Q2_K_BLOCK_BYTES, bytes.len() / Q2_K_BLOCK_BYTES),
        DTYPE_Q3_K => (Q3_K_BLOCK_BYTES, bytes.len() / Q3_K_BLOCK_BYTES),
        DTYPE_Q4_K => (Q4_K_BLOCK_BYTES, bytes.len() / Q4_K_BLOCK_BYTES),
        DTYPE_Q5_K => (Q5_K_BLOCK_BYTES, bytes.len() / Q5_K_BLOCK_BYTES),
        DTYPE_Q6_K => (Q6_K_BLOCK_BYTES, bytes.len() / Q6_K_BLOCK_BYTES),
        other => return Err(format!("unsupported K-quant dtype {other}")),
    };
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(format!(
            "K-quant payload length {} is not a multiple of block size {block_bytes}",
            bytes.len()
        ));
    }
    let expected_values = block_count
        .checked_mul(QK_K)
        .ok_or_else(|| "K-quant output length overflow".to_string())?;
    if expected_values != out.len() {
        return Err(format!(
            "K-quant payload dequantizes to {} values but output has {}",
            expected_values,
            out.len()
        ));
    }
    for (i, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let out_block = &mut out[i * QK_K..(i + 1) * QK_K];
        match dtype {
            DTYPE_Q2_K => dequant_q2_k(block, out_block),
            DTYPE_Q3_K => dequant_q3_k(block, out_block),
            DTYPE_Q4_K => dequant_q4_k(block, out_block),
            DTYPE_Q5_K => dequant_q5_k(block, out_block),
            DTYPE_Q6_K => dequant_q6_k(block, out_block),
            _ => unreachable!("validated above"),
        }
    }
    Ok(())
}

/// K-family dtypes with a native compressed-resident path in v0.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantDtype {
    /// GGML_TYPE_Q4_K (code 12), 144 bytes per 256-value super-block.
    Q4K,
    /// GGML_TYPE_Q6_K (code 14), 210 bytes per 256-value super-block.
    Q6K,
}

impl KQuantDtype {
    /// Map a GGUF dtype code onto a native K-quant dtype.
    ///
    /// Only dtypes with a compressed-resident path in v0.3 map here;
    /// Q2_K/Q3_K/Q5_K/Q8_K remain eager-f32-only and return `None`.
    pub fn from_gguf(code: u32) -> Option<Self> {
        match code {
            DTYPE_Q4_K => Some(Self::Q4K),
            DTYPE_Q6_K => Some(Self::Q6K),
            _ => None,
        }
    }

    /// The GGUF dtype code for this type.
    pub fn gguf_code(self) -> u32 {
        match self {
            Self::Q4K => DTYPE_Q4_K,
            Self::Q6K => DTYPE_Q6_K,
        }
    }

    /// Bytes per 256-value super-block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    /// Lowercase GGUF type name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
        }
    }
}

/// Load-time execution policy for K-family tensors (Q2_K..Q6_K, Q8_K).
///
/// The policy is applied per tensor: Q4_K_M files mix Q4_K and Q6_K, and
/// dtypes without a native kernel in v0.3 (Q2_K/Q3_K/Q5_K/Q8_K) stay on
/// the eager-f32 reference path. `--k-strategy` never governs non-K
/// dtypes (Q8_0 keeps its own compressed path unconditionally).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KStrategy {
    /// Dequantize every K-family tensor to f32 at load (v0.1/v0.2
    /// reference path).
    EagerF32,
    /// Compressed resident, scalar kernels.
    Scalar,
    /// Compressed resident, AVX2 kernels (x86 feature set required).
    X86,
    /// Best available per tensor: compressed when a native kernel exists,
    /// eager-f32 otherwise, with every choice recorded.
    Auto,
}

impl KStrategy {
    /// CLI-facing name.
    pub fn name(self) -> &'static str {
        match self {
            Self::EagerF32 => "eager-f32",
            Self::Scalar => "compressed-scalar",
            Self::X86 => "compressed-x86",
            Self::Auto => "auto",
        }
    }

    /// Parse the `--k-strategy` CLI value.
    pub fn from_cli(value: &str) -> Result<Self, String> {
        match value {
            "eager-f32" => Ok(Self::EagerF32),
            "scalar" => Ok(Self::Scalar),
            "x86" => Ok(Self::X86),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "unknown K strategy '{other}'; expected one of: eager-f32, scalar, x86, auto"
            )),
        }
    }
}

/// The per-tensor execution decision made by the loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KExecution {
    /// Dequantized to f32 at load; the eager-f32 reference path.
    EagerF32,
    /// Packed bytes resident, scalar kernels.
    CompressedScalar,
    /// Packed bytes resident, AVX2 kernels (x86 feature set required).
    CompressedX86,
}

/// One loader decision for a K-family tensor (recorded, never silent).
#[derive(Debug, Clone)]
pub struct KTensorDecision {
    /// Original GGUF dtype code.
    pub gguf_dtype: u32,
    /// Chosen execution path.
    pub execution: KExecution,
    /// Why the requested strategy was not honored, if it was not.
    pub fallback_reason: Option<String>,
}

/// A Q4_K or Q6_K weight matrix kept in raw block-compressed form.
///
/// Mirrors `QuantizedWeight` (Q8_0): the loader stores the packed GGUF
/// bytes (owned or mmap-backed) and kernels dequantize at super-block
/// granularity during matmul. There is no persistent f32 expansion on
/// this path.
///
/// Layout: GGUF dims are reversed from `[in, out]` to `[out, in]` so
/// super-blocks (256 values each) are contiguous per output feature.
/// `shape[0]` is `out_features`, `shape[1]` is `in_features` (always a
/// multiple of `QK_K`).
#[derive(Clone, Debug)]
pub struct KQuantWeight {
    data: QuantizedData,
    shape: [usize; 2],
    dtype: KQuantDtype,
    /// Per-tensor execution decision recorded at load; the matmul
    /// dispatch executes this decision and never silently downgrades.
    execution: KExecution,
}

impl KQuantWeight {
    /// Create a weight from raw packed bytes and logical shape
    /// `[out_features, in_features]`.
    pub fn new(data: Vec<u8>, shape: [usize; 2], dtype: KQuantDtype) -> Self {
        Self::try_new(data, shape, dtype).expect("invalid K-quant weight")
    }

    /// Fallible constructor for weights loaded from model files.
    pub fn try_new(data: Vec<u8>, shape: [usize; 2], dtype: KQuantDtype) -> Result<Self> {
        Self::try_new_storage(
            QuantizedData::Owned(data.into()),
            shape,
            dtype,
            KExecution::CompressedScalar,
        )
    }

    /// Construct from a shared read-only file mapping (loader path).
    #[allow(dead_code)] // wired into the loader in the compressed-dispatch commit
    pub(crate) fn try_from_mmap(
        mmap: Arc<Mmap>,
        range: Range<usize>,
        shape: [usize; 2],
        dtype: KQuantDtype,
        execution: KExecution,
    ) -> Result<Self> {
        if range.start > range.end || range.end > mmap.len() {
            bail!(
                "KQuantWeight: mmap range {:?} exceeds mapping length {}",
                range,
                mmap.len()
            );
        }
        Self::try_new_storage(
            QuantizedData::Mapped { mmap, range },
            shape,
            dtype,
            execution,
        )
    }

    fn try_new_storage(
        data: QuantizedData,
        shape: [usize; 2],
        dtype: KQuantDtype,
        execution: KExecution,
    ) -> Result<Self> {
        if shape[0] == 0 || shape[1] == 0 {
            bail!("KQuantWeight: dimensions must be non-zero, got {:?}", shape);
        }
        if !shape[1].is_multiple_of(QK_K) {
            bail!(
                "KQuantWeight: in_features ({}) must be a multiple of {}",
                shape[1],
                QK_K
            );
        }
        let expected_elements = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| anyhow::anyhow!("KQuantWeight: shape product overflow"))?;
        let expected_blocks = expected_elements / QK_K;
        let expected_len = expected_blocks
            .checked_mul(dtype.block_bytes())
            .ok_or_else(|| anyhow::anyhow!("KQuantWeight: byte length overflow"))?;
        if data.as_slice().len() != expected_len {
            bail!(
                "KQuantWeight: data len ({}) != expected ({}) for {}",
                data.as_slice().len(),
                expected_len,
                dtype.name()
            );
        }
        Ok(Self {
            data,
            shape,
            dtype,
            execution,
        })
    }

    /// Raw packed storage.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Compressed byte size of this weight.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.as_slice().len()
    }

    /// Whether this weight directly references a read-only file mapping.
    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(&self.data, QuantizedData::Mapped { .. })
    }

    /// Per-tensor GGUF type.
    #[inline]
    pub fn dtype(&self) -> KQuantDtype {
        self.dtype
    }

    /// Per-tensor execution decision recorded at load.
    #[inline]
    pub fn execution(&self) -> KExecution {
        self.execution
    }

    /// Override the recorded execution decision (used by the loader when
    /// it resolves the requested strategy against CPU features).
    pub fn with_execution(mut self, execution: KExecution) -> Self {
        self.execution = execution;
        self
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    /// Super-blocks per output feature (`in_features / QK_K`).
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.shape[1] / QK_K
    }

    /// Dequantize one output-feature row (or one embedding row) into
    /// `dst`, which must have length `in_features`.
    ///
    /// Rows are contiguous: output feature `j` occupies
    /// `blocks_per_row` super-blocks at byte offset
    /// `j * blocks_per_row * block_bytes`.
    pub fn dequantize_row(&self, row: usize, dst: &mut [f32]) {
        assert!(
            row < self.out_features(),
            "KQuantWeight row {} out of bounds for {} rows",
            row,
            self.out_features()
        );
        assert_eq!(
            dst.len(),
            self.in_features(),
            "KQuantWeight dequantize_row destination len ({}) != in_features ({})",
            dst.len(),
            self.in_features()
        );
        let block_bytes = self.dtype.block_bytes();
        let row_bytes = row * self.blocks_per_row() * block_bytes;
        let data = self.data();
        for b in 0..self.blocks_per_row() {
            let block = &data[row_bytes + b * block_bytes..row_bytes + (b + 1) * block_bytes];
            let out_block = &mut dst[b * QK_K..(b + 1) * QK_K];
            match self.dtype {
                KQuantDtype::Q4K => dequant_q4_k(block, out_block),
                KQuantDtype::Q6K => dequant_q6_k(block, out_block),
            }
        }
    }

    /// Fully dequantize to an f32 `CpuTensor` with shape
    /// `[out_features, in_features]`, data contiguous per output feature.
    ///
    /// Explicit reference conversion (the eager-f32 path); never used by
    /// the compressed-resident execution path.
    pub fn dequantize_all(&self) -> CpuTensor {
        let element_count = self
            .out_features()
            .checked_mul(self.in_features())
            .expect("KQuantWeight shape product validated at construction");
        let mut data = vec![0.0f32; element_count];
        dequant_tensor(self.dtype.gguf_code(), self.data(), &mut data)
            .expect("KQuantWeight payload validated at construction");
        CpuTensor::from_data(self.shape.to_vec(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation of the same math, kept structurally close to
    /// the llama.cpp source (transcription check, not an independent oracle;
    /// the end-to-end logits comparison against llama.cpp is the real gate).
    fn ref_q4_k(block: &[u8]) -> Vec<f32> {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut out = Vec::with_capacity(QK_K);
        let mut is = 0;
        let mut q = 0;
        for _ in 0..4 {
            for (half, take_high) in [(0usize, false), (1usize, true)] {
                let (sc, m) = get_scale_min_k4(is + half, scales);
                let d1 = d * f32::from(sc);
                let m1 = min * f32::from(m);
                for l in 0..32 {
                    let nib = if take_high {
                        qs[q + l] >> 4
                    } else {
                        qs[q + l] & 0x0F
                    };
                    out.push(d1 * f32::from(nib) - m1);
                }
            }
            q += 32;
            is += 2;
        }
        out
    }

    fn ref_q6_k(block: &[u8]) -> Vec<f32> {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let signed_scale = |b: u8| i32::from(i8::from_le_bytes([b]));
        let mut out = vec![0.0f32; QK_K];
        let mut y = 0;
        let mut q = 0;
        let mut h = 0;
        let mut s = 0;
        for _ in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[q + l] & 0x0F) | ((qh[h + l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[q + l + 32] & 0x0F) | (((qh[h + l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[q + l] >> 4) | (((qh[h + l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[q + l + 32] >> 4) | (((qh[h + l] >> 6) & 3) << 4)) - 32;
                out[y + l] = d * (signed_scale(scales[s + is]) as f32) * (q1 as f32);
                out[y + l + 32] = d * (signed_scale(scales[s + is + 2]) as f32) * (q2 as f32);
                out[y + l + 64] = d * (signed_scale(scales[s + is + 4]) as f32) * (q3 as f32);
                out[y + l + 96] = d * (signed_scale(scales[s + is + 6]) as f32) * (q4 as f32);
            }
            y += 128;
            q += 64;
            h += 32;
            s += 8;
        }
        out
    }

    fn ref_q3_k(block: &[u8]) -> Vec<f32> {
        let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let s0 = u32::from_le_bytes(block[96..100].try_into().unwrap());
        let s1 = u32::from_le_bytes(block[100..104].try_into().unwrap());
        let tmp = u32::from_le_bytes(block[104..108].try_into().unwrap());
        let a0 = (s0 & 0x0f0f0f0f) | ((tmp & 0x03030303) << 4);
        let a1 = (s1 & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
        let a2 = ((s0 >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        let a3 = ((s1 >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        let mut scales = [0u8; 16];
        scales[0..4].copy_from_slice(&a0.to_le_bytes());
        scales[4..8].copy_from_slice(&a1.to_le_bytes());
        scales[8..12].copy_from_slice(&a2.to_le_bytes());
        scales[12..16].copy_from_slice(&a3.to_le_bytes());
        let signed = |b: u8| i32::from(i8::from_le_bytes([b]));
        let mut out = Vec::with_capacity(QK_K);
        let mut is = 0;
        let mut m = 1u8;
        for chunk in 0..2 {
            let q = chunk * 32;
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * (signed(scales[is]) - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = f32::from((qs[q + l] >> shift) & 3);
                    let sub = if hmask[l] & m != 0 { 0.0 } else { 4.0 };
                    out.push(dl * (qv - sub));
                }
                let dl = d_all * (signed(scales[is]) - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = f32::from((qs[q + 16 + l] >> shift) & 3);
                    let sub = if hmask[16 + l] & m != 0 { 0.0 } else { 4.0 };
                    out.push(dl * (qv - sub));
                }
                shift += 2;
                m <<= 1;
            }
        }
        out
    }

    fn ref_q2_k(block: &[u8]) -> Vec<f32> {
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let scales = &block[0..16];
        let qs = &block[16..80];
        let mut out = Vec::with_capacity(QK_K);
        let mut is = 0;
        let mut q = 0;
        for _ in 0..2 {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    out.push(dl * f32::from((qs[q + l] >> shift) & 3) - ml);
                }
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    out.push(dl * f32::from((qs[q + 16 + l] >> shift) & 3) - ml);
                }
                shift += 2;
            }
            q += 32;
        }
        out
    }

    fn ref_q5_k(block: &[u8]) -> Vec<f32> {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut out = Vec::with_capacity(QK_K);
        let mut is = 0;
        let mut ql = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in 0..4 {
            for (half, u) in [(0usize, u1), (1usize, u2)] {
                let (sc, m) = get_scale_min_k4(is + half, scales);
                let d1 = d * f32::from(sc);
                let m1 = min * f32::from(m);
                for l in 0..32 {
                    let hi = if qh[l] & u != 0 { 16 } else { 0 };
                    let nib = if half == 0 {
                        qs[ql + l] & 0x0F
                    } else {
                        qs[ql + l] >> 4
                    };
                    out.push(d1 * f32::from(nib + hi) - m1);
                }
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        out
    }

    #[test]
    fn k_dequants_match_reference_implementations() {
        // deterministic pseudo-random blocks
        let mut seed = 0x5eed_1234u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        type BlockRef = fn(&[u8]) -> Vec<f32>;
        let cases: &[(u32, usize, BlockRef)] = &[
            (DTYPE_Q2_K, Q2_K_BLOCK_BYTES, ref_q2_k),
            (DTYPE_Q3_K, Q3_K_BLOCK_BYTES, ref_q3_k),
            (DTYPE_Q4_K, Q4_K_BLOCK_BYTES, ref_q4_k),
            (DTYPE_Q5_K, Q5_K_BLOCK_BYTES, ref_q5_k),
            (DTYPE_Q6_K, Q6_K_BLOCK_BYTES, ref_q6_k),
        ];
        for (dtype, block_bytes, reference) in cases {
            for _ in 0..8 {
                let mut block: Vec<u8> = (0..*block_bytes).map(|_| next()).collect();
                // sanitize the f16 scale fields so random bytes cannot
                // produce NaN/Inf scales (both impls agree, but NaN != NaN)
                let f16_offsets: &[usize] = match *dtype {
                    DTYPE_Q2_K => &[80, 82],
                    DTYPE_Q3_K => &[108],
                    DTYPE_Q4_K => &[0, 2],
                    DTYPE_Q5_K => &[0, 2],
                    DTYPE_Q6_K => &[208],
                    _ => &[],
                };
                for offset in f16_offsets {
                    let bits = u16::from_le_bytes([block[*offset], block[*offset + 1]]);
                    let bits = bits & 0x7FFF;
                    let bits = if bits >= 0x7C00 { 0x3C00 } else { bits };
                    block[*offset..*offset + 2].copy_from_slice(&bits.to_le_bytes());
                }
                let mut out = vec![0.0f32; QK_K];
                dequant_tensor(*dtype, &block, &mut out).unwrap();
                let expected = reference(&block);
                assert_eq!(out, expected, "dtype {dtype} mismatch");
            }
        }
    }

    /// End-to-end diagnostic: dequantize every K-quant tensor of a real GGUF
    /// (path from EMBER_K_TEST_GGUF) and report finiteness and value ranges.
    /// Not a numerical oracle — the llama.cpp logits comparison is — but it
    /// catches block-offset mistakes on real data instantly.
    #[test]
    fn real_file_dequant_is_finite() {
        let Ok(path) = std::env::var("EMBER_K_TEST_GGUF") else {
            eprintln!("skipped: EMBER_K_TEST_GGUF not set");
            return;
        };
        let loader = crate::loader::load_gguf(&path).unwrap();
        let mut checked = 0usize;
        for (name, tensor) in &loader.tensors {
            let crate::loader::LoadedTensor::F32(tensor) = tensor else {
                continue;
            };
            let values = tensor.data();
            if values.is_empty() {
                continue;
            }
            checked += 1;
            let nan = values.iter().filter(|v| v.is_nan()).count();
            let inf = values.iter().filter(|v| v.is_infinite()).count();
            let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            assert!(
                nan == 0 && inf == 0,
                "{name}: {nan} NaN, {inf} Inf (min {min}, max {max})"
            );
        }
        assert!(checked > 0, "no f32 tensors found in {path}");
        eprintln!("checked {checked} dequantized tensors: all finite");
    }

    /// Value-level check against the fp16 source tensor: dequantize the
    /// K-quant tensor and correlate with the f16 tensor of the same name
    /// (EMBER_K_TEST_GGUF + EMBER_F16_TEST_GGUF). The first 4096 values must
    /// correlate > 0.9 with the fp16 reference (quantization noise only).
    #[test]
    fn real_file_dequant_matches_fp16_source() {
        let Ok(q_path) = std::env::var("EMBER_K_TEST_GGUF") else {
            eprintln!("skipped: EMBER_K_TEST_GGUF not set");
            return;
        };
        let Ok(f16_path) = std::env::var("EMBER_F16_TEST_GGUF") else {
            eprintln!("skipped: EMBER_F16_TEST_GGUF not set");
            return;
        };
        let q_loader = crate::loader::load_gguf(&q_path).unwrap();
        let f16_loader = crate::loader::load_gguf(&f16_path).unwrap();
        let mut checked = 0usize;
        for (name, tensor) in &q_loader.tensors {
            let crate::loader::LoadedTensor::F32(tensor) = tensor else {
                continue;
            };
            let Some(crate::loader::LoadedTensor::F32(reference)) = f16_loader.tensors.get(name)
            else {
                continue;
            };
            let values = &tensor.data()[..4096.min(tensor.data().len())];
            let reference = &reference.data()[..4096.min(reference.data().len())];
            if values.len() < 256 || reference.len() < 256 {
                continue;
            }
            // Pearson correlation
            let n = values.len();
            let mean_v: f64 = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
            let mean_r: f64 = reference.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
            let cov: f64 = values
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (*a as f64 - mean_v) * (*b as f64 - mean_r))
                .sum();
            let var_v: f64 = values.iter().map(|a| (*a as f64 - mean_v).powi(2)).sum();
            let var_r: f64 = reference.iter().map(|b| (*b as f64 - mean_r).powi(2)).sum();
            let corr = if var_v > 0.0 && var_r > 0.0 {
                cov / (var_v * var_r).sqrt()
            } else {
                0.0
            };
            eprintln!("{name}: corr {corr:.4}");
            if corr < 0.9 {
                panic!("{name}: dequant correlation {corr:.4} with fp16 source");
            }
            checked += 1;
        }
        assert!(checked > 0, "no comparable tensors");
        eprintln!("correlation check passed for {checked} tensors");
    }

    /// Debug dump: print the first real Q3_K block bytes + dequant values
    /// (EMBER_K_TEST_GGUF + tensor name from EMBER_K_DEBUG_TENSOR).
    #[test]
    fn debug_dump_q3_block() {
        let Ok(q_path) = std::env::var("EMBER_K_TEST_GGUF") else {
            return;
        };
        let Ok(name) = std::env::var("EMBER_K_DEBUG_TENSOR") else {
            return;
        };
        let loader = crate::loader::load_gguf(&q_path).unwrap();
        let crate::loader::LoadedTensor::F32(tensor) = &loader.tensors[&name] else {
            return;
        };
        let values = tensor.data();
        eprintln!("{name}: first 12 values = {:?}", &values[..12]);
        // dump the raw q3 scale bytes from the loader: find the tensor's
        // raw payload via a second load with raw access — instead, reconstruct
        // from the dequantized values: verify scale consistency below.
        let raw_path = std::env::var("EMBER_K_DEBUG_RAW").ok();
        if let Some(raw_path) = raw_path {
            let raw = std::fs::read(&raw_path).unwrap();
            let scale_bytes = &raw[96..108];
            eprintln!("raw scales[96..108]: {:02x?}", scale_bytes);
            let s0 = u32::from_le_bytes(scale_bytes[0..4].try_into().unwrap());
            let s1 = u32::from_le_bytes(scale_bytes[4..8].try_into().unwrap());
            let tmp = u32::from_le_bytes(scale_bytes[8..12].try_into().unwrap());
            let k1 = 0x0303_0303u32;
            let k2 = 0x0f0f_0f0fu32;
            let a0 = (s0 & k2) | ((tmp & k1) << 4);
            let a1 = (s1 & k2) | (((tmp >> 2) & k1) << 4);
            let a2 = ((s0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            let a3 = ((s1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            eprintln!("reshuffled: a0={a0:08x} a1={a1:08x} a2={a2:08x} a3={a3:08x}");
            let signed = |b: u8| i32::from(i8::from_le_bytes([b]));
            let l: Vec<i32> = [a0, a1, a2, a3]
                .iter()
                .flat_map(|a| a.to_le_bytes())
                .map(signed)
                .collect();
            eprintln!("l values (signed): {l:?}");
        }
        let f16_path = std::env::var("EMBER_F16_TEST_GGUF").unwrap();
        let f16_loader = crate::loader::load_gguf(&f16_path).unwrap();
        let crate::loader::LoadedTensor::F32(reference) = &f16_loader.tensors[&name] else {
            return;
        };
        let reference = reference.data();
        for block in 0..8 {
            let start = block * 256;
            let v = &values[start..start + 256];
            let r = &reference[start..start + 256];
            let n = 256usize;
            let mean_v: f64 = v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let mean_r: f64 = r.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let cov: f64 = v
                .iter()
                .zip(r.iter())
                .map(|(a, b)| (*a as f64 - mean_v) * (*b as f64 - mean_r))
                .sum();
            let var_v: f64 = v.iter().map(|a| (*a as f64 - mean_v).powi(2)).sum();
            let var_r: f64 = r.iter().map(|b| (*b as f64 - mean_r).powi(2)).sum();
            let corr = if var_v > 0.0 && var_r > 0.0 {
                cov / (var_v * var_r).sqrt()
            } else {
                0.0
            };
            eprintln!("  block {block}: corr {corr:.4}");
            if block == 0 {
                let mut diffs = Vec::new();
                for idx in 0..256 {
                    if (v[idx] - r[idx]).abs() > 0.01 {
                        diffs.push((idx, v[idx], r[idx]));
                    }
                    if diffs.len() >= 12 {
                        break;
                    }
                }
                eprintln!("  first big diffs (idx, deq, fp16): {diffs:?}");
            }
        }
    }

    /// Debug: print dequantized Q8_0 tensor rows (EMBER_K_TEST_GGUF +
    /// EMBER_K_DEBUG_TENSOR + EMBER_K_DEBUG_ROWS).
    #[test]
    fn debug_dump_q8_rows() {
        let Ok(q_path) = std::env::var("EMBER_K_TEST_GGUF") else {
            return;
        };
        let Ok(name) = std::env::var("EMBER_K_DEBUG_TENSOR") else {
            return;
        };
        let rows: usize = std::env::var("EMBER_K_DEBUG_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let loader = crate::loader::load_gguf(&q_path).unwrap();
        let crate::loader::LoadedTensor::Q8_0(weight) = &loader.tensors[&name] else {
            eprintln!("{name}: not Q8_0");
            return;
        };
        let mut row = vec![0.0f32; weight.shape[1]];
        for r in 0..rows.min(weight.shape[0]) {
            weight.dequantize_row(r, &mut row);
            eprintln!("{name} row {r}: {:?}", &row[..8.min(row.len())]);
        }
    }

    #[test]
    fn dequant_tensor_validates_lengths() {
        let mut out = vec![0.0f32; QK_K];
        assert!(dequant_tensor(DTYPE_Q4_K, &[0u8; 143], &mut out).is_err());
        assert!(dequant_tensor(DTYPE_Q4_K, &[0u8; 144], &mut out[..128]).is_err());
        assert!(dequant_tensor(99, &[0u8; 144], &mut out).is_err());
        let mut out2 = vec![0.0f32; 2 * QK_K];
        assert!(dequant_tensor(DTYPE_Q4_K, &[0u8; 288], &mut out2).is_ok());
    }
}

#[cfg(test)]
mod resident_tests {
    use super::*;

    /// Deterministic pseudo-random block payload.
    fn seeded_block_bytes(dtype: KQuantDtype, blocks: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut bytes = vec![0u8; blocks * dtype.block_bytes()];
        for byte in &mut bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        // sanitize the f16 scale fields so random bytes cannot produce
        // NaN/Inf scales (mirrors the existing dequant reference tests)
        let f16_offsets: &[usize] = match dtype {
            KQuantDtype::Q4K => &[0, 2],
            KQuantDtype::Q6K => &[208],
        };
        for block in bytes.chunks_exact_mut(dtype.block_bytes()) {
            for &offset in f16_offsets {
                let bits = u16::from_le_bytes([block[offset], block[offset + 1]]);
                let bits = bits & 0x7FFF;
                let bits = if bits >= 0x7C00 { 0x3C00 } else { bits };
                block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn k_dtype_mapping_roundtrips() {
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q6K] {
            assert_eq!(KQuantDtype::from_gguf(dtype.gguf_code()), Some(dtype));
        }
        // no native kernel in v0.3: Q8_0, Q2_K, Q3_K, Q5_K, Q8_K
        for code in [8, 10, 11, 13, 15] {
            assert_eq!(KQuantDtype::from_gguf(code), None);
        }
        assert_eq!(KQuantDtype::Q4K.block_bytes(), Q4_K_BLOCK_BYTES);
        assert_eq!(KQuantDtype::Q6K.block_bytes(), Q6_K_BLOCK_BYTES);
        assert_eq!(KQuantDtype::Q4K.name(), "q4_k");
        assert_eq!(KQuantDtype::Q6K.name(), "q6_k");
    }

    #[test]
    fn k_quant_weight_accepts_valid_layouts() {
        // [4, 256] q4_k: 4 super-blocks x 144 bytes
        let q4 = KQuantWeight::try_new(
            seeded_block_bytes(KQuantDtype::Q4K, 4, 1),
            [4, 256],
            KQuantDtype::Q4K,
        )
        .unwrap();
        assert_eq!(q4.out_features(), 4);
        assert_eq!(q4.in_features(), 256);
        assert_eq!(q4.blocks_per_row(), 1);
        assert_eq!(q4.byte_len(), 4 * Q4_K_BLOCK_BYTES);
        assert_eq!(q4.dtype(), KQuantDtype::Q4K);
        assert!(!q4.is_mapped());

        // [2, 512] q6_k: 4 super-blocks x 210 bytes
        let q6 = KQuantWeight::try_new(
            seeded_block_bytes(KQuantDtype::Q6K, 4, 2),
            [2, 512],
            KQuantDtype::Q6K,
        )
        .unwrap();
        assert_eq!(q6.blocks_per_row(), 2);
        assert_eq!(q6.byte_len(), 4 * Q6_K_BLOCK_BYTES);

        // clone shares storage (Arc-backed), so tied-head reuse is cheap
        let clone = q4.clone();
        assert_eq!(clone.data().as_ptr(), q4.data().as_ptr());
    }

    #[test]
    fn k_quant_weight_rejects_malformed_layouts() {
        let ok = seeded_block_bytes(KQuantDtype::Q4K, 4, 3);
        // zero dimensions
        assert!(KQuantWeight::try_new(ok.clone(), [0, 256], KQuantDtype::Q4K).is_err());
        assert!(KQuantWeight::try_new(ok.clone(), [4, 0], KQuantDtype::Q4K).is_err());
        // in_features not 256-aligned
        assert!(KQuantWeight::try_new(ok.clone(), [4, 128], KQuantDtype::Q4K).is_err());
        // wrong byte length for the declared dtype (q6 blocks where q4 expected)
        assert!(KQuantWeight::try_new(ok.clone(), [4, 256], KQuantDtype::Q6K).is_err());
        // truncated payload
        assert!(
            KQuantWeight::try_new(ok[..ok.len() - 1].to_vec(), [4, 256], KQuantDtype::Q4K).is_err()
        );
        // shape product overflow
        assert!(KQuantWeight::try_new(ok, [usize::MAX, 256], KQuantDtype::Q4K).is_err());
    }

    #[test]
    fn k_quant_weight_mmap_range_is_checked() {
        let bytes = seeded_block_bytes(KQuantDtype::Q4K, 2, 4);
        let mut path = std::env::temp_dir();
        path.push("ember-kquant-mmap-test.bin");
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // Safety: the mapping outlives every weight constructed from it below.
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });

        // valid range
        let weight = KQuantWeight::try_from_mmap(
            std::sync::Arc::clone(&mmap),
            0..bytes.len(),
            [2, 256],
            KQuantDtype::Q4K,
            KExecution::CompressedScalar,
        )
        .unwrap();
        assert!(weight.is_mapped());
        assert_eq!(weight.data(), &bytes[..]);

        // range past the mapping
        assert!(KQuantWeight::try_from_mmap(
            std::sync::Arc::clone(&mmap),
            bytes.len()..bytes.len() + 10,
            [2, 256],
            KQuantDtype::Q4K,
            KExecution::CompressedScalar,
        )
        .is_err());
        // reversed range
        let (reverse_start, reverse_end) = (10usize, 5usize);
        assert!(KQuantWeight::try_from_mmap(
            std::sync::Arc::clone(&mmap),
            reverse_start..reverse_end,
            [2, 256],
            KQuantDtype::Q4K,
            KExecution::CompressedScalar,
        )
        .is_err());
        // range shorter than the shape requires
        assert!(KQuantWeight::try_from_mmap(
            std::sync::Arc::clone(&mmap),
            0..bytes.len() - 1,
            [2, 256],
            KQuantDtype::Q4K,
            KExecution::CompressedScalar,
        )
        .is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dequantize_all_matches_dequant_tensor() {
        for (dtype, blocks, shape) in [
            (KQuantDtype::Q4K, 3, [3usize, 256]),
            (KQuantDtype::Q6K, 6, [3usize, 512]),
        ] {
            let bytes = seeded_block_bytes(dtype, blocks, 7);
            let weight = KQuantWeight::try_new(bytes.clone(), shape, dtype).unwrap();
            let expanded = weight.dequantize_all();
            assert_eq!(expanded.shape(), &shape[..]);

            let mut direct = vec![0.0f32; shape[0] * shape[1]];
            dequant_tensor(dtype.gguf_code(), &bytes, &mut direct).unwrap();
            assert_eq!(expanded.data(), &direct[..]);
        }
    }
}

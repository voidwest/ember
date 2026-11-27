use super::*;

// ---------------------------------------------------------------------------
// fused Q8_0 decode (seq_len = 1)
// ---------------------------------------------------------------------------
//
// When the input is a single row (decode), quantize it once to Q8_0 and use
// packed integer dot products against each compressed weight row.
//
//   out[j] = Σblocks (d_x · d_w · Σₖ q_x[k] · q_w[j][k])

/// Fused Q8_0 dot product for single-row input (decode, seq_len = 1).
///
/// Computes a Q8_0 × Q8_0 matrix-vector product for every output row.
///
/// # Panics
///
/// Panics if `x` is not a valid quantized input row, `out.len() !=
/// out_features`, or the
/// weight data is not a valid Q8_0 encoding.
#[inline]
pub(crate) fn matmul_q8_0_decode(x: &[u8], w: &QuantizedWeight, out: &mut [f32]) {
    let encoded_row_len = (w.in_features() / Q8_0_BLOCK_SIZE)
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 decode input size overflow");
    assert_eq!(x.len(), encoded_row_len);
    assert_eq!(out.len(), w.out_features());

    let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
    if should_parallel_q8_decode(w.out_features(), w.in_features()) {
        matmul_q8_0_decode_parallel(x, w, out);
        return;
    }

    matmul_q8_0_decode_dispatch_chunk(x, w.data(), blocks_per_row, out);
}

/// Packed Q8_0 matrix multiply for multiple activation rows.
///
/// Rows are scheduled together so prompt prefill enters Rayon once instead
/// of launching and joining a separate parallel projection for every token.
pub(crate) fn matmul_q8_0_batch(x: &[u8], rows: usize, w: &QuantizedWeight, out: &mut [f32]) {
    let encoded_row_len = (w.in_features() / Q8_0_BLOCK_SIZE)
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("q8_0 batch row size overflow");
    let expected_input_len = rows
        .checked_mul(encoded_row_len)
        .expect("q8_0 batch input size overflow");
    let expected_output_len = rows
        .checked_mul(w.out_features())
        .expect("q8_0 batch output size overflow");
    assert_eq!(x.len(), expected_input_len);
    assert_eq!(out.len(), expected_output_len);

    #[cfg(target_arch = "aarch64")]
    if rows > 1 && crate::q8_gemm::supported() {
        // Bit-identical tiled GEMM (same per-element block order and scale
        // arithmetic as the decode and legacy batch kernels).
        let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
        crate::q8_gemm::matmul(x, rows, w.data(), w.out_features(), blocks_per_row, out);
        return;
    }
    matmul_q8_0_batch_legacy(x, rows, w, out);
}

/// The pre-GEMM batch schedule: row tiles of up to four activation rows, each
/// streaming the whole weight matrix. Kept for CPUs without the tiled GEMM
/// and as the bit-identity oracle in tests.
pub(crate) fn matmul_q8_0_batch_legacy(
    x: &[u8],
    rows: usize,
    w: &QuantizedWeight,
    out: &mut [f32],
) {
    let encoded_row_len = (w.in_features() / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
    let tile_rows = q8_batch_tile_rows(rows);

    if tile_rows == 1 {
        x.par_chunks(encoded_row_len)
            .zip(out.par_chunks_mut(w.out_features()))
            .for_each(|(x_row, out_row)| {
                crate::decode_pool::without_team(|| matmul_q8_0_decode(x_row, w, out_row))
            });
        return;
    }

    let input_tile_len = encoded_row_len
        .checked_mul(tile_rows)
        .expect("q8_0 batch input tile size overflow");
    let output_tile_len = w
        .out_features()
        .checked_mul(tile_rows)
        .expect("q8_0 batch output tile size overflow");
    let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
    x.par_chunks(input_tile_len)
        .zip(out.par_chunks_mut(output_tile_len))
        .for_each(|(x_tile, out_tile)| {
            let tile_len = x_tile.len() / encoded_row_len;
            matmul_q8_0_batch_dispatch_tile(
                x_tile,
                tile_len,
                w.data(),
                w.out_features(),
                blocks_per_row,
                out_tile,
            );
        });
}

fn q8_batch_tile_rows(rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    if rows >= threads.saturating_mul(4) {
        4
    } else if rows >= threads.saturating_mul(2) {
        2
    } else {
        1
    }
}

pub(super) fn matmul_q8_0_batch_dispatch_tile(
    x: &[u8],
    rows: usize,
    data: &[u8],
    out_features: usize,
    blocks_per_row: usize,
    out: &mut [f32],
) {
    // The x86 kernels read `x` and `data` through raw pointers.
    let row_bytes = blocks_per_row * Q8_0_TYPE_SIZE;
    assert!(
        x.len() >= rows * row_bytes,
        "q8_0 batch tile input too short"
    );
    assert!(
        data.len() >= out_features * row_bytes,
        "q8_0 batch tile weight too short"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: features checked; the asserts above bound `x` and
            // `data`.
            unsafe {
                return x86_64::matmul_q8_0_batch_avx512_vnni(
                    x,
                    rows,
                    data,
                    out_features,
                    blocks_per_row,
                    out,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            // SAFETY: features checked; the asserts above bound `x` and
            // `data`.
            unsafe {
                return x86_64::matmul_q8_0_batch_avx2(
                    x,
                    rows,
                    data,
                    out_features,
                    blocks_per_row,
                    out,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("dotprod") && is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime checks guard both extensions; the batch caller validates shapes.
        unsafe {
            match rows {
                4 => {
                    return aarch64::matmul_q8_0_batch_dotprod::<4>(
                        x,
                        data,
                        out_features,
                        blocks_per_row,
                        out,
                    )
                }
                2 => {
                    return aarch64::matmul_q8_0_batch_dotprod::<2>(
                        x,
                        data,
                        out_features,
                        blocks_per_row,
                        out,
                    )
                }
                _ => {}
            }
        }
    }

    let encoded_row_len = blocks_per_row * Q8_0_TYPE_SIZE;
    for (x_row, out_row) in x
        .chunks_exact(encoded_row_len)
        .zip(out.chunks_exact_mut(out_features))
    {
        matmul_q8_0_decode_dispatch_chunk(x_row, data, blocks_per_row, out_row);
    }
}

fn should_parallel_q8_decode(out_features: usize, in_features: usize) -> bool {
    rayon::current_num_threads() > 1
        && out_features.saturating_mul(in_features) >= PARALLEL_Q8_DECODE_MIN_WORK
}

#[inline]
pub(crate) fn q8_decode_uses_row_parallel(out_features: usize, in_features: usize) -> bool {
    should_parallel_q8_decode(out_features, in_features)
}

fn matmul_q8_0_decode_parallel(x: &[u8], w: &QuantizedWeight, out: &mut [f32]) {
    matmul_q8_0_decode_tasks(&mut [Q8DecodeTask {
        x,
        weight: Q8DecodeWeight::Rows(w),
        out,
        columns: 1,
    }]);
}

// -- decode activation preparation ----------------------------------------------

/// Q8_0 blocks per team chunk when preparing decode activations.
const PREPARE_CHUNK_BLOCKS: usize = 32;

/// Quantize whole rows of `src` into `dst` (resized), splitting the blocks
/// across the decode team. Byte-identical to
/// [`crate::quant::quantize_q8_0_into`]: every block is encoded
/// independently by the same code.
pub(crate) fn quantize_q8_0_decode_into(src: &[f32], dst: &mut Vec<u8>) {
    assert!(src.len().is_multiple_of(Q8_0_BLOCK_SIZE));
    let blocks = src.len() / Q8_0_BLOCK_SIZE;
    dst.resize(blocks * Q8_0_TYPE_SIZE, 0);
    let chunks = blocks.div_ceil(PREPARE_CHUNK_BLOCKS);
    let out = crate::decode_pool::SharedMut::new(dst);
    crate::decode_pool::run_or_serial(chunks, &|chunk| {
        let first = chunk * PREPARE_CHUNK_BLOCKS;
        let count = PREPARE_CHUNK_BLOCKS.min(blocks - first);
        // SAFETY: chunks cover disjoint block ranges of `dst`.
        let encoded = unsafe { out.range(first * Q8_0_TYPE_SIZE, count * Q8_0_TYPE_SIZE) };
        crate::quant::quantize_q8_0_blocks(
            &src[first * Q8_0_BLOCK_SIZE..(first + count) * Q8_0_BLOCK_SIZE],
            encoded,
        );
    });
}

/// `gated = silu(gate) * up`, then quantize `gated` into `dst` (resized),
/// fused per block range on the decode team. Element-wise identical to
/// [`silu_mul_into`] followed by [`crate::quant::quantize_q8_0_into`].
pub(crate) fn silu_mul_quantize_q8_0_decode_into(
    gate: &[f32],
    up: &[f32],
    gated: &mut [f32],
    dst: &mut Vec<u8>,
) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), gated.len());
    assert!(gate.len().is_multiple_of(Q8_0_BLOCK_SIZE));
    let blocks = gate.len() / Q8_0_BLOCK_SIZE;
    dst.resize(blocks * Q8_0_TYPE_SIZE, 0);
    let chunks = blocks.div_ceil(PREPARE_CHUNK_BLOCKS);
    let out = crate::decode_pool::SharedMut::new(dst);
    let gated = crate::decode_pool::SharedMut::new(gated);
    crate::decode_pool::run_or_serial(chunks, &|chunk| {
        let first = chunk * PREPARE_CHUNK_BLOCKS;
        let count = PREPARE_CHUNK_BLOCKS.min(blocks - first);
        let values = first * Q8_0_BLOCK_SIZE..(first + count) * Q8_0_BLOCK_SIZE;
        // SAFETY: chunks cover disjoint element and block ranges.
        let (gated, encoded) = unsafe {
            (
                gated.range(values.start, values.len()),
                out.range(first * Q8_0_TYPE_SIZE, count * Q8_0_TYPE_SIZE),
            )
        };
        // Chunk boundaries are multiples of 32 elements, so SIMD SiLU
        // variants see the same full-width lanes as one whole-row call.
        silu_mul_into(&gate[values.clone()], &up[values], gated);
        crate::quant::quantize_q8_0_blocks(gated, encoded);
    });
}

// -- decode regions ------------------------------------------------------------

/// Weight layouts a decode region can read.
#[derive(Clone, Copy)]
pub(crate) enum Q8DecodeWeight<'a> {
    /// Row-contiguous GGUF Q8_0 blocks.
    Rows(&'a QuantizedWeight),
    /// Four-row interleaved stripes (the LM head's decode copy).
    Interleaved(&'a QuantizedWeightInterleaved),
    /// Sixteen-output packed records (x86 AVX-512 VNNI only).
    Packed16(&'a QuantizedWeightVnni),
}

impl Q8DecodeWeight<'_> {
    fn out_features(self) -> usize {
        match self {
            Self::Rows(w) => w.out_features(),
            Self::Interleaved(w) => w.out_features(),
            Self::Packed16(w) => w.out_features(),
        }
    }

    fn in_features(self) -> usize {
        match self {
            Self::Rows(w) => w.in_features(),
            Self::Interleaved(w) => w.in_features(),
            Self::Packed16(w) => w.in_features(),
        }
    }

    /// Row alignment of a chunk start: keeps every kernel on its widest tile.
    fn row_alignment(self) -> usize {
        match self {
            Self::Rows(_) => 8,
            Self::Interleaved(_) => crate::quant::INTERLEAVE,
            Self::Packed16(_) => VNNI_OUT_TILE,
        }
    }

    /// Compute output rows `row_start..row_start + out.len()` serially.
    ///
    /// Every kernel computes each output row with the same arithmetic no
    /// matter which range it is called with, so chunking never changes bits.
    fn compute_rows(self, x: &[u8], row_start: usize, out: &mut [f32]) {
        match self {
            Self::Rows(w) => {
                let blocks_per_row = w.in_features() / Q8_0_BLOCK_SIZE;
                let data = &w.data()[row_start * blocks_per_row * Q8_0_TYPE_SIZE..];
                matmul_q8_0_decode_dispatch_chunk(x, data, blocks_per_row, out);
            }
            Self::Interleaved(w) => matmul_q8_0_decode_interleaved(x, w, out, row_start),
            Self::Packed16(w) => {
                assert!(packed_q8_0_vnni_supported());
                assert!(row_start.is_multiple_of(VNNI_OUT_TILE));
                #[cfg(target_arch = "x86_64")]
                // SAFETY: runtime feature checks above cover every enabled ISA
                // extension. Repacking pads the last tile and guarantees one
                // complete record for every aligned output range addressed here.
                unsafe {
                    x86_64::matmul_q8_0_decode_packed16_avx512_vnni(
                        x,
                        w.packed_bytes(),
                        w.blocks_per_row,
                        out,
                        row_start,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                let _ = (w, x, out);
            }
        }
    }
}

impl Q8DecodeWeight<'_> {
    /// Weight rows `row_start..row_end` against `columns` activation rows,
    /// writing column `c` row `r` to `out[c * out_stride + r]`.
    ///
    /// ARM reuses each loaded weight block across up to four columns with the
    /// single-row arithmetic; other targets run the single-row kernel once per
    /// column over the (now cache-resident) weight rows.
    ///
    /// # Safety
    /// `out` must be valid for those writes and no other thread may access
    /// them concurrently.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel signature: buffers, strides and shape stay separate arguments on the hot path"
    )]
    unsafe fn compute_rows_columns(
        self,
        x: &[u8],
        columns: usize,
        row_start: usize,
        row_end: usize,
        out: *mut f32,
        out_stride: usize,
    ) {
        let blocks_per_row = self.in_features() / Q8_0_BLOCK_SIZE;
        let row_bytes = blocks_per_row * Q8_0_TYPE_SIZE;
        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("dotprod") && is_aarch64_feature_detected!("fp16") {
            match self {
                Self::Rows(w) => {
                    let layout = aarch64::RowsBlocks {
                        data: w.data().as_ptr(),
                        blocks: blocks_per_row,
                    };
                    // SAFETY: features checked above; the layout addresses
                    // rows below `out_features` of a validated weight, and
                    // the caller guarantees the output writes.
                    unsafe {
                        aarch64::matmul_q8_0_decode_columns_dotprod(
                            &layout, x, row_bytes, columns, row_start, row_end, out, out_stride,
                        );
                    }
                    return;
                }
                Self::Interleaved(w) => {
                    let layout = aarch64::InterleavedBlocks {
                        quants: w.quants().as_ptr(),
                        scales: w.scales().as_ptr(),
                        blocks: blocks_per_row,
                    };
                    // SAFETY: as above; interleaved stripes are padded to a
                    // multiple of four rows.
                    unsafe {
                        aarch64::matmul_q8_0_decode_columns_dotprod(
                            &layout, x, row_bytes, columns, row_start, row_end, out, out_stride,
                        );
                    }
                    return;
                }
                Self::Packed16(_) => {}
            }
        }
        for column in 0..columns {
            // SAFETY: the caller guarantees exclusive access to these writes.
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    out.add(column * out_stride + row_start),
                    row_end - row_start,
                )
            };
            self.compute_rows(
                &x[column * row_bytes..(column + 1) * row_bytes],
                row_start,
                out,
            );
        }
    }
}

/// One matvec of a decode region: `out = weight · x` for `columns`
/// quantized activation rows.
///
/// `x` holds `columns` encoded rows back to back and `out` holds `columns`
/// output rows of `weight.out_features()` values (`out[c * out_features + r]`
/// is weight row `r` against activation `c`). Each column is bit-identical to
/// a one-column task on that activation alone.
pub(crate) struct Q8DecodeTask<'a, 'o> {
    pub(crate) x: &'a [u8],
    pub(crate) weight: Q8DecodeWeight<'a>,
    pub(crate) out: &'o mut [f32],
    pub(crate) columns: usize,
}

impl<'a, 'o> Q8DecodeTask<'a, 'o> {
    /// A task over `columns` activation rows (see the type docs for layout).
    pub(crate) fn columns(
        x: &'a [u8],
        weight: Q8DecodeWeight<'a>,
        columns: usize,
        out: &'o mut [f32],
    ) -> Self {
        Self {
            x,
            weight,
            out,
            columns,
        }
    }
}

/// Most matvecs one decode region can fuse (Q/K/V is the widest user).
pub(crate) const MAX_DECODE_TASKS: usize = 4;

/// Target weight bytes per scheduled chunk. Small enough that a preempted
/// worker only delays the region by one short chunk, large enough that
/// claiming a chunk stays negligible and each worker streams long runs
/// (M1 Pro sweep: 16 KiB 62.7, 32 KiB 63.3, 64 KiB 65.2, 128 KiB 65.9,
/// 256 KiB 66.2, 384 KiB 64.4 tok/s).
const DECODE_CHUNK_BYTES: usize = 128 * 1024;

fn decode_chunk_rows(weight: Q8DecodeWeight<'_>) -> usize {
    let row_bytes = (weight.in_features() / Q8_0_BLOCK_SIZE) * Q8_0_TYPE_SIZE;
    (DECODE_CHUNK_BYTES / row_bytes.max(1))
        .max(1)
        .next_multiple_of(weight.row_alignment())
}

struct DecodeRegionTask<'a> {
    x: &'a [u8],
    weight: Q8DecodeWeight<'a>,
    out: *mut f32,
    columns: usize,
    rows: usize,
    chunk_rows: usize,
    first_chunk: usize,
}

struct DecodeRegion<'a> {
    tasks: [Option<DecodeRegionTask<'a>>; MAX_DECODE_TASKS],
    chunks: usize,
}

// SAFETY: the raw output pointers come from disjoint `&mut [f32]` borrows held
// for the whole region, and every chunk index maps to a disjoint row range of
// exactly one task, so concurrent chunks never alias.
unsafe impl Sync for DecodeRegion<'_> {}

impl DecodeRegion<'_> {
    fn run_chunk(&self, chunk: usize) {
        let task = self
            .tasks
            .iter()
            .flatten()
            .rev()
            .find(|task| task.first_chunk <= chunk)
            .expect("decode chunk index within region");
        let row_start = (chunk - task.first_chunk) * task.chunk_rows;
        let row_end = (row_start + task.chunk_rows).min(task.rows);
        if task.columns != 1 {
            // SAFETY: see `unsafe impl Sync for DecodeRegion`; every column's
            // row range lies inside the task's output slice.
            unsafe {
                task.weight.compute_rows_columns(
                    task.x,
                    task.columns,
                    row_start,
                    row_end,
                    task.out,
                    task.rows,
                );
            }
            return;
        }
        // SAFETY: see `unsafe impl Sync for DecodeRegion`; the range lies
        // inside the task's output slice.
        let out =
            unsafe { std::slice::from_raw_parts_mut(task.out.add(row_start), row_end - row_start) };
        task.weight.compute_rows(task.x, row_start, out);
    }
}

/// Several row-contiguous projections of one quantized activation row as one
/// decode region (bit-identical to calling [`matmul_q8_0_decode`] on each).
pub(crate) fn matmul_q8_0_decode_rows(
    x: &[u8],
    projections: &mut [(&QuantizedWeight, &mut [f32])],
) {
    let mut tasks: [Option<Q8DecodeTask<'_, '_>>; MAX_DECODE_TASKS] = [None, None, None, None];
    assert!(
        projections.len() <= MAX_DECODE_TASKS,
        "too many decode tasks"
    );
    let used = projections.len();
    for (slot, (weight, out)) in tasks.iter_mut().zip(projections.iter_mut()) {
        *slot = Some(Q8DecodeTask {
            x,
            weight: Q8DecodeWeight::Rows(weight),
            out,
            columns: 1,
        });
    }
    run_task_slots(&mut tasks[..used]);
}

/// Several 16-output packed projections of one activation row as one region.
pub(crate) fn matmul_q8_0_decode_packed16_many(
    x: &[u8],
    projections: &mut [(&QuantizedWeightVnni, &mut [f32])],
) {
    assert!(packed_q8_0_vnni_supported());
    let mut tasks: [Option<Q8DecodeTask<'_, '_>>; MAX_DECODE_TASKS] = [None, None, None, None];
    assert!(
        projections.len() <= MAX_DECODE_TASKS,
        "too many decode tasks"
    );
    let used = projections.len();
    for (slot, (weight, out)) in tasks.iter_mut().zip(projections.iter_mut()) {
        *slot = Some(Q8DecodeTask {
            x,
            weight: Q8DecodeWeight::Packed16(weight),
            out,
            columns: 1,
        });
    }
    run_task_slots(&mut tasks[..used]);
}

fn run_task_slots(slots: &mut [Option<Q8DecodeTask<'_, '_>>]) {
    match slots {
        [Some(a)] => matmul_q8_0_decode_tasks(std::slice::from_mut(a)),
        [Some(a), Some(b)] => {
            let (a, b) = (a.reborrow(), b.reborrow());
            matmul_q8_0_decode_tasks(&mut [a, b]);
        }
        [Some(a), Some(b), Some(c)] => {
            let (a, b, c) = (a.reborrow(), b.reborrow(), c.reborrow());
            matmul_q8_0_decode_tasks(&mut [a, b, c]);
        }
        [Some(a), Some(b), Some(c), Some(d)] => {
            let (a, b, c, d) = (a.reborrow(), b.reborrow(), c.reborrow(), d.reborrow());
            matmul_q8_0_decode_tasks(&mut [a, b, c, d]);
        }
        [] => {}
        _ => unreachable!("decode task slots are filled in order"),
    }
}

impl<'a> Q8DecodeTask<'a, '_> {
    fn reborrow(&mut self) -> Q8DecodeTask<'a, '_> {
        Q8DecodeTask {
            x: self.x,
            weight: self.weight,
            out: self.out,
            columns: self.columns,
        }
    }
}

/// Run up to [`MAX_DECODE_TASKS`] single-row Q8_0 matvecs as one parallel
/// region.
///
/// Fusing independent projections (Q/K/V, gate/up) into one region pays the
/// scheduling cost once. Large regions run on the spin-waiting decode team
/// with fine-grained chunks; small ones stay serial. Each output row is
/// computed by the same serial kernel with the same arithmetic in every
/// case, so results are bit-identical to [`matmul_q8_0_decode`] on each task.
pub(crate) fn matmul_q8_0_decode_tasks(tasks: &mut [Q8DecodeTask<'_, '_>]) {
    assert!(tasks.len() <= MAX_DECODE_TASKS, "too many decode tasks");
    for task in tasks.iter() {
        let blocks_per_row = task.weight.in_features() / Q8_0_BLOCK_SIZE;
        assert!(task.columns >= 1, "q8_0 decode task needs a column");
        assert_eq!(
            task.x.len(),
            task.columns * blocks_per_row * Q8_0_TYPE_SIZE,
            "q8_0 decode input length mismatch"
        );
        assert_eq!(
            task.out.len(),
            task.columns * task.weight.out_features(),
            "q8_0 decode output length mismatch"
        );
    }
    let parallel = tasks.iter().any(|task| {
        should_parallel_q8_decode(
            task.weight.out_features(),
            task.weight.in_features().saturating_mul(task.columns),
        )
    });
    if !parallel && tasks.iter().all(|task| task.columns == 1) {
        for task in tasks.iter_mut() {
            task.weight.compute_rows(task.x, 0, task.out);
        }
        return;
    }
    let mut region = DecodeRegion {
        tasks: [None, None, None, None],
        chunks: 0,
    };
    for (slot, task) in region.tasks.iter_mut().zip(tasks.iter_mut()) {
        let chunk_rows = decode_chunk_rows(task.weight);
        let rows = task.weight.out_features();
        *slot = Some(DecodeRegionTask {
            x: task.x,
            weight: task.weight,
            out: task.out.as_mut_ptr(),
            columns: task.columns,
            rows,
            chunk_rows,
            first_chunk: region.chunks,
        });
        region.chunks += rows.div_ceil(chunk_rows);
    }
    let region = &region;
    if !parallel {
        for chunk in 0..region.chunks {
            region.run_chunk(chunk);
        }
        return;
    }
    if !crate::decode_pool::run(region.chunks, &|chunk| region.run_chunk(chunk)) {
        (0..region.chunks)
            .into_par_iter()
            .for_each(|chunk| region.run_chunk(chunk));
    }
}

pub(super) fn matmul_q8_0_decode_dispatch_chunk(
    x: &[u8],
    data: &[u8],
    blocks_per_row: usize,
    out: &mut [f32],
) {
    // The x86 kernels read `x` and `data` through raw pointers.
    let row_bytes = blocks_per_row * Q8_0_TYPE_SIZE;
    assert!(x.len() >= row_bytes, "q8_0 decode input too short");
    assert!(
        data.len() >= out.len() * row_bytes,
        "q8_0 decode weight too short"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: features checked; the asserts above bound `x` and
            // `data`.
            unsafe {
                return x86_64::matmul_q8_0_decode_avx512_vnni(
                    x,
                    data,
                    out.len(),
                    blocks_per_row,
                    out,
                );
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            // SAFETY: features checked; the asserts above bound `x` and
            // `data`.
            unsafe {
                return x86_64::matmul_q8_0_decode_avx2(x, data, out.len(), blocks_per_row, out);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("dotprod") && is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime detection guarantees the dot-product extension.
        return unsafe { aarch64::matmul_q8_0_decode_dotprod(x, data, blocks_per_row, out) };
    }
    matmul_q8_0_decode_scalar(x, data, out.len(), blocks_per_row, out);
}

// -- interleaved Q8_0 dispatch -----------------------------------------------

use crate::quant::QuantizedWeightInterleaved;
use crate::quant::{QuantizedWeightVnni, VNNI_OUT_TILE};

/// Matrix-vector multiply using the 16-output packed Q8_0 layout.
pub(crate) fn matmul_q8_0_decode_packed16_parallel(
    x: &[u8],
    weight: &QuantizedWeightVnni,
    out: &mut [f32],
) {
    assert!(packed_q8_0_vnni_supported());
    matmul_q8_0_decode_tasks(&mut [Q8DecodeTask {
        x,
        weight: Q8DecodeWeight::Packed16(weight),
        out,
        columns: 1,
    }]);
}

/// Matrix multiply using interleaved Q8_0 weight layout.
/// Dispatches to the fastest available kernel.
/// `global_row_start` is the absolute row index within the full weight matrix
/// that `out[0]` corresponds to. Pass 0 when processing all rows.
#[inline]
pub(crate) fn matmul_q8_0_decode_interleaved(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    out: &mut [f32],
    global_row_start: usize,
) {
    let out_rows = out.len();
    if out_rows == 0 {
        return;
    }
    let input_len = w
        .blocks_per_row
        .checked_mul(Q8_0_TYPE_SIZE)
        .expect("interleaved q8 input length overflow");
    assert_eq!(x.len(), input_len, "interleaved q8 input length mismatch");
    let row_end = global_row_start
        .checked_add(out_rows)
        .expect("interleaved q8 output range overflow");
    assert!(
        row_end <= w.out_features(),
        "interleaved q8 output range {global_row_start}..{row_end} exceeds {} rows",
        w.out_features()
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: features checked; the input length and row range are
            // asserted above.
            unsafe {
                return x86_64::matmul_q8_0_decode_interleaved_avx512_vnni(
                    x,
                    w.quants(),
                    w.scales(),
                    out_rows,
                    w.blocks_per_row,
                    out,
                    global_row_start,
                );
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if interleaved_q8_0_supported() {
        // SAFETY: range checks above and feature gate validate this call.
        return unsafe { aarch64::matmul_q8_0_interleaved_dotprod(x, w, out, global_row_start) };
    }
    // Fallback: dequantize to f32 and use standard matmul (unlikely path)
    for (row, out_val) in out.iter_mut().enumerate() {
        let global_row = global_row_start + row;
        let stripe = global_row / crate::quant::INTERLEAVE;
        let lane = global_row % crate::quant::INTERLEAVE;
        let q_stripe =
            &w.quants()[stripe * w.blocks_per_row * crate::quant::INTERLEAVE * Q8_0_BLOCK_SIZE..];
        let s_stripe = &w.scales()[stripe * w.blocks_per_row * crate::quant::INTERLEAVE * 2..];
        let mut sum = 0.0f32;
        for b in 0..w.blocks_per_row {
            let q_off = b * crate::quant::INTERLEAVE * Q8_0_BLOCK_SIZE + lane * Q8_0_BLOCK_SIZE;
            let s_off = b * crate::quant::INTERLEAVE * 2 + lane * 2;
            let ws =
                f16::from_bits(u16::from_le_bytes([s_stripe[s_off], s_stripe[s_off + 1]])).to_f32();
            let xs = f16::from_bits(u16::from_le_bytes([
                x[b * Q8_0_TYPE_SIZE],
                x[b * Q8_0_TYPE_SIZE + 1],
            ]))
            .to_f32();
            for j in 0..Q8_0_BLOCK_SIZE {
                sum += (x[b * Q8_0_TYPE_SIZE + 2 + j] as i8) as f32
                    * (q_stripe[q_off + j] as i8) as f32
                    * ws
                    * xs;
            }
        }
        *out_val = sum;
    }
}

/// Parallel version: splits output rows across the decode team.
pub(crate) fn matmul_q8_0_decode_interleaved_parallel(
    x: &[u8],
    w: &QuantizedWeightInterleaved,
    out: &mut [f32],
) {
    matmul_q8_0_decode_tasks(&mut [Q8DecodeTask {
        x,
        weight: Q8DecodeWeight::Interleaved(w),
        out,
        columns: 1,
    }]);
}

// -- scalar fallback --------------------------------------------------------

pub(super) fn matmul_q8_0_decode_scalar(
    x: &[u8],
    data: &[u8],
    _out_features: usize,
    blocks_per_row: usize,
    out: &mut [f32],
) {
    for (row, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let row_start = row * blocks_per_row;
        for b in 0..blocks_per_row {
            let byte_offset = (row_start + b) * Q8_0_TYPE_SIZE;
            let x_offset = b * Q8_0_TYPE_SIZE;
            let weight_scale = f16::from_bits(u16::from_le_bytes(
                data[byte_offset..byte_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let input_scale = f16::from_bits(u16::from_le_bytes(
                x[x_offset..x_offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            let mut block_sum = 0i32;
            for j in 0..Q8_0_BLOCK_SIZE {
                let weight = data[byte_offset + 2 + j] as i8 as i32;
                let input = x[x_offset + 2 + j] as i8 as i32;
                block_sum += weight * input;
            }
            sum += block_sum as f32 * weight_scale * input_scale;
        }
        *out_val = sum;
    }
}

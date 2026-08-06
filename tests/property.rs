//! Property-based tests (proptest) added in the Luminal-review cleanup.
//!
//! Luminal lesson #10/#47: shape/op logic and layout code deserve property
//! coverage with small ranges and `prop_assume!` to exclude degenerate
//! inputs, all seeded (proptest prints its seed on failure). Targets:
//!
//! - tensor shape ops (reshape/transpose round-trips, add_broadcast and
//!   softmax against hand-rolled references),
//! - the v0.4 decode arena (disjoint, aligned, non-aliasing regions; error
//!   contract on bad requests),
//! - K-quant dequant kernels (finite, deterministic, zero-block identity).
//!
//! Run: `cargo test --release --test property` (release avoids debug-assert
//! overhead in the SIMD paths used by the kernels).

use ember::plan::{DecodeArena, ScratchPlan, ScratchRegion};
use ember::quant_k::{
    dequant_q4_k, dequant_q6_k, k_block_bytes, DTYPE_Q4_K, DTYPE_Q6_K, Q4_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, QK_K,
};
use ember::tensor::CpuTensor;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// strategies
// ---------------------------------------------------------------------------

/// A 2-D shape `[rows, cols]` with row-major data of exactly `rows*cols`
/// finite values (no NaN/Inf inputs).
fn tensor_2d() -> impl Strategy<Value = (Vec<usize>, Vec<f32>)> {
    (1usize..=6, 1usize..=6).prop_flat_map(|(rows, cols)| {
        let n = rows * cols;
        (
            Just(vec![rows, cols]),
            prop::collection::vec(-10.0f32..10.0, n),
        )
    })
}

/// An N-D shape (1-3 dims, each 1..=4, product <= 64) with matching data.
fn tensor_nd() -> impl Strategy<Value = (Vec<usize>, Vec<f32>)> {
    prop::collection::vec(1usize..=4, 1..=3).prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        (Just(shape), prop::collection::vec(-10.0f32..10.0, n))
    })
}

// ---------------------------------------------------------------------------
// tensor shape ops
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// reshape to any compatible shape and back preserves the data exactly.
    #[test]
    fn reshape_roundtrip_preserves_data((shape, data) in tensor_nd()) {
        let n: usize = shape.iter().product();
        let t = CpuTensor::from_data(shape.clone(), data.clone());
        let flat: Vec<usize> = vec![1, n];
        let r = t.reshape(&flat).reshape(&shape);
        prop_assert_eq!(r.shape(), &shape[..]);
        prop_assert_eq!(r.data(), &data[..]);
    }

    /// transpose is an involution on 2-D tensors.
    #[test]
    fn transpose_double_is_identity((shape, data) in tensor_2d()) {
        let t = CpuTensor::from_data(shape.clone(), data);
        let t2 = t.transpose().transpose();
        prop_assert_eq!(t2.shape(), &shape[..]);
        prop_assert_eq!(t2.data(), t.data());
    }

    /// add_broadcast matches an explicit per-row reference loop.
    #[test]
    fn add_broadcast_matches_reference(
        (shape, data) in tensor_2d(),
        bias in prop::collection::vec(-5.0f32..5.0, 1..=6),
    ) {
        let (rows, cols) = (shape[0], shape[1]);
        prop_assume!(bias.len() == cols); // only pairs where the bias width matches
        let a = CpuTensor::from_data(shape, data.clone());
        let b = CpuTensor::from_data(vec![cols], bias.clone());
        let out = a.add_broadcast(&b);
        for r in 0..rows {
            for c in 0..cols {
                let expected = data[r * cols + c] + bias[c];
                prop_assert!(
                    (out.data()[r * cols + c] - expected).abs() < 1e-4,
                    "row {} col {}: got {} expected {}",
                    r,
                    c,
                    out.data()[r * cols + c],
                    expected
                );
            }
        }
    }

    /// softmax rows are a probability distribution (sum to 1) and stay finite.
    #[test]
    fn softmax_rows_sum_to_one((shape, data) in tensor_2d()) {
        let (rows, cols) = (shape[0], shape[1]);
        let t = CpuTensor::from_data(shape, data);
        let s = t.softmax();
        for r in 0..rows {
            let row_sum: f32 = s.data()[r * cols..(r + 1) * cols].iter().sum();
            prop_assert!((row_sum - 1.0).abs() < 1e-3, "row {} sum {}", r, row_sum);
            for c in 0..cols {
                prop_assert!(s.data()[r * cols + c].is_finite());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v0.4 decode arena
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Disjoint, aligned, correctly-sized regions; no cross-region aliasing
    /// (sentinel writes to one region leave the others untouched).
    #[test]
    fn arena_regions_are_disjoint_aligned_and_isolated(
        alignment in prop::sample::select(vec![4usize, 64]),
        sizes in prop::collection::vec((1usize..=8).prop_map(|n| n * 64), 2..=8),
    ) {
        let mut regions: Vec<ScratchRegion> = Vec::new();
        let mut cursor = 0usize;
        for (i, size) in sizes.iter().enumerate() {
            regions.push(ScratchRegion {
                name: format!("r{i}"),
                offset: cursor,
                size: *size,
                alignment,
                first_op: i,
                last_op: i + 1,
                shared_with: None,
            });
            cursor += size;
        }
        let total_bytes = cursor;
        let plan = ScratchPlan {
            total_bytes,
            alignment,
            seq_capacity: 1,
            regions: regions.clone(),
            tensor_regions: Default::default(),
        };

        let mut arena = DecodeArena::new(&plan);

        // Lengths match sizes; pointers are alignment-aligned.
        for (i, region) in regions.iter().enumerate() {
            let s = arena.region_f32(i).unwrap();
            prop_assert_eq!(s.len(), region.size / 4);
            prop_assert_eq!(
                (s.as_ptr() as usize) % alignment,
                0,
                "region {} not {}-aligned",
                i,
                alignment
            );
        }

        // Isolation: write a sentinel pattern into each region, then confirm
        // every region reads back its own pattern (i.e. regions never alias).
        for i in 0..sizes.len() {
            let s = arena.region_f32(i).unwrap();
            for (k, v) in s.iter_mut().enumerate() {
                *v = i as f32 * 1000.0 + k as f32;
            }
        }
        for i in 0..sizes.len() {
            let s = arena.region_f32(i).unwrap();
            for (k, v) in s.iter().enumerate() {
                prop_assert_eq!(
                    *v,
                    i as f32 * 1000.0 + k as f32,
                    "region {} element {} corrupted",
                    i,
                    k
                );
            }
        }
    }

    /// The error contract: out-of-range regions and unaligned sizes return
    /// Err (never panic), and requesting the same region twice is an
    /// aliasing error.
    #[test]
    fn arena_rejects_bad_requests(alignment in prop::sample::select(vec![4usize, 64])) {
        let plan = ScratchPlan {
            total_bytes: 128,
            alignment,
            seq_capacity: 1,
            regions: vec![
                ScratchRegion { name: "a".into(), offset: 0, size: 64, alignment, first_op: 0, last_op: 1, shared_with: None },
                ScratchRegion { name: "b".into(), offset: 64, size: 64, alignment, first_op: 0, last_op: 1, shared_with: None },
            ],
            tensor_regions: Default::default(),
        };
        let mut arena = DecodeArena::new(&plan);
        prop_assert!(arena.region_f32(2).is_err(), "out-of-range index must be an error");

        let bad = ScratchPlan {
            total_bytes: 128,
            alignment,
            seq_capacity: 1,
            regions: vec![ScratchRegion {
                name: "a".into(),
                offset: 0,
                size: 6, // not a multiple of 4: not f32-addressable
                alignment,
                first_op: 0,
                last_op: 1,
                shared_with: None,
            }],
            tensor_regions: Default::default(),
        };
        let mut arena = DecodeArena::new(&bad);
        prop_assert!(arena.region_f32(0).is_err(), "size not multiple of 4 must be an error");

        let mut arena = DecodeArena::new(&plan);
        prop_assert!(
            arena.regions_f32([0, 0]).is_err(),
            "requesting the same region twice must be an aliasing error"
        );
    }
}

// ---------------------------------------------------------------------------
// K-quant dequant kernels
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Q4_K: 256 finite values; deterministic; all-zero block -> all zeros.
    ///
    /// Contract note: the per-block scale (`d`) and offset (`min`) are stored
    /// as f16 in the block header. A *random* header can be an f16 NaN/Inf
    /// pattern, which propagates through the dequant math exactly as it does
    /// in llama.cpp — real GGUF files always carry finite headers (validated
    /// at load). We therefore `prop_assume!` finite headers, the non-degenerate
    /// input class the kernel actually promises to handle.
    #[test]
    fn q4k_dequant_is_finite_deterministic_and_zero_identity(
        bytes in prop::collection::vec(any::<u8>(), Q4_K_BLOCK_BYTES),
    ) {
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
        let min = half::f16::from_bits(u16::from_le_bytes([bytes[2], bytes[3]])).to_f32();
        prop_assume!(d.is_finite() && min.is_finite());
        let mut out1 = vec![0.0f32; QK_K];
        let mut out2 = vec![0.0f32; QK_K];
        dequant_q4_k(&bytes, &mut out1);
        dequant_q4_k(&bytes, &mut out2);
        prop_assert_eq!(&out1, &out2, "dequant must be deterministic");
        for v in &out1 {
            prop_assert!(v.is_finite(), "Q4_K produced non-finite {v}");
        }
        let zero_block = vec![0u8; Q4_K_BLOCK_BYTES];
        let mut zeros = vec![f32::NAN; QK_K];
        dequant_q4_k(&zero_block, &mut zeros);
        for v in &zeros {
            prop_assert_eq!(*v, 0.0, "zero Q4_K block must dequantize to zero");
        }
    }

    /// Q6_K: same contract (scales are int8 here — sign-extension edge cases
    /// are covered by the random bytes).
    ///
    /// Contract note: same f16 `d` header assumption as Q4_K above, but note
    /// the Q6_K layout puts `d` at the END of the block (ql[128], qh[64],
    /// scales[16], d[2] — per llama.cpp `block_q6_K`); the per-16 int8
    /// scales are exercised exhaustively by the random bytes.
    #[test]
    fn q6k_dequant_is_finite_deterministic_and_zero_identity(
        bytes in prop::collection::vec(any::<u8>(), Q6_K_BLOCK_BYTES),
    ) {
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[208], bytes[209]])).to_f32();
        prop_assume!(d.is_finite());
        let mut out1 = vec![0.0f32; QK_K];
        let mut out2 = vec![0.0f32; QK_K];
        dequant_q6_k(&bytes, &mut out1);
        dequant_q6_k(&bytes, &mut out2);
        prop_assert_eq!(&out1, &out2, "dequant must be deterministic");
        for v in &out1 {
            prop_assert!(v.is_finite(), "Q6_K produced non-finite {v}");
        }
        let zero_block = vec![0u8; Q6_K_BLOCK_BYTES];
        let mut zeros = vec![f32::NAN; QK_K];
        dequant_q6_k(&zero_block, &mut zeros);
        for v in &zeros {
            prop_assert_eq!(*v, 0.0, "zero Q6_K block must dequantize to zero");
        }
    }
}

// ---------------------------------------------------------------------------
// robustness: untrusted-input boundaries must never panic (fuzz-style)
// ---------------------------------------------------------------------------
//
// The GGUF loader and the v0.5 spec parser are the two external-input
// boundaries of the crate (Luminal lesson: `Result` at external input, and
// fuzz it). These proptests replace a cargo-fuzz setup for CI: arbitrary
// bytes/strings must produce `Ok` or `Err`, never a panic. The loader is
// OOM-safe by construction: tensor/metadata counts are validated against the
// input length before any allocation (`loader.rs` header parsing uses
// `try_reserve`).

use std::io::Cursor;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Arbitrary bytes fed to the GGUF loader: Ok or Err, never a panic.
    #[test]
    fn gguf_loader_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        let mut cursor = Cursor::new(bytes);
        let _ = ember::loader::load_gguf_from_reader(&mut cursor);
    }

    /// Arbitrary TOML-ish text fed to the v0.5 spec parser: Ok or Err, never
    /// a panic; and any spec that *does* parse must also resolve or fail
    /// cleanly (resolve() is pure — no filesystem access).
    #[test]
    fn v05_spec_parser_never_panics(text in prop::collection::vec(any::<char>(), 0..2048)) {
        let text: String = text.into_iter().collect();
        if let Ok(spec) = ember::v05::spec::RawExperimentSpec::from_toml_str(&text) {
            let _ = spec.resolve();
        }
    }

    /// Arbitrary bytes written as a .npy file: read_npy_2d must return Ok or
    /// Err, never panic (the parser validates magic/version/header bounds
    /// before any slicing; this guards regressions in that ordering).
    #[test]
    fn npy_reader_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ember_npy_fuzz_{}_{}.npy",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, &bytes).expect("temp write");
        let _ = ember::npy::read_npy_2d(path.to_str().expect("temp path is utf8"));
        let _ = std::fs::remove_file(&path);
    }
}

/// The dtype-code helpers agree with the byte-size constants (load-time
/// contract: `k_block_bytes(code) == block_bytes(code)` for the K family).
#[test]
fn k_block_bytes_agree_with_constants() {
    assert_eq!(k_block_bytes(DTYPE_Q4_K), Some(Q4_K_BLOCK_BYTES));
    assert_eq!(k_block_bytes(DTYPE_Q6_K), Some(Q6_K_BLOCK_BYTES));
}

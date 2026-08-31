//! Golden tests for audio preprocessing against the HuggingFace Whisper
//! feature extractor (CPU path), captured by
//! `tests/fixtures/audio/*.npy` (see the generator in git history or
//! regenerate with transformers' `WhisperFeatureExtractor`).

use ember::multimodal::audio::{
    decode_wav, log_mel_spectrogram, resample, to_mono_16k, AudioInput,
};

fn load_npy_f32(path: &std::path::Path) -> (Vec<usize>, Vec<f32>) {
    // minimal .npy v1 reader for float32 arrays
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[0..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = String::from_utf8_lossy(&bytes[10..10 + header_len]).to_string();
    // parse shape from the python-literal-ish header
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    // fortran_order False, dtype '<f4'
    assert!(header.contains("f4"), "expected f32 npy");
    let data_start = 10 + header_len;
    let mut data = vec![0.0f32; (bytes.len() - data_start) / 4];
    for (i, v) in data.iter_mut().enumerate() {
        let o = data_start + i * 4;
        *v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    }
    (shape, data)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn log_mel_matches_whisper_reference_on_golden_signals() {
    for name in ["chirp", "silence", "noise", "tone_mix"] {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/audio/{name}_mel.npy"));
        let (shape, reference) = load_npy_f32(&fixture);
        assert_eq!(shape.len(), 2, "mel fixture must be [mels, frames]");

        let samples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/audio/{name}_samples.npy"));
        let (_, samples) = load_npy_f32(&samples_path);

        let got = log_mel_spectrogram(&samples).unwrap();
        assert_eq!(
            got.shape(),
            &shape,
            "{name}: mel shape mismatch vs reference"
        );
        let d = max_abs_diff(got.data(), &reference);
        // The fixtures were captured through HF's default CPU path, which
        // runs its torch STFT in float32; ember computes the same pipeline
        // in f64 (matching the numpy reference implementation to <1e-6).
        // Residual 4e-5 on [~0, 1]-normalized outputs is that f32 STFT
        // rounding, not a pipeline difference.
        assert!(d < 2e-4, "{name}: mel diverges from reference: max_abs={d}");
    }
}

#[test]
fn wav_decode_roundtrip_matches_saved_samples() {
    for name in ["chirp", "silence", "noise", "tone_mix"] {
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/audio/{name}.wav"));
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        // the fixtures were written as int16 PCM; compare against the saved
        // f32 originals with int16 quantization slack (~1/32767 plus wav
        // clipping of anything outside [-1, 1])
        let samples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("fixtures/audio/{name}_samples.npy"));
        let samples_path = if samples_path.exists() {
            samples_path
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/fixtures/audio/{name}_samples.npy"))
        };
        let (_, original) = load_npy_f32(&samples_path);
        assert_eq!(decoded.samples.len(), original.len());
        let d = max_abs_diff(&decoded.samples, &original);
        assert!(d < 1.0 / 32767.0 + 1e-3, "{name}: wav decode drift {d}");
    }
}

#[test]
fn resample_identity_and_length() {
    let x: Vec<f32> = (0..16_000).map(|i| ((i as f32) * 0.01).sin()).collect();
    // identity at equal rates
    assert_eq!(resample(&x, 16_000, 16_000).unwrap(), x);
    // 2x upsample doubles length; energy roughly preserved
    let up = resample(&x, 8_000, 16_000).unwrap();
    assert!((up.len() as i64 - 32_000).abs() <= 1);
    let rms_in = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let rms_out = (up.iter().map(|v| v * v).sum::<f32>() / up.len() as f32).sqrt();
    assert!(
        (rms_in - rms_out).abs() / rms_in < 0.05,
        "rms drift after upsampling: {rms_in} vs {rms_out}"
    );
}

#[test]
fn to_mono_16k_accepts_all_input_forms() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio");
    let via_file = to_mono_16k(&AudioInput::File(dir.join("chirp.wav"))).unwrap();
    let bytes = std::fs::read(dir.join("chirp.wav")).unwrap();
    let via_bytes = to_mono_16k(&AudioInput::Bytes(bytes)).unwrap();
    assert_eq!(via_file.samples, via_bytes.samples);

    let raw: Vec<f32> = (0..8000).map(|i| ((i as f32) * 0.05).cos()).collect();
    let via_samples = to_mono_16k(&AudioInput::Samples {
        data: raw.clone(),
        sample_rate: 8_000,
    })
    .unwrap();
    assert_eq!(via_samples.sample_rate, 16_000);
    assert!((via_samples.samples.len() as i64 - 16_000).abs() <= 1);
}

#[test]
#[ignore]
fn debug_print_chirp_mel() {
    let samples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/audio/chirp_samples.npy");
    let (_, samples) = load_npy_f32(&samples_path);
    let got = log_mel_spectrogram(&samples).unwrap();
    println!("shape {:?}", got.shape());
    println!("row0[:8] {:?}", &got.data()[..8]);
    println!(
        "row64[:4] {:?}",
        &got.data()[64 * got.shape()[1]..64 * got.shape()[1] + 4]
    );
}

// ---------------------------------------------------------------------------
// long-form chunking layout (Track D2)
// ---------------------------------------------------------------------------

#[test]
fn long_form_windows_boundary_math() {
    use ember::multimodal::audio::{long_form_windows, MAX_FRAMES};

    // below/at the context: single window
    assert_eq!(long_form_windows(1, MAX_FRAMES), vec![(0, 1)]);
    assert_eq!(long_form_windows(2999, MAX_FRAMES), vec![(0, 2999)]);
    assert_eq!(long_form_windows(3000, MAX_FRAMES), vec![(0, 3000)]);
    // one frame over: continuation window with a single valid frame
    assert_eq!(
        long_form_windows(3001, MAX_FRAMES),
        vec![(0, 3000), (3000, 1)]
    );
    // mid-length
    assert_eq!(
        long_form_windows(4500, MAX_FRAMES),
        vec![(0, 3000), (3000, 1500)]
    );
    // exact multiple: no empty trailing window
    assert_eq!(
        long_form_windows(6000, MAX_FRAMES),
        vec![(0, 3000), (3000, 3000)]
    );
    // generic invariants
    for total in [29_00, 30_00, 30_01, 45_12, 60_00, 61_237] {
        let wins = long_form_windows(total, MAX_FRAMES);
        assert_eq!(wins.iter().map(|(_, v)| v).sum::<usize>(), total);
        assert!(wins[0].1 == MAX_FRAMES.min(total));
        for (i, &(s, v)) in wins.iter().enumerate() {
            if i > 0 {
                let (ps, pv) = wins[i - 1];
                assert_eq!(s, ps + pv, "windows must be contiguous");
            }
            assert!(v <= MAX_FRAMES);
        }
    }
}

#[test]
fn log_mel_full_matches_guarded_variant_for_short_input() {
    use ember::multimodal::audio::{log_mel_spectrogram, log_mel_spectrogram_full};
    let n = 16_000; // 1 s
    let x: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
        .collect();
    let a = log_mel_spectrogram(&x).unwrap();
    let b = log_mel_spectrogram_full(&x).unwrap();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(
        a.data(),
        b.data(),
        "full variant must be identical below the guard"
    );
}

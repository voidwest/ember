//! Audio preprocessing, isolated from the model runtime (the audio
//! counterpart of [`crate::multimodal::image`]).
//!
//! Pipeline: decode WAV (PCM u8/i16/i24/i32, f32/f64) → channel
//! normalization (mean of channels) → optional resampling to 16 kHz →
//! log-mel spectrogram. The mel path reproduces HuggingFace's CPU feature
//! extractor for Whisper-family models exactly enough to validate at
//! ~1e-6: periodic Hann window (`np.hanning(401)[:-1]`), center padding by
//! `n_fft/2` in reflect mode, f64 STFT (DFT via precomputed tables), power
//! spectrum, Slaney-scale/norm mel filterbank (201 bins -> 128 mels,
//! 0..8000 Hz @ 16 kHz), `log10`, drop of the last frame, global
//! `max - 8` floor and `(x + 4) / 4` normalization.
//!
//! The resampler is a windowed-sinc design chosen by ember; it is *not*
//! validated against a specific external reference (references disagree
//! with each other). Numerical validation feeds 16 kHz sources so the
//! resampler stays off the reference path.

use crate::tensor::CpuTensor;
use anyhow::{anyhow, ensure, Result};
use std::path::Path;

/// Whisper-feature-extractor constants shared by every supported model
/// (16 kHz mono input).
pub const TARGET_SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
/// One-sided spectrum bins for [`N_FFT`] real samples.
pub const FREQ_BINS: usize = N_FFT / 2 + 1;
/// Maximum encoder context: whisper encoders take at most 3000 frames
/// (30 s); longer inputs are chunked by [`long_form_windows`].
pub const MAX_FRAMES: usize = 3000;

/// The periodic Hann window (`np.hanning(401)[:-1]`) shared by the static
/// and streaming frontends.
pub(crate) fn window_fn() -> Vec<f64> {
    periodic_hann(N_FFT)
}

/// Long-form window layout for `total` mel frames: `(start, valid_len)`
/// per 30 s window. Continuation windows (`start > 0`) shorter than
/// `context` get zero-padded by the caller before encoding; this function
/// reports their *valid* length only.
pub fn long_form_windows(total: usize, context: usize) -> Vec<(usize, usize)> {
    if total <= context {
        return vec![(0, total)];
    }
    let n = total.div_ceil(context);
    (0..n)
        .map(|c| {
            let start = c * context;
            let end = (start + context).min(total);
            (start, end - start)
        })
        .collect()
}

/// Decoded audio samples plus their source sample rate. Samples are mono
/// f32 in nominal [-1, 1].
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// A raw audio input for a multimodal request: a file path or in-memory
/// samples (already decoded, any rate). In-memory variants keep agents and
/// tools from having to touch the filesystem.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioInput {
    File(std::path::PathBuf),
    /// In-memory WAV bytes (a complete RIFF file).
    Bytes(Vec<u8>),
    /// Already-decoded samples at an arbitrary rate; normalized/resampled
    /// as needed.
    Samples {
        data: Vec<f32>,
        sample_rate: u32,
    },
}

// ---------------------------------------------------------------------------
// WAV decoding
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(
            self.pos + n <= self.data.len(),
            "WAV truncated: wanted {n} bytes at offset {}",
            self.pos
        );
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.read_bytes(n)?;
        Ok(())
    }
    fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }
}

/// Decode a WAV file (RIFF/WAVE): PCM 8/16/24/32-bit and IEEE float 32/64,
/// including WAVE_FORMAT_EXTENSIBLE wrappers. Multi-channel audio is mixed
/// down to mono by the mean over channels.
///
/// Integer PCM is scaled to nominal [-1, 1] by dividing through the format
/// midpoint (matching librosa/soundfile conventions: symmetric around zero,
/// so positive full scale is +1.0 and negative full scale is exactly -1.0).
/// Maximum WAV file size accepted by the path-backed `decode_wav` reader.
///
/// Real recordings are far smaller than this bound. Keeping a finite limit
/// prevents an attacker-controlled path from forcing an unbounded read and
/// decode allocation before audio processing starts. In-memory
/// [`AudioInput::Bytes`] inputs are already bounded by the caller.
pub const MAX_WAV_FILE_BYTES: u64 = 256 * 1024 * 1024;

pub fn decode_wav(path: &Path) -> Result<DecodedAudio> {
    use std::io::Read;

    // One-descriptor bounded read with path identity checks (same contract
    // as the tokenizer/NPY readers): reject symlinks up front, then verify
    // the opened descriptor and the path still refer to the same regular
    // file after reading.
    let path_metadata = std::fs::symlink_metadata(path)
        .map_err(|e| anyhow!("failed to stat wav {}: {e}", path.display()))?;
    anyhow::ensure!(
        path_metadata.file_type().is_file(),
        "wav {} is not a regular file",
        path.display()
    );
    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow!("failed to read wav {}: {e}", path.display()))?;
    let initial = file
        .metadata()
        .map_err(|e| anyhow!("failed to stat wav {}: {e}", path.display()))?;
    anyhow::ensure!(
        initial.file_type().is_file()
            && initial.len() == path_metadata.len()
            && initial.modified().ok() == path_metadata.modified().ok(),
        "wav file changed while opening {}",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            initial.dev() == path_metadata.dev() && initial.ino() == path_metadata.ino(),
            "wav file changed while opening {}",
            path.display()
        );
    }
    let length = initial.len();
    anyhow::ensure!(
        length <= MAX_WAV_FILE_BYTES,
        "wav file {} is {length} bytes, exceeding the {MAX_WAV_FILE_BYTES} byte limit",
        path.display()
    );
    let capacity =
        usize::try_from(length).map_err(|_| anyhow!("wav file length exceeds address space"))?;
    let max_bytes = usize::try_from(MAX_WAV_FILE_BYTES)
        .map_err(|_| anyhow!("wav byte limit exceeds address space"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|e| anyhow!("failed to reserve wav buffer: {e}"))?;
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|e| anyhow!("failed to read wav {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        anyhow::ensure!(
            bytes.len() <= max_bytes.saturating_sub(read),
            "wav file {} grew beyond the {MAX_WAV_FILE_BYTES} byte limit while reading",
            path.display()
        );
        bytes
            .try_reserve_exact(read)
            .map_err(|e| anyhow!("failed to grow wav buffer: {e}"))?;
        bytes.extend_from_slice(&chunk[..read]);
    }
    let final_metadata = file
        .metadata()
        .map_err(|e| anyhow!("failed to stat wav {} after reading: {e}", path.display()))?;
    let final_path_metadata = std::fs::symlink_metadata(path)
        .map_err(|e| anyhow!("failed to stat wav {} after reading: {e}", path.display()))?;
    anyhow::ensure!(
        final_metadata.len() == length
            && final_metadata.modified().ok() == initial.modified().ok()
            && final_path_metadata.file_type().is_file()
            && final_path_metadata.len() == initial.len()
            && final_path_metadata.modified().ok() == initial.modified().ok(),
        "wav file changed while reading {}",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            final_path_metadata.dev() == initial.dev()
                && final_path_metadata.ino() == initial.ino(),
            "wav file changed while reading {}",
            path.display()
        );
    }
    decode_wav_bytes(&bytes)
}

/// Decode WAV from memory ([`AudioInput::Bytes`]).
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut cur = Cursor::new(bytes);
    let riff = cur.read_bytes(12)?;
    ensure!(&riff[0..4] == b"RIFF", "not a RIFF file");
    ensure!(&riff[8..12] == b"WAVE", "not a WAVE file");

    // find fmt and data chunks
    let mut format_tag: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_range: Option<(usize, usize)> = None;
    while !cur.eof() {
        let chunk_id = cur.read_bytes(4)?;
        let chunk_size = cur.read_u32()? as usize;
        match chunk_id {
            b"fmt " => {
                format_tag = cur.read_u16()?;
                channels = cur.read_u16()?;
                sample_rate = cur.read_u32()?;
                let _byte_rate = cur.read_u32()?;
                let _block_align = cur.read_u16()?;
                bits_per_sample = cur.read_u16()?;
                if format_tag == 0xFFFE {
                    // WAVE_FORMAT_EXTENSIBLE: cbSize, valid bits, channel mask,
                    // then a 16-byte subformat GUID whose first 2 bytes carry
                    // the real format tag
                    let cb = cur.read_u16()? as usize;
                    ensure!(cb >= 22, "extensible fmt too small");
                    cur.skip(2)?; // valid bits
                    cur.skip(4)?; // channel mask
                    let guid = cur.read_bytes(14)?.to_vec();
                    cur.skip(cb - 20)?;
                    ensure!(
                        guid[..14]
                            == [
                                0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00,
                                0x38, 0x9B, 0x71
                            ],
                        "unsupported extensible subformat"
                    );
                    // KSDATAFORMAT_SUBTYPE_PCM or _IEEE_FLOAT both supported;
                    // distinguish by bits
                }
            }
            b"data" => {
                let start = cur.pos;
                cur.skip(chunk_size)?;
                data_range = Some((start, chunk_size));
                // chunks are word-aligned
                if chunk_size % 2 == 1 {
                    cur.skip(1).ok();
                }
            }
            _ => {
                // skip unknown chunks (LIST, fact, ...)
                cur.skip(chunk_size)?;
                if chunk_size % 2 == 1 {
                    cur.skip(1).ok();
                }
            }
        }
    }

    let (data_start, data_len) = data_range.ok_or_else(|| anyhow!("WAV has no data chunk"))?;
    let data = &bytes[data_start..data_start + data_len];
    ensure!(channels > 0, "WAV has zero channels");

    // resolve the effective format: raw tag or the extensible GUID mapping
    let effective_tag = if format_tag == 0xFFFE {
        // PCM if integer bit depth, float if 32/64-bit IEEE payload
        match bits_per_sample {
            32 | 64 => 3, // IEEE float
            _ => 1,       // PCM integer
        }
    } else {
        format_tag
    };

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    ensure!(
        bytes_per_sample > 0 && data.len().is_multiple_of(bytes_per_sample),
        "WAV data length {} not aligned to {}-bit samples",
        data.len(),
        bits_per_sample
    );
    let frames = data.len() / bytes_per_sample / channels as usize;

    let read_one = |frame: usize, ch: usize| -> f32 {
        let off = (frame * channels as usize + ch) * bytes_per_sample;
        let b = &data[off..off + bytes_per_sample];
        match (effective_tag, bits_per_sample) {
            (1, 8) => (b[0] as f32 - 128.0) / 128.0, // unsigned 8-bit
            (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            (1, 24) => {
                let v = ((b[2] as i32) << 24 | (b[1] as i32) << 16 | (b[0] as i32) << 8) >> 8;
                v as f32 / 8388608.0
            }
            (1, 32) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0,
            (3, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (3, 64) => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            _ => panic!("unsupported wav format tag {effective_tag} at {bits_per_sample} bits"),
        }
    };

    let mut samples = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..channels as usize {
            acc += read_one(f, c);
        }
        samples.push(acc / channels as f32);
    }
    Ok(DecodedAudio {
        samples,
        sample_rate,
    })
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample mono audio from `from_rate` to `to_rate` with a Hann-windowed
/// sinc kernel (half-width 16 output samples, quality comparable to common
/// high-quality settings). This is ember's own converter; it is validated
/// for bandwidth preservation and identity at equal rates rather than
/// bit-exactness against any external library.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = vec![0.0f32; out_len];
    // sinc cutoff relative to the source grid; when downsampling we widen
    // the kernel to avoid aliasing
    let cutoff = ratio.min(1.0);
    let half_width = 16usize;
    let time_step = 1.0 / ratio;
    for (o, slot) in out.iter_mut().enumerate() {
        let t_src = o as f64 * time_step;
        let i0 = t_src.floor() as isize;
        let frac = t_src - i0 as f64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for k in -(half_width as isize)..=(half_width as isize) {
            let idx = i0 + k;
            if idx < 0 || idx as usize >= samples.len() {
                continue;
            }
            let x = (frac - k as f64) / cutoff;
            // Hann-windowed sinc: the window peaks at 1.0 at the tap center
            let s = if x == 0.0 {
                1.0
            } else if x.abs() < half_width as f64 {
                let w = 0.5 * (1.0 + (std::f64::consts::PI * x / half_width as f64).cos());
                w * (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            } else {
                0.0
            };
            acc += samples[idx as usize] as f64 * s;
            wsum += s;
        }
        *slot = if wsum != 0.0 {
            (acc / wsum) as f32
        } else {
            0.0
        };
    }
    out
}

/// Normalize any [`AudioInput`] to mono f32 at 16 kHz: decode, mean-of-
/// channels, resample. This is the single entry point the model wrapper
/// uses for all sources.
pub fn to_mono_16k(input: &AudioInput) -> Result<DecodedAudio> {
    let decoded = match input {
        AudioInput::File(p) => decode_wav(p)?,
        AudioInput::Bytes(b) => decode_wav_bytes(b)?,
        AudioInput::Samples { data, sample_rate } => DecodedAudio {
            samples: data.clone(),
            sample_rate: *sample_rate,
        },
    };
    if decoded.sample_rate == TARGET_SAMPLE_RATE as u32 {
        return Ok(decoded);
    }
    Ok(DecodedAudio {
        sample_rate: TARGET_SAMPLE_RATE as u32,
        samples: resample(
            &decoded.samples,
            decoded.sample_rate,
            TARGET_SAMPLE_RATE as u32,
        ),
    })
}

// ---------------------------------------------------------------------------
// Log-mel spectrogram (Whisper-compatible)
// ---------------------------------------------------------------------------

fn periodic_hann(n: usize) -> Vec<f64> {
    // np.hanning(n+1)[:-1]: 0.5 - 0.5*cos(2*pi*k/(n+1-1)) = 2*pi*k/n
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
        .collect()
}

fn hz_to_mel_slaney(freqs: &[f64]) -> Vec<f64> {
    freqs
        .iter()
        .map(|&f| {
            if f >= 1000.0 {
                15.0 + (f / 1000.0).ln() * (27.0 / 6.4f64.ln())
            } else {
                3.0 * f / 200.0
            }
        })
        .collect()
}

fn mel_to_hz_slaney(mels: &[f64]) -> Vec<f64> {
    mels.iter()
        .map(|&m| {
            if m >= 15.0 {
                1000.0 * (6.4f64.ln() / 27.0 * (m - 15.0)).exp()
            } else {
                200.0 * m / 3.0
            }
        })
        .collect()
}

/// The Slaney-scale/norm mel filterbank used by Whisper:
/// `[num_freq_bins, num_mels]`, row-major (bin-major).
pub fn mel_filterbank(num_freq_bins: usize, num_mels: usize, sampling_rate: u32) -> Vec<f64> {
    assert_eq!(num_freq_bins, 1 + N_FFT / 2);
    let mel_min = hz_to_mel_slaney(&[0.0])[0];
    let mel_max = hz_to_mel_slaney(&[8000.0])[0];
    let mel_points: Vec<f64> = (0..num_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64)
        .collect();
    let filter_freqs = mel_to_hz_slaney(&mel_points);

    // frequencies of FFT bins: np.linspace(0, sr//2, num_freq_bins)
    let fft_freqs: Vec<f64> = (0..num_freq_bins)
        .map(|i| (sampling_rate / 2) as f64 * i as f64 / (num_freq_bins - 1) as f64)
        .collect();

    // triangular bank
    let mut filters = vec![0.0f64; num_freq_bins * num_mels];
    for j in 0..num_mels {
        let lower = filter_freqs[j];
        let center = filter_freqs[j + 1];
        let upper = filter_freqs[j + 2];
        for (i, &f) in fft_freqs.iter().enumerate().take(num_freq_bins) {
            let down = if center > lower {
                (f - lower) / (center - lower)
            } else {
                0.0
            };
            let up = if upper > center {
                (upper - f) / (upper - center)
            } else {
                0.0
            };
            filters[i * num_mels + j] = down.min(up).max(0.0);
        }
    }
    // Slaney area normalization: enorm = 2 / (filter_freqs[j+2] - filter_freqs[j])
    for j in 0..num_mels {
        let enorm = 2.0 / (filter_freqs[j + 2] - filter_freqs[j]);
        for i in 0..num_freq_bins {
            let idx = i * num_mels + j;
            filters[idx] *= enorm;
        }
    }
    filters
}

/// One-sided DFT of real frames via precomputed cos/sin tables (f64,
/// matching numpy's float64 internal precision). `frames` is row-major
/// `[num_frames, N_FFT]`; returns `[num_frames, N_FFT/2 + 1]` magnitudes².
fn dft_power(frames: &[f64]) -> Vec<f64> {
    let num_frames = frames.len() / N_FFT;
    let bins = N_FFT / 2 + 1;
    // cos/sin lookup tables per frequency bin: [bins][N_FFT]
    let mut cos_tab = vec![0.0f64; bins * N_FFT];
    let mut sin_tab = vec![0.0f64; bins * N_FFT];
    for k in 0..bins {
        let angle = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
        for n in 0..N_FFT {
            let a = angle * n as f64;
            cos_tab[k * N_FFT + n] = a.cos();
            sin_tab[k * N_FFT + n] = a.sin();
        }
    }
    let mut out = vec![0.0f64; num_frames * bins];
    for (row, frame) in frames.chunks_exact(N_FFT).enumerate() {
        for k in 0..bins {
            let ct = &cos_tab[k * N_FFT..(k + 1) * N_FFT];
            let st = &sin_tab[k * N_FFT..(k + 1) * N_FFT];
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (n, &x) in frame.iter().enumerate() {
                re += x * ct[n];
                im += x * st[n];
            }
            out[row * bins + k] = re * re + im * im;
        }
    }
    out
}

/// Compute the Whisper-style log-mel spectrogram of mono 16 kHz samples.
///
/// Returns `[N_MELS, T]` row-major where `T = ceil(len/160)` (the final
/// frame of the raw STFT is dropped after the mel projection, matching the
/// reference pipeline). Frames beyond [`MAX_FRAMES`] indicate audio longer
/// than 30 s; callers chunk before calling this function.
pub fn log_mel_spectrogram(samples: &[f32]) -> Result<CpuTensor> {
    let mel = log_mel_spectrogram_full(samples)?;
    ensure!(
        mel.shape()[1] <= MAX_FRAMES,
        "audio too long: {} frames exceeds the {}-frame (30 s) encoder context; use the chunked long-form path",
        mel.shape()[1],
        MAX_FRAMES
    );
    Ok(mel)
}

/// [`log_mel_spectrogram`] without the 30 s guard, for long-form chunking:
/// the whole signal is featurized with ONE global `max - 8` floor (exactly
/// like the reference processor, which computes mel over the full audio and
/// slices chunks afterwards), then the caller splits frames into
/// encoder-sized windows.
pub fn log_mel_spectrogram_full(samples: &[f32]) -> Result<CpuTensor> {
    ensure!(!samples.is_empty(), "log_mel_spectrogram: empty waveform");

    // 1. center-pad with reflect by n_fft/2 on both sides
    // (np.pad mode='reflect': mirror without repeating the edge sample)
    let pad = N_FFT / 2;
    let n = samples.len();
    ensure!(
        pad < n,
        "log_mel_spectrogram: waveform shorter than {} samples",
        pad + 1
    );
    let mut padded = vec![0.0f64; n + 2 * pad];
    for (g, p) in padded.iter_mut().enumerate() {
        *p = if g < pad {
            // left mirror: padded[i] = x[pad - i]
            samples[pad - g] as f64
        } else if g < pad + n {
            samples[g - pad] as f64
        } else {
            // right mirror (no edge repeat): padded[n+k] = x[n-2-k]
            let k = g - (pad + n);
            samples[n - 2 - k] as f64
        };
    }

    // 2. frame (400 samples, hop 160), window
    let num_frames_raw = 1 + (padded.len() - N_FFT) / HOP_LENGTH;
    let window = periodic_hann(N_FFT);
    let mut frames = vec![0.0f64; num_frames_raw * N_FFT];
    for t in 0..num_frames_raw {
        let src = &padded[t * HOP_LENGTH..t * HOP_LENGTH + N_FFT];
        let dst = &mut frames[t * N_FFT..(t + 1) * N_FFT];
        for (d, (&s, &w)) in dst.iter_mut().zip(src.iter().zip(window.iter())) {
            *d = s * w;
        }
    }

    // 3. power spectrum + mel projection (both f64, like numpy)
    let power = dft_power(&frames);
    let bins = N_FFT / 2 + 1;
    let fb = mel_filterbank(bins, N_MELS, TARGET_SAMPLE_RATE as u32);
    let mut mel = vec![0.0f64; num_frames_raw * N_MELS];
    for t in 0..num_frames_raw {
        let spec = &power[t * bins..(t + 1) * bins];
        for (j, slot) in mel[t * N_MELS..(t + 1) * N_MELS].iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &s) in spec.iter().enumerate() {
                acc += fb[k * N_MELS + j] * s;
            }
            // mel_floor applied during the reference's dot-product stage
            *slot = acc.max(1e-10);
        }
    }

    // 4. log10, drop last frame, global max-8 floor, (x+4)/4
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
    }
    let usable = num_frames_raw - 1;
    let max_log = mel[..usable * N_MELS]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let floor = max_log - 8.0;
    // pack as [N_MELS, usable] (row-major mel-major), transposing the
    // [time][mel] working buffer
    let mut data = vec![0.0f32; usable * N_MELS];
    for t in 0..usable {
        for j in 0..N_MELS {
            let v = ((mel[t * N_MELS + j].max(floor) + 4.0) / 4.0) as f32;
            data[j * usable + t] = v;
        }
    }

    Ok(CpuTensor::from_data(vec![N_MELS, usable], data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn decode_wav_rejects_symlinks_and_oversized_files() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ember-wav-path-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).expect("create wav fixture dir");
        let path = dir.join("clip.wav");
        std::fs::write(&path, b"not a wav").expect("write wav fixture");
        #[cfg(unix)]
        {
            let link = dir.join("clip-link.wav");
            std::os::unix::fs::symlink(&path, &link).expect("create wav symlink");
            let error = decode_wav(&link).expect_err("symlinked wav paths must be rejected");
            assert!(error.to_string().contains("regular file"), "{error}");
        }
        // A regular file passes the path checks and fails at parse time.
        let error = decode_wav(&path)
            .expect_err("a regular file must pass path checks and fail at parse time");
        assert!(!error.to_string().contains("regular file"), "{error}");

        // A sparse oversized file is rejected by the size cap before reading.
        let big = dir.join("big.wav");
        let file = std::fs::File::create(&big).expect("create sparse wav fixture");
        file.set_len(MAX_WAV_FILE_BYTES + 1)
            .expect("extend sparse wav fixture");
        drop(file);
        let error = decode_wav(&big).expect_err("an oversized wav must be rejected before reading");
        assert!(error.to_string().contains("byte limit"), "{error}");

        std::fs::remove_dir_all(&dir).expect("remove wav fixture dir");
    }
}

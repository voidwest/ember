//! Streaming audio input frontend.
//!
//! Honest streaming on top of the validated static preprocessing: PCM
//! arrives incrementally, the resampler and log-mel frontend keep state
//! across pushes, and finalized feature frames never change retroactively.
//!
//! Two precision contracts are pinned by unit tests:
//!
//! 1. **Resampler**: a stream fed arbitrary chunk partitions produces
//!    bit-identical output to the one-shot [`super::audio::resample`] —
//!    outputs are emitted early only when their full sinc-tap support is
//!    resident, and every tail output at [`AudioStream::finish`] is
//!    computed with exactly the one-shot's boundary arithmetic (tap skip +
//!    weight renormalization over in-range taps).
//! 2. **Log-mel**: streamed mel over randomly partitioned PCM is
//!    bit-identical to one-shot [`super::audio::log_mel_spectrogram_full`].
//!    A frame is finalized only once its support lies inside received
//!    samples *plus* ≥200 samples of margin: near-end frames read the
//!    right reflect mirror, which depends on the final length, so they
//!    stay pending until [`AudioStream::finish`].
//!
//! The global Whisper normalization (`max − 8` floor, `(x+4)/4`) spans all
//! usable frames by construction, so normalized features exist at finish;
//! [`AudioStream::provisional_mel`] exposes unstable partial features for
//! UIs, explicitly not for inference.
//!
//! The Whisper-family *encoder* has no recurrent state; what gets
//! re-encoded when is an above-front-end scheduling decision. This module
//! guarantees the frontend state machine never recomputes or mutates
//! history, and reports the offsets needed to schedule honestly.

use crate::multimodal::audio::{long_form_windows, MAX_FRAMES, TARGET_SAMPLE_RATE};
use crate::tensor::CpuTensor;
use anyhow::{ensure, Result};

const SINC_HALF_WIDTH: isize = 16;
/// End-margin (samples) required beyond a frame's support before it may be
/// finalized while streaming: the right reflect mirror reaches back to
/// `x[n - 201]`, so 200 spare samples guarantee immutability.
const MEL_FINALIZE_MARGIN: usize = 200;

// ---------------------------------------------------------------------------
// Incremental windowed-sinc resampler
// ---------------------------------------------------------------------------

/// Stateful half of [`super::audio::resample`]: identical per-output
/// arithmetic, executed as input arrives.
pub(crate) struct StreamingResampler {
    ratio: f64,
    time_step: f64,
    cutoff: f64,
    buf: Vec<f32>,
    /// Global source index of `buf[0]`.
    buf_start: usize,
    /// Next output index to emit.
    next_out: usize,
    total_in: usize,
}

impl StreamingResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        let ratio = f64::from(to_rate) / f64::from(from_rate);
        Self {
            ratio,
            time_step: 1.0 / ratio,
            cutoff: ratio.min(1.0),
            buf: Vec::new(),
            buf_start: 0,
            next_out: 0,
            total_in: 0,
        }
    }

    /// Push source samples; emit every output whose full tap window now
    /// lies inside the received prefix. Early outputs can never differ
    /// from one-shot: all taps exist, so no boundary branch runs.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(samples);
        self.total_in += samples.len();
        let mut out = Vec::new();
        loop {
            let t_src = self.next_out as f64 * self.time_step;
            let i0 = t_src.floor() as isize;
            if i0 + SINC_HALF_WIDTH >= self.total_in as isize {
                break;
            }
            out.push(self.output_at(t_src, i0));
            self.next_out += 1;
        }
        self.retire();
        out
    }

    /// Flush remaining outputs against the true final length, using the
    /// one-shot boundary behavior exactly.
    pub fn finish(mut self) -> Vec<f32> {
        let out_len = ((self.total_in as f64) * self.ratio).round() as usize;
        let total = self.total_in;
        let mut out = Vec::with_capacity(out_len.saturating_sub(self.next_out));
        while self.next_out < out_len {
            let t_src = self.next_out as f64 * self.time_step;
            let i0 = t_src.floor() as isize;
            out.push(self.output_at_bounded(t_src, i0, total));
            self.next_out += 1;
        }
        out
    }

    fn output_at(&self, t_src: f64, i0: isize) -> f32 {
        self.output_at_bounded(t_src, i0, self.total_in)
    }

    /// Arithmetic mirrors `audio::resample` exactly: same f64 ops, same k
    /// order, same skip/renormalize branches, taps ascending.
    fn output_at_bounded(&self, t_src: f64, i0: isize, len: usize) -> f32 {
        let frac = t_src - i0 as f64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for k in -SINC_HALF_WIDTH..=SINC_HALF_WIDTH {
            let idx = i0 + k;
            if idx < 0 || idx as usize >= len {
                continue;
            }
            let x = (frac - k as f64) / self.cutoff;
            let s = if x == 0.0 {
                1.0
            } else if x.abs() < SINC_HALF_WIDTH as f64 {
                let w = 0.5 * (1.0 + (std::f64::consts::PI * x / SINC_HALF_WIDTH as f64).cos());
                w * (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            } else {
                0.0
            };
            acc += f64::from(self.buf[idx as usize - self.buf_start]) * s;
            wsum += s;
        }
        if wsum != 0.0 {
            (acc / wsum) as f32
        } else {
            0.0
        }
    }

    /// Drop source samples no future output can touch: output `next_out`
    /// reads down to `floor(next_out * dt) - 16`.
    fn retire(&mut self) {
        let lowest_needed = ((self.next_out as f64 * self.time_step).floor() as isize
            - SINC_HALF_WIDTH)
            .max(0) as usize;
        if lowest_needed > self.buf_start {
            let drop = (lowest_needed - self.buf_start).min(self.buf.len());
            self.buf.drain(..drop);
            self.buf_start += drop;
        }
    }
}

// ---------------------------------------------------------------------------
// Incremental log-mel (Whisper-compatible)
// ---------------------------------------------------------------------------

/// Stateful half of [`super::audio::log_mel_spectrogram_full`] (16 kHz mono
/// domain). Raw pre-floor log10 mel columns become immutable as their input
/// windows leave the uncertainty zone; normalization happens once, at
/// finish, because the reference floor is global.
pub(crate) struct MelStream {
    /// Retained source samples (global index `tail_start .. total`).
    tail: Vec<f32>,
    tail_start: usize,
    total: usize,
    /// Raw log10 mel values, time-major: `columns[t * N_MELS + j]`.
    columns: Vec<f64>,
    running_max_log: f64,
    filterbank: Vec<f64>,
    window: Vec<f64>,
    scratch_frame: Vec<f64>,
}

impl MelStream {
    pub fn new() -> Self {
        Self {
            tail: Vec::new(),
            tail_start: 0,
            total: 0,
            columns: Vec::new(),
            running_max_log: f64::NEG_INFINITY,
            filterbank: crate::multimodal::audio::mel_filterbank(
                crate::multimodal::audio::FREQ_BINS,
                crate::multimodal::audio::N_MELS,
                TARGET_SAMPLE_RATE as u32,
            ),
            window: crate::multimodal::audio::window_fn(),
            scratch_frame: vec![0.0; crate::multimodal::audio::N_FFT],
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.tail.extend_from_slice(samples);
        self.total += samples.len();
        let target = self.ready_frames();
        while self.finalized_frames() < target {
            self.append_immutable_column(self.finalized_frames());
        }
        self.retire();
    }

    /// Frame `t` reads source `[t*160 - 200, t*160 + 199]`; it is
    /// immutable iff `t*160 + 400 + MARGIN <= total`. The margin keeps any
    /// finalized frame strictly below the last raw frame index
    /// (`floor(total/160)`), which the reference pipeline drops.
    fn ready_frames(&self) -> usize {
        let horizon = crate::multimodal::audio::N_FFT + MEL_FINALIZE_MARGIN;
        if self.total < horizon {
            return 0;
        }
        (self.total - horizon) / 160 + 1
    }

    fn append_immutable_column(&mut self, t: usize) {
        debug_assert!(t < self.ready_frames() + self.finalized_frames());
        self.compute_column(t, false);
    }

    /// Shared column computation. `allow_right_mirror` selects streaming
    /// (never reads the mirror; caller guarantees reachability) versus
    /// finish (mirrors active, matching one-shot formulas).
    fn compute_column(&mut self, t: usize, allow_right_mirror: bool) {
        let n_fft = crate::multimodal::audio::N_FFT;
        // element i of frame t sits at padded index t*160 + i; in source
        // coordinates s = t*160 + i - 200:
        //   s < 0          -> left reflect  x[-s]
        //   0 <= s < n     -> real sample   x[s]
        //   s >= n         -> right reflect x[2n - s - 2]
        let start = t as isize * 160 - 200;
        for i in 0..n_fft {
            let s = start + i as isize;
            let v = if s < 0 {
                self.source_at((-s) as usize)
            } else if (s as usize) < self.total {
                self.source_at(s as usize)
            } else if allow_right_mirror {
                let k = (s as usize) - self.total;
                self.source_at(self.total - 2 - k)
            } else {
                unreachable!("immutable frame reached into the right mirror");
            };
            self.scratch_frame[i] = v * self.window[i];
        }
        let power = dft_power_one(&self.scratch_frame, n_fft);
        for j in 0..crate::multimodal::audio::N_MELS {
            let mut acc = 0.0f64;
            for (k, &s) in power.iter().enumerate() {
                acc += self.filterbank[k * crate::multimodal::audio::N_MELS + j] * s;
            }
            let v = acc.max(1e-10).log10();
            self.running_max_log = self.running_max_log.max(v);
            self.columns.push(v);
        }
    }

    fn source_at(&self, idx: usize) -> f64 {
        f64::from(self.tail[idx - self.tail_start])
    }

    /// Immutable raw log-mel frame count. A frame here can never change
    /// value again; only the global floor applied at finish can rescale it.
    pub fn finalized_frames(&self) -> usize {
        self.columns.len() / crate::multimodal::audio::N_MELS
    }

    /// Drop retained samples below the next unfinalized frame's support
    /// start. Left-mirror lookups (frames 0/1 reading x[1..=200]) are safe:
    /// they only occur while `finalized_frames() < 2`, at which point
    /// nothing has been dropped.
    fn retire(&mut self) {
        let lowest = self
            .finalized_frames()
            .saturating_mul(160)
            .saturating_sub(200);
        if lowest > self.tail_start {
            let drop = (lowest - self.tail_start).min(self.tail.len());
            self.tail.drain(..drop);
            self.tail_start += drop;
        }
    }

    /// Flush pending frames against the true length (right mirror live),
    /// drop the final raw frame, apply the global floor + normalization.
    /// Returns `[N_MELS, T]` bit-identical to the one-shot path plus the
    /// exact floor that was applied.
    pub fn finish_with_floor(mut self) -> Result<(CpuTensor, f64)> {
        ensure!(
            self.total > 200,
            "streamed audio too short: {} samples (need > 200)",
            self.total
        );
        let raw = 1 + self.total / 160;
        while self.finalized_frames() < raw {
            let t = self.finalized_frames();
            self.compute_column(t, true);
        }
        let n_mels = crate::multimodal::audio::N_MELS;
        let usable = raw - 1;
        let max_log = self.columns[..usable * n_mels]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let floor = max_log - 8.0;
        let mut data = vec![0.0f32; usable * n_mels];
        for t in 0..usable {
            for j in 0..n_mels {
                let v = ((self.columns[t * n_mels + j].max(floor) + 4.0) / 4.0) as f32;
                data[j * usable + t] = v;
            }
        }
        Ok((CpuTensor::from_data(vec![n_mels, usable], data), floor))
    }

    /// Normalized mel for frames `[start, start+len)` under an explicit
    /// floor (`max_log − 8`): the exact per-element arithmetic of
    /// [`Self::finish`], applied to immutable raw columns. Encoder
    /// scheduling above this module uses this to (re)build windows from a
    /// floor it names, so finish-time validation can prove which cached
    /// encodes remain bit-valid.
    pub(crate) fn normalized_window(&self, start: usize, len: usize, floor: f64) -> CpuTensor {
        let n_mels = crate::multimodal::audio::N_MELS;
        assert!(
            start + len <= self.finalized_frames(),
            "normalized_window [{start}, {}) exceeds finalized frames {}",
            start + len,
            self.finalized_frames()
        );
        let mut data = vec![0.0f32; len * n_mels];
        for t in 0..len {
            for j in 0..n_mels {
                let v = ((self.columns[(start + t) * n_mels + j].max(floor) + 4.0) / 4.0) as f32;
                data[j * len + t] = v;
            }
        }
        CpuTensor::from_data(vec![n_mels, len], data)
    }

    /// The running max over all finalized raw columns (including any final
    /// raw frame later dropped from usable output). Provisional floors are
    /// `running_max_log() − 8.0`; see [`AudioStream::running_floor`].
    pub(crate) fn running_max_log(&self) -> f64 {
        self.running_max_log
    }

    /// Unstable partial features over finalized frames using the running
    /// max as floor. Later audio can raise the max and rescale history;
    /// for display, not inference.
    pub fn provisional(&self) -> CpuTensor {
        let n_mels = crate::multimodal::audio::N_MELS;
        let n = self.finalized_frames();
        let floor = if n > 0 {
            self.running_max_log - 8.0
        } else {
            0.0
        };
        let mut data = vec![0.0f32; n * n_mels];
        for t in 0..n {
            for j in 0..n_mels {
                let v = ((self.columns[t * n_mels + j].max(floor) + 4.0) / 4.0) as f32;
                data[j * n + t] = v;
            }
        }
        CpuTensor::from_data(vec![n_mels, n], data)
    }
}

/// One-frame power spectrum via per-bin angles (bit-identical values to
/// `audio::dft_power`, which precomputes the same cos/sin tables).
fn dft_power_one(frame: &[f64], n_fft: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let mut out = vec![0.0f64; bins];
    for (k, slot) in out.iter_mut().enumerate() {
        let angle = -2.0 * std::f64::consts::PI * k as f64 / n_fft as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (n, &x) in frame.iter().enumerate() {
            let a = angle * n as f64;
            re += x * a.cos();
            im += x * a.sin();
        }
        *slot = re * re + im * im;
    }
    out
}

// ---------------------------------------------------------------------------
// AudioStream facade
// ---------------------------------------------------------------------------

/// Stream configuration, fixed at construction: a mid-stream sample-rate
/// change is a protocol error by definition (the frontend's phase and
/// feature-frame bookkeeping have no meaning across rates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioStreamConfig {
    /// Input PCM sample rate (mono f32, nominal [-1, 1]).
    pub sample_rate: u32,
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: TARGET_SAMPLE_RATE as u32,
        }
    }
}

/// Progress after one `push_pcm`: absolute offsets plus the finalized /
/// pending split needed for honest encoder scheduling above this module.
#[derive(Debug, Clone, Copy)]
pub struct StreamProgress {
    /// Absolute input samples accepted so far (caller's sample clock).
    pub input_samples: usize,
    /// Absolute 16 kHz-domain samples produced so far.
    pub samples_16k: usize,
    /// Immutable raw log-mel frames available so far.
    pub finalized_frames: usize,
    /// Seconds received but not yet finalizable (tail overlap + margin).
    pub pending_seconds: f64,
    /// Total received duration in the 16 kHz domain.
    pub seconds_received: f64,
}

/// Final result of a completed stream.
#[derive(Debug)]
pub struct StreamedAudio {
    /// Normalized log-mel `[N_MELS, T]`, bit-identical to one-shot
    /// preprocessing of the concatenated PCM.
    pub mel: CpuTensor,
    /// Long-form window layout `(start, valid_len)` over mel frames.
    pub encoder_windows: Vec<(usize, usize)>,
    /// The exact global normalization floor (`max_log − 8`) applied to
    /// `mel`. Window encodes whose recorded floor differs from this value
    /// must be recomputed for bit-exact parity with static preprocessing.
    pub floor_log: f64,
    pub input_samples: usize,
    pub input_sample_rate: u32,
    pub samples_16k: usize,
    pub input_seconds: f64,
}

/// Incremental audio input: `open -> push_pcm* -> finish`. Mono f32 PCM in
/// memory; WAV files are deliberately not involved. Tracks the absolute
/// sample offset, the finalized/pending feature split, and (once finished)
/// the exact long-form window layout the encoder consumes.
pub struct AudioStream {
    config: AudioStreamConfig,
    resampler: Option<StreamingResampler>,
    mel: MelStream,
    input_samples: usize,
    samples_16k: usize,
    finalized_resampler_samples: usize,
    finished: bool,
}

impl AudioStream {
    pub fn open(config: AudioStreamConfig) -> Result<Self> {
        ensure!(config.sample_rate > 0, "sample rate must be positive");
        let resampler = if config.sample_rate == TARGET_SAMPLE_RATE as u32 {
            None
        } else {
            Some(StreamingResampler::new(
                config.sample_rate,
                TARGET_SAMPLE_RATE as u32,
            ))
        };
        Ok(Self {
            config,
            resampler,
            mel: MelStream::new(),
            input_samples: 0,
            samples_16k: 0,
            finalized_resampler_samples: 0,
            finished: false,
        })
    }

    pub fn config(&self) -> &AudioStreamConfig {
        &self.config
    }

    /// Push mono f32 PCM at the configured rate. Returns progress with the
    /// finalized/pending split.
    pub fn push_pcm(&mut self, samples: &[f32]) -> Result<StreamProgress> {
        ensure!(!self.finished, "push_pcm after finish");
        self.input_samples += samples.len();
        let fresh = match &mut self.resampler {
            Some(r) => r.push(samples),
            None => samples.to_vec(),
        };
        self.finalized_resampler_samples += fresh.len();
        self.samples_16k += fresh.len();
        self.mel.push(&fresh);
        Ok(self.progress())
    }

    /// Finish the stream and produce the definitive mel spectrogram.
    ///
    /// Exactly what is recomputed here: only the tail frames whose values
    /// depend on the right reflect mirror (plus the dropped final raw
    /// frame). Everything finalized during streaming is reused verbatim.
    pub fn finish(mut self) -> Result<StreamedAudio> {
        self.finished = true;
        if let Some(r) = self.resampler.take() {
            let tail = r.finish();
            self.samples_16k += tail.len();
            self.mel.push(&tail);
        }
        let (mel, floor_log) = self.mel.finish_with_floor()?;
        let frames = mel.shape()[1];
        Ok(StreamedAudio {
            encoder_windows: long_form_windows(frames, MAX_FRAMES),
            mel,
            floor_log,
            input_samples: self.input_samples,
            input_sample_rate: self.config.sample_rate,
            samples_16k: self.samples_16k,
            input_seconds: self.input_samples as f64 / f64::from(self.config.sample_rate),
        })
    }

    /// Unstable partial mel over finalized frames (display only — see
    /// [`MelStream::provisional`]).
    pub fn provisional_mel(&self) -> Option<CpuTensor> {
        let m = self.mel.provisional();
        if m.shape()[1] == 0 {
            None
        } else {
            Some(m)
        }
    }

    // -----------------------------------------------------------------
    // Encoder-scheduling peeks (Track C4)
    //
    // These expose exactly the information an above-front-end scheduler
    // needs: how many 30 s windows are fully determined, and a way to
    // rebuild any finalized frame range under a named normalization floor.
    // They never mutate stream state.
    // -----------------------------------------------------------------

    /// Immutable mel frames available so far (before finish).
    pub fn finalized_frames(&self) -> usize {
        self.mel.finalized_frames()
    }

    /// Number of complete [`MAX_FRAMES`]-frame long-form windows whose every
    /// frame is already immutable. Window `k` covers
    /// `[k*MAX_FRAMES, (k+1)*MAX_FRAMES)`; these windows are fully
    /// determined regardless of future audio. The trailing partial window
    /// (`finalized_frames % MAX_FRAMES`) remains mutable until finish.
    pub fn fixed_windows_finalized(&self) -> usize {
        self.mel.finalized_frames() / MAX_FRAMES
    }

    /// Normalized mel for the finalized frame range `[start, start+len)`
    /// under the explicit floor `floor_log` (a value of the form
    /// `max_log − 8`). Bit-identical to slicing the finished spectrogram
    /// whenever `floor_log` equals its final floor.
    pub fn window_mel_with_floor(
        &self,
        start: usize,
        len: usize,
        floor_log: f64,
    ) -> Result<CpuTensor> {
        ensure!(
            !self.finished,
            "stream finished: use StreamedAudio::mel instead"
        );
        ensure!(len > 0, "empty window request");
        ensure!(
            start + len <= self.finalized_frames(),
            "window [{start}, {}) exceeds finalized frames {}",
            start + len,
            self.finalized_frames()
        );
        Ok(self.mel.normalized_window(start, len, floor_log))
    }

    /// Current provisional normalization floor: `running_max_log() − 8`.
    /// Encodes built under this floor are provisional — later audio can
    /// raise the global max and invalidate them (detected at finish by
    /// comparing recorded floors).
    pub fn running_floor(&self) -> Option<f64> {
        let max = self.mel.running_max_log();
        if max.is_finite() {
            Some(max - 8.0)
        } else {
            None
        }
    }

    /// Absolute-offset snapshot for scheduler bookkeeping.
    pub fn progress(&self) -> StreamProgress {
        let seconds_received = self.samples_16k as f64 / TARGET_SAMPLE_RATE as f64;
        let finalized_secs = self.mel.finalized_frames() as f64 / TARGET_SAMPLE_RATE as f64;
        StreamProgress {
            input_samples: self.input_samples,
            samples_16k: self.samples_16k,
            finalized_frames: self.mel.finalized_frames(),
            pending_seconds: (seconds_received - finalized_secs).max(0.0),
            seconds_received,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::audio::{log_mel_spectrogram_full, resample};

    /// Deterministic pseudo-random signal in [-1, 1].
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn chunk_len(&mut self, max: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % max + 1
        }
    }

    fn signal(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| rng.next_f32()).collect()
    }

    fn assert_bit_exact(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: sample {i} differs");
        }
    }

    fn run_partitioned_resample(sig: &[f32], parts: &[usize]) -> Vec<f32> {
        let mut r = StreamingResampler::new(44_100, 16_000);
        let mut out = Vec::new();
        let mut pos = 0;
        for &len in parts {
            let end = (pos + len).min(sig.len());
            out.extend(r.push(&sig[pos..end]));
            pos = end;
        }
        assert_eq!(pos, sig.len());
        out.extend(r.finish());
        out
    }

    #[test]
    fn streamed_resampler_matches_one_shot_bit_exact() {
        let sig = signal(50_000, 42); // ~1.13 s @ 44.1 kHz
        let one_shot = resample(&sig, 44_100, 16_000);
        assert!(!one_shot.is_empty());

        // 1-sample chunks
        let parts = vec![1; sig.len()];
        assert_bit_exact(
            &run_partitioned_resample(&sig, &parts),
            &one_shot,
            "1-sample chunks",
        );

        // odd chunk sizes
        let odd: Vec<usize> = (0..sig.len())
            .map(|i| if i % 3 == 0 { 7 } else { 13 })
            .collect();
        assert_bit_exact(
            &run_partitioned_resample(&sig, &odd),
            &one_shot,
            "odd chunks",
        );

        // random chunk sizes
        let mut rng = Lcg(7);
        let mut random_parts = Vec::new();
        let mut covered = 0;
        while covered < sig.len() {
            let len = rng.chunk_len(997);
            random_parts.push(len);
            covered += len;
        }
        assert_bit_exact(
            &run_partitioned_resample(&sig, &random_parts),
            &one_shot,
            "random chunks",
        );

        // large chunks
        let big: Vec<usize> = std::iter::repeat_n(8192, sig.len() / 8192 + 1).collect();
        assert_bit_exact(
            &run_partitioned_resample(&sig, &big),
            &one_shot,
            "large chunks",
        );
    }

    #[test]
    fn streamed_mel_matches_one_shot_bit_exact_across_partitions() {
        // awkward, non-hop-aligned length (~3.70 s @ 16 kHz)
        let sig = signal(59_201, 1234);
        let one_shot = log_mel_spectrogram_full(&sig).expect("one-shot mel");
        let expected: Vec<f32> = one_shot.data().to_vec();

        let run = |parts: &[usize]| -> Vec<f32> {
            let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
            let mut pos = 0;
            for &len in parts {
                let end = (pos + len).min(sig.len());
                stream.push_pcm(&sig[pos..end]).unwrap();
                pos = end;
            }
            assert_eq!(pos, sig.len());
            stream.finish().unwrap().mel.data().to_vec()
        };

        let single = vec![sig.len()];
        assert_eq!(run(&single), expected, "single push");

        let ones = vec![1; sig.len()];
        assert_eq!(run(&ones), expected, "1-sample chunks");

        let odd: Vec<usize> = (0..sig.len())
            .map(|i| if i % 5 == 0 { 3 } else { 17 })
            .collect();
        assert_eq!(run(&odd), expected, "odd chunks");

        let mut rng = Lcg(99);
        let mut parts = Vec::new();
        let mut covered = 0;
        while covered < sig.len() {
            let len = rng.chunk_len(1333);
            parts.push(len);
            covered += len;
        }
        assert_eq!(run(&parts), expected, "random chunks");
    }

    #[test]
    fn resampled_audio_stream_matches_static_pipeline_bit_exact() {
        // 44.1 kHz source through the full streamed pipeline vs
        // resample-then-mel on the whole buffer.
        let sig = signal(44_100, 777); // 1.0 s @ 44.1 kHz
        let expected = log_mel_spectrogram_full(&resample(&sig, 44_100, 16_000))
            .unwrap()
            .data()
            .to_vec();
        let mut stream = AudioStream::open(AudioStreamConfig {
            sample_rate: 44_100,
        })
        .unwrap();
        let mut pos = 0;
        let mut rng = Lcg(5);
        while pos < sig.len() {
            let len = rng.chunk_len(1000).min(sig.len() - pos);
            stream.push_pcm(&sig[pos..pos + len]).unwrap();
            pos += len;
        }
        let got = stream.finish().unwrap().mel.data().to_vec();
        assert_eq!(got, expected);
    }

    #[test]
    fn finalized_prefix_never_shrinks() {
        let sig = signal(24_000, 31); // 1.5 s
        let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
        let mut last_frames = 0;
        for chunk in sig.chunks(480) {
            let p = stream.push_pcm(chunk).unwrap();
            assert!(p.finalized_frames >= last_frames);
            last_frames = p.finalized_frames;
        }
        let done = stream.finish().unwrap();
        assert_eq!(done.input_samples, sig.len());
        assert_eq!(done.input_sample_rate, 16_000);
        assert_eq!(done.encoder_windows, vec![(0, done.mel.shape()[1])]);
        // the consumed stream cannot accept more audio
        assert!(done.mel.shape()[1] > 0);
    }

    #[test]
    fn too_short_stream_errors_like_one_shot() {
        let sig = signal(150, 3);
        let one_shot = log_mel_spectrogram_full(&sig);
        assert!(one_shot.is_err(), "one-shot must reject short input");
        let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
        stream.push_pcm(&sig).unwrap();
        assert!(stream.finish().is_err(), "streamed must reject short input");
    }

    #[test]
    fn provisional_features_exist_before_finish() {
        let sig = signal(48_000, 55); // 3 s
        let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
        let mut saw_provisional = false;
        for (i, chunk) in sig.chunks(3200).enumerate() {
            stream.push_pcm(chunk).unwrap();
            if i >= 4 {
                assert!(stream.provisional_mel().is_some());
                saw_provisional = true;
            }
        }
        assert!(saw_provisional);
    }
}

#[cfg(test)]
mod window_floor_tests {
    use super::*;

    /// Deterministic signal whose global max lands at ~1 s (well before
    /// any tail effects): a single impulse followed by low-level noise.
    fn impulse_signal(n: usize) -> Vec<f32> {
        let mut sig = vec![0.0f32; n];
        let peak = 16_000; // 1 s
        if peak < n {
            sig[peak] = 0.98;
        }
        // tiny deterministic ripple afterwards
        for (i, s) in sig.iter_mut().enumerate().skip(peak + 1) {
            *s = 1e-4 * ((i % 13) as f32);
        }
        sig
    }

    #[test]
    fn window_slice_matches_finished_mel_when_floor_settled() {
        // ~2.5 s: impulse early, quiet after; once everything is finalized
        // and the running max has settled, an explicit-floor slice of ALL
        // finalized frames must be bit-equal to the finished spectrogram.
        let sig = impulse_signal(40_000);
        let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
        stream.push_pcm(&sig).unwrap();
        let finalized = stream.finalized_frames();
        assert!(finalized > 0);
        let floor = stream.running_floor().unwrap();

        let sliced = stream
            .window_mel_with_floor(0, finalized, floor)
            .unwrap()
            .data()
            .to_vec();

        let done = stream.finish().unwrap();
        // every finalized frame must appear identically in the finish output
        let n_mels = crate::multimodal::audio::N_MELS;
        let width = done.mel.shape()[1];
        for t in 0..finalized {
            for j in 0..n_mels {
                let a = sliced[j * finalized + t];
                let b = done.mel.data()[j * width + t];
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "frame {t} mel {j} differs between explicit-floor slice and finish"
                );
            }
        }
    }

    #[test]
    fn running_floor_rises_and_slices_track_it() {
        // quiet prefix first, loud spike late: slices taken under the early
        // floor differ from finish output (floor rose), which is exactly
        // what finish-time staleness detection keys on.
        let mut sig = vec![0.001f32; 30_000];
        sig[29_000] = 0.95;
        let mut stream = AudioStream::open(AudioStreamConfig::default()).unwrap();
        // push only up to just past the finalize margin of the first frames
        stream.push_pcm(&sig[..24_000]).unwrap();
        let early_floor = stream.running_floor().unwrap();
        let finalized = stream.finalized_frames();
        let sliced = stream
            .window_mel_with_floor(0, finalized, early_floor)
            .unwrap();

        stream.push_pcm(&sig[24_000..]).unwrap();
        let done = stream.finish().unwrap();
        let final_floor = done.floor_log;
        assert!(
            final_floor > early_floor,
            "late spike must raise the global max"
        );
        // same raw columns under a higher floor => clamped columns change
        let n_mels = crate::multimodal::audio::N_MELS;
        let width = done.mel.shape()[1];
        let diffs = (0..finalized)
            .filter(|&t| {
                (0..n_mels)
                    .any(|j| sliced.data()[j * finalized + t] != done.mel.data()[j * width + t])
            })
            .count();
        assert!(diffs > 0, "stale-floor slice must differ from finish");
    }
}

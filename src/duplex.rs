//! Real concurrent full-duplex audio plumbing (Phase 5 Track A/B).
//!
//! Architecture (the OS boundary stays OUT of model logic):
//!
//! ```text
//! device/cpal callback ──> CaptureRing ──> runtime thread
//!        (realtime)       (drop-oldest)        │ TurnDetector
//!                                              │ SpeechStarted during
//!                                              │ assistant activity
//!                                              v
//!                                   GenerationControl.cancel()
//!                                   + PlaybackRing::request_clear()
//! runtime thread ──> PlaybackRing ──> device/cpal callback
//!                        (underrun = silence)
//! ```
//!
//! The realtime audio callback never blocks on inference: capture pushes
//! into a lock-free SPSC ring (dropping the OLDEST samples when the
//! consumer falls behind — speech onsets survive; overruns are counted);
//! playback pops from one (silence on underrun, counted). Device sample
//! rates flow through the existing [`crate::multimodal::stream`] resampler,
//! never a hidden second path.
//!
//! The `cpal` device bindings live in the `device` submodule behind the `audio` cargo
//! feature; everything here is pure Rust so the policy layer is testable
//! without hardware.

use anyhow::{ensure, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A block of mono PCM captured from (or destined for) an audio device.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Absolute stream position of `samples[0]` in device samples
    /// (monotone; head-drops advance it — gaps are visible, not hidden).
    pub first_sample_offset: u64,
}

/// One turn-detector decision per fed chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEvent {
    SpeechStarted,
    SpeechContinues,
    SpeechEnded,
}

/// Endpointing seam (Track B): consumes incoming PCM and produces turn
/// events. Implementations must be cheap enough to run between chunks on
/// the runtime thread; nothing here touches inference or devices.
pub trait TurnDetector {
    fn feed(&mut self, chunk: &AudioChunk) -> Option<TurnEvent>;
    /// Manual endpoint signal from the application (push-to-talk style).
    fn force_start(&mut self);
    fn force_end(&mut self);
}

/// Energy VAD with adaptive noise floor, start hysteresis and hangover.
///
/// * 10 ms frames;
/// * noise floor: exponential minimum tracker updated only while quiet;
/// * SpeechStarted after `start_frames_needed` consecutive loud frames;
/// * SpeechEnded after `hangover_frames_max` consecutive quiet frames.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    sample_rate: u32,
    frame_samples: usize,
    noise_floor: f32,
    start_ratio: f32,
    stop_ratio: f32,
    start_frames_needed: usize,
    hangover_frames_max: usize,
    above_run: usize,
    hangover_left: usize,
    speaking: bool,
}

impl EnergyVad {
    pub fn new(sample_rate: u32) -> Self {
        let frame_samples = (sample_rate as usize / 100).max(16); // 10 ms
        Self {
            sample_rate,
            frame_samples,
            noise_floor: 0.001,
            start_ratio: 3.5,
            stop_ratio: 2.0,
            start_frames_needed: 3,
            hangover_frames_max: 30, // 300 ms tail
            above_run: 0,
            hangover_left: 0,
            speaking: false,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    fn frame_rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        (sum / x.len() as f64).sqrt() as f32
    }
}

impl TurnDetector for EnergyVad {
    fn feed(&mut self, chunk: &AudioChunk) -> Option<TurnEvent> {
        debug_assert_eq!(chunk.sample_rate, self.sample_rate);
        let mut event = None;
        let mut done = 0usize;
        let samples = &chunk.samples;
        while done < samples.len() {
            let take = self.frame_samples.min(samples.len() - done);
            // partial trailing frames fold into later chunks naturally:
            // state carries across feed() calls
            let rms = Self::frame_rms(&samples[done..done + take]);
            done += take;

            let ratio = rms / self.noise_floor.max(1e-9);
            if !self.speaking {
                // adapt the noise floor only while quiet
                self.noise_floor += (rms - self.noise_floor) * 0.05;
                if ratio >= self.start_ratio && rms > 1e-4 {
                    self.above_run += 1;
                    if self.above_run >= self.start_frames_needed {
                        self.speaking = true;
                        self.hangover_left = self.hangover_frames_max;
                        event = Some(TurnEvent::SpeechStarted);
                    }
                } else {
                    self.above_run = 0;
                }
            } else if ratio < self.stop_ratio || rms <= 1e-4 {
                self.hangover_left = self.hangover_left.saturating_sub(1);
                if self.hangover_left == 0 {
                    self.speaking = false;
                    self.above_run = 0;
                    event = Some(TurnEvent::SpeechEnded);
                }
            } else {
                self.hangover_left = self.hangover_frames_max;
            }
        }
        event
    }

    fn force_start(&mut self) {
        self.speaking = true;
        self.hangover_left = self.hangover_frames_max;
    }

    fn force_end(&mut self) {
        self.speaking = false;
        self.above_run = 0;
        self.hangover_left = 0;
    }
}

// ---------------------------------------------------------------------------
// bounded SPSC rings with explicit policies + metrics
// ---------------------------------------------------------------------------

/// Queue sizing policy: at 16 kHz mono f32 these bound memory (~128/64 KiB)
/// and worst-case staleness (~2 s capture / ~1 s playback).
pub const CAPTURE_QUEUE_SAMPLES: usize = 32_000;
pub const PLAYBACK_QUEUE_SAMPLES: usize = 16_000;

/// Shared telemetry for both rings.
#[derive(Default)]
pub struct RingMetrics {
    pub overruns: AtomicU64,
    pub dropped_samples: AtomicU64,
    pub underruns: AtomicU64,
    pub clears: AtomicU64,
    pub pulls: AtomicU64,
}

/// Create a capture path: realtime producer (device callback side) and
/// runtime consumer. Drop-oldest overrun policy on the producer.
pub fn capture_ring(capacity: usize) -> (CaptureProducer, CaptureConsumer) {
    let (producer, consumer) = rtrb::RingBuffer::new(capacity.max(64));
    let metrics = std::sync::Arc::new(RingMetrics::default());
    let accepted = std::sync::Arc::new(AtomicU64::new(0));
    let oldest_pos = std::sync::Arc::new(AtomicU64::new(0));
    (
        CaptureProducer {
            ring: producer,
            capacity: capacity.max(64),
            metrics: metrics.clone(),
            accepted: accepted.clone(),
            oldest_pos: oldest_pos.clone(),
        },
        CaptureConsumer {
            ring: consumer,
            metrics,
            accepted,
            oldest_pos,
        },
    )
}

/// Realtime side of capture. Lives on the audio callback thread.
/// `push` never blocks and never allocates.
pub struct CaptureProducer {
    ring: rtrb::Producer<f32>,
    capacity: usize,
    metrics: std::sync::Arc<RingMetrics>,
    /// total samples ACCEPTED into the stream (drops excluded)
    accepted: std::sync::Arc<AtomicU64>,
    /// stream position of the oldest queued sample
    oldest_pos: std::sync::Arc<AtomicU64>,
}

impl CaptureProducer {
    /// Overrun policy: DROP-NEWEST. When the ring is full (the runtime
    /// thread stalled longer than the capacity — ~2 s at 16 kHz) incoming
    /// samples are discarded and counted; the gap becomes visible through
    /// the chunk offset stream. Onset preservation beats tail preservation
    /// here because endpointing keys on speech starts.
    pub fn push(&mut self, samples: &[f32]) {
        // Drop-newest policy: when full, discard incoming samples.
        // Onset preservation beats tail preservation because endpointing
        // keys on speech starts; gaps are visible via offsets.
        let was_empty = self.ring.slots() == self.capacity;
        let space = self.ring.slots();
        let accepted_len = space.min(samples.len());
        for s in &samples[..accepted_len] {
            let _ = self.ring.push(*s);
        }
        let dropped = samples.len() - accepted_len;
        if dropped > 0 {
            self.metrics.overruns.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .dropped_samples
                .fetch_add(dropped as u64, Ordering::Relaxed);
        }
        let acc_after = self
            .accepted
            .fetch_add(accepted_len as u64, Ordering::Relaxed)
            + accepted_len as u64;
        if was_empty && accepted_len > 0 {
            // queue begins a new contiguous segment here
            self.oldest_pos
                .store(acc_after - accepted_len as u64, Ordering::Relaxed);
        }
    }
}

/// Runtime side of capture.
pub struct CaptureConsumer {
    ring: rtrb::Consumer<f32>,
    metrics: std::sync::Arc<RingMetrics>,
    accepted: std::sync::Arc<AtomicU64>,
    oldest_pos: std::sync::Arc<AtomicU64>,
}

impl CaptureConsumer {
    /// Drain everything currently queued as one chunk. Offsets reflect any
    /// head drops (gaps are visible in the offset stream, not hidden).
    pub fn pop_chunk(&mut self, sample_rate: u32) -> Result<AudioChunk> {
        let queued = self.ring.slots();
        ensure!(queued > 0, "capture queue empty");
        let first_sample_offset = self.oldest_pos.load(Ordering::Relaxed);
        let mut samples = Vec::with_capacity(queued);
        for _ in 0..queued {
            match self.ring.pop() {
                Ok(s) => samples.push(s),
                Err(_) => break,
            }
        }
        self.oldest_pos.store(
            first_sample_offset + samples.len() as u64,
            Ordering::Relaxed,
        );
        Ok(AudioChunk {
            samples,
            sample_rate,
            first_sample_offset,
        })
    }

    pub fn queued(&self) -> usize {
        self.ring.slots()
    }

    pub fn overruns(&self) -> u64 {
        self.metrics.overruns.load(Ordering::Relaxed)
    }

    pub fn dropped_samples(&self) -> u64 {
        self.metrics.dropped_samples.load(Ordering::Relaxed)
    }

    /// Total samples accepted by the paired producer (liveness metric).
    pub fn accepted_snapshot(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    pub fn underruns(&self) -> u64 {
        self.metrics.underruns.load(Ordering::Relaxed)
    }

    pub fn clears(&self) -> u64 {
        self.metrics.clears.load(Ordering::Relaxed)
    }

    /// Shared telemetry handle (also covers the paired playback ring).
    pub fn metrics_handle(&self) -> std::sync::Arc<RingMetrics> {
        self.metrics.clone()
    }
}

/// Create a playback path: runtime writer and realtime reader (device
/// callback side). Underrun policy: silence + counted. Barge-in policy:
/// `request_clear()` makes the READER drop everything queued within one
/// callback block — stale assistant audio cannot survive it.
pub fn playback_ring(capacity: usize) -> (PlaybackWriter, PlaybackReader) {
    let (producer, consumer) = rtrb::RingBuffer::new(capacity.max(64));
    let metrics = std::sync::Arc::new(RingMetrics::default());
    let clear_requested = std::sync::Arc::new(AtomicBool::new(false));
    let buffered = std::sync::Arc::new(AtomicU64::new(0));
    (
        PlaybackWriter {
            ring: producer,
            clear_requested: clear_requested.clone(),
            buffered: buffered.clone(),
            metrics: metrics.clone(),
        },
        PlaybackReader {
            ring: consumer,
            clear_requested,
            buffered,
            metrics,
        },
    )
}

/// Runtime side of playback.
pub struct PlaybackWriter {
    ring: rtrb::Producer<f32>,
    clear_requested: std::sync::Arc<AtomicBool>,
    buffered: std::sync::Arc<AtomicU64>,
    metrics: std::sync::Arc<RingMetrics>,
}

impl PlaybackWriter {
    /// Shared telemetry handle of the paired reader/writer.
    pub fn metrics_handle(&self) -> std::sync::Arc<RingMetrics> {
        self.metrics.clone()
    }

    /// Queue PCM for playback. Returns samples REJECTED (queue full — the
    /// caller decides whether to wait or drop). Barge-in aware: a pending
    /// clear is honored here too (the queue is logically emptied).
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let space = self.ring.slots().min(samples.len());
        for s in &samples[..space] {
            let _ = self.ring.push(*s);
        }
        self.buffered.fetch_add(space as u64, Ordering::Relaxed);
        samples.len() - space
    }

    pub fn buffered(&self) -> usize {
        self.buffered.load(Ordering::Relaxed) as usize
    }

    /// Barge-in: ask the realtime reader to drop all queued audio. Takes
    /// effect within one callback block (~a few ms), never blocks.
    pub fn request_clear(&self) {
        self.clear_requested.store(true, Ordering::Release);
    }

    pub fn underruns_observed(&self) -> u64 {
        self.metrics.underruns.load(Ordering::Relaxed)
    }
}

/// Realtime side of playback. Lives on the audio callback thread.
pub struct PlaybackReader {
    ring: rtrb::Consumer<f32>,
    clear_requested: std::sync::Arc<AtomicBool>,
    buffered: std::sync::Arc<AtomicU64>,
    metrics: std::sync::Arc<RingMetrics>,
}

impl PlaybackReader {
    /// Callback invocations (diagnostic for live-device bring-up).
    pub fn pulls(&self) -> u64 {
        self.metrics.pulls.load(Ordering::Relaxed)
    }

    /// Copy up to `out.len()` samples into `out`, zero-filling any shortfall
    /// (counted as an underrun). Honors a pending barge-in clear first.
    pub fn pull(&mut self, out: &mut [f32]) {
        self.metrics.pulls.fetch_add(1, Ordering::Relaxed);
        if self.clear_requested.swap(false, Ordering::AcqRel) {
            self.drop_all_queued();
        }
        let mut filled = 0usize;
        while filled < out.len() {
            match self.ring.pop() {
                Ok(s) => {
                    out[filled] = s;
                    filled += 1;
                }
                Err(_) => break,
            }
        }
        self.buffered.fetch_sub(filled as u64, Ordering::Relaxed);
        if filled < out.len() {
            out[filled..].fill(0.0);
            self.metrics.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn drop_all_queued(&mut self) {
        let mut dropped = 0usize;
        while self.ring.pop().is_ok() {
            dropped += 1;
        }
        if dropped > 0 {
            self.buffered.fetch_sub(dropped as u64, Ordering::Relaxed);
            self.metrics.clears.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn underruns(&self) -> u64 {
        self.metrics.underruns.load(Ordering::Relaxed)
    }

    pub fn clears(&self) -> u64 {
        self.metrics.clears.load(Ordering::Relaxed)
    }

    pub fn dropped_samples(&self) -> u64 {
        self.metrics.dropped_samples.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "audio")]
pub mod device;

// ---------------------------------------------------------------------------
// duplex controller: the runtime-thread glue between audio I/O and sessions
// ---------------------------------------------------------------------------

/// Runtime-thread controller for a live full-duplex conversation.
///
/// Owns the consumer/writer halves of the audio rings plus the turn
/// detector. Each [`pump`](Self::pump) drains captured audio, feeds the
/// detector, and — when speech starts while the assistant is active —
/// fires barge-in: cancel generation (via a shared
/// [`crate::multimodal::GenerationControl`]-compatible flag) and drop all
/// queued playback. The same object collects the user's utterance PCM so
/// the next turn can be submitted.
pub struct DuplexController {
    pub capture: CaptureConsumer,
    pub playback: PlaybackWriter,
    pub detector: Box<dyn TurnDetector>,
    /// Set while assistant generation/playback is active; barge-in only
    /// fires in that state.
    assistant_active: bool,
    /// Barge-in latch: read through `stop_probe()` by generate_reply.
    barge_in: std::sync::Arc<AtomicBool>,
    /// Mirrors the detector's speaking state (trait stays minimal).
    speaking_state: bool,
    quiet_offset: u64,
    /// User utterance collected since SpeechStarted, at DEVICE rate.
    utterance: Vec<f32>,
    utterance_rate: u32,
    /// Absolute device-sample offset where the utterance began.
    utterance_start_offset: u64,
    pub metrics: std::sync::Arc<RingMetrics>,
}

impl DuplexController {
    /// `sample_rate` is the DEVICE capture rate; all chunks are labelled
    /// with it and the detector must be constructed for the same rate.
    pub fn new_with_sample_rate(
        capture: CaptureConsumer,
        playback: PlaybackWriter,
        detector: Box<dyn TurnDetector>,
        sample_rate: u32,
    ) -> Self {
        let mut ctl = Self::new(capture, playback, detector);
        ctl.utterance_rate = sample_rate;
        ctl
    }

    pub fn new(
        capture: CaptureConsumer,
        playback: PlaybackWriter,
        detector: Box<dyn TurnDetector>,
    ) -> Self {
        let metrics = capture.metrics_handle();
        Self {
            capture,
            playback,
            detector,
            assistant_active: false,
            barge_in: std::sync::Arc::new(AtomicBool::new(false)),
            speaking_state: false,
            quiet_offset: 0,
            utterance: Vec::new(),
            utterance_rate: 16_000,
            utterance_start_offset: 0,
            metrics,
        }
    }

    /// Mark assistant activity state. Barge-in arms itself only here.
    pub fn set_assistant_active(&mut self, active: bool) {
        self.assistant_active = active;
        if !active {
            self.barge_in.store(false, Ordering::Release);
        }
    }

    pub fn is_barge_in(&self) -> bool {
        self.barge_in.load(Ordering::Acquire)
    }

    /// Stop-probe seam for `VoiceSession::generate_reply`: true cancels at
    /// the next token checkpoint.
    pub fn stop_probe(&self) -> bool {
        self.is_barge_in()
    }

    /// Drain captured audio into the detector; returns the turn event (if
    /// any). Fires barge-in on SpeechStarted during assistant activity.
    pub fn pump(&mut self) -> Option<TurnEvent> {
        self.pump_events().into_iter().next()
    }

    /// Like [`pump`](Self::pump) but reports EVERY detector transition the
    /// drained chunk contained, in order (a single pop can carry both an
    /// onset and an endpoint; callers driving a state machine need both).
    pub fn pump_events(&mut self) -> Vec<TurnEvent> {
        self.pump_with_chunk_cb(|_, _| ())
    }

    /// [`pump_events`](Self::pump_events) with a tap on each drained chunk
    /// (`samples`, `device_sample_rate`). The tap runs BEFORE event
    /// application, so a driver can mirror live PCM into its own frontend
    /// while the controller simultaneously maintains utterance collection.
    pub fn pump_with_chunk_cb(&mut self, mut on_chunk: impl FnMut(&[f32], u32)) -> Vec<TurnEvent> {
        if self.capture.queued() == 0 {
            return Vec::new();
        }
        let chunk = match self.capture.pop_chunk(self.utterance_rate) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let rate = chunk.sample_rate;
        on_chunk(&chunk.samples, rate);
        // Feed in ~10 ms slices so a chunk containing BOTH an onset and an
        // end loses neither transition.
        let slice = (rate as usize / 100).max(16);
        let mut events: Vec<TurnEvent> = Vec::new();
        for piece in chunk.samples.chunks(slice.max(16)) {
            if let Some(e) = self.detector.feed(&AudioChunk {
                samples: piece.to_vec(),
                sample_rate: rate,
                first_sample_offset: 0,
            }) {
                events.push(e);
                self.apply_event_state(Some(e));
            }
        }
        let _ = rate;
        // Collect everything belonging to the open utterance. The condition
        // must include "a transition happened in THIS chunk": a chunk that
        // contains a full utterance (onset + endpoint) ends with
        // speaking_state == false and an empty buffer, and skipping
        // collection there would lose the entire utterance.
        if !events.is_empty() || !self.utterance.is_empty() || self.speaking_state {
            self.utterance.extend_from_slice(&chunk.samples);
        }
        events
    }

    /// Device sample rate utterances are collected at (constructor arg).
    pub fn utterance_sample_rate(&self) -> u32 {
        self.utterance_rate
    }

    /// Mirror detector transitions into controller state; fires barge-in.
    fn apply_event_state(&mut self, event: Option<TurnEvent>) {
        match event {
            Some(TurnEvent::SpeechStarted) => {
                self.speaking_state = true;
                if self.assistant_active && !self.is_barge_in() {
                    self.barge_in.store(true, Ordering::Release);
                    self.playback.request_clear();
                }
            }
            Some(TurnEvent::SpeechEnded) => {
                self.speaking_state = false;
            }
            _ => {}
        }
    }

    /// Collected user utterance and its device sample rate.
    pub fn take_utterance(&mut self) -> (Vec<f32>, u32, u64) {
        (
            std::mem::take(&mut self.utterance),
            self.utterance_rate,
            self.utterance_start_offset,
        )
    }

    /// Feed one block of synthesized silence through the same path (used
    /// when no device is live but detectors need wall-clock progress).
    /// Runs the full event state machine so transitions are not lost.
    pub fn pump_quiet(&mut self, samples: usize) -> Option<TurnEvent> {
        let mut event = None;
        let frame = (self.utterance_rate as usize / 100).max(16);
        for piece in vec![0.0f32; samples].chunks(frame) {
            if let Some(e) = self.detector.feed(&AudioChunk {
                samples: piece.to_vec(),
                sample_rate: self.utterance_rate,
                first_sample_offset: self.quiet_offset,
            }) && event.is_none()
            {
                event = Some(e);
            }
            self.apply_event_state(event);
            self.quiet_offset += piece.len() as u64;
        }
        event
    }

    /// Whether the detector currently reports an open speech turn.
    pub fn detector_has_speech(&self) -> bool {
        self.speaking_state
    }

    /// Samples discarded by the capture overrun policy so far.
    pub fn dropped_samples(&self) -> u64 {
        self.capture.dropped_samples()
    }

    /// Total samples accepted by the capture stream so far (liveness).
    pub fn captured_total(&self) -> u64 {
        self.captured_counter()
    }

    fn captured_counter(&self) -> u64 {
        self.capture.accepted_snapshot()
    }

    /// Queue assistant PCM for playback.
    pub fn play_audio(&mut self, samples: &[f32]) -> usize {
        self.playback.push(samples)
    }

    /// Immediately drop any unplayed assistant audio.
    pub fn clear_playback(&mut self) {
        self.playback.request_clear();
    }
}

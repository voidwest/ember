//! Live audio device bindings (Phase 5 Track A2/A3/A4) — `cpal` backend.
//!
//! Backend decision (documented for the phase report): cpal over ALSA.
//! This host runs PipeWire, which exposes itself through the standard
//! ALSA pcm interface (`pipewire-alsa`), so one ALSA path serves PipeWire,
//! JACK and bare-hw setups; cpal's callback model matches the realtime
//! requirements of Tracks A3/A4; CoreAudio arrives free on a future macOS
//! port. Raw ALSA FFI and pipewire-rs were rejected as larger surfaces for
//! the same behavior.
//!
//! Everything in here is plumbing between device callbacks and the
//! lock-free rings in [`super`]: no inference, no blocking, no allocation
//! on the realtime side beyond what the ring push performs.

use super::{
    capture_ring, playback_ring, CaptureConsumer, PlaybackWriter, RingMetrics,
    CAPTURE_QUEUE_SAMPLES, PLAYBACK_QUEUE_SAMPLES,
};
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;

/// Keep-alive guard for the two device streams plus their telemetry.
///
/// Dropping this stops capture/playback; hold it for the session lifetime.
pub struct LiveStreamGuard {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    metrics: Arc<RingMetrics>,
    pub out_cb_probe: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl LiveStreamGuard {
    pub fn underruns(&self) -> u64 {
        self.metrics
            .underruns
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn clears(&self) -> u64 {
        self.metrics
            .clears
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Output-callback invocations (diagnostic).
    pub fn output_pulls(&self) -> u64 {
        self.metrics
            .pulls
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Everything the runtime thread needs from a live duplex opening.
pub struct LiveDuplex {
    pub capture: CaptureConsumer,
    pub playback: PlaybackWriter,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    guard: LiveStreamGuard,
}

impl LiveDuplex {
    /// Open default input+output at their native rates. Device rate
    /// conversion to/from Ember's 16 kHz/24 kHz domains flows through the
    /// existing streaming resampler on the session side — never here.
    pub fn open_default() -> Result<Self> {
        let host = cpal::default_host();
        let in_dev = host
            .default_input_device()
            .context("no default input (microphone) device")?;
        let out_dev = host
            .default_output_device()
            .context("no default output (speaker) device")?;

        let in_cfg = in_dev.default_input_config().context("input config")?;
        let out_cfg = out_dev.default_output_config().context("output config")?;
        let in_rate = in_cfg.sample_rate().0;
        let out_rate = out_cfg.sample_rate().0;
        let (capture_producer, capture_consumer) = capture_ring(CAPTURE_QUEUE_SAMPLES);
        let (playback_writer, playback_reader) = playback_ring(PLAYBACK_QUEUE_SAMPLES);
        // the guard reports PLAYBACK telemetry; capture counters are read
        // through CaptureConsumer's own accessors
        let metrics = playback_writer.metrics_handle();

        // -- playback callback: pull from the ring, fan out to channels --
        let out_cb_probe = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let out_cb_probe_ref = out_cb_probe.clone();

        let mut playback_reader = playback_reader;
        let out_channels = out_cfg.channels() as usize;
        let mut scratch: Vec<f32> = Vec::new();
        let err_cb = |e| eprintln!("audio output stream error: {e}");
        let stream_out = match out_cfg.sample_format() {
            cpal::SampleFormat::F32 => out_dev.build_output_stream(
                &out_cfg.into(),
                move |data: &mut [f32], _| {
                    out_cb_probe_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let frames = data.len() / out_channels;
                    scratch.resize(frames, 0.0);
                    playback_reader.pull(&mut scratch);
                    fanout(data, &scratch, out_channels);
                },
                err_cb,
                None,
            )?,
            other => anyhow::bail!("unsupported output sample format {other:?}"),
        };
        stream_out.play().context("start output stream")?;
        // -- capture callback: mixdown to mono f32, push into the ring --
        let mut capture_producer = capture_producer;
        let in_channels = in_cfg.channels() as usize;
        let err_cb = |e| eprintln!("audio input stream error: {e}");
        let stream_in = match in_cfg.sample_format() {
            cpal::SampleFormat::F32 => in_dev.build_input_stream(
                &in_cfg.into(),
                move |data: &[f32], _| {
                    capture_producer.push(&mixdown(data, in_channels));
                },
                err_cb,
                None,
            )?,
            cpal::SampleFormat::I16 => in_dev.build_input_stream(
                &in_cfg.into(),
                move |data: &[i16], _| {
                    let f: Vec<f32> = data.iter().map(|&v| v as f32 / 32768.0).collect();
                    capture_producer.push(&mixdown(&f, in_channels));
                },
                err_cb,
                None,
            )?,
            other => anyhow::bail!("unsupported input sample format {other:?}"),
        };
        stream_in.play().context("start input stream")?;

        Ok(Self {
            capture: capture_consumer,
            playback: playback_writer,
            input_sample_rate: in_rate,
            output_sample_rate: out_rate,
            guard: LiveStreamGuard {
                _input_stream: stream_in,
                _output_stream: stream_out,
                metrics,
                out_cb_probe: out_cb_probe.clone(),
            },
        })
    }

    /// Decompose into the runtime-thread halves plus the stream keep-alive
    /// guard. Dropping the guard stops both device streams.
    pub fn into_parts(self) -> (CaptureConsumer, PlaybackWriter, LiveStreamGuard, u32, u32) {
        (
            self.capture,
            self.playback,
            self.guard,
            self.input_sample_rate,
            self.output_sample_rate,
        )
    }
}

/// Average channels into mono (kept near the callback that uses it).
fn mixdown(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect()
}

/// Spread mono across output channels (duplicate to all).
fn fanout(out: &mut [f32], mono: &[f32], channels: usize) {
    if channels == 1 {
        out.copy_from_slice(mono);
        return;
    }
    for (frame, &m) in out.chunks_mut(channels).zip(mono) {
        frame.fill(m);
    }
}

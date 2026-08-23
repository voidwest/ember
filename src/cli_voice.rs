//! `ember voice` — live full-duplex audio (Phase 5 Track A).
//!
//! Compiled only with the `audio` cargo feature. Subcommands:
//!
//! * `--list-devices`: enumerate input/output devices via cpal.
//! * `--duplex-smoke <seconds>`: open default devices, run the duplex
//!   pipeline (capture ring → VAD, tone playback) for N seconds and print
//!   the telemetry that proves CONCURRENT operation: capture keeps
//!   delivering samples while playback is sounding.
//!
//! The realtime audio callbacks live on cpal's own threads and only touch
//! the lock-free rings; all model work stays on the runtime thread.

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};

use ember::duplex::{device::LiveDuplex, DuplexController, EnergyVad, TurnEvent};

#[derive(Parser)]
pub struct VoiceCommand {
    /// enumerate audio devices and exit
    #[arg(long)]
    pub list_devices: bool,
    /// run a concurrent capture+playback smoke test for N seconds
    #[arg(long)]
    pub duplex_smoke: Option<f64>,
}

pub fn run_voice_command(command: &VoiceCommand) -> Result<()> {
    if command.list_devices {
        return list_devices();
    }
    if let Some(seconds) = command.duplex_smoke {
        return duplex_smoke(seconds);
    }
    anyhow::bail!("nothing to do: pass --list-devices or --duplex-smoke <seconds>")
}

fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("host: {}", host.id().name());
    println!("-- input devices --");
    match host.input_devices() {
        Ok(devs) => {
            for (i, d) in devs.enumerate() {
                let name = d.name().unwrap_or_else(|e| format!("<err {e}>"));
                println!("  [{i}] {name}");
            }
        }
        Err(e) => println!("  <error enumerating: {e}>"),
    }
    println!("-- output devices --");
    match host.output_devices() {
        Ok(devs) => {
            for (i, d) in devs.enumerate() {
                let name = d.name().unwrap_or_else(|e| format!("<err {e}>"));
                println!("  [{i}] {name}");
            }
        }
        Err(e) => println!("  <error enumerating: {e}>"),
    }
    Ok(())
}

fn duplex_smoke(seconds: f64) -> Result<()> {
    let live =
        LiveDuplex::open_default().context("opening default audio devices for duplex smoke")?;
    let in_rate = live.input_sample_rate;
    let out_rate = live.output_sample_rate;
    println!("duplex open: input {in_rate} Hz, output {out_rate} Hz");
    let (capture, playback, guard, _in_rate, _out_rate) = live.into_parts();

    // controller at the DEVICE input rate; VAD sees device-rate PCM
    let mut ctl = DuplexController::new_with_sample_rate(
        capture,
        playback,
        Box::new(EnergyVad::new(in_rate)),
        in_rate,
    );

    let duration = std::time::Duration::from_secs_f64(seconds.max(0.5));
    let t0 = std::time::Instant::now();
    let mut last_tone = t0;
    let mut events: Vec<(std::time::Duration, TurnEvent)> = Vec::new();
    let mut full_rejections = 0usize;
    let phase = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Simulated assistant speech: 250 ms tones every second. In the real
    // loop these are TTS chunks; here they prove playback while capturing.
    ctl.set_assistant_active(true);

    while t0.elapsed() < duration {
        if let Some(e) = ctl.pump() {
            events.push((t0.elapsed(), e));
            match e {
                TurnEvent::SpeechStarted => {
                    println!(
                        "[{:>7.1} ms] SpeechStarted (assistant active -> barge-in latch: {})",
                        t0.elapsed().as_secs_f64() * 1e3,
                        ctl.is_barge_in()
                    );
                }
                other => println!("[{:>7.1} ms] {other:?}", t0.elapsed().as_secs_f64() * 1e3),
            }
        }

        // queue a tone burst every ~1 s ("assistant talking")
        if last_tone.elapsed().as_millis() > 1000 {
            last_tone = std::time::Instant::now();
            let rate = out_rate;
            let n = rate as usize / 4; // 250 ms
            let start = phase.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tone: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / rate as f32 + start as f32 * 0.25;
                    0.15 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect();
            let rejected = ctl.play_audio(&tone);
            if rejected > 0 {
                full_rejections += rejected;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    if full_rejections > 0 {
        println!("playback queue rejections: {full_rejections} samples (burst pacing)");
    }
    let (utterance_samples, utt_rate, utt_off) = ctl.take_utterance();
    println!("--- duplex smoke summary ---");
    println!(
        "capture accepted : {} samples @ {in_rate} Hz",
        ctl.captured_total()
    );
    println!("capture overruns : {}", ctl.capture.overruns());
    println!("capture dropped  : {} samples", ctl.dropped_samples());
    println!(
        "playback underruns/clears: {}/{}",
        guard.underruns(),
        guard.clears()
    );
    println!(
        "output callback pulls: {} (raw cb entries: {})",
        guard.output_pulls(),
        guard
            .out_cb_probe
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "turn events      : {:?}",
        events
            .iter()
            .map(|(d, e)| (d.as_millis(), *e))
            .collect::<Vec<_>>()
    );
    println!(
        "collected utterance: {} samples @ {utt_rate} Hz from offset {utt_off}",
        utterance_samples.len()
    );
    drop(guard); // stop the streams last
    Ok(())
}

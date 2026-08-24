//! `ember voice` — live full-duplex audio (Phase 5 Track A).
//!
//! Compiled only with the `audio` cargo feature. Subcommands:
//!
//! * `--list-devices`: enumerate input/output devices via cpal.
//! * `--duplex-smoke <seconds>`: open default devices, run the duplex
//!   pipeline (capture ring → VAD, tone playback) for N seconds and print
//!   the telemetry that proves CONCURRENT operation: capture keeps
//!   delivering samples while playback is sounding.
//! * `--converse <seconds>`: the full model-in-the-loop conversation loop —
//!   mic → streaming frontend → Ultravox → VoiceSession → LLM → OuteTTS →
//!   speaker, with live barge-in from concurrent microphone activity.
//!
//! The realtime audio callbacks live on cpal's own threads and only touch
//! the lock-free rings; all model work stays on the runtime thread.

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait};

use ember::duplex::device::LiveDuplex;
use ember::duplex::{DuplexController, EnergyVad, TurnEvent};
use ember::multimodal::converse::{ConverseConfig, ConverseEvent, VoiceConversation};
use ember::multimodal::VoiceSession;

#[derive(Parser)]
pub struct VoiceCommand {
    /// enumerate audio devices and exit
    #[arg(long)]
    pub list_devices: bool,
    /// run a concurrent capture+playback smoke test for N seconds
    #[arg(long)]
    pub duplex_smoke: Option<f64>,
    /// run the full model-in-the-loop voice conversation for N seconds
    #[arg(long)]
    pub converse: Option<f64>,

    // -- converse model paths (same conventions as `ember audio` / `ember tts`)
    /// text LLM GGUF (llama arch) for Ultravox
    #[arg(long)]
    pub text_gguf: Option<String>,
    /// ultravox audio mmproj GGUF
    #[arg(long)]
    pub audio_gguf: Option<String>,
    /// tokenizer.json of the text model
    #[arg(long)]
    pub tokenizer: Option<String>,
    /// OuteTTS speech GGUF (qwen2 arch)
    #[arg(long)]
    pub tts_gguf: Option<String>,
    /// OuteTTS tokenizer.json
    #[arg(long)]
    pub tts_tokenizer: Option<String>,
    /// WavTokenizer codec decoder GGUF
    #[arg(long)]
    pub codec: Option<String>,
    /// MMS-TTS (VITS) GGUF: use the Arabic-capable engine instead of OuteTTS
    #[arg(long)]
    pub vits_model: Option<String>,

    // -- converse behavior knobs
    /// user-turn prompt; <|audio|> binds streamed features
    #[arg(long, default_value = "<|audio|>")]
    pub prompt: String,
    /// max assistant reply tokens per turn
    #[arg(long, default_value_t = 96)]
    pub max_reply_tokens: usize,
    /// codec tokens per streamed TTS chunk
    #[arg(long, default_value_t = 24)]
    pub chunk_tokens: usize,
    /// minimum ms between provisional transcripts while capturing (0 = off)
    #[arg(long, default_value_t = 0)]
    pub partial_every_ms: u64,
    /// save each turn's user PCM + reply WAV files under this directory
    #[arg(long)]
    pub save_turns: Option<String>,
}

pub fn run_voice_command(command: &VoiceCommand) -> Result<()> {
    if command.list_devices {
        return list_devices();
    }
    if let Some(seconds) = command.duplex_smoke {
        return duplex_smoke(seconds);
    }
    if let Some(seconds) = command.converse {
        return converse(command, seconds);
    }
    anyhow::bail!(
        "nothing to do: pass --list-devices, --duplex-smoke <seconds> or --converse <seconds>"
    )
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

fn need<'a>(flag: &'a Option<String>, name: &str) -> Result<&'a String> {
    flag.as_ref()
        .ok_or_else(|| anyhow::anyhow!("--{name} is required for --converse"))
}

fn converse(command: &VoiceCommand, seconds: f64) -> Result<()> {
    let text_gguf = need(&command.text_gguf, "text-gguf")?;
    let audio_gguf = need(&command.audio_gguf, "audio-gguf")?;
    let tokenizer = need(&command.tokenizer, "tokenizer")?;

    println!("loading models…");
    println!("  text   : {text_gguf}");
    println!("  audio  : {audio_gguf}");
    let backend = ember::backend::CpuBackend;
    let ultravox = ember::ultravox::Ultravox::from_ggufs(
        std::path::Path::new(text_gguf),
        std::path::Path::new(audio_gguf),
    )
    .context("loading ultravox")?;
    let tok = ember::tokenizer::EmberTokenizer::from_file(std::path::Path::new(tokenizer))
        .context("loading tokenizer")?;

    // Speech engine: MMS-VITS (Arabic-capable) when given; else OuteTTS.
    enum Engines {
        Oute(Box<ember::tts::outetts::OuteTts>),
        Mms(Box<ember::tts::vits::MmsVits>),
    }
    let _engines;
    let tts: &dyn ember::tts::SpeechOut = if let Some(v) = &command.vits_model {
        println!("  tts    : {v} (mms-vits)");
        _engines = Engines::Mms(Box::new(
            ember::tts::vits::MmsVits::from_gguf(std::path::Path::new(v))
                .context("loading mms-vits")?,
        ));
        match &_engines {
            Engines::Mms(m) => &**m as &dyn ember::tts::SpeechOut,
            _ => unreachable!(),
        }
    } else {
        let tts_gguf = need(&command.tts_gguf, "tts-gguf")?;
        let tts_tokenizer = need(&command.tts_tokenizer, "tts-tokenizer")?;
        let codec = need(&command.codec, "codec")?;
        println!("  tts    : {tts_gguf}");
        println!("  codec  : {codec}");
        _engines = Engines::Oute(Box::new(
            ember::tts::outetts::OuteTts::from_gguf(
                std::path::Path::new(tts_gguf),
                std::path::Path::new(tts_tokenizer),
                std::path::Path::new(codec),
                ember::quant_k::KStrategy::Auto,
            )
            .context("loading tts")?,
        ));
        match &_engines {
            Engines::Oute(o) => &**o as &dyn ember::tts::SpeechOut,
            _ => unreachable!(),
        }
    };

    let live =
        LiveDuplex::open_default().context("opening default audio devices for conversation")?;
    let in_rate = live.input_sample_rate;
    let out_rate = live.output_sample_rate;
    println!("devices: input {in_rate} Hz, output {out_rate} Hz");
    let (capture, playback, guard, _in_rate, _out_rate) = live.into_parts();

    let session = VoiceSession::new(&ultravox, &backend, &tok, 2048, 64 * 1024 * 1024)
        .context("creating voice session")?;
    let config = ConverseConfig {
        prompt: command.prompt.clone(),
        max_reply_tokens: command.max_reply_tokens,
        chunk_tokens: command.chunk_tokens,
        partial_every_ms: command.partial_every_ms,
        partial_max_tokens: 24,
    };
    let detector = Box::new(EnergyVad::new(in_rate));
    let ctl = DuplexController::new_with_sample_rate(capture, playback, detector, in_rate);
    let mut conv = VoiceConversation::new(session, tts, ctl, config);

    let saver = command.save_turns.as_deref().map(std::path::PathBuf::from);
    if let Some(dir) = &saver {
        std::fs::create_dir_all(dir).context("create --save-turns dir")?;
    }

    println!(
        "listening ({} s budget; Ctrl-C to stop). Speak after the prompt.",
        seconds
    );
    let t0 = std::time::Instant::now();
    let duration = std::time::Duration::from_secs_f64(seconds.max(1.0));
    let mut turn_index = 0usize;
    while t0.elapsed() < duration && conv.turns_completed() < 4096 {
        for event in conv.pump() {
            print_event(event, &saver, &mut turn_index);
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let stats = conv.session.stats();
    let ctl = conv.duplex.borrow();
    println!("--- conversation summary ---");
    println!("turns completed : {}", conv.turns_completed());
    println!(
        "user/assistant  : {}/{} committed, {} cancelled replies",
        stats.user_turns_committed,
        stats.assistant_replies_completed,
        stats.assistant_replies_cancelled
    );
    println!("kv cursor       : {} tokens", conv.session.committed_len());
    println!(
        "provisional     : {} pulses, {:.0} ms total",
        stats.provisional_transcripts, stats.provisional_ms
    );
    println!(
        "media features  : {} hits / {} misses",
        stats.media_feature_hits, stats.media_feature_misses
    );
    println!(
        "capture         : {} accepted, {} dropped ({})",
        ctl.captured_total(),
        ctl.dropped_samples(),
        if ctl.dropped_samples() > 0 {
            "OVERRUNS — runtime too slow for this device"
        } else {
            "clean"
        }
    );
    println!(
        "playback        : underruns {}, clears {}, output pulls {}",
        guard.underruns(),
        guard.clears(),
        guard.output_pulls()
    );
    drop(guard); // stop the streams last
    Ok(())
}

fn print_event(event: ConverseEvent, saver: &Option<std::path::PathBuf>, turn_index: &mut usize) {
    match event {
        ConverseEvent::SpeechStarted => println!("🎤 listening…"),
        ConverseEvent::PartialTranscript { text } => {
            println!("   · partial: {text}")
        }
        ConverseEvent::UserCommitted { audio_seconds } => {
            println!("✓ user turn committed ({audio_seconds:.1} s audio)")
        }
        ConverseEvent::ReplyTextDelta { piece } => {
            use std::io::Write;
            print!("{piece}");
            let _ = std::io::stdout().flush();
        }
        ConverseEvent::AssistantAudioStart { samples } => {
            println!();
            println!("🔊 speaking ({samples} samples queued so far)…")
        }
        ConverseEvent::BargeIn { during_generation } => {
            println!();
            println!(
                "⛔ barge-in ({})",
                if during_generation {
                    "generation cancelled + rolled back"
                } else {
                    "playback dropped, reply kept"
                }
            );
        }
        ConverseEvent::TurnComplete {
            reply_text,
            end,
            timings,
        } => {
            *turn_index += 1;
            println!(
                "— turn {}: {:?} | reply \"{}\" | utt {:.1}s → token {:.0} ms → audio {:.0} ms → done {:.0} ms | codes {}",
                *turn_index,
                end,
                reply_text.trim(),
                timings.utterance_seconds,
                timings.end_to_first_token_ms,
                timings.end_to_first_audio_ms,
                timings.end_to_turn_done_ms,
                timings.reply_codes
            );
            if let (Some(dir), true) = (saver, !reply_text.trim().is_empty()) {
                save_turn(dir, *turn_index, &reply_text);
            }
        }
    }
}

/// Persist the assistant reply text for evidence collection. (User PCM of a
/// live turn is not retained by the runtime; see the report for why.)
fn save_turn(dir: &std::path::Path, index: usize, reply_text: &str) {
    let path = dir.join(format!("turn_{index:03}.txt"));
    if let Err(e) = std::fs::write(path, reply_text) {
        eprintln!("save-turn failed: {e}");
    }
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

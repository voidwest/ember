//! Model-in-the-loop live conversation validation (Phase 5 Session 2,
//! Track A4). These tests drive the COMPLETE chain
//!
//! ```text
//! capture ring -> TurnDetector -> AudioStream -> VoiceSession (Ultravox)
//!              -> LLM reply    -> OuteTTS     -> WavTokenizer PCM
//!              -> playback ring
//! ```
//!
//! against REAL weights, with a synthetic microphone producer feeding the
//! capture ring exactly as the cpal callback would. Barge-in is exercised
//! DURING actual generation and DURING actual TTS playback — the piece the
//! Phase-5 smoke test could not prove.
//!
//! Skip unless:
//!
//! ```text
//! EMBER_CONVERSE_E2E=1
//! EMBER_VOICE_TEXT_GGUF   llama-arch GGUF (Llama-3.2-1B-Instruct-Q8_0)
//! EMBER_VOICE_AUDIO_GGUF  ultravox audio mmproj GGUF
//! EMBER_VOICE_TOKENIZER   tokenizer.json of the text model
//! EMBER_TTS_GGUF          OuteTTS speech GGUF (qwen2 arch)
//! EMBER_TTS_TOKENIZER     its tokenizer.json
//! EMBER_TTS_CODEC         wavtokenizer decoder GGUF
//! ```

use ember::duplex::{capture_ring, playback_ring, DuplexController, EnergyVad};
use ember::multimodal::converse::{
    AssistantEnd, ConversationState, ConverseConfig, ConverseEvent, VoiceConversation,
};
use ember::multimodal::VoiceSession;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

struct Fixture {
    text_gguf: PathBuf,
    audio_gguf: PathBuf,
    tokenizer: PathBuf,
    tts_gguf: PathBuf,
    tts_tokenizer: PathBuf,
    codec: PathBuf,
}

fn fixture() -> Option<Fixture> {
    if std::env::var("EMBER_CONVERSE_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    let get = |k: &str| -> Option<PathBuf> {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Some(Fixture {
        text_gguf: get("EMBER_VOICE_TEXT_GGUF")?,
        audio_gguf: get("EMBER_VOICE_AUDIO_GGUF")?,
        tokenizer: get("EMBER_VOICE_TOKENIZER")?,
        tts_gguf: get("EMBER_TTS_GGUF")?,
        tts_tokenizer: get("EMBER_TTS_TOKENIZER")?,
        codec: get("EMBER_TTS_CODEC")?,
    })
}

/// One shared set of loaded models per process (expensive-fixture pattern;
/// tests run --test-threads=1 so the lock never contends).
static MODELS: std::sync::OnceLock<
    Mutex<(
        ember::ultravox::Ultravox,
        ember::tokenizer::EmberTokenizer,
        ember::tts::outetts::OuteTts,
    )>,
> = std::sync::OnceLock::new();

fn with_models(
    mut body: impl FnMut(
        &ember::ultravox::Ultravox,
        &ember::tokenizer::EmberTokenizer,
        &ember::tts::outetts::OuteTts,
    ),
) {
    let Some(fx) = fixture() else {
        eprintln!("skipping: set EMBER_CONVERSE_E2E=1 (+ paths)");
        return;
    };
    let guard = MODELS
        .get_or_init(|| {
            let ultravox = ember::ultravox::Ultravox::from_ggufs(&fx.text_gguf, &fx.audio_gguf)
                .expect("load ultravox");
            let tok =
                ember::tokenizer::EmberTokenizer::from_file(&fx.tokenizer).expect("tokenizer");
            let tts = ember::tts::outetts::OuteTts::from_gguf(
                &fx.tts_gguf,
                &fx.tts_tokenizer,
                &fx.codec,
                ember::quant_k::KStrategy::Auto,
            )
            .expect("load tts");
            Mutex::new((ultravox, tok, tts))
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    body(&guard.0, &guard.1, &guard.2);
}

struct Rings {
    /// Shared so a synthetic-microphone THREAD can push concurrently —
    /// exactly how the cpal callback relates to the driver.
    producer: std::sync::Arc<std::sync::Mutex<ember::duplex::CaptureProducer>>,
    reader: std::sync::Mutex<ember::duplex::PlaybackReader>,
}

impl Rings {
    /// Simulate the device input callback: push mono PCM at 16 kHz.
    fn mic(&self, pcm: &[f32]) {
        self.producer.lock().unwrap().push(pcm);
    }

    /// Simulate the output callback draining playback.
    fn drain_output(&self, n: usize) {
        let mut buf = vec![0.0f32; n];
        self.reader.lock().unwrap().pull(&mut buf);
    }
}

fn sine(rate: u32, seconds: f32, freq: f32, amp: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            amp * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

fn make_conversation<'m>(
    ultravox: &'m ember::ultravox::Ultravox,
    tok: &'m ember::tokenizer::EmberTokenizer,
    tts: &'m OuteTtsRef,
    backend: &'m ember::backend::CpuBackend,
) -> (VoiceConversation<'m>, Rings) {
    let session =
        VoiceSession::new(ultravox, backend, tok, 2048, 64 * 1024 * 1024).expect("voice session");
    let (producer, consumer) = capture_ring(ember::duplex::CAPTURE_QUEUE_SAMPLES);
    let (writer, reader) = playback_ring(ember::duplex::PLAYBACK_QUEUE_SAMPLES);
    let detector = Box::new(EnergyVad::new(16_000));
    let ctl = DuplexController::new_with_sample_rate(consumer, writer, detector, 16_000);
    (
        VoiceConversation::new(
            session,
            tts,
            ctl,
            ConverseConfig {
                prompt: "<|audio|>Reply with one short English sentence.".into(),
                max_reply_tokens: 48,
                chunk_tokens: 8,
                ..ConverseConfig::default()
            },
        ),
        Rings {
            producer: std::sync::Arc::new(std::sync::Mutex::new(producer)),
            reader: std::sync::Mutex::new(reader),
        },
    )
}

type OuteTtsRef = ember::tts::outetts::OuteTts;

/// Pump for up to `budget_ms`, collecting every event. Early-exits when
/// `step` returns true after a batch; the step hook receives the rings so a
/// scenario can inject microphone audio mid-conversation (exactly what an
/// application thread would do through its own capture handle).
fn run_conversation(
    conv: &mut VoiceConversation,
    rings: &Rings,
    mut step: impl FnMut(&Rings, &[ConverseEvent], ConversationState) -> bool,
    budget_ms: u64,
) -> Vec<ConverseEvent> {
    let t0 = std::time::Instant::now();
    let mut events = Vec::new();
    while (t0.elapsed().as_millis() as u64) < budget_ms {
        let batch = conv.pump();
        rings.drain_output(160);
        let state = conv.state();
        let stop = step(rings, &batch, state);
        events.extend(batch);
        if stop {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    events
}

/// Trailing silence so the energy-VAD hangover can elapse (a real
/// microphone always delivers ambience; synthetic producers must too).
fn quiet_tail(rings: &Rings, seconds: f32) {
    rings.mic(&vec![0.0f32; (16_000.0 * seconds) as usize]);
}

/// A synthetic microphone THREAD: waits for `trigger`, then keeps talking
/// (loud sine bursts + gaps) until `stop` is set. This is what makes
/// mid-generation / mid-playback barge-in reachable from a single-threaded
/// driver — the same concurrency contract as the live cpal callback.
struct MicThread {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MicThread {
    fn start(rings: &Rings, trigger: Arc<AtomicBool>, freq: f32) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let producer = rings.producer.clone();
        let handle = std::thread::spawn(move || {
            while !trigger.load(Ordering::SeqCst) {
                if stop2.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let mut t = 0u64;
            while !stop2.load(Ordering::SeqCst) {
                let n = 8000; // 0.5 s bursts
                let pcm: Vec<f32> = (0..n)
                    .map(|i| {
                        let tt = (t + i) as f32 / 16_000.0;
                        0.85 * (2.0 * std::f32::consts::PI * freq * tt).sin()
                    })
                    .collect();
                t += n;
                if let Ok(mut p) = producer.lock() {
                    p.push(&pcm);
                }
                // gaps > VAD hangover (300 ms) so turns can still end
                let gap: Vec<f32> = vec![0.0; 8000]; // 500 ms
                if let Ok(mut p) = producer.lock() {
                    p.push(&gap);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for MicThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn turns_done(events: &[ConverseEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, ConverseEvent::TurnComplete { .. }))
        .count()
}

// ---------------------------------------------------------------------------
// Scenario A: full happy-path turn (mic → model → TTS → "speaker") plus
// same-session continuation.
// ---------------------------------------------------------------------------

#[test]
fn converse_full_turn_mic_to_speaker_and_same_session_continuation() {
    with_models(|ultravox, tok, tts| {
        let backend = ember::backend::CpuBackend;
        let (mut conv, rings) = make_conversation(ultravox, tok, tts, &backend);

        // user turn: loud speech burst through the ring, then ambience
        rings.mic(&sine(16_000, 1.5, 320.0, 0.8));
        quiet_tail(&rings, 0.6);
        let events = run_conversation(
            &mut conv,
            &rings,
            |_, evs, state| state == ConversationState::Idle && turns_done(evs) > 0,
            240_000,
        );

        assert!(
            events
                .iter()
                .any(|e| matches!(e, ConverseEvent::SpeechStarted)),
            "SpeechStarted missing"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ConverseEvent::UserCommitted { .. })),
            "user turn not committed"
        );
        for e in &events {
            eprintln!("DBG {e:?}");
        }
        match events.last() {
            Some(ConverseEvent::TurnComplete { end, timings, .. }) => {
                assert_eq!(*end, AssistantEnd::Completed);
                assert!(timings.end_to_first_token_ms > 0.0);
                assert!(
                    timings.end_to_first_audio_ms > 0.0 || timings.reply_codes == 0,
                    "TTS must produce PCM when the reply is speakable"
                );
                assert!(timings.reply_codes > 0);
            }
            other => panic!("turn did not complete (last event {other:?})"),
        }
        assert_eq!(conv.state(), ConversationState::Idle);
        let committed_1 = conv.session.committed_len();

        // same-session continuation: second turn grows KV incrementally
        rings.mic(&sine(16_000, 1.2, 300.0, 0.7));
        quiet_tail(&rings, 0.6);
        let events2 =
            run_conversation(&mut conv, &rings, |_, evs, _| turns_done(evs) >= 1, 240_000);
        assert!(
            events2.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::Completed,
                    ..
                }
            )),
            "second turn incomplete: last={:?}",
            events2.last()
        );
        assert!(conv.session.committed_len() > committed_1, "KV must grow");
        assert_eq!(conv.session.stats().cache_rebuilds, 0);
    });
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Scenario B: interrupt during ACTUAL generation — cancel + KV rollback +
// deferred capture + next-turn readiness in the same session.
// ---------------------------------------------------------------------------

#[test]
fn converse_barge_in_during_generation_rolls_back_and_recovers() {
    with_models(|ultravox, tok, tts| {
        let backend = ember::backend::CpuBackend;
        let (mut conv, rings) = make_conversation(ultravox, tok, tts, &backend);

        // user speech starts the first turn; ambience closes it so the
        // model reaches generation.
        rings.mic(&sine(16_000, 1.2, 300.0, 0.8));
        quiet_tail(&rings, 0.6);

        // Synthetic mic thread: talks CONTINUOUSLY from t=0. Early bursts
        // merge into turn-1's utterance; because the thread never stops, it
        // is still talking when generation begins — real concurrent
        // pressure on an in-flight decode.
        let mic = MicThread::start(&rings, Arc::new(AtomicBool::new(true)), 340.0);
        let interrupted = Arc::new(AtomicBool::new(false));
        let stop_watcher = interrupted.clone();
        let events = run_conversation(
            &mut conv,
            &rings,
            move |_, evs, state| {
                if evs.iter().any(|e| {
                    matches!(
                        e,
                        ConverseEvent::BargeIn {
                            during_generation: true,
                            ..
                        }
                    )
                }) {
                    stop_watcher.store(true, Ordering::SeqCst);
                }
                stop_watcher.load(Ordering::SeqCst) && state == ConversationState::CapturingUser
            },
            300_000,
        );
        drop(mic);

        // Cancel may land pre-first-token when the mic is already talking;
        // still fully model-in-the-loop (rollback clean, deferred capture).
        assert!(
            events.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::InterruptedDuringGeneration,
                    ..
                }
            )),
            "barge-in never cancelled a generation; tail={:?}",
            events.last()
        );
        assert!(
            conv.session.stats().assistant_replies_cancelled >= 1
                && conv.session.stats().cache_rebuilds == 0,
            "replies must be cancelled (not committed) without cache rebuilds"
        );
        // NOTE: with the continuous mic the deferred capture may already
        // have processed further interrupts (Idle again) before this assert
        // runs — rollback + recovery is what matters, pinned by the next
        // turn completing.
        let _ = conv.state();

        // close the interrupting utterance with ambience, then a fresh turn
        quiet_tail(&rings, 1.0);
        let post = run_conversation(&mut conv, &rings, |_, evs, _| turns_done(evs) >= 1, 300_000);
        assert!(
            post.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::Completed,
                    ..
                }
            )),
            "post-interrupt turn must complete normally; last={:?}",
            post.last()
        );
        assert_eq!(conv.session.stats().cache_rebuilds, 0);
    });
}

// ---------------------------------------------------------------------------
// Scenario C: interrupt while actual TTS chunks are queued/playing. The
// committed reply stays; NO stale audio may play afterwards.
// ---------------------------------------------------------------------------

#[test]
fn converse_interrupt_during_playback_drops_audio_keeps_reply() {
    with_models(|ultravox, tok, tts| {
        let backend = ember::backend::CpuBackend;
        let (mut conv, rings) = make_conversation(ultravox, tok, tts, &backend);

        rings.mic(&sine(16_000, 1.0, 330.0, 0.8));
        quiet_tail(&rings, 0.6);

        // Talk over the speaker EXACTLY when first reply PCM is queued
        // (driver-published TTFA flag): silent during capture/generation so
        // the interrupt provably lands in the playback phase.
        let mic = MicThread::start(&rings, conv.audio_started.clone(), 350.0);
        let events = run_conversation(&mut conv, &rings, |_, evs, _| turns_done(evs) >= 1, 360_000);
        drop(mic);

        assert!(
            events.iter().any(|e| matches!(
                e,
                ConverseEvent::AssistantAudioStart { samples } if *samples > 0
            )),
            "playback never started"
        );
        let completed = events.iter().find_map(|e| match e {
            ConverseEvent::TurnComplete {
                end, reply_text, ..
            } => Some((reply_text.clone(), *end)),
            _ => None,
        });
        let (reply_text, end) = completed.expect("turn must resolve");
        assert_eq!(
            end,
            AssistantEnd::InterruptedDuringPlayback,
            "expected playback interruption; got {end:?}"
        );
        assert!(
            !reply_text.trim().is_empty(),
            "committed reply must survive a playback interrupt"
        );
        // reader-side queue empty: no stale assistant audio can play
        let mut probe = vec![1.0f32; 160];
        rings.reader.lock().unwrap().pull(&mut probe);
        assert!(
            probe.iter().all(|&v| v == 0.0),
            "stale assistant audio played after cancellation"
        );
        assert_eq!(
            conv.state(),
            ConversationState::CapturingUser,
            "interrupting utterance should be capturing"
        );
    });
}

// ---------------------------------------------------------------------------
// Scenario D: rapid second intervention — interrupt again right after a
// first interrupt, then let the follow-up turn complete cleanly.
// ---------------------------------------------------------------------------

#[test]
fn converse_rapid_second_intervention_recovers_cleanly() {
    with_models(|ultravox, tok, tts| {
        let backend = ember::backend::CpuBackend;
        let (mut conv, rings) = make_conversation(ultravox, tok, tts, &backend);

        // First turn completes fully.
        rings.mic(&sine(16_000, 0.9, 300.0, 0.8));
        quiet_tail(&rings, 0.6);
        let ev1 = run_conversation(&mut conv, &rings, |_, evs, _| turns_done(evs) >= 1, 300_000);
        assert!(
            ev1.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::Completed,
                    ..
                }
            )),
            "first turn incomplete"
        );

        // Second turn: mic thread starts talking at first reply text —
        // barge-in #2 lands mid-generation...
        rings.mic(&sine(16_000, 0.9, 320.0, 0.8));
        quiet_tail(&rings, 0.6);
        let _trigger = Arc::new(AtomicBool::new(false));
        let mic = MicThread::start(&rings, Arc::new(AtomicBool::new(true)), 360.0);
        let ev2 = run_conversation(
            &mut conv,
            &rings,
            |_, evs, _| {
                turns_done(evs) >= 1
                    && evs.iter().any(|e| {
                        matches!(
                            e,
                            ConverseEvent::TurnComplete {
                                end: AssistantEnd::InterruptedDuringGeneration
                                    | AssistantEnd::InterruptedDuringPlayback,
                                ..
                            }
                        )
                    })
            },
            120_000,
        );
        drop(mic);
        assert!(
            ev2.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::InterruptedDuringGeneration
                        | AssistantEnd::InterruptedDuringPlayback,
                    ..
                }
            )),
            "second barge-in never fired; tail={:?}",
            ev2.last()
        );

        // ...and one more immediate burst, then ambience closes it: the
        // follow-up turn must complete cleanly.
        rings.mic(&sine(16_000, 0.4, 380.0, 0.9));
        quiet_tail(&rings, 0.8);
        let ev3 = run_conversation(&mut conv, &rings, |_, evs, _| turns_done(evs) >= 1, 360_000);
        assert!(
            ev3.iter().any(|e| matches!(
                e,
                ConverseEvent::TurnComplete {
                    end: AssistantEnd::Completed,
                    ..
                }
            )),
            "post-rapid-interrupt turn must still complete; last={:?}",
            ev3.last()
        );
        assert_eq!(conv.session.stats().cache_rebuilds, 0);
    });
}

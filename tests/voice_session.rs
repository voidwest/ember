//! End-to-end voice-session validation (Phase 4 session 2).
//!
//! These tests drive the full interactive loop — persistent session,
//! streaming audio input, multiturn KV reuse, cancellation rollback and
//! barge-in policy — against REAL weights. They skip silently unless:
//!
//! ```text
//! EMBER_VOICE_E2E=1
//! EMBER_VOICE_TEXT_GGUF   llama-3.2-1b Q8_0 (or any llama-arch GGUF)
//! EMBER_VOICE_AUDIO_GGUF  ultravox audio mmproj GGUF
//! EMBER_VOICE_TOKENIZER   tokenizer.json for the text model
//! ```
//!
//! Hermetic `cargo test` runs skip them; the session-2 report records the
//! executed run against Llama-3.2-1B-Instruct-Q8_0 + ultravox audio tower f32.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct VoiceFixture {
    text_gguf: PathBuf,
    audio_gguf: PathBuf,
    tokenizer: PathBuf,
}

fn fixture() -> Option<VoiceFixture> {
    if std::env::var("EMBER_VOICE_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    let get = |k: &str| -> Option<PathBuf> {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Some(VoiceFixture {
        text_gguf: get("EMBER_VOICE_TEXT_GGUF")?,
        audio_gguf: get("EMBER_VOICE_AUDIO_GGUF")?,
        tokenizer: get("EMBER_VOICE_TOKENIZER")?,
    })
}

/// One shared loaded model for the whole test binary (sessions borrow it).
/// The wrapper is not Sync (interior mutability), hence the Mutex — this is
/// the standard expensive-fixture pattern; nothing here leaks into
/// production code paths. Tests run with --test-threads=1 so the lock never
/// contends in practice.
static MODEL: std::sync::OnceLock<
    std::sync::Mutex<(ember::ultravox::Ultravox, ember::tokenizer::EmberTokenizer)>,
> = std::sync::OnceLock::new();

fn model_and_tokenizer() -> Option<
    std::sync::MutexGuard<'static, (ember::ultravox::Ultravox, ember::tokenizer::EmberTokenizer)>,
> {
    use std::sync::Mutex;
    let fx = fixture()?;
    Some(
        MODEL
            .get_or_init(|| {
                let model = ember::ultravox::Ultravox::from_ggufs(&fx.text_gguf, &fx.audio_gguf)
                    .expect("load ultravox");
                let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&fx.tokenizer)
                    .expect("load tokenizer");
                Mutex::new((model, tokenizer))
            })
            .lock()
            .unwrap_or_else(|e| e.into_inner()),
    )
}

fn signal_1s() -> Vec<f32> {
    // deterministic speech-ish burst: 1 s of decaying sinusoid mix
    (0..16_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin() * (-t * 1.5).exp()
                + 0.15 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
        })
        .collect()
}

#[test]
fn multiturn_session_reuses_kv_without_rebuild() {
    let Some(guard) = model_and_tokenizer() else {
        eprintln!("skipping: set EMBER_VOICE_E2E=1 (+ paths)");
        return;
    };
    let (model, tokenizer) = (&guard.0, &guard.1);
    let backend = ember::backend::CpuBackend;
    let mut session =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 1024, 64 * 1024 * 1024)
            .expect("session");
    let sys_len = session.committed_len();
    assert!(sys_len > 0, "system prefix must be committed at creation");

    // turn 1: text-only user turn + reply
    session.begin_user_turn();
    session
        .set_turn_prompt("Say hello in one short word.".to_string())
        .unwrap();
    let (span1, _) = session.commit_user_turn().unwrap();
    assert_eq!(span1.0, sys_len);

    let control = ember::multimodal::GenerationControl::new();
    let (text1, cancelled) = session
        .generate_reply(&control, 24, |_| {}, || false)
        .unwrap();
    assert!(!cancelled);
    assert!(!text1.is_empty());

    // turn 2: KV must grow incrementally — no re-prefill of turn 1
    let after_turn1 = session.committed_len();
    session.begin_user_turn();
    session
        .set_turn_prompt("Now say goodbye briefly.".to_string())
        .unwrap();
    let (span2, _) = session.commit_user_turn().unwrap();
    assert_eq!(
        span2.0, after_turn1,
        "turn 2 prefill starts exactly at the boundary"
    );

    let stats = session.stats();
    assert_eq!(stats.cache_rebuilds, 0);
    assert!(stats.prefilled_tokens > 0);
}

#[test]
fn cancellation_rolls_back_kv_and_keeps_history_metadata() {
    let Some(guard) = model_and_tokenizer() else {
        eprintln!("skipping: set EMBER_VOICE_E2E=1 (+ paths)");
        return;
    };
    let (model, tokenizer) = (&guard.0, &guard.1);
    let backend = ember::backend::CpuBackend;
    let mut session =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 1024, 64 * 1024 * 1024)
            .expect("session");
    session.begin_user_turn();
    session
        .set_turn_prompt("Count from one to ten slowly.".to_string())
        .unwrap();
    session.commit_user_turn().unwrap();
    let boundary = session.committed_len();

    // cancel before the first checkpoint: nothing may be generated/committed
    let control = ember::multimodal::GenerationControl::new();
    control.cancel();
    let (text, cancelled) = session
        .generate_reply(&control, 32, |_| {}, || false)
        .unwrap();
    assert!(cancelled);
    assert!(text.is_empty());
    assert_eq!(
        session.committed_len(),
        boundary,
        "KV cursor must roll back to the pre-generation boundary"
    );
    let turns = session.turns();
    assert_eq!(
        turns.last().unwrap().state,
        ember::multimodal::TurnState::Cancelled
    );
    assert_eq!(turns.last().unwrap().live_tokens(), 0);

    // the session remains valid: a follow-up turn works and appends cleanly
    session.begin_user_turn();
    session
        .set_turn_prompt("Say one word.".to_string())
        .unwrap();
    let (span, _) = session.commit_user_turn().unwrap();
    assert_eq!(span.0, boundary, "new turn fills the rolled-back region");

    let control = ember::multimodal::GenerationControl::new();
    let (_, cancelled2) = session
        .generate_reply(&control, 16, |_| {}, || false)
        .unwrap();
    assert!(!cancelled2);
}

#[test]
fn barge_in_during_generation_cancels_reply() {
    let Some(guard) = model_and_tokenizer() else {
        eprintln!("skipping: set EMBER_VOICE_E2E=1 (+ paths)");
        return;
    };
    let (model, tokenizer) = (&guard.0, &guard.1);
    let backend = ember::backend::CpuBackend;
    let mut session =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 1024, 64 * 1024 * 1024)
            .expect("session");
    session.begin_user_turn();
    session
        .set_turn_prompt("Tell me a very long story.".to_string())
        .unwrap();
    session.commit_user_turn().unwrap();
    let boundary = session.committed_len();

    // barge-in fires immediately: reply must cancel, nothing committed
    let control = ember::multimodal::GenerationControl::new();
    let (_, cancelled) = session
        .generate_reply(&control, 48, |_| {}, || true)
        .unwrap();
    assert!(cancelled);
    assert_eq!(session.committed_len(), boundary);
}

#[test]
fn streaming_audio_through_session_matches_static_commit() {
    let Some(guard) = model_and_tokenizer() else {
        eprintln!("skipping: set EMBER_VOICE_E2E=1 (+ paths)");
        return;
    };
    let (model, tokenizer) = (&guard.0, &guard.1);
    let backend = ember::backend::CpuBackend;
    let sig = signal_1s();

    // streamed path
    let mut session =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 1024, 64 * 1024 * 1024)
            .expect("session");
    session.begin_user_turn();
    session
        .open_streaming_audio(ember::multimodal::stream::AudioStreamConfig::default())
        .unwrap();
    for chunk in sig.chunks(1600) {
        session.push_streaming_audio(chunk).unwrap();
        session.update_stream_encoder(false).unwrap();
    }
    let rows = session.finalize_streaming_audio().unwrap();
    assert!(rows > 0);
    session
        .set_turn_prompt("<|audio|>What did you hear?".to_string())
        .unwrap();
    session.commit_user_turn().unwrap();
    let control = ember::multimodal::GenerationControl::new();
    let (reply_streamed, cancelled) = session
        .generate_reply(&control, 24, |_| {}, || false)
        .unwrap();
    assert!(!cancelled);
    assert!(!reply_streamed.is_empty());

    // static path with identical PCM must produce the same reply
    let mut session2 =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 1024, 64 * 1024 * 1024)
            .expect("session 2");
    session2.begin_user_turn();
    session2
        .attach_static_audio(&ember::multimodal::audio::AudioInput::Samples {
            data: sig.clone(),
            sample_rate: 16_000,
        })
        .unwrap();
    session2
        .set_turn_prompt("<|audio|>What did you hear?".to_string())
        .unwrap();
    session2.commit_user_turn().unwrap();
    let control2 = ember::multimodal::GenerationControl::new();
    let (reply_static, _) = session2
        .generate_reply(&control2, 24, |_| {}, || false)
        .unwrap();
    assert_eq!(
        reply_streamed, reply_static,
        "streamed vs static audio turn must produce identical greedy replies"
    );
}

#[test]
fn provisional_transcript_never_touches_committed_state() {
    let Some(guard) = model_and_tokenizer() else {
        eprintln!("skipping: set EMBER_VOICE_E2E=1 (+ paths)");
        return;
    };
    let (model, tokenizer) = (&guard.0, &guard.1);
    let backend = ember::backend::CpuBackend;
    let mut session =
        ember::multimodal::VoiceSession::new(model, &backend, tokenizer, 2048, 64 * 1024 * 1024)
            .expect("session");
    session.begin_user_turn();
    session.open_streaming_audio(Default::default()).unwrap();

    let counter = Arc::new(AtomicUsize::new(0));
    let c2 = counter.clone();
    let sig = signal_1s();
    for chunk in sig.chunks(3200) {
        session.push_streaming_audio(chunk).unwrap();
        session.update_stream_encoder(true).unwrap();
        c2.fetch_add(1, Ordering::Relaxed);
        let before = session.committed_len();
        let t_before = session.stats().provisional_transcripts;
        let _ = session.provisional_transcript(8).unwrap();
        assert_eq!(session.committed_len(), before, "committed boundary frozen");
        assert_eq!(session.stats().provisional_transcripts, t_before + 1);
    }
    assert!(counter.load(Ordering::Relaxed) > 0);
    assert!(session.stats().provisional_ms >= 0.0);
}

// ---------------------------------------------------------------------------
// Track F: streaming speech output — TTFA and honest deviation accounting
// ---------------------------------------------------------------------------

fn tts_fixture() -> Option<(PathBuf, PathBuf, PathBuf)> {
    if std::env::var("EMBER_TTS_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    let get = |k: &str| -> Option<PathBuf> {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Some((
        get("EMBER_TTS_GGUF")?,
        get("EMBER_TTS_TOKENIZER")?,
        get("EMBER_TTS_CODEC")?,
    ))
}

static TTS_MODEL: std::sync::OnceLock<std::sync::Mutex<ember::tts::outetts::OuteTts>> =
    std::sync::OnceLock::new();

#[test]
fn streaming_synthesis_ttfa_and_chunk_deviation() {
    let Some((gguf, tokenizer, codec)) = tts_fixture() else {
        eprintln!("skipping: set EMBER_TTS_E2E=1 (+ EMBER_TTS_GGUF/TOKENIZER/CODEC)");
        return;
    };
    let backend = ember::backend::CpuBackend;
    let guard = TTS_MODEL.get_or_init(|| {
        std::sync::Mutex::new(
            ember::tts::outetts::OuteTts::from_gguf(
                &gguf,
                &tokenizer,
                &codec,
                ember::quant_k::KStrategy::Auto,
            )
            .expect("load outetts"),
        )
    });
    let tts = guard.lock().unwrap_or_else(|e| e.into_inner());

    let text = "Streaming audio arrives before the words are finished.";
    let t0 = std::time::Instant::now();
    let mut ttfa_ms: f64 = 0.0;
    let mut streamed_len = 0usize;
    let mut n_chunks = 0usize;

    // collect the streamed concatenation AND the final single-pass decode
    use std::cell::RefCell;
    let streamed = RefCell::new(Vec::<f32>::new());
    let (final_pcm, ids, timings) = {
        let s = &streamed;
        tts.synthesize_streaming(
            &backend,
            text,
            512,
            16,
            |chunk| {
                if !chunk.pcm.is_empty() && ttfa_ms == 0.0 {
                    ttfa_ms = t0.elapsed().as_secs_f64() * 1e3;
                }
                n_chunks += 1;
                streamed_len += chunk.pcm.len();
                s.borrow_mut().extend_from_slice(&chunk.pcm);
                true
            },
            |_| true,
        )
        .expect("streaming synthesis")
    };

    assert!(
        ttfa_ms > 0.0 && streamed_len > 0 && n_chunks > 1,
        "streaming must emit several chunks before completion"
    );

    // Deviation of the STREAMED concatenation vs the FINAL single-pass
    // decode over the same codes. Chunked decoding carries left context but
    // cannot see future tokens (global attention inside the codec backbone),
    // so a bounded difference is expected and is REPORTED, not hidden.
    let borrowed = streamed.borrow();
    let n: usize = borrowed.len().min(final_pcm.len());
    let a = &borrowed[..n];
    let b = &final_pcm[..n];
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("TTFA {ttfa_ms:.0} ms, chunks {n_chunks}, streamed {streamed_len} samples");
    println!("codes {}", timings.n_codes);
    println!("streamed-vs-final decode deviation: max_abs {max_abs:.5} over {n} samples");
    let _ = ids;
}

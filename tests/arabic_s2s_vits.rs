//! Arabic speech-to-speech end-to-end (Phase 5 Session 2 Track D rerun).
//!
//! Drives the complete S2S chain with the FIXED MmsVits engine behind the
//! engine-agnostic `SpeechOut` seam:
//!
//! ```text
//! bank WAV -> streaming audio in -> Ultravox transcript
//!          -> VoiceSession.generate_reply (Arabic reply)
//!          -> &dyn SpeechOut (MmsVits) -> PCM chunks
//! ```
//!
//! Skips silently unless the real-weight fixture is present:
//!
//! ```text
//! EMBER_VOICE_E2E=1
//! EMBER_VOICE_TEXT_GGUF / EMBER_VOICE_AUDIO_GGUF / EMBER_VOICE_TOKENIZER
//! EMBER_VITS_GGUF       mms-tts ara GGUF
//! ```
//!
//! Set `EMBER_VOICE_E2E_REQUIRED=1` to make an absent fixture a failure
//! rather than a skip, so a release job cannot report green without having
//! run the Arabic speech chain. See `tests/common/mod.rs`.

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;

fn fixture() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    match common::gate(
        "EMBER_VOICE_E2E_REQUIRED",
        Some("EMBER_VOICE_E2E"),
        &[
            "EMBER_VOICE_TEXT_GGUF",
            "EMBER_VOICE_AUDIO_GGUF",
            "EMBER_VOICE_TOKENIZER",
            "EMBER_VITS_GGUF",
        ],
    ) {
        Ok(Some(paths)) => Some((
            paths[0].clone(),
            paths[1].clone(),
            paths[2].clone(),
            paths[3].clone(),
        )),
        Ok(None) => None,
        Err(why) => panic!("ember arabic_s2s_vits: {why}"),
    }
}

#[test]
fn arabic_s2s_vits_full_chain_bank_audio_to_speech() {
    use ember::tts::SpeechOut;

    let Some((text_gguf, audio_gguf, tokenizer, vits_gguf)) = fixture() else {
        eprintln!(
            "skipping: set EMBER_VOICE_E2E=1 (+ TEXT/AUDIO/TOKENIZER paths, EMBER_VITS_GGUF)"
        );
        return;
    };
    let backend = ember::backend::CpuBackend;
    let t0 = std::time::Instant::now();
    let mark = |what: &str| println!("[s2s {:>7.1}s] {what}", t0.elapsed().as_secs_f64());

    let model = ember::ultravox::Ultravox::from_ggufs(&text_gguf, &audio_gguf)
        .expect("load ultravox tower");
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&tokenizer).expect("tokenizer");
    let vits = ember::tts::vits::MmsVits::from_gguf(&vits_gguf).expect("load mms-vits");
    mark("models loaded");

    let mut session =
        ember::multimodal::VoiceSession::new(&model, &backend, &tokenizer, 2048, 64 << 20)
            .expect("session");

    // user turn: real Arabic bank audio streamed through the validated path
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("research/banks/arabic_speech_001/ar_eg_test_0000.wav");
    // bank clips are mono s16le PCM (scripts/build_arabic_speech_bank.py)
    let clip = ember::multimodal::audio::decode_wav(&wav).expect("read bank clip");
    let pcm = clip.samples;
    assert_eq!(clip.sample_rate, 16_000, "bank clips are 16 kHz");
    session.begin_user_turn();
    session.open_streaming_audio(Default::default()).unwrap();
    // fill the stream first; the tower encode happens once at finalize
    // (per-chunk forced active-window inference would re-encode the whole
    // prefix through the encoder on every push — a harness cadence issue,
    // not a runtime one).
    for chunk in pcm.chunks(3200) {
        session.push_streaming_audio(chunk).unwrap();
    }
    mark("audio pushed");
    session.finalize_streaming_audio().unwrap();
    mark("stream finalized");
    session.set_turn_prompt("<|audio|>".to_string()).unwrap();
    let (_span, _tokens) = session.commit_user_turn().unwrap();
    mark("turn committed");

    let control = ember::multimodal::GenerationControl::new();
    let (reply, cancelled) = session
        .generate_reply(&control, 48, |_| {}, || false)
        .unwrap();
    assert!(!cancelled);
    assert!(!reply.trim().is_empty(), "model must produce a reply");
    println!("arabic reply: {reply}");

    // speak the reply through the engine-agnostic seam (fixed VITS engine)
    let speech: &dyn SpeechOut = &vits;
    assert_eq!(speech.sample_rate(), 16_000);
    let mut chunks = 0usize;
    let mut first_audio: Option<usize> = None;
    let (out_pcm, codes, timings) = speech
        .stream_speech(
            &backend,
            &reply,
            4096,
            64,
            &mut |meta| {
                chunks += 1;
                if first_audio.is_none() {
                    first_audio = Some(meta.first_sample);
                }
                true
            },
            &mut |_| true,
        )
        .expect("vits stream_speech");
    mark("speech streamed");
    println!(
        "s2s speech: {chunks} chunks, {} samples @16 kHz, ttfa {:.0} ms",
        out_pcm.len(),
        timings.time_to_first_audio_ms
    );
    assert!(codes.is_empty(), "vits emits raw PCM, not codec tokens");
    assert!(out_pcm.len() >= 8_000, "expected at least 0.5 s of audio");
    assert_eq!(chunks > 0, first_audio.is_some());
    for s in out_pcm.iter() {
        assert!(
            s.is_finite() && s.abs() <= 1.0,
            "PCM must be finite tanh output"
        );
    }
}

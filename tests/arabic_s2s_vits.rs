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

use std::path::PathBuf;

fn fixture() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    if std::env::var("EMBER_VOICE_E2E").ok().as_deref() != Some("1") {
        return None;
    }
    let get = |k: &str| -> Option<PathBuf> {
        std::env::var(k)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Some((
        get("EMBER_VOICE_TEXT_GGUF")?,
        get("EMBER_VOICE_AUDIO_GGUF")?,
        get("EMBER_VOICE_TOKENIZER")?,
        get("EMBER_VITS_GGUF")?,
    ))
}

/// Minimal 16-bit mono PCM WAV reader (bank clips are 16 kHz s16le).
fn read_wav_mono(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    // chunk payload start = tag position + 4 (tag) + 4 (size field)
    let find =
        |tag: &[u8; 4]| -> Option<usize> { bytes.windows(4).position(|w| w == tag).map(|p| p + 8) };
    let fmt_at = find(b"fmt ").ok_or_else(|| anyhow::anyhow!("no fmt chunk"))?;
    let channels = u16::from_le_bytes([bytes[fmt_at + 2], bytes[fmt_at + 3]]) as usize;
    let sr = u32::from_le_bytes([
        bytes[fmt_at + 4],
        bytes[fmt_at + 5],
        bytes[fmt_at + 6],
        bytes[fmt_at + 7],
    ]);
    let data_at = find(b"data").ok_or_else(|| anyhow::anyhow!("no data chunk"))?;
    let pcm: Vec<f32> = bytes[data_at..]
        .chunks_exact(2 * channels)
        .map(|c| {
            let s = i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
            // bank is mono; if stereo ever appears, average the frame
            if channels == 2 {
                let s2 = i16::from_le_bytes([c[2], c[3]]) as f32 / 32768.0;
                (s + s2) * 0.5
            } else {
                s
            }
        })
        .collect();
    Ok((pcm, sr))
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
    let (pcm, sr) = read_wav_mono(&wav).expect("read bank clip");
    assert_eq!(sr, 16_000, "bank clips are 16 kHz");
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

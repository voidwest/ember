//! Probe: does running Ultravox session inference corrupt OuteTTS prefill
//! in the same process? (Phase 5 Session 2 Track D blocker investigation.)
use ember::backend::CpuBackend;

fn main() -> anyhow::Result<()> {
    let text_gguf = std::env::var("EMBER_VOICE_TEXT_GGUF").unwrap();
    let audio_gguf = std::env::var("EMBER_VOICE_AUDIO_GGUF").unwrap();
    let tok_path = std::env::var("EMBER_VOICE_TOKENIZER").unwrap();
    let tts_gguf = std::env::var("EMBER_TTS_GGUF").unwrap();
    let tts_tok = std::env::var("EMBER_TTS_TOKENIZER").unwrap();
    let codec = std::env::var("EMBER_TTS_CODEC").unwrap();

    // 1. TTS BEFORE any ultravox work
    let backend = CpuBackend;
    let tts_a = ember::tts::outetts::OuteTts::from_gguf(
        std::path::Path::new(&tts_gguf),
        std::path::Path::new(&tts_tok),
        std::path::Path::new(&codec),
        ember::quant_k::KStrategy::Auto,
    )?;
    let (pcm_a, ids_a, t_a) = tts_a.synthesize(&backend, "I'm ready to respond.", 208, |_| true)?;
    println!(
        "BEFORE ultravox: codes={} tokens={} gen_ms={:.0}",
        t_a.n_codes,
        ids_a.len(),
        t_a.generate_ms
    );
    drop(pcm_a);
    drop(tts_a);

    // 2. ultravox session inference
    let ultravox = ember::ultravox::Ultravox::from_ggufs(
        std::path::Path::new(&text_gguf),
        std::path::Path::new(&audio_gguf),
    )?;
    let tok = ember::tokenizer::EmberTokenizer::from_file(std::path::Path::new(&tok_path))?;
    let mut session =
        ember::multimodal::VoiceSession::new(&ultravox, &backend, &tok, 1024, 64 << 20)?;
    session.begin_user_turn();
    session.set_turn_prompt("<|audio|>".into())?;
    let tone: Vec<f32> = (0..16_000)
        .map(|i| 0.6 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
        .collect();
    session.attach_static_audio(&ember::multimodal::audio::AudioInput::Samples {
        data: tone,
        sample_rate: 16_000,
    })?;
    session.commit_user_turn()?;
    let ctl = ember::multimodal::GenerationControl::new();
    let (reply, _) = session.generate_reply(&ctl, 24, |_| {}, || false)?;
    println!("session reply: {reply:?}");

    // 3. fresh TTS instance AFTER ultravox work
    let tts_b = ember::tts::outetts::OuteTts::from_gguf(
        std::path::Path::new(&tts_gguf),
        std::path::Path::new(&tts_tok),
        std::path::Path::new(&codec),
        ember::quant_k::KStrategy::Auto,
    )?;
    let (_, ids_b, t_b) = tts_b.synthesize_streaming(
        &backend,
        "I'm ready to respond.",
        208,
        8,
        |_| true,
        |_| true,
    )?;
    println!(
        "AFTER ultravox(streaming): codes={} tokens={} gen_ms={:.0}",
        t_b.n_codes,
        ids_b.len(),
        t_b.generate_ms
    );
    // AND: fresh-process-equivalent control -> streaming BEFORE any session
    let tts_c = ember::tts::outetts::OuteTts::from_gguf(
        std::path::Path::new(&tts_gguf),
        std::path::Path::new(&tts_tok),
        std::path::Path::new(&codec),
        ember::quant_k::KStrategy::Auto,
    )?;
    let (_, ids_c, t_c) = tts_c.synthesize_streaming(
        &backend,
        "I'm ready to respond.",
        208,
        8,
        |_| true,
        |_| true,
    )?;
    println!(
        "CONTROL streaming (fresh instance): codes={} tokens={}",
        t_c.n_codes,
        ids_c.len()
    );
    Ok(())
}

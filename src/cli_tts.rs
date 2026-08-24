//! `ember tts`: speech-output commands.
//!
//! Two modes:
//!
//! 1. `--codec-selftest` — decodes deterministic LCG code sequences through
//!    the WavTokenizer decoder and dumps every validation boundary
//!    (mirroring `scripts/ref_wavtokenizer.py`) for the parity ladder.
//! 2. full OuteTTS synthesis (`--model` + text) — LLM emits codec tokens,
//!    the codec turns them into PCM, written as a WAV file. Streams
//!    `OutputEvent::AudioChunk`s while decoding (Track F).

use crate::Args;
use anyhow::{Context, Result};
use clap::Args as ClapArgs;

#[derive(ClapArgs)]
pub(crate) struct TtsCommand {
    /// path to the OuteTTS GGUF (qwen2-family codec-token generator)
    #[arg(long)]
    model: Option<String>,

    /// path to the tokenizer.json for the TTS model
    #[arg(long)]
    tokenizer: Option<String>,

    /// path to the wavtokenizer-decoder GGUF
    #[arg(long)]
    codec: String,

    /// speaker profile JSON (optional; omit for the model's default voice)
    #[arg(long)]
    speaker: Option<String>,

    /// text to synthesize
    #[arg(long)]
    text: Option<String>,

    /// write PCM to this WAV path
    #[arg(long, default_value = "ember-tts.wav")]
    out: String,

    /// greedy token budget for the speech generator
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// run the codec self-test instead of synthesis: decode LCG codes and
    /// dump all ladder boundaries
    #[arg(long, default_value_t = false)]
    codec_selftest: bool,

    /// self-test code lengths (tokens per sequence)
    #[arg(long, default_value_t = 0)]
    tokens: usize,

    /// stream PCM in chunks while generating; prints time-to-first-audio
    /// and per-chunk cadence instead of single-shot timings
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// codec tokens per streamed chunk
    #[arg(long, default_value_t = 16)]
    chunk_tokens: usize,

    /// dump progressive-validation artifacts here (self-test or synthesis)
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// MMS-TTS (VITS) GGUF: switch to the Arabic-capable engine for
    /// synthesis (`--text` + `--out`), skipping the OuteTTS path entirely
    #[arg(long)]
    vits_model: Option<String>,
}

use std::path::{Path, PathBuf};

pub(crate) fn run_tts_command(command: &TtsCommand, _args: &Args) -> Result<()> {
    use ember::tts::wavtokenizer::WavTokenizerDecoder;

    if let Some(vits_path) = &command.vits_model {
        return vits_synthesize(command, vits_path);
    }

    if command.codec_selftest {
        let decoder = WavTokenizerDecoder::from_gguf(std::path::Path::new(&command.codec))
            .context("failed to load codec gguf")?;
        let backend = ember::backend::CpuBackend;
        codec_selftest(&decoder, &backend, command)?;
        return Ok(());
    }

    let model_path = command.model.as_deref().context("--model is required")?;
    let tok_path = command
        .tokenizer
        .as_deref()
        .context("--tokenizer is required")?;
    let text = command.text.as_deref().context("--text is required")?;

    use ember::tts::outetts::OuteTts;
    println!("loading OuteTTS (K-quant resident)...");
    let tts = OuteTts::from_gguf(
        std::path::Path::new(model_path),
        std::path::Path::new(tok_path),
        std::path::Path::new(&command.codec),
        ember::quant_k::KStrategy::Auto,
    )?;
    let backend = ember::backend::CpuBackend;

    let prompt = tts.build_prompt(text)?;
    println!(
        "prompt: {} chars -> template preview: {:.140}",
        prompt.len(),
        prompt
    );

    let sr = tts.codec.config.sample_rate;
    let t0 = std::time::Instant::now();

    let (pcm, ids, timings) = if command.stream {
        use std::sync::{Arc, Mutex};
        let all_pcm: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = all_pcm.clone();
        let ttfa: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
        let ttfa_sink = ttfa.clone();
        let gen_start = std::time::Instant::now();
        let cadence: Arc<Mutex<Vec<(usize, f64)>>> = Arc::new(Mutex::new(Vec::new()));
        let cadence_sink = cadence.clone();
        let (full, ids2, timings) = tts.synthesize_streaming(
            &backend,
            text,
            command.max_tokens,
            command.chunk_tokens,
            move |chunk| {
                if !chunk.pcm.is_empty() {
                    let mut tt = ttfa_sink.lock().expect("ttfa");
                    if tt.is_none() {
                        *tt = Some(gen_start.elapsed().as_secs_f64() * 1e3);
                    }
                }
                cadence_sink
                    .lock()
                    .expect("cadence")
                    .push((chunk.pcm.len(), gen_start.elapsed().as_secs_f64() * 1e3));
                if let Ok(mut s) = sink.lock() {
                    s.extend_from_slice(&chunk.pcm);
                }
                true
            },
            |_| true,
        )?;
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let mut timings = timings;
        timings.time_to_first_audio_ms = ttfa.lock().map(|t| t.unwrap_or(0.0)).unwrap_or(0.0);
        let chunks = cadence.lock().map(|c| c.clone()).unwrap_or_default();
        // persist the naive concatenation for offline drift analysis
        {
            let streamed_path = std::path::PathBuf::from(format!("{}.streamed.wav", command.out));
            if let Ok(s) = all_pcm.lock()
                && !s.is_empty()
                && let Err(e) = write_wav(&streamed_path, &s, sr)
            {
                eprintln!("warn: could not write {streamed_path:?}: {e}");
            }
        }
        println!(
            "streamed {} chunks -> {} samples | time-to-first-audio {:.0} ms | wall {wall:.0} ms",
            chunks.len(),
            all_pcm.lock().map(|s| s.len()).unwrap_or(0),
            timings.time_to_first_audio_ms
        );
        for (i, (n, at)) in chunks.iter().enumerate().take(6) {
            println!("  chunk {i}: {n} samples at {at:.0} ms");
        }
        println!(
            "streamed-vs-final: max_abs {:.4} rms_rel {:.2e} corr {:.6} | refined: max_abs {:.4} rms_rel {:.2e}",
            timings.streamed_max_abs,
            timings.streamed_rms_rel,
            timings.streamed_corr,
            timings.refined_max_abs,
            timings.refined_rms_rel
        );
        println!(
            "drift-by-distance (tokens behind frontier; mean |Δ|): <4: {:.2e}  <8: {:.2e}  <16: {:.2e}  <32: {:.2e}  32+: {:.2e}",
            timings.drift_by_distance[0],
            timings.drift_by_distance[1],
            timings.drift_by_distance[2],
            timings.drift_by_distance[3],
            timings.drift_by_distance[4]
        );
        (full, ids2, timings)
    } else {
        let (pcm, ids, timings) = tts.synthesize(&backend, text, command.max_tokens, |_| true)?;
        (pcm, ids, timings)
    };
    let wall = t0.elapsed().as_secs_f64() * 1e3;

    write_wav(std::path::Path::new(&command.out), &pcm, sr)?;
    println!(
        "generated {} tokens / {} codec codes -> {} samples ({:.2} s) @{} Hz",
        timings.n_tokens,
        timings.n_codes,
        pcm.len(),
        pcm.len() as f64 / f64::from(sr),
        sr
    );
    println!(
        "timings: prompt {:.0} ms | prefill {:.0} ms | generate {:.0} ms | codec {:.0} ms | TTFA {:.0} ms | wall {wall:.0} ms | RTF {:.2}x",
        timings.prompt_ms,
        timings.prefill_ms,
        timings.generate_ms,
        timings.codec_ms,
        timings.time_to_first_audio_ms,
        timings.rtf()
    );

    if let Some(dir) = &command.dump_dir {
        std::fs::create_dir_all(dir)?;
        let as_f32 = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let prompt = tts.build_prompt(text)?;
        let pids = tts.tokenizer.encode_no_special(&prompt)?;
        for (name, data) in [
            ("prompt_ids", as_f32(&pids)),
            ("gen_ids", as_f32(&ids)),
            ("codes", as_f32(&tts.extract_codes(&ids))),
            ("waveform", pcm.clone()),
        ] {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in &data {
                bytes.extend(v.to_le_bytes());
            }
            std::fs::write(dir.join(format!("{name}.bin")), &bytes)?;
        }
        println!("ladder dumps written to {}", dir.display());
    }

    println!("wrote {}", command.out);
    Ok(())
}

/// Deterministic code sequence identical to scripts/ref_wavtokenizer.py.
fn lcg_codes(n_tokens: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(((s >> 33) % 4096) as u32);
    }
    out
}

fn codec_selftest(
    decoder: &ember::tts::wavtokenizer::WavTokenizerDecoder,
    backend: &ember::backend::CpuBackend,
    command: &TtsCommand,
) -> Result<()> {
    use std::time::Instant;

    let dir = command
        .dump_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("tts-selftest"));
    std::fs::create_dir_all(&dir)?;
    let lengths: Vec<usize> = if command.tokens > 0 {
        vec![command.tokens]
    } else {
        vec![37, 150]
    };
    let mut manifest = serde_json::Map::new();
    for n in lengths {
        let codes = lcg_codes(n, n as u64);
        let t0 = Instant::now();
        let (pcm, trace) = decoder.decode_traced(backend, &codes, true)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let code_f32: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
        write_bin(&dir, &format!("codes_{n}"), &code_f32, vec![codes.len()])?;
        if let Some(t) = &trace.features {
            write_bin(
                &dir,
                &format!("0_features_{n}"),
                t.data(),
                t.shape().to_vec(),
            )?;
        }
        if let Some(t) = &trace.embed {
            write_bin(&dir, &format!("1_embed_{n}"), t.data(), t.shape().to_vec())?;
        }
        if let Some(t) = &trace.pos_net {
            write_bin(&dir, &format!("2_posnet_{n}"), t.data(), t.shape().to_vec())?;
        }
        if let Some(t) = &trace.adanorm {
            write_bin(
                &dir,
                &format!("3_adanorm_{n}"),
                t.data(),
                t.shape().to_vec(),
            )?;
        }
        for (i, t) in &trace.convnext_blocks {
            write_bin(
                &dir,
                &format!("4_convnext_{i}_{n}"),
                t.data(),
                t.shape().to_vec(),
            )?;
        }
        if let Some(t) = &trace.backbone_final {
            write_bin(
                &dir,
                &format!("5_backbone_final_{n}"),
                t.data(),
                t.shape().to_vec(),
            )?;
        }
        if let (Some(m), Some(p)) = (&trace.mag, &trace.phase) {
            write_bin(&dir, &format!("6_mag_{n}"), m.data(), m.shape().to_vec())?;
            write_bin(&dir, &format!("6_phase_{n}"), p.data(), p.shape().to_vec())?;
        }
        // waveform + wav file
        write_bin(&dir, &format!("7_waveform_{n}"), &pcm, vec![pcm.len()])?;
        let sr = decoder.config.sample_rate;
        let wav_path = dir.join(format!("out_{n}.wav"));
        write_wav(&wav_path, &pcm, sr)?;
        println!(
            "[{n}] {} tokens -> {} samples @{} Hz | decode {wall_ms:.0} ms ({:.3}x RTF) -> {}",
            n,
            pcm.len(),
            sr,
            wall_ms / 1000.0 / (pcm.len() as f64 / sr as f64),
            wav_path.display()
        );
        manifest.insert(
            format!("{n}"),
            serde_json::json!({
                "tokens": n,
                "samples": pcm.len(),
                "decode_ms": wall_ms,
            }),
        );
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&serde_json::Value::Object(manifest))?,
    )?;
    Ok(())
}

fn write_wav(path: &std::path::Path, pcm: &[f32], sample_rate: u32) -> Result<()> {
    let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&((36 + pcm.len() * 2) as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(pcm.len() as u32 * 2).to_le_bytes());
    for &s in pcm {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn write_bin(dir: &Path, name: &str, data: &[f32], _shape: Vec<usize>) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend(v.to_le_bytes());
    }
    std::fs::write(dir.join(format!("{name}.bin")), &bytes)?;
    Ok(())
}

/// MMS-VITS synthesis path (Phase 5 Session 2 Track C): Arabic-capable
/// deterministic text-to-speech with optional ladder dumps.
fn vits_synthesize(command: &TtsCommand, vits_path: &str) -> Result<()> {
    let backend = ember::backend::CpuBackend;
    let model = ember::tts::vits::MmsVits::from_gguf(Path::new(vits_path))
        .context("loading mms-vits gguf")?;
    let text = command
        .text
        .as_deref()
        .context("--text is required with --vits-model")?;
    let t0 = std::time::Instant::now();
    let result = if command.stream {
        let mut first_audio_ms = None;
        let mut chunks = 0usize;
        let (pcm, _codes, timings) = model.synthesize_streaming(
            &backend,
            text,
            4096,
            command.chunk_tokens.max(8),
            |_meta| {
                chunks += 1;
                if first_audio_ms.is_none() {
                    first_audio_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
                }
                true
            },
            |_| true,
        )?;
        println!(
            "vits streaming: {} chunks, TTFA {:.0} ms, wall {:.0} ms",
            chunks,
            first_audio_ms.unwrap_or(0.0),
            t0.elapsed().as_secs_f64() * 1e3
        );
        (pcm, timings)
    } else {
        let r = model.synthesize(&backend, text, command.dump_dir.is_some())?;
        (r.pcm, r.timings)
    };
    let (pcm, timings) = result;
    write_wav(Path::new(&command.out), &pcm, model.config.sample_rate)?;
    println!(
        "vits: \"{}\" -> {} samples @{} Hz ({:.2} s audio) in {:.0} ms | prompt {:.0} prefill {:.0} gen {:.0} codec {:.0} ms | RTF {:.2}",
        text,
        pcm.len(),
        model.config.sample_rate,
        pcm.len() as f64 / model.config.sample_rate as f64,
        t0.elapsed().as_secs_f64() * 1e3,
        timings.prompt_ms,
        timings.prefill_ms,
        timings.generate_ms,
        timings.codec_ms,
        timings.rtf()
    );

    if let Some(dir) = &command.dump_dir {
        std::fs::create_dir_all(dir)?;
        // ladder mirror of scripts/ref_vits.py dumps
        let r = model.synthesize(&backend, text, true)?;
        let write_npy = |name: &str, data: &[f32], shape: Vec<usize>| -> Result<()> {
            // simple .npy writer (float32, C order)
            let mut header = vec![0x93u8; 0];
            header.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y']);
            header.push(1u8);
            header.push(0u8);
            let descr = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},)}}",
                shape.iter().product::<usize>()
            );
            let hlen = (descr.len() + 1) as u16;
            header.extend(&(hlen).to_le_bytes());
            header.extend(descr.as_bytes());
            header.push(b'\n');
            let mut bytes = header;
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join(format!("{name}.npy")), bytes)?;
            Ok(())
        };
        write_npy("00_input_ids", &[0.0], vec![1])?; // ids dumped via manifest below
        if let Some(tr) = Some(&r.trace) {
            if let Some(x) = &tr.embed_scaled {
                write_npy(
                    "01_embed_scaled",
                    x,
                    vec![x.len() / model.config.hidden_size, model.config.hidden_size],
                )?;
            }
            if let Some(x) = &tr.encoder_out {
                write_npy(
                    "02_encoder_out",
                    x,
                    vec![x.len() / model.config.hidden_size, model.config.hidden_size],
                )?;
            }
            if let Some(x) = &tr.prior_means {
                write_npy(
                    "03_prior_means",
                    x,
                    vec![x.len() / model.config.flow_size, model.config.flow_size],
                )?;
            }
            if let Some(ld) = &tr.log_duration {
                write_npy("05_log_duration", ld, vec![ld.len()])?;
            }
            if let Some(x) = &tr.expanded_hidden {
                write_npy(
                    "07_expanded_hidden",
                    x,
                    vec![x.len() / model.config.hidden_size, model.config.hidden_size],
                )?;
            }
            if let Some(z) = &tr.flow_z {
                write_npy(
                    "09_flow_z",
                    z,
                    vec![model.config.flow_size, z.len() / model.config.flow_size],
                )?;
            }
            if let Some(w) = &tr.waveform {
                write_npy("10_waveform", w, vec![w.len()])?;
            }
        }
        let manifest = serde_json::json!({
            "engine": "mms-vits",
            "text": text,
            "ids": r.trace.ids.clone().unwrap_or_default(),
            "waveform_samples": r.trace.waveform.as_ref().map(|w| w.len()).unwrap_or(0),
            "sample_rate": model.config.sample_rate,
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        println!("dumped ladder artifacts to {}", dir.display());
    }
    Ok(())
}

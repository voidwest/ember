//! `ember audio`: audio-conditioned generation and validation dumps for
//! the first voice-capable model (Ultravox v0.5, llama-3.2-1b).

use crate::Args;
use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use ember::backend::{Backend, CpuBackend};
use ember::multimodal::audio::AudioInput;
use ember::tokenizer::EmberTokenizer;
use ember::ultravox::{Ultravox, AUDIO_PLACEHOLDER};
use std::path::PathBuf;

#[derive(ClapArgs)]
pub(crate) struct AudioCommand {
    /// path to the text LLM GGUF (llama arch; Llama-3.2-1B-Instruct)
    #[arg(long)]
    model: String,

    /// path to the audio mmproj GGUF (see tools/convert_ultravox_audio.py)
    #[arg(long)]
    audio_model: String,

    /// path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// WAV file(s); repeat for multiple segments. Each binds to one
    /// <|audio|> placeholder in the prompt, in order.
    #[arg(long = "audio", value_name = "FILE")]
    audio_files: Vec<String>,

    /// user prompt; use <|audio|> where each segment should be bound
    /// (default: one placeholder before the prompt)
    #[arg(long, default_value = "What is being said in this audio?")]
    prompt: String,

    /// number of tokens to generate
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// write progressive-validation artifacts to this directory
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// feed every segment through the incremental AudioStream frontend +
    /// window scheduler under several partition patterns, then require
    /// bit-exact parity with the static pipeline (mel -> features ->
    /// assembled embeddings -> first logits -> greedy tokens).
    #[arg(long, default_value_t = false)]
    stream_validate: bool,
}

pub(crate) fn run_audio_command(command: &AudioCommand, _args: &Args) -> Result<()> {
    anyhow::ensure!(
        !command.audio_files.is_empty(),
        "--audio <file.wav> is required (repeat for multiple segments)"
    );
    let backend = CpuBackend;
    let model = Ultravox::from_ggufs(
        std::path::Path::new(&command.model),
        std::path::Path::new(&command.audio_model),
    )?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)
        .with_context(|| format!("failed to load tokenizer {}", command.tokenizer))?;

    // bind segments to placeholders in order; default to one leading
    // placeholder when the prompt has none and exactly one segment is given
    let placeholders = command.prompt.matches(AUDIO_PLACEHOLDER).count();
    let prompt = if placeholders == 0 && command.audio_files.len() == 1 {
        format!("{AUDIO_PLACEHOLDER}{}", command.prompt)
    } else {
        anyhow::ensure!(
            placeholders == command.audio_files.len(),
            "prompt has {placeholders} {AUDIO_PLACEHOLDER} placeholders but {} --audio flags were given",
            command.audio_files.len()
        );
        command.prompt.clone()
    };
    let audios: Vec<AudioInput> = command
        .audio_files
        .iter()
        .map(|p| AudioInput::File(std::path::PathBuf::from(p)))
        .collect();

    println!(
        "transcribing: {} segment(s), prompt {:?}",
        audios.len(),
        prompt.replace(AUDIO_PLACEHOLDER, "<|audio|>")
    );
    let (generated, text, timings) =
        model.generate_with_audio(&backend, &tokenizer, &prompt, &audios, command.max_tokens)?;

    println!("generated ({} tokens): {}", generated.len(), text);
    print_timings(&timings);

    if command.stream_validate {
        stream_validate(&model, &backend, &tokenizer, &prompt, &command.audio_files)?;
    }

    if let Some(dir) = &command.dump_dir {
        dump_validation_artifacts(&model, &backend, &tokenizer, command, &prompt, dir)?;
    }
    Ok(())
}

/// Track C4 end-to-end validation: the streamed path (AudioStream ->
/// schedule_stream_update -> stream_finish) must reproduce the static
/// pipeline bit-exactly for every partition pattern.
fn stream_validate(
    model: &Ultravox,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    prompt: &str,
    audio_files: &[String],
) -> Result<()> {
    use ember::multimodal::stream::{AudioStream, AudioStreamConfig};
    use std::hint::black_box;

    let decoded = ember::multimodal::audio::to_mono_16k(&AudioInput::File(
        std::path::PathBuf::from(&audio_files[0]),
    ))?;
    println!(
        "stream-validate: {} samples @ {} Hz ({:.2} s)",
        decoded.samples.len(),
        decoded.sample_rate,
        decoded.samples.len() as f64 / f64::from(decoded.sample_rate)
    );

    // static reference once: mel -> chunked encode
    let t = std::time::Instant::now();
    let mel_static = ember::multimodal::audio::log_mel_spectrogram_full(&decoded.samples)?;
    let (feat_static, _, _, _, n_chunks_static) = model.encode_mel_chunked(backend, &mel_static)?;
    println!(
        "static path: {:.1} ms, {} window(s), feature rows {}",
        t.elapsed().as_secs_f64() * 1e3,
        n_chunks_static,
        feat_static.shape()[0]
    );
    let static_ref = greedy_from_features(model, backend, tokenizer, prompt, &feat_static)?;

    // partition patterns over the SAME pcm
    let len = decoded.samples.len();
    let mut patterns: Vec<(String, Vec<usize>)> = Vec::new();
    patterns.push(("single push".into(), vec![len]));
    {
        let c = decoded.sample_rate as usize / 10; // ~100 ms
        patterns.push((
            "100 ms chunks".into(),
            (0..len).step_by(c).map(|s| c.min(len - s)).collect(),
        ));
    }
    {
        // deterministic pseudo-random partitions (network-like)
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut parts = Vec::new();
        let mut covered = 0usize;
        while covered < len {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let n = ((seed >> 33) as usize) % 3331 + 1;
            parts.push(n.min(len - covered));
            covered += n.min(len - covered);
        }
        patterns.push(("random 1-3331".into(), parts));
    }
    if len > 31 * decoded.sample_rate as usize {
        // long input: exercise multi-window streaming too
        patterns.push(("2 s chunks".into(), {
            let c = 2 * decoded.sample_rate as usize;
            (0..len).step_by(c).map(|s| c.min(len - s)).collect()
        }));
    }

    let mut all_ok = true;
    for (name, parts) in &patterns {
        let mut stream = AudioStream::open(AudioStreamConfig {
            sample_rate: decoded.sample_rate,
        })?;
        let mut sched = Ultravox::stream_schedule_new();
        let mut pos = 0usize;
        let mut infer_active_pulses = 0u32;
        for (i, &n) in parts.iter().enumerate() {
            stream.push_pcm(&decoded.samples[pos..pos + n])?;
            pos += n;
            // update cadence ~every 0.5 s of input; provisional active
            // inference on a slow pulse (it is pure overhead for parity)
            let per_sec = decoded.sample_rate as usize;
            if i * n % per_sec.max(1) < n && i > 0 {
                let infer = if infer_active_pulses.is_multiple_of(4) {
                    infer_active_pulses += 1;
                    true
                } else {
                    false
                };
                let u = model.stream_update(backend, &mut sched, &stream, infer)?;
                black_box(u.new_audio_samples);
            }
        }
        let streamed = stream.finish()?;
        let t = std::time::Instant::now();
        let (feat_streamed, stats, _, n_windows) = model.stream_finish(backend, sched, streamed)?;
        let finish_ms = t.elapsed().as_secs_f64() * 1e3;

        let mel_ok = feat_static.shape() == feat_streamed.shape()
            && feat_static.data() == feat_streamed.data();
        let streamed_ref = greedy_from_features(model, backend, tokenizer, prompt, &feat_streamed)?;
        let gen_ok = static_ref.ids == streamed_ref.ids;

        println!(
            "[{name}] windows {n_windows} | mel/features {}: logits {}: generation {} | finish {finish_ms:.0} ms | fresh {:.1}s reencoded {:.1}s enc {:.0}ms proj {:.0}ms",
            if mel_ok { "BIT-EXACT" } else { "MISMATCH" },
            if static_ref.first_logits == streamed_ref.first_logits { "BIT-EXACT" } else { "MISMATCH" },
            if gen_ok { format!("{}/{} tokens", streamed_ref.ids.len(), static_ref.ids.len()) } else { "MISMATCH".into() },
            stats.freshly_encoded_seconds,
            stats.reencoded_seconds,
            stats.encoder_wall_ms,
            stats.projector_wall_ms,
        );
        all_ok &= mel_ok && gen_ok;
    }

    anyhow::ensure!(all_ok, "stream-vs-static parity FAILED");
    println!("stream-validate: ALL PATTERNS BIT-EXACT vs static path");
    Ok(())
}

struct GreedyRef {
    ids: Vec<u32>,
    first_logits: Vec<f32>,
}

/// Assemble `prompt` + precomputed features and run greedy generation —
/// mirrors `generate_with_audio` but skips preprocessing entirely.
fn greedy_from_features(
    model: &Ultravox,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    prompt: &str,
    features: &ember::tensor::CpuTensor,
) -> Result<GreedyRef> {
    let rendered = model.assembler.render_chat_template(prompt);
    let assembled = model.assembler.assemble(
        backend,
        tokenizer,
        &rendered,
        &[ember::ultravox::AudioFeatures {
            features: features.clone(),
        }],
        &model.llm.embed_tokens,
    )?;
    let max_tokens = 16;
    let mut cache = model
        .llm
        .create_request_cache(backend, assembled.input_ids.len(), max_tokens);
    let mut logits = model.llm.forward_last_logits_embeddings_with_cache(
        backend,
        &assembled.embeddings,
        &mut cache,
        0,
    )?;
    let first_logits = logits.data().to_vec();
    let eos_ids = tokenizer.eos_token_ids();
    let start_pos = assembled.input_ids.len();
    let mut ids: Vec<u32> = Vec::new();
    for step in 0..max_tokens {
        let best = ember::sampler::argmax_token(backend.data(&logits));
        let best = u32::try_from(best)?;
        ids.push(best);
        if eos_ids.contains(&best) || step + 1 >= max_tokens {
            break;
        }
        logits = model.llm.forward_last_logits_with_cache(
            backend,
            &[best],
            &mut cache,
            start_pos + step,
        )?;
    }
    Ok(GreedyRef { ids, first_logits })
}

fn print_timings(t: &ember::ultravox::AudioTimings) {
    println!("timings:");
    println!("  audio decode + normalize : {:8.1} ms", t.decode_ms);
    println!("  mel features             : {:8.1} ms", t.features_ms);
    println!(
        "  encoder                  : {:8.1} ms  ({:.2}x real time)",
        t.encoder_ms,
        t.encoder_real_time_factor()
    );
    println!("  projector                : {:8.1} ms", t.projector_ms);
    println!("  LLM prefill              : {:8.1} ms", t.llm_prefill_ms);
    println!("  audio duration           : {:8.2} s", t.audio_seconds);
    println!("  time to first token      : {:8.1} ms", t.ttft_ms);
    if t.decode_tok_s > 0.0 {
        println!(
            "  decode                   : {:8.2} tok/s ({} tokens)",
            t.decode_tok_s, t.n_decode_tokens
        );
    }
}

/// Progressive-validation dump matching `scripts/ref_ultravox.py`:
/// 0 waveform samples, 2 mel features, conv1 output, selected encoder
/// layers, final encoder output, projector output, assembled embeddings,
/// first logits, per-step logits + generation ids.
fn dump_validation_artifacts(
    model: &Ultravox,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    command: &AudioCommand,
    prompt: &str,
    dir: &std::path::Path,
) -> Result<()> {
    use ember::model::ForwardModel;
    use ember::sampler::argmax_token;

    std::fs::create_dir_all(dir)?;
    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();

    // 0. normalized waveform samples (16 kHz mono f32)
    let audios: Vec<AudioInput> = command
        .audio_files
        .iter()
        .map(|p| AudioInput::File(std::path::PathBuf::from(p)))
        .collect();
    let decoded = ember::multimodal::audio::to_mono_16k(&audios[0])?;
    write_bin(
        dir,
        "0_waveform",
        &tensor_from_vec(decoded.samples.clone()),
        &mut shapes,
    )?;
    let seconds = decoded.samples.len() as f64 / 16_000.0;

    // build traced inputs through the standard pipeline
    let (mel, decode_ms, features_ms, _) = model.build_mel(&audios[0])?;
    let _ = (decode_ms, features_ms);
    write_bin(dir, "2_mel_features", &mel, &mut shapes)?;

    let (projected, trace, _enc_ms, _proj_ms, n_chunks) =
        model.encode_mel_chunked(backend, &mel)?;
    if let Some(c) = &trace.conv1_output {
        write_bin(dir, "3_conv1_output", c, &mut shapes)?;
    }
    for (i, layer_out) in trace.layer_outputs.iter().enumerate() {
        if matches!(i, 0 | 1 | 5 | 15 | 31) || i == trace.layer_outputs.len() - 1 {
            write_bin(dir, &format!("4_layer_{i}"), layer_out, &mut shapes)?;
        }
    }
    if let Some(e) = &trace.encoder_output {
        // long-form: this is the final (masked, padded) window; keep both
        // the canonical ladder name and the explicit long-form name
        write_bin(dir, "5_encoder_output", e, &mut shapes)?;
        if n_chunks > 1 {
            write_bin(dir, "5_encoder_output_last_window", e, &mut shapes)?;
        }
    }
    write_bin(dir, "6_projector_output", &projected, &mut shapes)?;

    // assemble + prefill logits
    let rendered = model.assembler.render_chat_template(prompt);
    let assembled = model.assembler.assemble(
        backend,
        tokenizer,
        &rendered,
        &[ember::ultravox::AudioFeatures {
            features: projected.clone(),
        }],
        &model.llm.embed_tokens,
    )?;
    write_bin(
        dir,
        "7_assembled_embeddings",
        &assembled.embeddings,
        &mut shapes,
    )?;

    let mut cache = model
        .llm
        .create_request_cache(backend, assembled.input_ids.len(), 1);
    let logits = model.llm.forward_last_logits_embeddings_with_cache(
        backend,
        &assembled.embeddings,
        &mut cache,
        0,
    )?;
    // flatten to [vocab] to match the reference dump layout
    let logits_flat = tensor_from_vec(logits.data().to_vec());
    write_bin(dir, "8_first_logits", &logits_flat, &mut shapes)?;

    // greedy generation with per-step logits
    let (generated, text, timings) =
        model.generate_with_audio(backend, tokenizer, prompt, &audios, command.max_tokens)?;
    {
        let vocab = model.llm.vocab_size(backend);
        let mut step_logits: Vec<f32> = Vec::new();
        let mut cache =
            model
                .llm
                .create_request_cache(backend, assembled.input_ids.len(), command.max_tokens);
        let eos_ids = tokenizer.eos_token_ids();
        let start_pos = assembled.input_ids.len();
        let mut logits = model.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &assembled.embeddings,
            &mut cache,
            0,
        )?;
        for step in 0..command.max_tokens {
            let data = backend.data(&logits);
            step_logits.extend_from_slice(&data[..vocab]);
            let best = argmax_token(data);
            let best = u32::try_from(best)?;
            if eos_ids.contains(&best) {
                break;
            }
            if step + 1 < command.max_tokens {
                logits = model.llm.forward_last_logits_with_cache(
                    backend,
                    &[best],
                    &mut cache,
                    start_pos + step,
                )?;
            }
        }
        let mut bytes = Vec::with_capacity(step_logits.len() * 4);
        for v in &step_logits {
            bytes.extend(v.to_le_bytes());
        }
        std::fs::write(dir.join("step_logits.bin"), &bytes)?;
        let n_steps = step_logits.len() / vocab;
        shapes.push(("step_logits".into(), vec![n_steps, vocab]));
    }

    let manifest = serde_json::json!({
        "model": command.model,
        "audio_model": command.audio_model,
        "audio": command.audio_files,
        "prompt": prompt,
        "max_tokens": command.max_tokens,
        "input_ids": assembled.input_ids,
        "input_ids_len": assembled.input_ids.len(),
        "generation_ids": generated,
        "generated_text": text,
        "audio_seconds": seconds,
        "n_encoder_chunks": n_chunks,
        "timings_ms": serde_json::to_value(&timings)?,
        "shapes": shapes
            .into_iter()
            .map(|(k, v)| (k, serde_json::json!(v)))
            .collect::<serde_json::Map<_, _>>(),
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!("validation artifacts written to {}", dir.display());
    Ok(())
}

fn tensor_from_vec(data: Vec<f32>) -> ember::tensor::CpuTensor {
    let n = data.len();
    ember::tensor::CpuTensor::from_data(vec![n], data)
}

fn write_bin(
    dir: &std::path::Path,
    name: &str,
    tensor: &ember::tensor::CpuTensor,
    shapes: &mut Vec<(String, Vec<usize>)>,
) -> Result<()> {
    let mut bytes = Vec::with_capacity(tensor.len() * 4);
    for v in tensor.data() {
        bytes.extend(v.to_le_bytes());
    }
    std::fs::write(dir.join(format!("{name}.bin")), &bytes)?;
    shapes.push((name.to_string(), tensor.shape().to_vec()));
    Ok(())
}

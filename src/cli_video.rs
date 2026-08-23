//! `ember video`: video-conditioned generation and validation dumps for
//! the first video-capable model (SmolVLM2-256M-Video-Instruct).
//!
//! Frames come from a directory of numbered PNGs (`--frames-dir`) plus a
//! declared source fps — decoded-video input without a codec dependency in
//! core. Sampling, preprocessing, encoding, assembly and generation run
//! through the same API an application would use
//! (`SmolVlmVideo::generate_with_parts` over `ContentPart`s).

use crate::Args;
use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use ember::backend::CpuBackend;
use ember::multimodal::request::{ContentPart, VideoFrames, VideoInput};
use ember::multimodal::video::FrameSampling;
use ember::smolvlm_video::SmolVlmVideo;
use ember::tensor::CpuTensor;
use ember::tokenizer::EmberTokenizer;
use std::path::{Path, PathBuf};

#[derive(ClapArgs)]
pub(crate) struct VideoCommand {
    /// path to the text LLM GGUF (llama arch; SmolLM2 backbone)
    #[arg(long)]
    model: String,

    /// path to the vision mmproj GGUF
    #[arg(long)]
    mmproj: String,

    /// path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// directory of numbered PNG frames (decoded video)
    #[arg(long)]
    frames_dir: String,

    /// declared source frame rate of the frames (labels timestamps)
    #[arg(long, default_value_t = 8.0)]
    source_fps: f64,

    /// sampling policy: "uniform" or "fps"
    #[arg(long, default_value = "uniform")]
    sampling: String,

    /// max frames kept by uniform sampling
    #[arg(long, default_value_t = 64)]
    max_frames: usize,

    /// fps for the "fps" sampling policy
    #[arg(long, default_value_t = 1.0)]
    fps: f64,

    /// user prompt; use <video> where each video should be bound
    #[arg(long, default_value = "<video>What happens in this video?")]
    prompt: String,

    /// number of tokens to generate
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    /// write progressive-validation artifacts to this directory
    #[arg(long)]
    dump_dir: Option<PathBuf>,
}

pub(crate) fn run_video_command(command: &VideoCommand, _args: &Args) -> Result<()> {
    let backend = CpuBackend;
    let model = SmolVlmVideo::from_ggufs(Path::new(&command.model), Path::new(&command.mmproj))?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)
        .with_context(|| format!("failed to load tokenizer {}", command.tokenizer))?;

    // load decoded frames from the directory
    let mut names: Vec<_> = std::fs::read_dir(&command.frames_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    names.sort();
    anyhow::ensure!(!names.is_empty(), "no PNG frames in {}", command.frames_dir);
    let mut frames = Vec::with_capacity(names.len());
    for p in &names {
        frames.push(ember::multimodal::image::decode_rgb(p)?);
    }
    let n = frames.len();
    let timestamps_ms: Vec<f64> = (0..n)
        .map(|i| i as f64 * 1000.0 / command.source_fps)
        .collect();
    let video = VideoInput::Frames(VideoFrames {
        frames,
        timestamps_ms,
        source_fps: Some(command.source_fps),
        source_duration_s: Some(n as f64 / command.source_fps),
    });

    let sampling = match command.sampling.as_str() {
        "uniform" => FrameSampling::Uniform {
            max_frames: command.max_frames,
        },
        "fps" => FrameSampling::FixedFps {
            fps: command.fps,
            max_frames: command.max_frames,
        },
        other => anyhow::bail!("unknown --sampling {other} (uniform|fps)"),
    };

    // bind videos to <video> placeholders in order (single-video CLI here;
    // multiple placeholders fail closed inside render_prompt)
    anyhow::ensure!(
        command.prompt.matches("<video>").count() >= 1,
        "prompt must contain at least one <video> placeholder"
    );
    let parts = vec![
        ContentPart::Text(command.prompt.clone()),
        ContentPart::Video(video),
    ];

    // apply the wrapper's sampling policy by pre-sampling is NOT how the
    // API works — instead clone the model's policy via a scoped override:
    // SmolVlmVideo exposes `sampling` publicly; set it before building.
    let mut model = model;
    model.sampling = sampling;

    let (generated, text, timings) =
        model.generate_with_parts(&backend, &tokenizer, &parts, command.max_tokens)?;

    println!("generated ({} tokens): {}", generated.len(), text);
    println!("timings:");
    println!("  preprocess          : {:8.1} ms", timings.preprocess_ms);
    println!("  vision encoder      : {:8.1} ms", timings.vision_ms);
    println!("  LLM prefill         : {:8.1} ms", timings.llm_prefill_ms);
    println!("  time to first token : {:8.1} ms", timings.ttft_ms);
    if timings.decode_tok_s > 0.0 {
        println!(
            "  decode              : {:8.2} tok/s ({} tokens)",
            timings.decode_tok_s, timings.n_decode_tokens
        );
    }

    if let Some(dir) = &command.dump_dir {
        dump_validation_artifacts(&model, &backend, &tokenizer, command, dir)?;
    }
    Ok(())
}

/// Progressive-validation dumps matching scripts/ref_smolvlm2_video.py.
fn dump_validation_artifacts(
    model: &SmolVlmVideo,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    command: &VideoCommand,
    dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();

    // rebuild through build_inputs_parts for the trace
    let mut names: Vec<_> = std::fs::read_dir(&command.frames_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    names.sort();
    let mut frames = Vec::with_capacity(names.len());
    for p in &names {
        frames.push(ember::multimodal::image::decode_rgb(p)?);
    }
    let n = frames.len();
    let video = VideoInput::Frames(VideoFrames {
        frames,
        timestamps_ms: (0..n)
            .map(|i| i as f64 * 1000.0 / command.source_fps)
            .collect(),
        source_fps: Some(command.source_fps),
        source_duration_s: Some(n as f64 / command.source_fps),
    });
    let parts = vec![
        ContentPart::Text(command.prompt.clone()),
        ContentPart::Video(video),
    ];
    let (trace, sequence, _t) = model.build_inputs_parts(backend, tokenizer, &parts, 0)?;

    write_bin(dir, "1_pixels", &trace.pixels, &mut shapes)?;
    if let Some(e) = &trace.vision.encoder_output {
        write_bin(dir, "4_encoder_output", e, &mut shapes)?;
    }
    write_bin(
        dir,
        "5_projector_output",
        &trace.projector_output,
        &mut shapes,
    )?;
    write_bin(dir, "6_assembled", &trace.assembled_embeddings, &mut shapes)?;

    let mut cache = model
        .llm
        .create_request_cache(backend, trace.input_ids.len(), 1);
    let logits = model.llm.forward_last_logits_embeddings_with_cache(
        backend,
        &sequence.embeddings,
        &mut cache,
        0,
    )?;
    write_bin(
        dir,
        "7_first_logits",
        &CpuTensor::from_data(vec![logits.len()], logits.data().to_vec()),
        &mut shapes,
    )?;

    let manifest = serde_json::json!({
        "model": command.model,
        "mmproj": command.mmproj,
        "frames_dir": command.frames_dir,
        "source_fps": command.source_fps,
        "sampling": command.sampling,
        "max_frames": command.max_frames,
        "prompt": command.prompt,
        "input_ids_len": trace.input_ids.len(),
        "sampled_indices": trace.sampled.source_indices,
        "sampled_timestamps_ms": trace.sampled.timestamps_ms,
        "total_source_frames": trace.sampled.total_source_frames,
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

fn write_bin(
    dir: &Path,
    name: &str,
    tensor: &CpuTensor,
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

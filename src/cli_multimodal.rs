//! `ember multimodal`: image-conditioned generation and validation dumps
//! for the first image-capable model (SmolVLM-256M-Instruct).

use crate::Args;
use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use ember::backend::{Backend, CpuBackend};
use ember::smolvlm::{MultimodalTimings, SmolVlm};
use ember::tokenizer::EmberTokenizer;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub(crate) struct MultimodalCommand {
    /// path to the text LLM GGUF (llama arch)
    #[arg(long)]
    model: String,

    /// path to the vision mmproj GGUF (see tools/convert_smolvlm_mmproj.py)
    #[arg(long)]
    mmproj: String,

    /// path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// image file(s) (PNG/JPEG); repeat the flag for multiple images. Each
    /// image binds to one `<image>` placeholder in the prompt, in order.
    #[arg(long = "image", value_name = "FILE")]
    image: Vec<String>,

    /// user prompt; use <image> as the image placeholder (image-first)
    #[arg(long, default_value = "What is shown in this image?")]
    prompt: String,

    /// number of tokens to generate
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// write progressive-validation artifacts to this directory
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// run the traced pipeline once and print per-op stage timings
    #[arg(long)]
    profile: bool,
}

pub(crate) fn run_multimodal_command(command: &MultimodalCommand, _args: &Args) -> Result<()> {
    anyhow::ensure!(
        !command.image.is_empty(),
        "--image is required (repeat the flag for multiple images)"
    );
    let backend = CpuBackend;
    let model = SmolVlm::from_ggufs(
        std::path::Path::new(&command.model),
        std::path::Path::new(&command.mmproj),
    )?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)
        .with_context(|| format!("failed to load tokenizer {}", command.tokenizer))?;

    // Bind images to <image> placeholders in order. With exactly one image
    // and no explicit placeholder, prepend one (the historical behavior:
    // reference messages are image-first). With multiple images the prompt
    // must carry every placeholder explicitly.
    let mut placeholders = command.prompt.matches("<image>").count();
    let mut prompt = command.prompt.clone();
    if command.image.len() == 1 && placeholders == 0 {
        prompt = format!("<image>{prompt}");
        placeholders = 1;
    }
    anyhow::ensure!(
        placeholders == command.image.len(),
        "prompt has {placeholders} <image> placeholders but {} --image flags were given",
        command.image.len()
    );
    let parts: Vec<ember::multimodal::InputPart> =
        std::iter::once(ember::multimodal::InputPart::Text(prompt.to_string()))
            .chain(
                command
                    .image
                    .iter()
                    .map(|p| ember::multimodal::InputPart::Image(std::path::PathBuf::from(p))),
            )
            .collect();
    let (generated, text, timings) =
        model.generate_with_parts(&backend, &tokenizer, &parts, command.max_tokens)?;

    println!("generated ({} tokens): {}", generated.len(), text);
    print_timings(&timings);

    if command.profile {
        print_op_profile(
            &model,
            &backend,
            &tokenizer,
            &prompt,
            std::path::Path::new(&command.image[0]),
        )?;
    }

    if let Some(dir) = &command.dump_dir {
        dump_validation_artifacts(&model, &backend, &tokenizer, command, &prompt, dir)?;
    }
    Ok(())
}

/// One traced pass over the whole multimodal input pipeline, printing
/// per-op vision timings and preprocess sub-stage timings.
fn print_op_profile(
    model: &SmolVlm,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    prompt: &str,
    image: &std::path::Path,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let decoded = ember::multimodal::image::decode_rgb(image)?;
    let processed = ember::multimodal::image::preprocess(&decoded, &model.preprocess_config)?;
    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(processed);

    let (trace, _seq) = model.build_inputs_with_tokenizer(backend, tokenizer, image, prompt, 0)?;

    println!("op profile (one traced pass):");
    println!("  image decode            : {:8.1} ms", decode_ms);
    println!(
        "  preprocess resize       : {:8.1} ms",
        trace
            .images
            .first()
            .map(|p| p.timings.resize_ms)
            .unwrap_or(0.0)
    );
    println!(
        "  preprocess tile         : {:8.1} ms",
        trace
            .images
            .first()
            .map(|p| p.timings.tile_ms)
            .unwrap_or(0.0)
    );
    println!(
        "  preprocess normalize    : {:8.1} ms",
        trace
            .images
            .first()
            .map(|p| p.timings.normalize_ms)
            .unwrap_or(0.0)
    );
    let t = &trace.vision.op_timings;
    println!("  patch embed (im2col+mm) : {:8.1} ms", t.patch_embed_ms);
    println!("  position embed add      : {:8.1} ms", t.pos_embed_ms);
    println!("  layernorms (2/layer)    : {:8.1} ms", t.ln_ms);
    println!("  q/k/v projections       : {:8.1} ms", t.qkv_proj_ms);
    println!("  attn scores (qk^T)      : {:8.1} ms", t.attn_scores_ms);
    println!("  attn softmax            : {:8.1} ms", t.softmax_ms);
    println!("  attn values (pv)        : {:8.1} ms", t.attn_values_ms);
    println!("  out projection          : {:8.1} ms", t.out_proj_ms);
    println!("  residual adds + slicing : {:8.1} ms", t.residual_add_ms);
    println!("  mlp fc1                 : {:8.1} ms", t.fc1_ms);
    println!("  gelu (tanh)             : {:8.1} ms", t.gelu_ms);
    println!("  mlp fc2                 : {:8.1} ms", t.fc2_ms);
    println!("  post layernorm          : {:8.1} ms", t.post_ln_ms);
    println!("  vision total (accounted): {:8.1} ms", t.total_ms());
    Ok(())
}

fn print_timings(t: &MultimodalTimings) {
    println!("timings:");
    println!("  image preprocessing : {:8.1} ms", t.preprocess_ms);
    println!("  vision encoder      : {:8.1} ms", t.vision_ms);
    println!("  projector           : {:8.1} ms", t.projector_ms);
    println!("  LLM prefill         : {:8.1} ms", t.llm_prefill_ms);
    println!("  time to first token : {:8.1} ms", t.ttft_ms);
    if t.decode_tok_s > 0.0 {
        println!(
            "  decode              : {:8.2} tok/s ({} tokens)",
            t.decode_tok_s, t.n_decode_tokens
        );
    }
}

/// Progressive-validation dump: every boundary the reference script
/// (`scripts/ref_smolvlm.py`) also captures, as raw f32 little-endian
/// payloads + a shapes JSON. See docs/multimodal-foundation-plan.md §7.
fn dump_validation_artifacts(
    model: &SmolVlm,
    backend: &CpuBackend,
    tokenizer: &EmberTokenizer,
    command: &MultimodalCommand,
    prompt: &str,
    dir: &std::path::Path,
) -> Result<()> {
    use ember::model::ForwardModel;
    std::fs::create_dir_all(dir)?;

    let (trace, sequence) = {
        let parts: Vec<ember::multimodal::InputPart> =
            std::iter::once(ember::multimodal::InputPart::Text(prompt.to_string()))
                .chain(
                    command
                        .image
                        .iter()
                        .map(|p| ember::multimodal::InputPart::Image(std::path::PathBuf::from(p))),
                )
                .collect();
        model.build_inputs_parts(backend, tokenizer, &parts, 0)?
    };

    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();

    // 1. processed pixel tensor: all tiles of all images concatenated
    //    ([total_tiles, 3, 512, 512]; matches the reference pixel_values)
    if !trace.images.is_empty() {
        let first = &trace.images[0].tiles;
        let (channels, height, width) = (first.shape()[1], first.shape()[2], first.shape()[3]);
        let tile_len = channels * height * width;
        let total_tiles: usize = trace.images.iter().map(|p| p.tiles.shape()[0]).sum();
        let mut pixels = vec![0.0f32; total_tiles * tile_len];
        let mut off = 0usize;
        for p in &trace.images {
            pixels[off..off + p.tiles.len()].copy_from_slice(p.tiles.data());
            off += p.tiles.len();
        }
        let tiles =
            ember::tensor::CpuTensor::from_data(vec![total_tiles, channels, height, width], pixels);
        write_bin(dir, "1_pixels", &tiles, &mut shapes)?;
    }

    // 2. patch embeddings [n*1024, 768]
    if let Some(p) = &trace.vision.patch_embeddings {
        write_bin(dir, "2_patch_embeddings", p, &mut shapes)?;
    }

    // 3. selected layer outputs
    for (i, layer_out) in trace.vision.layer_outputs.iter().enumerate() {
        if matches!(i, 0 | 1 | 5 | 11) {
            write_bin(dir, &format!("3_layer_{i}"), layer_out, &mut shapes)?;
        }
    }

    // 4. encoder output
    if let Some(e) = &trace.vision.encoder_output {
        write_bin(dir, "4_encoder_output", e, &mut shapes)?;
    }

    // 5. projector output [n*64, 576]
    write_bin(
        dir,
        "5_projector_output",
        &trace.projector_output,
        &mut shapes,
    )?;

    // 6. assembled LLM input embeddings [seq, 576]
    write_bin(
        dir,
        "6_assembled_embeddings",
        &trace.assembled_embeddings,
        &mut shapes,
    )?;

    // 7. first LLM logits (last position, [1, vocab])
    let mut cache = model
        .llm
        .create_cache(backend, model.llm.max_seq_len(backend));
    let logits = model.llm.forward_last_logits_embeddings_with_cache(
        backend,
        &sequence.embeddings,
        &mut cache,
        0,
    )?;
    write_bin(dir, "7_first_logits", &logits, &mut shapes)?;

    // 8. greedy generation ids + per-step logits (for near-tie analysis)
    let parts: Vec<ember::multimodal::InputPart> =
        std::iter::once(ember::multimodal::InputPart::Text(prompt.to_string()))
            .chain(
                command
                    .image
                    .iter()
                    .map(|p| ember::multimodal::InputPart::Image(std::path::PathBuf::from(p))),
            )
            .collect();
    let (generated, text, timings) =
        model.generate_with_parts(backend, tokenizer, &parts, command.max_tokens)?;
    {
        // replay the decode loop on a fresh cache to capture step logits.
        // Step 0 is the prefill's last-position logits; subsequent steps
        // decode generated tokens (matching the reference generate()).
        let mut cache = model
            .llm
            .create_cache(backend, model.llm.max_seq_len(backend));
        let eos_ids = tokenizer.eos_token_ids();
        let vocab = model.llm.vocab_size(backend);
        let mut step_logits: Vec<f32> = Vec::new();
        let start_pos = trace.input_ids.len();
        let mut logits = model.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &sequence.embeddings,
            &mut cache,
            0,
        )?;
        for step in 0..command.max_tokens {
            let data = backend.data(&logits);
            step_logits.extend_from_slice(&data[..vocab]);
            let best = ember::sampler::argmax_token(data);
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
        "mmproj": command.mmproj,
        "image": command.image,
        "prompt": prompt,
        "max_tokens": command.max_tokens,
        "input_ids": trace.input_ids,
        "input_ids_len": trace.input_ids.len(),
        "generation_ids": generated,
        "generated_text": text,
        "tile_grids": trace
            .images
            .iter()
            .map(|p| serde_json::json!([p.tile_grid.0, p.tile_grid.1]))
            .collect::<Vec<_>>(),
        "original_dims": trace
            .images
            .iter()
            .map(|p| serde_json::json!([p.original_dims.0, p.original_dims.1]))
            .collect::<Vec<_>>(),
        "resized_dims": trace
            .images
            .iter()
            .map(|p| serde_json::json!([p.resized_dims.0, p.resized_dims.1]))
            .collect::<Vec<_>>(),
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

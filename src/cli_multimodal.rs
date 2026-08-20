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

    /// image file (PNG/JPEG)
    #[arg(long)]
    image: String,

    /// user prompt; use <image> as the image placeholder (image-first)
    #[arg(long, default_value = "What is shown in this image?")]
    prompt: String,

    /// number of tokens to generate
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// write progressive-validation artifacts to this directory
    #[arg(long)]
    dump_dir: Option<PathBuf>,
}

pub(crate) fn run_multimodal_command(command: &MultimodalCommand, _args: &Args) -> Result<()> {
    let backend = CpuBackend;
    let model = SmolVlm::from_ggufs(
        std::path::Path::new(&command.model),
        std::path::Path::new(&command.mmproj),
    )?;
    let tokenizer = EmberTokenizer::from_file(&command.tokenizer)
        .with_context(|| format!("failed to load tokenizer {}", command.tokenizer))?;

    // Reference message structure: content = [image, text], so the prompt
    // is image-first. Prepend the placeholder when the user did not include
    // one (the assembler itself fails closed without it).
    let prompt = if command.prompt.contains("<image>") {
        command.prompt.clone()
    } else {
        format!("<image>{}", command.prompt)
    };
    let (generated, text, timings) = model.generate_with_image(
        &backend,
        &tokenizer,
        std::path::Path::new(&command.image),
        &prompt,
        command.max_tokens,
    )?;

    println!("generated ({} tokens): {}", generated.len(), text);
    print_timings(&timings);

    if let Some(dir) = &command.dump_dir {
        dump_validation_artifacts(&model, &backend, &tokenizer, command, &prompt, dir)?;
    }
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

    let (trace, sequence) = model.build_inputs_with_tokenizer(
        backend,
        tokenizer,
        std::path::Path::new(&command.image),
        prompt,
        0,
    )?;

    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();

    // 1. processed pixel tensor [n, 3, 512, 512]
    write_bin(dir, "1_pixels", &trace.processed.tiles, &mut shapes)?;

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
    let (generated, text, timings) = model.generate_with_image(
        backend,
        tokenizer,
        std::path::Path::new(&command.image),
        prompt,
        command.max_tokens,
    )?;
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
        "tile_grid": [trace.processed.tile_grid.0, trace.processed.tile_grid.1],
        "original_dims": [trace.processed.original_dims.0, trace.processed.original_dims.1],
        "resized_dims": [trace.processed.resized_dims.0, trace.processed.resized_dims.1],
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

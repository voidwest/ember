//! Probe-mode hidden-state extraction.
//! Split out of `main.rs` (2026-08-01) to keep the CLI dispatcher thin.

use crate::cli_commands::{effective_context_limit, ensure_sequence_fits};
use crate::cli_support::default_tokenizer_for_arch;
use crate::{Args, RunMetadata};
use anyhow::Context;
use ember::backend::Backend;
use ember::extraction::{git_commit, unix_timestamp};
use ember::model::ForwardModel;
use ember::npy::NpyStreamWriter;
use ember::sampler::argmax_token;
use std::fs;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(crate) enum ProbePosition {
    Last,
    Root,
    Pattern,
    PromptMean,
}

pub(crate) struct ProbeJob {
    template: String,
    position: ProbePosition,
    output_path: String,
}

pub(crate) struct ProbeOutput {
    position: ProbePosition,
    output_path: String,
}

pub(crate) struct ProbeGroupConfig<'a> {
    stimuli_path: &'a str,
    template: &'a str,
    outputs: Vec<ProbeOutput>,
    generate_tokens: usize,
    limit: Option<usize>,
    context_limit: usize,
    model_path: &'a str,
    arch: &'a str,
    tokenizer_path: &'a str,
    run_metadata: &'a RunMetadata,
}

pub(crate) struct LogitDumpConfig<'a> {
    pub(crate) prompt: &'a str,
    pub(crate) output_path: &'a str,
    pub(crate) max_seq_len: Option<usize>,
    pub(crate) model_path: &'a str,
    pub(crate) arch: &'a str,
    pub(crate) tokenizer_path: &'a str,
    pub(crate) run_metadata: &'a RunMetadata,
}

impl ProbePosition {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "last" => Ok(Self::Last),
            "root" => Ok(Self::Root),
            "pattern" => Ok(Self::Pattern),
            "prompt_mean" => Ok(Self::PromptMean),
            _ => anyhow::bail!(
                "unknown probe position '{}'; expected last, root, pattern, or prompt_mean",
                value
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Last => "last",
            Self::Root => "root",
            Self::Pattern => "pattern",
            Self::PromptMean => "prompt_mean",
        }
    }
}

pub(crate) fn split_probe_list(value: Option<&String>, fallback: &str) -> Vec<String> {
    value
        .map(|s| s.as_str())
        .unwrap_or(fallback)
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(crate) fn sanitize_probe_path_part(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn build_probe_jobs(args: &Args) -> anyhow::Result<Vec<ProbeJob>> {
    let templates = split_probe_list(args.probe_templates.as_ref(), &args.probe_template);
    let positions = split_probe_list(args.probe_positions.as_ref(), &args.probe_position);
    if templates.is_empty() {
        anyhow::bail!("probe template list is empty");
    }
    if positions.is_empty() {
        anyhow::bail!("probe position list is empty");
    }

    let is_batch = args.probe_templates.is_some()
        || args.probe_positions.is_some()
        || args.probe_output_dir.is_some()
        || templates.len() > 1
        || positions.len() > 1;
    if !is_batch {
        return Ok(vec![ProbeJob {
            template: templates[0].clone(),
            position: ProbePosition::parse(&positions[0])?,
            output_path: args.probe_output.clone(),
        }]);
    }

    let output_dir = args
        .probe_output_dir
        .clone()
        .unwrap_or_else(|| "data/probe_matrix".to_string());
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create probe output directory: {output_dir}"))?;

    let mut jobs = Vec::with_capacity(templates.len() * positions.len());
    let prefix = sanitize_probe_path_part(&args.probe_output_prefix);
    for template in templates {
        let template_part = sanitize_probe_path_part(&template);
        for position_value in &positions {
            let position = ProbePosition::parse(position_value)?;
            let output_path = format!(
                "{}/{}_{}_{}_activations.npy",
                output_dir,
                prefix,
                template_part,
                position.as_str()
            );
            jobs.push(ProbeJob {
                template: template.clone(),
                position,
                output_path,
            });
        }
    }
    Ok(jobs)
}

pub(crate) fn run_probe_jobs<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    args: &Args,
    run_metadata: &RunMetadata,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let jobs = build_probe_jobs(args)?;
    eprintln!("running {} probe extraction job(s)", jobs.len());

    let mut grouped: Vec<(String, Vec<&ProbeJob>)> = Vec::new();
    for job in &jobs {
        if let Some((_, group_jobs)) = grouped
            .iter_mut()
            .find(|(template, _)| template == &job.template)
        {
            group_jobs.push(job);
        } else {
            grouped.push((job.template.clone(), vec![job]));
        }
    }

    let total_groups = grouped.len();
    for (group_idx, (template, group_jobs)) in grouped.into_iter().enumerate() {
        let outputs = group_jobs
            .iter()
            .map(|job| ProbeOutput {
                position: job.position,
                output_path: job.output_path.clone(),
            })
            .collect::<Vec<_>>();
        let positions = outputs
            .iter()
            .map(|output| output.position.as_str())
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "\n=== probe job group {}/{}: template={} positions={} ===",
            group_idx + 1,
            total_groups,
            template,
            positions
        );
        probe_group_mode(
            backend,
            model,
            tokenizer,
            ProbeGroupConfig {
                stimuli_path: &args.probe_stimuli,
                template: &template,
                outputs,
                generate_tokens: args.probe_generate_tokens,
                limit: args.probe_limit,
                context_limit: effective_context_limit(backend, model, args),
                model_path: &args.model,
                arch: &args.arch,
                tokenizer_path: args
                    .tokenizer
                    .as_deref()
                    .unwrap_or_else(|| default_tokenizer_for_arch(&args.arch)),
                run_metadata,
            },
        )?;
    }
    Ok(())
}

pub(crate) fn token_indices_for_offsets(
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
) -> Vec<usize> {
    offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(tok_start, tok_end))| {
            if tok_start != tok_end && tok_start < end && tok_end > start {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn non_special_token_indices(
    offsets: &[(usize, usize)],
    token_count: usize,
) -> Vec<usize> {
    let indices: Vec<usize> = offsets
        .iter()
        .enumerate()
        .filter_map(|(i, &(start, end))| if start != end { Some(i) } else { None })
        .collect();
    if indices.is_empty() {
        (0..token_count).collect()
    } else {
        indices
    }
}

pub(crate) fn stimulus_text_field(
    stimulus: &serde_json::Value,
    field: &str,
) -> anyhow::Result<String> {
    stimulus[field]
        .as_str()
        .map(str::to_owned)
        .with_context(|| format!("stimulus missing string field '{}'", field))
}

pub(crate) fn select_probe_indices(
    prompt: &str,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    stimulus: &serde_json::Value,
    position: ProbePosition,
) -> anyhow::Result<Vec<usize>> {
    match position {
        ProbePosition::Last => {
            let indices = non_special_token_indices(offsets, token_ids.len());
            indices
                .last()
                .copied()
                .map(|i| vec![i])
                .context("cannot select last token from empty prompt")
        }
        ProbePosition::PromptMean => Ok(non_special_token_indices(offsets, token_ids.len())),
        ProbePosition::Root | ProbePosition::Pattern => {
            let field = match position {
                ProbePosition::Root => "root",
                ProbePosition::Pattern => "pattern",
                _ => unreachable!(),
            };
            let needle = stimulus_text_field(stimulus, field)?;
            let start = prompt.find(&needle).with_context(|| {
                format!(
                    "could not locate {} '{}' in selected prompt template",
                    field, needle
                )
            })?;
            let indices = token_indices_for_offsets(offsets, start, start + needle.len());
            if indices.is_empty() {
                anyhow::bail!(
                    "could not map {} '{}' to tokenizer offsets in selected prompt template",
                    field,
                    needle
                );
            }
            Ok(indices)
        }
    }
}

pub(crate) fn normalize_for_match(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

pub(crate) fn match_generated_text(generated: &str, expected: &str) -> (bool, bool) {
    let generated_norm = normalize_for_match(generated);
    let expected_norm = normalize_for_match(expected);
    if expected_norm.is_empty() {
        return (false, false);
    }
    (
        generated_norm == expected_norm,
        generated_norm.contains(&expected_norm),
    )
}

#[inline]
pub(crate) fn has_next_decode_evaluation(step: usize, max_tokens: usize) -> bool {
    step + 1 < max_tokens
}

pub(crate) fn generate_probe_continuation<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    context_limit: usize,
) -> anyhow::Result<(Vec<u32>, String)>
where
    B::Error: Send + Sync + 'static,
{
    if max_tokens == 0 {
        return Ok((Vec::new(), String::new()));
    }

    let prompt_len = prompt_tokens.len();
    let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_last_logits_with_cache(backend, prompt_tokens, &mut cache, 0)?;
    let vocab_size = backend.shape(&logits)[1];
    let mut generated = Vec::with_capacity(max_tokens);

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = &logit_data[..vocab_size];
        let next_token = argmax_token(last_logits);

        let eos_ids = tokenizer.eos_token_ids();
        if eos_ids.contains(&(next_token as u32)) {
            break;
        }

        generated.push(next_token as u32);
        if !has_next_decode_evaluation(step, max_tokens) {
            break;
        }
        logits = model.forward_last_logits_with_cache(
            backend,
            &[next_token as u32],
            &mut cache,
            prompt_len + step,
        )?;
    }

    let generated_text = tokenizer.decode(&generated)?;
    Ok((generated, generated_text))
}

/// probe mode: feed each stimulus prompt through the model and collect pooled
/// per-layer hidden states for one or more selected token positions.
///
/// Writes one 3d .npy file per requested position: `(n_stimuli, n_layers, embed_dim)`.
pub(crate) fn probe_group_mode<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    config: ProbeGroupConfig<'_>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    if config.outputs.is_empty() {
        anyhow::bail!("probe group has no outputs");
    }

    // -- load stimuli ------------------------------------------
    let stimuli_json = fs::read_to_string(config.stimuli_path)
        .with_context(|| format!("failed to read stimuli file: {}", config.stimuli_path))?;
    let mut stimuli: Vec<serde_json::Value> = serde_json::from_str(&stimuli_json)?;
    if let Some(limit) = config.limit {
        stimuli.truncate(limit);
    }
    eprintln!(
        "loaded {} stimuli from {}",
        stimuli.len(),
        config.stimuli_path
    );

    let n_layers = model.n_layers();
    let embed_dim = model.embed_dim();
    eprintln!("model: {} layers, {} hidden dim", n_layers, embed_dim);

    let shape = [stimuli.len(), n_layers, embed_dim];
    let row_floats = n_layers * embed_dim;
    eprintln!(
        "streaming {} activation file(s): {} floats per row ({:.1} KB per row)",
        config.outputs.len(),
        row_floats,
        row_floats as f64 * 4.0 / 1024.0
    );
    let mut activation_writers = config
        .outputs
        .iter()
        .map(|output| NpyStreamWriter::create(&output.output_path, &shape))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let zero_activation_row = vec![0.0f32; row_floats];

    // -- collect -----------------------------------------------
    let start = Instant::now();
    let mut correctness: Vec<Vec<serde_json::Value>> = config
        .outputs
        .iter()
        .map(|_| Vec::with_capacity(stimuli.len()))
        .collect();
    let mut token_selections: Vec<Vec<serde_json::Value>> = config
        .outputs
        .iter()
        .map(|_| Vec::with_capacity(stimuli.len()))
        .collect();

    // batched extraction: concatenate all stimuli into one sequence
    // with block-diagonal attention masking for independent processing.
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut block_boundaries: Vec<usize> = Vec::new();
    let mut block_token_counts: Vec<usize> = Vec::new();
    let mut stimulus_info: Vec<(String, serde_json::Value, Vec<Vec<usize>>)> = Vec::new();

    for (si, stimulus) in stimuli.iter().enumerate() {
        let prompt = stimulus["prompts"][config.template]
            .as_str()
            .with_context(|| {
                format!(
                    "stimulus {} missing prompt template '{}'",
                    si, config.template
                )
            })?;

        let (token_ids, offsets) = tokenizer.encode_with_offsets(prompt)?;
        if token_ids.is_empty() {
            eprintln!(
                "  [{}/{}] WARNING: empty tokenization, skipping",
                si + 1,
                stimuli.len()
            );
            // write zero activation row for this stimulus
            for writer in &mut activation_writers {
                writer.write_f32s(&zero_activation_row)?;
            }
            continue;
        }

        block_boundaries.push(all_token_ids.len());
        block_token_counts.push(token_ids.len());
        let probe_indices = config
            .outputs
            .iter()
            .map(|output| {
                select_probe_indices(prompt, &token_ids, &offsets, stimulus, output.position)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        all_token_ids.extend_from_slice(&token_ids);
        stimulus_info.push((prompt.to_string(), stimulus.clone(), probe_indices));
    }

    if block_boundaries.is_empty() {
        eprintln!("no valid stimuli to process");
        return Ok(());
    }
    let total_tokens = all_token_ids.len();
    let context_limit = config.context_limit;
    eprintln!(
        "batched {} stimuli into {} total tokens ({} blocks), context limit {}",
        stimulus_info.len(),
        total_tokens,
        block_boundaries.len(),
        context_limit
    );

    // split into chunks that fit within the context limit
    let n_outputs = config.outputs.len();
    let n_stimuli = stimulus_info.len();
    let mut chunk_start = 0usize; // stimulus index
    let mut global_stimulus_idx = 0usize;

    while chunk_start < n_stimuli {
        // find how many stimuli fit in this chunk
        let mut chunk_end = chunk_start;
        let mut chunk_tokens = 0usize;
        while chunk_end < n_stimuli {
            let next_tokens = chunk_tokens + block_token_counts[chunk_end];
            if next_tokens > context_limit && chunk_tokens > 0 {
                break;
            }
            chunk_tokens = next_tokens;
            chunk_end += 1;
        }

        let chunk_boundaries = &block_boundaries[chunk_start..chunk_end];
        // remap boundaries relative to chunk start token position
        let chunk_base = chunk_boundaries[0];
        let chunk_token_ids = &all_token_ids[chunk_base..chunk_base + chunk_tokens];
        let remapped_boundaries: Vec<usize> =
            chunk_boundaries.iter().map(|b| b - chunk_base).collect();

        // build probe index groups for this chunk only
        let mut chunk_probe_indices: Vec<Vec<usize>> =
            Vec::with_capacity((chunk_end - chunk_start) * n_outputs);

        for bi in chunk_start..chunk_end {
            let base = block_boundaries[bi];
            for local_indices in &stimulus_info[bi].2 {
                let absolute: Vec<usize> = local_indices
                    .iter()
                    .map(|i| (base - chunk_base) + i)
                    .collect();
                chunk_probe_indices.push(absolute);
            }
        }

        eprintln!(
            "  chunk [{}-{}]: {} stimuli, {} tokens",
            chunk_start,
            chunk_end - 1,
            chunk_end - chunk_start,
            chunk_tokens
        );

        // batched forward pass for this chunk
        let (pooled_states, logits) = model.forward_pooled_with_blocks(
            backend,
            chunk_token_ids,
            &remapped_boundaries,
            &chunk_probe_indices,
        )?;

        // write pooled states and correctness for each stimulus in this chunk
        for (local_bi, bi) in (chunk_start..chunk_end).enumerate() {
            let base = chunk_boundaries[local_bi];
            let n_tokens = block_token_counts[bi];
            let token_slice = &all_token_ids[base..base + n_tokens];

            // Block-aware probing returns one last-token logit row per stimulus.
            let logit_data = backend.data(&logits);
            let logit_shape = backend.shape(&logits);
            let vocab_size = logit_shape[1];
            let last_row_start = local_bi * vocab_size;
            let last_logits = &logit_data[last_row_start..last_row_start + vocab_size];
            let predicted_id = last_logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(max_i, max_v), (i, &v)| {
                    if v > max_v {
                        (i, v)
                    } else {
                        (max_i, max_v)
                    }
                })
                .0;
            let predicted_text = tokenizer.decode(&[predicted_id as u32])?;
            let (generated_ids, generated_text) = generate_probe_continuation(
                backend,
                model,
                tokenizer,
                token_slice,
                config.generate_tokens,
                config.context_limit,
            )?;
            let stimulus = &stimulus_info[bi].1;
            let expected = stimulus["expected_surface"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let (generated_exact_match, generated_contains_match) =
                match_generated_text(&generated_text, &expected);

            for (oi, output) in config.outputs.iter().enumerate() {
                // extract pooled states: index = local_bi * n_outputs + oi
                let pool_idx = local_bi * n_outputs + oi;
                let pooled_slice = &pooled_states[pool_idx];

                let probe_indices = &stimulus_info[bi].2[oi];

                correctness[oi].push(serde_json::json!({
                    "index": bi,
                    "root": stimulus["root"],
                    "pattern": stimulus["pattern"],
                    "expected": expected,
                    "predicted": predicted_text.trim().to_string(),
                    "predicted_id": predicted_id,
                    "next_token_predicted": predicted_text.trim().to_string(),
                    "next_token_id": predicted_id,
                    "generated": generated_text.trim().to_string(),
                    "generated_ids": generated_ids,
                    "generated_exact_match": generated_exact_match,
                    "generated_contains_match": generated_contains_match,
                    "correct": generated_exact_match || generated_contains_match,
                    "probe_template": config.template,
                    "probe_position": output.position.as_str(),
                    "probe_generate_tokens": config.generate_tokens,
                    "probe_token_indices": probe_indices,
                }));
                token_selections[oi].push(serde_json::json!({
                    "index": bi,
                    "token_count": n_tokens,
                    "probe_token_indices": probe_indices,
                }));

                activation_writers[oi].write_f32s(pooled_slice)?;
            }

            global_stimulus_idx += 1;
            if global_stimulus_idx.is_multiple_of(100) || global_stimulus_idx == n_stimuli {
                eprintln!(
                    "  [{:4}/{}] saved in {:.1}s",
                    global_stimulus_idx,
                    n_stimuli,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        chunk_start = chunk_end;
    } // end while chunk_start < n_stimuli

    // -- save --------------------------------------------------
    for (writer, output) in activation_writers.iter_mut().zip(&config.outputs) {
        writer.finish()?;
        eprintln!("saved activations to {}", output.output_path);
    }

    for (oi, output) in config.outputs.iter().enumerate() {
        let correct_count = correctness[oi]
            .iter()
            .filter(|c| {
                c["correct"]
                    .as_bool()
                    .unwrap_or(c["predicted"] == c["expected"])
            })
            .count();
        let correctness_pct = if correctness[oi].is_empty() {
            0.0
        } else {
            correct_count as f64 / correctness[oi].len() as f64 * 100.0
        };
        eprintln!(
            "correctness [{}]: {}/{} ({:.1}%)",
            output.position.as_str(),
            correct_count,
            correctness[oi].len(),
            correctness_pct
        );

        let correctness_path = output.output_path.replace(".npy", "_correctness.json");
        fs::write(
            &correctness_path,
            serde_json::to_string_pretty(&correctness[oi])?,
        )?;
        eprintln!("saved correctness to {}", correctness_path);

        let metadata = serde_json::json!({
            "model_path": config.model_path,
            "architecture": config.arch,
            "tokenizer_path": config.tokenizer_path,
            "tokenizer_sha256": config.run_metadata.tokenizer_sha256,
            "model_file_size_bytes": config.run_metadata.model_file_size_bytes,
            "model_sha256": config.run_metadata.model_sha256,
            "gguf_metadata": config.run_metadata.gguf_metadata,
            "run_manifest": config.run_metadata.run_manifest,
            "stimuli_path": config.stimuli_path,
            "output_path": output.output_path,
            "probe_template": config.template,
            "probe_position": output.position.as_str(),
            "probe_generate_tokens": config.generate_tokens,
            "probe_limit": config.limit,
            "context_limit": config.context_limit,
            "n_stimuli": stimuli.len(),
            "n_layers": n_layers,
            "embed_dim": embed_dim,
            "activation_shape": shape,
            "correctness_path": correctness_path,
            "token_selections": token_selections[oi],
            "run_timestamp_unix": unix_timestamp(),
            "git_commit": git_commit(),
            "batched_probe_extraction": true,
            "batched_probe_positions": config
                .outputs
                .iter()
                .map(|output| output.position.as_str())
                .collect::<Vec<_>>(),
        });
        let metadata_path = output.output_path.replace(".npy", "_metadata.json");
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        eprintln!("saved metadata to {}", metadata_path);
    }

    eprintln!("done in {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

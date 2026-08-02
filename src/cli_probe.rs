//! Probe-mode hidden-state extraction.
//! Split out of `main.rs` (2026-08-01) to keep the CLI dispatcher thin.

use crate::cli_commands::{effective_context_limit, ensure_sequence_fits};
use crate::cli_support::{sidecar_path, validate_token_ids_for_model, write_json_file};
use crate::{Args, RunMetadata};
use anyhow::Context;
use ember::backend::Backend;
use ember::extraction::{
    byte_span_to_character_span, git_commit, sha256_file_result, unique_substring_byte_span,
    unix_timestamp,
};
use ember::model::ForwardModel;
use ember::npy::NpyStreamWriter;
use ember::sampler::argmax_token;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ProbePosition {
    Last,
    Root,
    Pattern,
    PromptMean,
}

#[derive(Debug)]
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

pub(crate) struct TensorDumpConfig<'a> {
    pub(crate) prompt: &'a str,
    pub(crate) output_path: &'a str,
    pub(crate) max_seq_len: Option<usize>,
    pub(crate) model_path: &'a str,
    pub(crate) arch: &'a str,
    pub(crate) tokenizer_path: &'a str,
    pub(crate) run_metadata: &'a RunMetadata,
}

struct StimulusInfo {
    source_index: usize,
    prompt: String,
    stimulus: serde_json::Value,
    probe_indices: Vec<Vec<usize>>,
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
    let mut seen_templates = HashSet::new();
    for template in &templates {
        if !seen_templates.insert(template) {
            anyhow::bail!("duplicate probe template '{template}'");
        }
    }
    let parsed_positions = positions
        .iter()
        .map(|position| ProbePosition::parse(position))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut seen_positions = HashSet::new();
    for position in &parsed_positions {
        if !seen_positions.insert(*position) {
            anyhow::bail!("duplicate probe position '{}'", position.as_str());
        }
    }

    let is_batch = args.probe_templates.is_some()
        || args.probe_positions.is_some()
        || args.probe_output_dir.is_some()
        || templates.len() > 1
        || positions.len() > 1;
    if !is_batch {
        return Ok(vec![ProbeJob {
            template: templates[0].clone(),
            position: parsed_positions[0],
            output_path: args.probe_output.clone(),
        }]);
    }

    let output_dir = args
        .probe_output_dir
        .clone()
        .unwrap_or_else(|| "data/probe_matrix".to_string());

    let mut jobs = Vec::with_capacity(templates.len() * positions.len());
    let prefix = sanitize_probe_path_part(&args.probe_output_prefix);
    if prefix.is_empty() {
        anyhow::bail!("probe output prefix contains no filename-safe characters");
    }
    let mut output_paths = HashSet::new();
    for template in templates {
        let template_part = sanitize_probe_path_part(&template);
        if template_part.is_empty() {
            anyhow::bail!("probe template '{template}' contains no filename-safe characters");
        }
        for position in &parsed_positions {
            let output_path = Path::new(&output_dir)
                .join(format!(
                    "{}_{}_{}_activations.npy",
                    prefix,
                    template_part,
                    position.as_str()
                ))
                .to_string_lossy()
                .into_owned();
            if !output_paths.insert(output_path.clone()) {
                anyhow::bail!(
                    "probe output path collision after filename sanitization: {output_path}"
                );
            }
            jobs.push(ProbeJob {
                template: template.clone(),
                position: *position,
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
    tokenizer_path: &str,
    run_metadata: &RunMetadata,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let jobs = build_probe_jobs(args)?;
    for job in &jobs {
        if let Some(parent) = Path::new(&job.output_path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create probe output directory: {}",
                    parent.display()
                )
            })?;
        }
    }
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
                tokenizer_path,
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
            let byte_span = unique_substring_byte_span(prompt, &needle).with_context(|| {
                format!("could not select {field} '{needle}' in selected prompt template")
            })?;
            let [start, end] = byte_span_to_character_span(prompt, byte_span)?;
            let indices = token_indices_for_offsets(offsets, start, end);
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

fn validate_probe_logits<B: Backend>(
    backend: &B,
    logits: &B::Tensor,
    expected_rows: usize,
    expected_vocab_size: usize,
) -> anyhow::Result<()> {
    let shape = backend.shape(logits);
    if shape != [expected_rows, expected_vocab_size] {
        anyhow::bail!(
            "probe logits shape mismatch: expected [{expected_rows}, {expected_vocab_size}], got {shape:?}"
        );
    }
    let expected_values = expected_rows
        .checked_mul(expected_vocab_size)
        .context("probe logits shape product overflow")?;
    let values = backend.data(logits);
    if values.len() != expected_values {
        anyhow::bail!(
            "probe logits payload has {} values, expected {expected_values}",
            values.len()
        );
    }
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("probe logits contain non-finite value {value} at flat index {index}");
    }
    Ok(())
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
    if prompt_tokens.is_empty() {
        anyhow::bail!("cannot generate a probe continuation from an empty token sequence");
    }

    let prompt_len = prompt_tokens.len();
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    validate_token_ids_for_model(prompt_tokens, model_vocab_size, "probe prompt")?;
    let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_last_logits_with_cache(backend, prompt_tokens, &mut cache, 0)?;
    validate_probe_logits(backend, &logits, 1, model_vocab_size)?;
    let vocab_size = model_vocab_size;
    let mut generated = Vec::with_capacity(max_tokens);
    let eos_ids = tokenizer.eos_token_ids();

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = &logit_data[..vocab_size];
        let next_token = argmax_token(last_logits);
        let next_token = u32::try_from(next_token).context("probe token ID exceeds u32")?;
        if !tokenizer.contains_token_id(next_token) {
            anyhow::bail!(
                "model selected token ID {next_token}, but the tokenizer cannot decode it"
            );
        }

        if eos_ids.contains(&next_token) {
            break;
        }

        generated.push(next_token);
        if !has_next_decode_evaluation(step, max_tokens) {
            break;
        }
        logits = model.forward_last_logits_with_cache(
            backend,
            &[next_token],
            &mut cache,
            prompt_len + step,
        )?;
        validate_probe_logits(backend, &logits, 1, model_vocab_size)?;
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
    if stimuli.is_empty() {
        anyhow::bail!("probe stimuli are empty after applying --probe-limit");
    }
    eprintln!(
        "loaded {} stimuli from {}",
        stimuli.len(),
        config.stimuli_path
    );

    let n_layers = model.n_layers();
    let embed_dim = model.embed_dim();
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    if n_layers == 0 || embed_dim == 0 {
        anyhow::bail!(
            "probe model dimensions must be non-zero, got {n_layers} layers and hidden width {embed_dim}"
        );
    }
    eprintln!("model: {} layers, {} hidden dim", n_layers, embed_dim);

    let shape = [stimuli.len(), n_layers, embed_dim];
    let row_floats = n_layers
        .checked_mul(embed_dim)
        .context("probe activation row size overflow")?;
    eprintln!(
        "streaming {} activation file(s): {} floats per row ({:.1} KB per row)",
        config.outputs.len(),
        row_floats,
        row_floats as f64 * 4.0 / 1024.0
    );
    // batched extraction: concatenate all stimuli into one sequence
    // with block-diagonal attention masking for independent processing.
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut block_boundaries: Vec<usize> = Vec::new();
    let mut block_token_counts: Vec<usize> = Vec::new();
    let mut stimulus_info: Vec<StimulusInfo> = Vec::with_capacity(stimuli.len());

    for (si, stimulus) in stimuli.iter().enumerate() {
        if !stimulus.is_object() {
            anyhow::bail!("stimulus {si} must be a JSON object");
        }
        for required in ["root", "pattern", "expected_surface"] {
            stimulus[required]
                .as_str()
                .with_context(|| format!("stimulus {si} missing string field '{required}'"))?;
        }
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
            anyhow::bail!("stimulus {si} produced no token IDs");
        }
        validate_token_ids_for_model(&token_ids, model_vocab_size, &format!("stimulus {si}"))?;
        if token_ids.len() > config.context_limit {
            anyhow::bail!(
                "stimulus {si} has {} tokens, exceeding context limit {}",
                token_ids.len(),
                config.context_limit
            );
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
        all_token_ids
            .len()
            .checked_add(token_ids.len())
            .context("combined probe token count overflow")?;
        all_token_ids.extend_from_slice(&token_ids);
        stimulus_info.push(StimulusInfo {
            source_index: si,
            prompt: prompt.to_string(),
            stimulus: stimulus.clone(),
            probe_indices,
        });
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

    let mut activation_writers = config
        .outputs
        .iter()
        .map(|output| NpyStreamWriter::create(&output.output_path, &shape))
        .collect::<anyhow::Result<Vec<_>>>()?;

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
            let next_tokens = chunk_tokens
                .checked_add(block_token_counts[chunk_end])
                .context("probe chunk token count overflow")?;
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
            for local_indices in &stimulus_info[bi].probe_indices {
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
        let chunk_stimuli = chunk_end - chunk_start;
        let expected_pools = chunk_stimuli
            .checked_mul(n_outputs)
            .context("probe pool count overflow")?;
        if pooled_states.len() != expected_pools {
            anyhow::bail!(
                "model returned {} pooled activation rows, expected {expected_pools}",
                pooled_states.len()
            );
        }
        for (pool_index, values) in pooled_states.iter().enumerate() {
            if values.len() != row_floats {
                anyhow::bail!(
                    "pooled activation row {pool_index} has {} values, expected {row_floats}",
                    values.len()
                );
            }
            if let Some((value_index, value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                anyhow::bail!(
                    "pooled activation row {pool_index} contains non-finite value {value} at index {value_index}"
                );
            }
        }
        validate_probe_logits(backend, &logits, chunk_stimuli, model_vocab_size)?;

        // write pooled states and correctness for each stimulus in this chunk
        for (local_bi, bi) in (chunk_start..chunk_end).enumerate() {
            let base = chunk_boundaries[local_bi];
            let n_tokens = block_token_counts[bi];
            let token_slice = &all_token_ids[base..base + n_tokens];

            // Block-aware probing returns one last-token logit row per stimulus.
            let logit_data = backend.data(&logits);
            let vocab_size = model_vocab_size;
            let last_row_start = local_bi * vocab_size;
            let last_logits = &logit_data[last_row_start..last_row_start + vocab_size];
            let predicted_id = argmax_token(last_logits);
            let predicted_id_u32 =
                u32::try_from(predicted_id).context("predicted token ID exceeds u32")?;
            if !tokenizer.contains_token_id(predicted_id_u32) {
                anyhow::bail!(
                    "model predicted token ID {predicted_id}, but the tokenizer cannot decode it"
                );
            }
            let predicted_text = tokenizer.decode(&[predicted_id_u32])?;
            let (generated_ids, generated_text) = generate_probe_continuation(
                backend,
                model,
                tokenizer,
                token_slice,
                config.generate_tokens,
                config.context_limit,
            )?;
            let info = &stimulus_info[bi];
            let stimulus = &info.stimulus;
            let expected = stimulus["expected_surface"]
                .as_str()
                .expect("expected_surface validated during preprocessing")
                .to_string();
            let (generated_exact_match, generated_contains_match) =
                match_generated_text(&generated_text, &expected);

            for (oi, output) in config.outputs.iter().enumerate() {
                // extract pooled states: index = local_bi * n_outputs + oi
                let pool_idx = local_bi * n_outputs + oi;
                let pooled_slice = &pooled_states[pool_idx];

                let probe_indices = &info.probe_indices[oi];

                correctness[oi].push(serde_json::json!({
                    "index": info.source_index,
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
                    "index": info.source_index,
                    "prompt": info.prompt,
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
        if correctness[oi].len() != stimuli.len() || token_selections[oi].len() != stimuli.len() {
            anyhow::bail!(
                "probe row-alignment failure for {}: {} correctness rows and {} token selections for {} stimuli",
                output.output_path,
                correctness[oi].len(),
                token_selections[oi].len(),
                stimuli.len()
            );
        }
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

        let correctness_path = sidecar_path(&output.output_path, "_correctness.json")?;
        write_json_file(
            &correctness_path,
            &serde_json::Value::Array(correctness[oi].clone()),
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
            "stimuli_sha256": sha256_file_result(config.stimuli_path)?,
            "output_path": output.output_path,
            "activations_sha256": sha256_file_result(&output.output_path)?,
            "probe_template": config.template,
            "probe_position": output.position.as_str(),
            "probe_generate_tokens": config.generate_tokens,
            "probe_limit": config.limit,
            "context_limit": config.context_limit,
            "n_stimuli": stimuli.len(),
            "n_layers": n_layers,
            "embed_dim": embed_dim,
            "model_vocab_size": model_vocab_size,
            "activation_shape": shape,
            "correctness_path": correctness_path,
            "correctness_sha256": sha256_file_result(&correctness_path)?,
            "token_selections": token_selections[oi],
            "run_timestamp_unix": unix_timestamp(),
            "git_commit": git_commit(),
            "batched_probe_extraction": true,
            "row_order": "source stimulus array order",
            "row_indices": stimulus_info
                .iter()
                .map(|info| info.source_index)
                .collect::<Vec<_>>(),
            "batched_probe_positions": config
                .outputs
                .iter()
                .map(|output| output.position.as_str())
                .collect::<Vec<_>>(),
        });
        let metadata_path = sidecar_path(&output.output_path, "_metadata.json")?;
        write_json_file(&metadata_path, &metadata)?;
        eprintln!("saved metadata to {}", metadata_path);
    }

    eprintln!("done in {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn duplicate_probe_axes_are_rejected_before_outputs_are_created() {
        let duplicate_template =
            Args::try_parse_from(["ember", "--probe", "--probe-templates", "en_zero,en_zero"])
                .unwrap();
        assert!(build_probe_jobs(&duplicate_template)
            .unwrap_err()
            .to_string()
            .contains("duplicate probe template"));

        let duplicate_position =
            Args::try_parse_from(["ember", "--probe", "--probe-positions", "root,root"]).unwrap();
        assert!(build_probe_jobs(&duplicate_position)
            .unwrap_err()
            .to_string()
            .contains("duplicate probe position"));
    }

    #[test]
    fn sanitized_probe_filename_collisions_are_rejected() {
        let args =
            Args::try_parse_from(["ember", "--probe", "--probe-templates", "a/b,a?b"]).unwrap();
        assert!(build_probe_jobs(&args)
            .unwrap_err()
            .to_string()
            .contains("output path collision"));
    }

    #[test]
    fn probe_span_selection_uses_overlapping_non_special_offsets() {
        let stimulus = serde_json::json!({"root": "bc", "pattern": "x"});
        let indices = select_probe_indices(
            "abcd",
            &[99, 1, 2],
            &[(0, 0), (0, 2), (2, 4)],
            &stimulus,
            ProbePosition::Root,
        )
        .unwrap();
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn arabic_probe_span_uses_character_offsets_not_utf8_bytes() {
        let stimulus = serde_json::json!({"root": "كتب", "pattern": "x"});
        let indices = select_probe_indices(
            "سكتبx",
            &[1, 2, 3],
            &[(0, 1), (1, 3), (3, 5)],
            &stimulus,
            ProbePosition::Root,
        )
        .unwrap();
        assert_eq!(indices, vec![1, 2]);
    }
}

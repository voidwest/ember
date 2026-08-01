//! Generation, demo, and experiment execution paths.
//! Split out of `main.rs` (2026-08-01) to keep the CLI dispatcher thin.

use crate::cli_commands::{effective_context_limit, ensure_sequence_fits};
use crate::cli_probe::{has_next_decode_evaluation, LogitDumpConfig};
use crate::cli_support::{token_audit_json, write_json_file};
use crate::{rayon_current_num_threads, Args};
use anyhow::Context;
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::experiments::{
    ExecutionContext, ExecutionPhase, ExperimentRunner, ExperimentalForwardModel,
    GenerationContext, ModelContext, TracingState,
};
use ember::model::ForwardModel;
use ember::model::Gpt2;
use ember::npy::write_npy_2d;
use ember::sampler::{argmax_token, sample_token};
use ember::trace;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

pub(crate) fn run_single_prompt_with_experiment(
    backend: &CpuBackend,
    model: &impl ExperimentalForwardModel,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    args: &Args,
    model_context: ModelContext<'_>,
    runner: &mut ExperimentRunner,
) -> anyhow::Result<()> {
    let output = generate_with_experiment(
        backend,
        model,
        runner,
        model_context,
        tokenizer,
        &args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.benchmark,
        args.trace.is_some(),
        args.trace_out.as_deref(),
        args.trace_values == "summary",
        args.trace_run_metadata,
        rayon_current_num_threads(),
        effective_context_limit(backend, model, args),
    )?;
    println!("{}", output);
    Ok(())
}

pub(crate) fn run_single_prompt<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    args: &Args,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let output = generate(
        backend,
        model,
        tokenizer,
        &args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.benchmark,
        args.trace.is_some(),
        args.trace_out.as_deref(),
        args.trace_values == "summary",
        args.trace_run_metadata,
        rayon_current_num_threads(),
        effective_context_limit(backend, model, args),
    )?;
    println!("{}", output);
    Ok(())
}

/// run a curated demo showcasing the project.
///
/// uses greedy sampling (temperature 0) for deterministic, repeatable output -
/// ideal for screen recordings, benchmarks, and project demonstrations.
/// runs through a fixed set of prompts, printing each one with its completion
/// and per-prompt timing, then a summary table.
///
/// when `delay_ms > 0`, tokens are streamed one at a time with a typewriter
/// effect. ansi colors are used for visual distinction (`--color` cli flag or
/// terminal detection can be added to toggle).
pub(crate) fn demo_mode<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    max_tokens: usize,
    model_path: &str,
    delay_ms: u64,
    context_limit: usize,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    // -- ansi style helpers ----------------------------------------------
    // simple string concatenation to avoid macro complexity.
    // each "style" builder returns a formatted string with escape codes.
    const RST: &str = "\x1b[0m";
    const BLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[2m";
    const CYN: &str = "\x1b[36m";
    const GRN: &str = "\x1b[32m";
    const YLW: &str = "\x1b[33m";

    fn s(ansi: &str, text: &dyn std::fmt::Display) -> String {
        format!("{ansi}{text}{RST}")
    }
    fn s2(a: &str, b: &str, text: &dyn std::fmt::Display) -> String {
        format!("{a}{b}{text}{RST}")
    }

    // eprintln / print without newline helpers so we don't forget io::stdout().flush()
    macro_rules! eprint_flush { ($($arg:tt)*) => {{
        eprint!($($arg)*);
        let _ = io::stderr().flush();
    }}; }
    macro_rules! print_flush { ($($arg:tt)*) => {{
        print!($($arg)*);
        let _ = io::stdout().flush();
    }}; }

    let embed_dim = model.embed_dim();

    // -- header ------------------------------------------------------
    let header_border = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );
    let header_line = s2(
        BLD,
        CYN,
        &"|              ember  -  llm inference             |",
    );
    let header_sep = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );

    println!("{header_border}");
    println!("{header_line}");
    println!("{header_sep}");

    let kv = |k: &str, v: &dyn std::fmt::Display| {
        println!(
            "{} {} {:>37} {}",
            s2(DIM, CYN, &"|"),
            s(DIM, &k),
            s(BLD, &v),
            s2(DIM, CYN, &"|"),
        );
    };
    kv("model     ", &model_path);
    kv("layers    ", &model.n_layers());
    kv("embed_dim ", &embed_dim);
    kv("vocab     ", &tokenizer.vocab_size());
    kv("sampling  ", &"greedy (temp=0)");

    let header_foot = s2(
        DIM,
        CYN,
        &"+--------------------------------------------------+",
    );
    println!("{header_foot}");

    if delay_ms > 0 {
        println!();
        println!(
            "{}",
            s(
                DIM,
                &format!("  typewriter delay: {delay_ms} ms/token - press ctrl-c to exit")
            ),
        );
    }
    println!();

    let prompts: &[(&str, &str)] = &[
        ("Once upon a time, in a land far away,", "story generation"),
        (
            "The three primary colors of light are",
            "factual completion",
        ),
        (
            "// fibonacci sequence in python\ndef fib(n):",
            "code generation",
        ),
        ("The meaning of life is", "open-ended reasoning"),
    ];

    let spinner_chars = ['|', '/', '-', '\\'];

    let mut total_prefill_ms = 0.0;
    let mut total_decode_ms = 0.0;
    let mut total_prompt_tokens = 0usize;
    let mut total_generated = 0usize;
    let mut total_decode_evaluations = 0usize;

    for (i, (prompt, category)) in prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();
        let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;

        // -- prefill with spinner ----------------------------------
        let prefill_start = std::time::Instant::now();
        eprint_flush!(
            "{}  {}{}",
            s(CYN, &"*"),
            s(DIM, &"prefilling... "),
            spinner_chars[0],
        );

        let mut cache = model.create_cache(backend, max_seq_len);
        let mut logits =
            model.forward_last_logits_with_cache(backend, &prompt_tokens, &mut cache, 0)?;
        let vocab_size = backend.shape(&logits)[1];

        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprint_flush!("\r{}\n", s(GRN, &"prefill complete"));

        // -- decode with typewriter streaming ----------------------
        let decode_start = std::time::Instant::now();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut decode_evaluations = 0usize;
        let eos_ids = tokenizer.eos_token_ids();

        // print prompt card
        println!();
        let pn = i + 1;
        let card_width: usize = 50;
        let top_prefix = format!("+- prompt {pn} - {category} - ");
        let pad_len = card_width.saturating_sub(top_prefix.chars().count() + 1);
        let dashes = "-".repeat(pad_len);
        println!("{}", s2(BLD, CYN, &format!("{top_prefix}{dashes}+")),);
        println!("{}", s(DIM, &"|"));
        println!("{} {}", s(DIM, &"| prompt:    "), s(YLW, &prompt),);
        print_flush!(
            "{} {}",
            s(DIM, &"| completion:"),
            GRN, // start completion on a new line, green
        );

        for step in 0..max_tokens {
            let logit_data = backend.data(&logits);
            let last_logits = &logit_data[..vocab_size];

            let next = argmax_token(last_logits);

            if eos_ids.contains(&(next as u32)) {
                break;
            }

            generated.push(next as u32);

            // stream this single token now, before computing the next.
            // individual subword tokens may decode to replacement characters
            // (U+FFFD) when they're part of a multi-token UTF-8 sequence;
            // filter those out so the typewriter effect stays clean.
            let token_text = tokenizer.decode(&[next as u32])?;
            let cleaned: String = token_text.chars().filter(|c| *c != '\u{FFFD}').collect();
            if !cleaned.is_empty() {
                print_flush!("{}", cleaned);
            }

            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }

            if !has_next_decode_evaluation(step, max_tokens) {
                break;
            }
            logits = model.forward_last_logits_with_cache(
                backend,
                &[next as u32],
                &mut cache,
                prompt_len + step,
            )?;
            decode_evaluations += 1;
        }
        // reset color after completion
        println!("{RST}");
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        // -- per-prompt stats -------------------------------------
        println!("{}", s(DIM, &"|"));
        println!(
            "{} {} prompt + {} generated = {} total",
            s(DIM, &"| tokens:    "),
            prompt_len,
            generated.len(),
            prompt_len + generated.len(),
        );
        println!(
            "{} {:.1} ms ({:.0} tok/s)",
            s(DIM, &"| prefill:   "),
            prefill_ms,
            prompt_len as f64 / (prefill_ms / 1000.0)
        );
        println!(
            "{} {:.1} ms ({} evals, {:.0} eval/s)",
            s(DIM, &"| decode:    "),
            decode_ms,
            decode_evaluations,
            decode_evaluations as f64 / (decode_ms / 1000.0)
        );
        println!(
            "{}",
            s2(
                DIM,
                CYN,
                &"+------------------------------------------------+"
            )
        );
        println!();

        total_prefill_ms += prefill_ms;
        total_decode_ms += decode_ms;
        total_prompt_tokens += prompt_len;
        total_generated += generated.len();
        total_decode_evaluations += decode_evaluations;

        // brief pause between prompts so the viewer can absorb
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms * 5));
        }
    }

    // -- summary ----------------------------------------------------
    let total_ms = total_prefill_ms + total_decode_ms;
    let total_tokens = total_prompt_tokens + total_generated;

    println!(
        "{}",
        s2(
            BLD,
            YLW,
            &"=========================== summary =========================="
        ),
    );
    println!();
    println!("  prompts:       {}", prompts.len());
    println!(
        "  total tokens:  {} ({} prompt + {} generated)",
        total_tokens, total_prompt_tokens, total_generated
    );
    println!("  total time:    {:.1} ms", total_ms);
    println!(
        "  throughput:    {:.0} tok/s",
        total_tokens as f64 / (total_ms / 1000.0)
    );
    println!(
        "  prefill avg:   {:.1} ms - {:.0} tok/s",
        total_prefill_ms / prompts.len() as f64,
        total_prompt_tokens as f64 / (total_prefill_ms / 1000.0)
    );
    println!(
        "  decode avg:    {:.1} ms - {} evals at {:.0} eval/s",
        total_decode_ms / prompts.len() as f64,
        total_decode_evaluations,
        total_decode_evaluations as f64 / (total_decode_ms / 1000.0)
    );
    println!();
    println!(
        "{}",
        s2(
            DIM,
            YLW,
            &"=============================================================="
        ),
    );
    println!();

    // -- end-of-demo flicker ------------------------------------
    // prints a blinking cursor effect that persists for ~2 seconds
    // so the viewer knows the demo is complete and the terminal is
    // still live.
    if delay_ms > 0 {
        print_flush!("{}", s(DIM, &"demo complete. "));
        let cursor_chars = ['|', ' '];
        let flicker_start = std::time::Instant::now();
        let mut flicker_idx = 0usize;
        while flicker_start.elapsed().as_secs() < 2 {
            print_flush!(
                "\r{} {}",
                s(DIM, &"demo complete. "),
                cursor_chars[flicker_idx % 2],
            );
            flicker_idx += 1;
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        // clear the cursor line
        print_flush!("\r{}\r", s(DIM, &"demo complete."));
    }

    Ok(())
}

pub(crate) trait GenerationExecution<B, M>
where
    B: Backend,
    M: ForwardModel<B>,
{
    fn before_prefill(&mut self, prompt_token_count: usize) -> anyhow::Result<()>;

    fn forward_last_logits(
        &mut self,
        backend: &B,
        model: &M,
        token_ids: &[u32],
        cache: &mut ember::kv_cache::KVCache,
        start_pos: usize,
        phase: ExecutionPhase,
    ) -> Result<B::Tensor, B::Error>;

    fn generation_complete(
        &mut self,
        prompt_token_count: usize,
        generated_token_count: usize,
        decode_evaluations: usize,
        input_token_ids: &[u32],
        generated_token_ids: &[u32],
    ) -> anyhow::Result<()>;
}

pub(crate) struct StandardGeneration;

impl<B, M> GenerationExecution<B, M> for StandardGeneration
where
    B: Backend,
    M: ForwardModel<B>,
{
    #[inline(always)]
    fn before_prefill(&mut self, _prompt_token_count: usize) -> anyhow::Result<()> {
        Ok(())
    }

    #[inline(always)]
    fn forward_last_logits(
        &mut self,
        backend: &B,
        model: &M,
        token_ids: &[u32],
        cache: &mut ember::kv_cache::KVCache,
        start_pos: usize,
        _phase: ExecutionPhase,
    ) -> Result<B::Tensor, B::Error> {
        model.forward_last_logits_with_cache(backend, token_ids, cache, start_pos)
    }

    #[inline(always)]
    fn generation_complete(
        &mut self,
        _prompt_token_count: usize,
        _generated_token_count: usize,
        _decode_evaluations: usize,
        _input_token_ids: &[u32],
        _generated_token_ids: &[u32],
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

pub(crate) struct ActiveGeneration<'runner, 'model> {
    runner: &'runner mut ExperimentRunner,
    model_context: ModelContext<'model>,
    tracing: TracingState,
}

impl<M> GenerationExecution<CpuBackend, M> for ActiveGeneration<'_, '_>
where
    M: ExperimentalForwardModel,
{
    fn before_prefill(&mut self, prompt_token_count: usize) -> anyhow::Result<()> {
        let context = ExecutionContext::new(
            self.model_context,
            ExecutionPhase::Prefill,
            0,
            prompt_token_count,
            self.tracing,
        );
        self.runner.before_prefill(&context)?;
        Ok(())
    }

    fn forward_last_logits(
        &mut self,
        backend: &CpuBackend,
        model: &M,
        token_ids: &[u32],
        cache: &mut ember::kv_cache::KVCache,
        start_pos: usize,
        phase: ExecutionPhase,
    ) -> Result<ember::tensor::CpuTensor, ember::backend::CpuError> {
        let context = ExecutionContext::new(
            self.model_context,
            phase,
            start_pos,
            token_ids.len(),
            self.tracing,
        );
        model.forward_last_logits_with_experiment(
            backend,
            token_ids,
            cache,
            start_pos,
            context,
            self.runner,
        )
    }

    fn generation_complete(
        &mut self,
        prompt_token_count: usize,
        generated_token_count: usize,
        decode_evaluations: usize,
        input_token_ids: &[u32],
        generated_token_ids: &[u32],
    ) -> anyhow::Result<()> {
        let context = GenerationContext::new(
            self.model_context,
            prompt_token_count,
            generated_token_count,
            decode_evaluations,
            self.tracing,
            input_token_ids,
            generated_token_ids,
        );
        self.runner.on_generation_complete(&context)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    benchmark: bool,
    trace_ops: bool,
    trace_out: Option<&str>,
    trace_values_summary: bool,
    trace_run_metadata: bool,
    thread_count: usize,
    context_limit: usize,
) -> anyhow::Result<String>
where
    B::Error: Send + Sync + 'static,
{
    let mut execution = StandardGeneration;
    generate_with_execution(
        backend,
        model,
        &mut execution,
        tokenizer,
        prompt,
        max_tokens,
        temperature,
        top_k,
        top_p,
        benchmark,
        trace_ops,
        trace_out,
        trace_values_summary,
        trace_run_metadata,
        thread_count,
        context_limit,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_experiment(
    backend: &CpuBackend,
    model: &impl ExperimentalForwardModel,
    runner: &mut ExperimentRunner,
    model_context: ModelContext<'_>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    benchmark: bool,
    trace_ops: bool,
    trace_out: Option<&str>,
    trace_values_summary: bool,
    trace_run_metadata: bool,
    thread_count: usize,
    context_limit: usize,
) -> anyhow::Result<String> {
    let mut execution = ActiveGeneration {
        runner,
        model_context,
        tracing: TracingState::from(trace_ops),
    };
    generate_with_execution(
        backend,
        model,
        &mut execution,
        tokenizer,
        prompt,
        max_tokens,
        temperature,
        top_k,
        top_p,
        benchmark,
        trace_ops,
        trace_out,
        trace_values_summary,
        trace_run_metadata,
        thread_count,
        context_limit,
    )
}

/// run the full autoregressive generation loop.
///
/// operates in two phases:
/// 1. **prefill** - feeds the entire prompt through the model in one forward pass,
///    populating the kv cache with key/value projections for all prompt tokens.
/// 2. **decode** - generates one token at a time: samples from the last position's
///    logits, appends it, and runs a single-token forward pass reusing the cached
///    k/v from all previous positions. stops when `max_tokens` is reached or a
///    tokenizer-defined eos token is predicted.
///
/// temperature 0.0 uses greedy argmax; any positive value enables temperature
/// scaling with optional top-k and top-p filtering via [`sample_token`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_execution<B, M, E>(
    backend: &B,
    model: &M,
    execution: &mut E,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    benchmark: bool,
    trace_ops: bool,
    trace_out: Option<&str>,
    trace_values_summary: bool,
    trace_run_metadata: bool,
    thread_count: usize,
    context_limit: usize,
) -> anyhow::Result<String>
where
    B: Backend,
    M: ForwardModel<B>,
    E: GenerationExecution<B, M>,
    B::Error: Send + Sync + 'static,
{
    let mut rng = rand::thread_rng();

    let mut all_tokens = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    log::info!("prompt has {} tokens", all_tokens.len());

    let prompt_len = all_tokens.len();
    let max_seq_len = ensure_sequence_fits(prompt_len, max_tokens, context_limit)?;

    // -- 1. prefill: run the prompt through the transformer and fill kv cache.
    // Only the last prompt position needs logits for generation, so avoid
    // materializing a full [prompt_len, vocab_size] logits tensor.
    let prefill_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    log::info!("prefilling KV cache for {} tokens", prompt_len);

    // -- trace: prefill ----------------------------------------------------
    let mut prefill_trace: Option<trace::TraceReport> = None;
    let run_meta = if trace_run_metadata {
        Some(trace::collect_run_metadata(thread_count))
    } else {
        None
    };
    if trace_ops {
        if trace_values_summary {
            trace::set_values_level(trace::TraceValuesLevel::Summary);
        }
        trace::enable_tracing("prefill", 0);
    }
    let mut cache = model.create_cache(backend, max_seq_len);
    execution.before_prefill(prompt_len)?;
    let mut logits = execution.forward_last_logits(
        backend,
        model,
        &all_tokens,
        &mut cache,
        0,
        ExecutionPhase::Prefill,
    )?;
    let prefill_elapsed = prefill_start.map(|s| s.elapsed());
    if trace_ops {
        prefill_trace = trace::disable_tracing();
        if let Some(ref mut pt) = prefill_trace {
            pt.run_metadata = run_meta.clone();
        }
    }
    let vocab_size = backend.shape(&logits)[1];

    // -- 2. decode loop: one new token at a time --------------------------
    let decode_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    let mut generated = Vec::with_capacity(max_tokens);
    let mut next_token: usize;
    let mut decode_traces: Vec<trace::TraceReport> = Vec::new();
    let mut decode_evaluations = 0usize;

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = &logit_data[..vocab_size];

        next_token = if temperature == 0.0 {
            argmax_token(last_logits)
        } else {
            sample_token(last_logits, temperature, top_k, top_p, &mut rng)
        };

        log::debug!("step {}: predicted token {}", step, next_token);

        let eos_ids = tokenizer.eos_token_ids();
        if eos_ids.contains(&(next_token as u32)) {
            log::info!("eos token reached after {} generated tokens", step);
            break;
        }

        all_tokens.push(next_token as u32);
        generated.push(next_token as u32);

        if !has_next_decode_evaluation(step, max_tokens) {
            break;
        }

        // decode step: forward with just the new token, using cached K/V
        if trace_ops {
            trace::enable_tracing("decode", step);
        }
        logits = execution.forward_last_logits(
            backend,
            model,
            &[next_token as u32],
            &mut cache,
            prompt_len + step, // absolute position offset
            ExecutionPhase::Decode,
        )?;
        decode_evaluations += 1;
        if trace_ops {
            if let Some(report) = trace::disable_tracing() {
                decode_traces.push(report);
            }
        }
    }

    let output = tokenizer.decode(&generated)?;

    if benchmark {
        let prefill_ms = prefill_elapsed.unwrap().as_secs_f64() * 1000.0;
        let decode_ms = decode_start.unwrap().elapsed().as_secs_f64() * 1000.0;
        eprintln!("--- benchmark ---");
        eprintln!(
            "prefill: {} tokens in {:.1}ms -> {:.0} tok/s",
            prompt_len,
            prefill_ms,
            prompt_len as f64 / prefill_elapsed.unwrap().as_secs_f64()
        );
        eprintln!(
            "decode:  {} evals in {:.1}ms -> {:.0} eval/s",
            decode_evaluations,
            decode_ms,
            decode_evaluations as f64 / decode_start.unwrap().elapsed().as_secs_f64()
        );
    }

    // -- trace: emit reports -----------------------------------------------
    if trace_ops {
        // aggregate all decode traces into one
        let mut all_events: Vec<trace::OpTrace> = Vec::new();
        let mut total_decode_ns: u64 = 0;
        for report in &decode_traces {
            all_events.extend(report.events.clone());
            total_decode_ns += report.total_duration_ns;
        }
        let aggregated = trace::TraceReport {
            phase: "decode".to_string(),
            token_index: 0,
            events: all_events,
            total_duration_ns: total_decode_ns,
            run_metadata: run_meta.clone(),
        };

        // write JSON if requested
        if let Some(path) = trace_out {
            let json = aggregated.to_json();
            fs::write(path, json).context("failed to write trace JSON")?;
            eprintln!("trace JSON written to {}", path);
        }

        eprintln!(
            "--- trace: decode ({} tokens, {:.2} ms) ---",
            decode_traces.len(),
            total_decode_ns as f64 / 1_000_000.0
        );
        eprintln!("{}", aggregated.summary());

        if let Some(ref prefill) = prefill_trace {
            eprintln!(
                "\n--- trace: prefill ({:.2} ms) ---",
                prefill.total_duration_ns as f64 / 1_000_000.0
            );
            eprintln!("{}", prefill.summary());
        }
    }

    if log::log_enabled!(log::Level::Debug) {
        let decoded_prompt = tokenizer.decode(&all_tokens[..prompt_len])?;
        log::debug!("prompt: {:?}", decoded_prompt);
        log::debug!("generated: {:?}", output);
    }

    execution.generation_complete(
        prompt_len,
        generated.len(),
        decode_evaluations,
        &all_tokens[..prompt_len],
        &generated,
    )?;
    Ok(output)
}

pub(crate) fn dump_last_logits<B: Backend>(
    backend: &B,
    model: &impl ForwardModel<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    config: LogitDumpConfig<'_>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let token_ids = tokenizer
        .encode(config.prompt)
        .context("failed to tokenize prompt")?;
    let (offset_ids, offsets) = tokenizer
        .encode_with_offsets(config.prompt)
        .context("failed to tokenize prompt with offsets")?;
    if offset_ids != token_ids {
        anyhow::bail!("token audit failed: encode and encode_with_offsets emitted different ids");
    }
    if token_ids.is_empty() {
        anyhow::bail!("cannot dump logits for an empty prompt");
    }
    let context_limit = config
        .max_seq_len
        .unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let logits = model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
    let shape = backend.shape(&logits);
    if shape.len() != 2 || shape[0] != 1 {
        anyhow::bail!("expected last logits shape [1, vocab], got {:?}", shape);
    }
    write_npy_2d(
        config.output_path,
        backend.data(&logits),
        &[shape[0], shape[1]],
    )?;
    let metadata_path = config.output_path.replace(".npy", "_metadata.json");
    let metadata = serde_json::json!({
        "model_path": config.model_path,
        "architecture": config.arch,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_sha256": config.run_metadata.tokenizer_sha256,
        "model_file_size_bytes": config.run_metadata.model_file_size_bytes,
        "model_sha256": config.run_metadata.model_sha256,
        "gguf_metadata": config.run_metadata.gguf_metadata,
        "output_path": config.output_path,
        "prompt": config.prompt,
        "context_limit": context_limit,
        "logits_shape": [shape[0], shape[1]],
        "token_audit": token_audit_json(
            config.prompt,
            config.tokenizer_path,
            config.run_metadata.tokenizer_sha256.as_deref(),
            tokenizer.bos_token_id(),
            &token_ids,
            &offsets,
        ),
        "run_manifest": config.run_metadata.run_manifest,
    });
    write_json_file(&metadata_path, &metadata)?;
    eprintln!(
        "saved last logits for {} prompt tokens to {} with shape {:?}",
        token_ids.len(),
        config.output_path,
        shape
    );
    eprintln!("saved logits metadata to {}", metadata_path);
    Ok(())
}

pub(crate) fn bail_dump_layers_unsupported(arch: &str) -> anyhow::Result<()> {
    anyhow::bail!("--dump-layers is only supported with --arch gemma4, got --arch {arch}")
}

/// Dump per-layer hidden states (last prompt token) directly to a binary file.
///
/// ## Binary output format
///
///   dtype:      f32 (native endian)
///   shape:      [n_layers * embed_dim] flat, layer-major
///   layer count: model n_layers
///   hidden size: model embed_dim
///   row order:   layer 0 first, layer (n_layers-1) last
///
/// Boundary: after each block's final residual add and layer_output_scale.
/// Matches llama.cpp per-layer dump point (after `build_cvec` in gemma4.cpp).
pub(crate) fn dump_layers_gemma4<B: Backend>(
    backend: &B,
    model: &ember::gemma4::Gemma4<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    output_path: &str,
    max_seq_len: Option<usize>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let token_ids = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    if token_ids.is_empty() {
        anyhow::bail!("cannot dump layers for an empty prompt");
    }
    let context_limit = max_seq_len.unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let (layer_states, _logits) =
        model.forward_last_logits_with_layer_dump(backend, &token_ids, &mut cache, 0)?;
    let embed_dim = model.config.embed_dim;
    let n_layers = layer_states.len();
    let flat: Vec<f32> = layer_states.into_iter().flatten().collect();
    assert_eq!(flat.len(), n_layers * embed_dim);
    let file = std::fs::File::create(output_path)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut bytes = [0u8; 1024 * std::mem::size_of::<f32>()];
    for values in flat.chunks(1024) {
        for (slot, value) in bytes
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(values)
        {
            slot.copy_from_slice(&value.to_ne_bytes());
        }
        writer.write_all(&bytes[..std::mem::size_of_val(values)])?;
    }
    writer.flush()?;
    eprintln!(
        "saved {} layers × {} hidden = {} floats to {}",
        n_layers,
        embed_dim,
        flat.len(),
        output_path
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn interactive_mode<B: Backend>(
    backend: &B,
    model: &Gpt2<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    _initial_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    max_seq_len: Option<usize>,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    println!("ember interactive mode. type /quit to exit, /help for commands.");
    println!("max tokens per turn: {}", max_tokens);

    // warm-up with the initial prompt
    print!("> ");
    io::stdout().flush()?;

    loop {
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break; // ctrl-d
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            print!("> ");
            io::stdout().flush()?;
            continue;
        }

        match line.as_str() {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("/help   show this message");
                println!("/quit   exit interactive mode");
                println!("/stats  show model info");
                print!("> ");
                io::stdout().flush()?;
                continue;
            }
            "/stats" => {
                log::info!(
                    "wte shape: {:?}, blocks: {}",
                    backend.shape(&model.wte),
                    model.blocks.len()
                );
                print!("> ");
                io::stdout().flush()?;
                continue;
            }
            prompt => {
                let output = generate(
                    backend,
                    model,
                    tokenizer,
                    prompt,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    false, // benchmark not meaningful in interactive mode
                    false, // trace not meaningful in interactive mode
                    None,  // trace_out
                    false, // trace_values
                    false, // trace_run_metadata
                    1,     // thread_count
                    max_seq_len.unwrap_or_else(|| {
                        <Gpt2<B> as ForwardModel<B>>::max_seq_len(model, backend)
                    }),
                )?;
                println!("{}", output);
                print!("> ");
                io::stdout().flush()?;
            }
        }
    }

    Ok(())
}

// -- probe mode -------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_commands::validate_experiment_options;
    use crate::cli_support::{build_run_manifest, default_tokenizer_for_arch};
    use crate::{Commands, LifecycleModeArg, PackedSelectionArg};
    use clap::Parser;

    #[test]
    fn final_requested_token_needs_no_followup_evaluation() {
        assert!(!has_next_decode_evaluation(0, 1));
        assert!(has_next_decode_evaluation(0, 2));
        assert!(!has_next_decode_evaluation(1, 2));
    }

    #[test]
    fn default_tokenizer_tracks_architecture() {
        let gpt2 = Args::try_parse_from(["ember"]).expect("default args should parse");
        assert_eq!(
            gpt2.tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&gpt2.arch)),
            "tokenizer-gpt2.json"
        );

        let llama =
            Args::try_parse_from(["ember", "--arch", "llama"]).expect("llama args should parse");
        assert_eq!(
            llama
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&llama.arch)),
            "tokenizer.json"
        );

        let gemma4 =
            Args::try_parse_from(["ember", "--arch", "gemma4"]).expect("gemma4 args should parse");
        assert_eq!(
            gemma4
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&gemma4.arch)),
            "tokenizer-gemma4.json"
        );

        let qwen3 =
            Args::try_parse_from(["ember", "--arch", "qwen3"]).expect("qwen3 args should parse");
        assert_eq!(
            qwen3
                .tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&qwen3.arch)),
            "tokenizer-qwen3.json"
        );
    }

    #[test]
    fn cli_rejects_invalid_sampling_args() {
        assert!(Args::try_parse_from(["ember", "--temperature", "-0.1"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-k", "0"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-p", "0"]).is_err());
        assert!(Args::try_parse_from(["ember", "--top-p", "1.1"]).is_err());
        assert!(Args::try_parse_from(["ember", "--max-seq-len", "0"]).is_err());
    }

    #[test]
    fn zero_layer_output_cli_parses_typed_spec() {
        let args = Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--zero-layer-output",
            "4:attention",
        ])
        .unwrap();
        let spec = args.zero_layer_output.unwrap();
        assert_eq!(spec.layer(), 4);
        assert_eq!(
            spec.stage(),
            ember::experiments::ZeroLayerOutputStage::Attention
        );
        validate_experiment_options(&args).unwrap();
    }

    #[test]
    fn zero_layer_output_cli_rejects_malformed_and_incompatible_uses() {
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "llama",
            "--zero-layer-output",
            "4:residual",
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "gemma4",
            "--zero-layer-output",
            "4:attention",
            "--dump-layers",
            "layers.bin",
        ])
        .is_err());

        let gpt2 =
            Args::try_parse_from(["ember", "--arch", "gpt2", "--zero-layer-output", "0:layer"])
                .unwrap();
        assert!(validate_experiment_options(&gpt2)
            .unwrap_err()
            .to_string()
            .contains("not gpt2"));

        let subcommand = Args::try_parse_from([
            "ember",
            "--zero-layer-output",
            "0:layer",
            "bench-decode",
            "--model",
            "model.gguf",
            "--arch",
            "llama",
        ])
        .unwrap();
        assert!(validate_experiment_options(&subcommand)
            .unwrap_err()
            .to_string()
            .contains("normal generation"));
    }

    #[test]
    fn activation_stats_cli_parses_and_rejects_incompatible_uses() {
        let args = Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--activation-stats",
            "activation-stats.json",
        ])
        .unwrap();
        assert_eq!(
            args.activation_stats.as_deref(),
            Some("activation-stats.json")
        );
        validate_experiment_options(&args).unwrap();

        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--activation-stats",
            "stats.json",
            "--zero-layer-output",
            "1:mlp",
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "gemma4",
            "--activation-stats",
            "stats.json",
            "--dump-layers",
            "layers.bin",
        ])
        .is_err());

        let gpt2 = Args::try_parse_from([
            "ember",
            "--arch",
            "gpt2",
            "--activation-stats",
            "stats.json",
        ])
        .unwrap();
        assert!(validate_experiment_options(&gpt2)
            .unwrap_err()
            .to_string()
            .contains("not gpt2"));
    }

    #[test]
    fn run_manifest_omits_disabled_experiment_and_records_active_one() {
        let normal = Args::try_parse_from(["ember", "--arch", "llama"]).unwrap();
        let normal_manifest = build_run_manifest(
            &normal,
            "tokenizer.json",
            None,
            None,
            &serde_json::json!({}),
        );
        assert!(normal_manifest["execution"].get("experiment").is_none());

        let active =
            Args::try_parse_from(["ember", "--arch", "gemma4", "--zero-layer-output", "2:mlp"])
                .unwrap();
        let active_manifest = build_run_manifest(
            &active,
            "tokenizer.json",
            None,
            None,
            &serde_json::json!({}),
        );
        assert_eq!(
            active_manifest["execution"]["experiment"],
            serde_json::json!({
                "name": "zero-layer-output",
                "layer": 2,
                "stage": "mlp",
                "modifies_execution": true,
            })
        );

        let observation = Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--activation-stats",
            "stats.json",
        ])
        .unwrap();
        let observation_manifest = build_run_manifest(
            &observation,
            "tokenizer-qwen3.json",
            None,
            None,
            &serde_json::json!({}),
        );
        assert_eq!(
            observation_manifest["execution"]["experiment"],
            serde_json::json!({
                "name": "activation-stats",
                "output": "stats.json",
                "modifies_execution": false,
            })
        );
    }

    #[test]
    fn bench_decode_subcommand_parses_matched_timing_options() {
        let args = Args::try_parse_from([
            "ember",
            "bench-decode",
            "--model",
            "model.gguf",
            "--arch",
            "gemma4",
            "--tokens",
            "16",
            "--warmups",
            "1",
            "--repetitions",
            "3",
        ])
        .unwrap();
        match args.command {
            Some(Commands::BenchDecode(command)) => {
                assert_eq!(command.tokens, 16);
                assert_eq!(command.warmups, 1);
                assert_eq!(command.repetitions, 3);
            }
            _ => panic!("expected bench-decode command"),
        }
    }

    #[test]
    fn bench_lifecycle_subcommand_parses_explicit_experiment_modes() {
        let args = Args::try_parse_from([
            "ember",
            "bench-lifecycle",
            "--model",
            "model.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--lifecycle",
            "pack-before-prefill-reevict",
            "--selection",
            "attention-gate-up",
            "--tokens",
            "32",
        ])
        .unwrap();
        match args.command {
            Some(Commands::BenchLifecycle(command)) => {
                assert!(matches!(
                    command.lifecycle,
                    LifecycleModeArg::PackBeforePrefillReevict
                ));
                assert!(matches!(
                    command.selection,
                    PackedSelectionArg::AttentionGateUp
                ));
                assert_eq!(command.tokens, 32);
                assert!(!command.timing_only);
            }
            _ => panic!("expected bench-lifecycle command"),
        }
    }
}

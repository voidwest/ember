//! Generation, demo, and experiment execution paths.
//! Split out of `main.rs` (2026-08-01) to keep the CLI dispatcher thin.

use crate::cli_commands::{effective_context_limit, ensure_sequence_fits};
use crate::cli_probe::{has_next_decode_evaluation, TensorDumpConfig};
use crate::cli_support::{
    sidecar_path, token_audit_json, validate_token_ids_for_model, write_json_file,
};
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
use rand::SeedableRng;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// v0.5 seeded-sampling RNG: StdRng when a seed is given (deterministic),
/// the thread RNG otherwise.
enum SeededRng {
    Std(Box<rand::rngs::StdRng>),
    Thread(rand::rngs::ThreadRng),
}

impl rand::RngCore for SeededRng {
    fn next_u32(&mut self) -> u32 {
        match self {
            SeededRng::Std(rng) => rng.next_u32(),
            SeededRng::Thread(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            SeededRng::Std(rng) => rng.next_u64(),
            SeededRng::Thread(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            SeededRng::Std(rng) => rng.fill_bytes(dest),
            SeededRng::Thread(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            SeededRng::Std(rng) => rng.try_fill_bytes(dest),
            SeededRng::Thread(rng) => rng.try_fill_bytes(dest),
        }
    }
}

static RAW_DUMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TraceCleanup;

impl Drop for TraceCleanup {
    fn drop(&mut self) {
        if trace::is_tracing() {
            let _ = trace::disable_tracing();
        }
        trace::set_values_level(trace::TraceValuesLevel::None);
    }
}

fn validate_last_logits<B: Backend>(
    backend: &B,
    logits: &B::Tensor,
    expected_vocab_size: usize,
) -> anyhow::Result<()> {
    let shape = backend.shape(logits);
    if shape != [1, expected_vocab_size] {
        anyhow::bail!("expected last logits shape [1, {expected_vocab_size}], got {shape:?}");
    }
    let data = backend.data(logits);
    if data.len() != expected_vocab_size {
        anyhow::bail!(
            "last-logits payload has {} values, expected {expected_vocab_size}",
            data.len()
        );
    }
    if let Some((index, value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("last logits contain non-finite value {value} at vocabulary index {index}");
    }
    Ok(())
}

fn validate_generated_token(
    tokenizer: &ember::tokenizer::EmberTokenizer,
    token: usize,
    model_vocab_size: usize,
) -> anyhow::Result<u32> {
    if token >= model_vocab_size {
        anyhow::bail!(
            "model selected token ID {token} outside its vocabulary size {model_vocab_size}"
        );
    }
    let token = u32::try_from(token).context("selected token ID exceeds u32")?;
    if !tokenizer.contains_token_id(token) {
        anyhow::bail!(
            "model selected token ID {token}, but the tokenizer cannot decode it; model/tokenizer vocabularies are incompatible"
        );
    }
    Ok(token)
}

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
        None,
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
    B::Error: Send + Sync + 'static,
{
    fn before_prefill(&mut self, prompt_token_ids: &[u32]) -> anyhow::Result<()>;

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

    /// Greedy next-token inference, fused with the decode forward.
    ///
    /// Default: full-logits forward plus an in-place argmax (identical to
    /// the ordinary greedy path, so experiment/capture hooks still see the
    /// full logits). `StandardGeneration` overrides this to the model's
    /// fused fast path, which avoids materializing the vocabulary.
    fn forward_greedy_token(
        &mut self,
        backend: &B,
        model: &M,
        token_ids: &[u32],
        cache: &mut ember::kv_cache::KVCache,
        start_pos: usize,
        phase: ExecutionPhase,
    ) -> anyhow::Result<(u32, f32)> {
        let logits =
            self.forward_last_logits(backend, model, token_ids, cache, start_pos, phase)?;
        let data = backend.data(&logits);
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &value) in data.iter().enumerate() {
            if value > best_val {
                best_val = value;
                best = i;
            }
        }
        Ok((best as u32, best_val))
    }
}

pub(crate) struct StandardGeneration;

impl<B, M> GenerationExecution<B, M> for StandardGeneration
where
    B: Backend,
    M: ForwardModel<B>,
    B::Error: Send + Sync + 'static,
{
    #[inline(always)]
    fn before_prefill(&mut self, _prompt_token_ids: &[u32]) -> anyhow::Result<()> {
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

    fn forward_greedy_token(
        &mut self,
        _backend: &B,
        model: &M,
        token_ids: &[u32],
        cache: &mut ember::kv_cache::KVCache,
        start_pos: usize,
        _phase: ExecutionPhase,
    ) -> anyhow::Result<(u32, f32)> {
        model
            .greedy_next_token_with_cache(_backend, token_ids, cache, start_pos)
            .map_err(anyhow::Error::msg)
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
    fn before_prefill(&mut self, prompt_token_ids: &[u32]) -> anyhow::Result<()> {
        let context = ExecutionContext::new_with_token_ids(
            self.model_context,
            ExecutionPhase::Prefill,
            0,
            prompt_token_ids,
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
        let context = ExecutionContext::new_with_token_ids(
            self.model_context,
            phase,
            start_pos,
            token_ids,
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
        None,
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
    rng_seed: Option<u64>,
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
        rng_seed,
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
    rng_seed: Option<u64>,
) -> anyhow::Result<String>
where
    B: Backend,
    M: ForwardModel<B>,
    E: GenerationExecution<B, M>,
    B::Error: Send + Sync + 'static,
{
    // The fused greedy decode path (branch-and-bound argmax lm_head) is
    // off by default: its exact-scalar accumulation can differ from the
    // SIMD full-logits path's argmax on near-tie tokens, which would change
    // greedy output. Opt in with EMBER_FUSED_GREEDY=1.
    let fused_greedy = std::env::var("EMBER_FUSED_GREEDY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // v0.5: a fixed seed makes temperature sampling deterministic
    // (StdRng/ChaCha); None keeps the historical thread-local RNG.
    let mut rng = match rng_seed {
        Some(seed) => SeededRng::Std(Box::new(rand::rngs::StdRng::seed_from_u64(seed))),
        None => SeededRng::Thread(rand::thread_rng()),
    };

    let mut all_tokens = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    if all_tokens.is_empty() {
        anyhow::bail!("cannot generate from a prompt that produces no token IDs");
    }
    let model_vocab_size = model.vocab_size(backend);
    validate_token_ids_for_model(&all_tokens, model_vocab_size, "prompt")?;
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
    let _trace_cleanup = trace_ops.then_some(TraceCleanup);
    let mut cache = model.create_cache(backend, max_seq_len);
    execution.before_prefill(&all_tokens)?;
    if trace_ops {
        trace::set_values_level(if trace_values_summary {
            trace::TraceValuesLevel::Summary
        } else {
            trace::TraceValuesLevel::None
        });
        if !trace::enable_tracing("prefill", 0) {
            anyhow::bail!("cannot start prefill trace because a trace is already active");
        }
    }
    let mut logits = execution.forward_last_logits(
        backend,
        model,
        &all_tokens,
        &mut cache,
        0,
        ExecutionPhase::Prefill,
    )?;
    validate_last_logits(backend, &logits, model_vocab_size)?;
    let prefill_elapsed = prefill_start.map(|s| s.elapsed());
    if trace_ops {
        prefill_trace = trace::disable_tracing();
        if let Some(ref mut pt) = prefill_trace {
            pt.run_metadata = run_meta.clone();
        }
    }
    let vocab_size = model_vocab_size;

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
    let eos_ids = tokenizer.eos_token_ids();

    for step in 0..max_tokens {
        // Greedy decode steps after the first can use the fused argmax path
        // (the model may compute only the top token instead of the full
        // vocabulary). Tracing and sampling always use the full path.
        let greedy_fused = fused_greedy && step > 0 && temperature == 0.0 && !trace_ops;
        if greedy_fused {
            let (token, logit) = execution.forward_greedy_token(
                backend,
                model,
                &[all_tokens[all_tokens.len() - 1]],
                &mut cache,
                prompt_len + step - 1, // position of the previous token
                ExecutionPhase::Decode,
            )?;
            next_token = token as usize;
            if !logit.is_finite() {
                anyhow::bail!(
                    "fused greedy decode returned non-finite logit {logit} for token {token}"
                );
            }
            decode_evaluations += 1;
            log::debug!(
                "step {}: greedy token {} (logit {logit:.4})",
                step,
                next_token
            );
        } else {
            let logit_data = backend.data(&logits);
            let last_logits = &logit_data[..vocab_size];

            next_token = if temperature == 0.0 {
                argmax_token(last_logits)
            } else {
                sample_token(last_logits, temperature, top_k, top_p, &mut rng)
            };

            log::debug!("step {}: predicted token {}", step, next_token);
        }

        let next_token = validate_generated_token(tokenizer, next_token, model_vocab_size)?;
        if eos_ids.contains(&next_token) {
            log::info!("eos token reached after {} generated tokens", step);
            break;
        }

        all_tokens.push(next_token);
        generated.push(next_token);

        if !has_next_decode_evaluation(step, max_tokens) {
            break;
        }

        let next_step_uses_fused = fused_greedy && temperature == 0.0 && !trace_ops && step == 0;
        if !greedy_fused && !next_step_uses_fused {
            // decode step: forward with just the new token, using cached K/V
            if trace_ops && !trace::enable_tracing("decode", step) {
                anyhow::bail!("cannot start decode trace because a trace is already active");
            }
            logits = execution.forward_last_logits(
                backend,
                model,
                &[next_token],
                &mut cache,
                prompt_len + step, // absolute position offset
                ExecutionPhase::Decode,
            )?;
            validate_last_logits(backend, &logits, model_vocab_size)?;
            decode_evaluations += 1;
            if trace_ops {
                if let Some(report) = trace::disable_tracing() {
                    decode_traces.push(report);
                }
            }
        }
    }

    let output = tokenizer.decode(&generated)?;

    if benchmark {
        // Snapshot each duration once so the printed latency and derived rate
        // use exactly the same interval. This matters to machine parsers and
        // is especially visible for short smoke-model runs.
        let prefill_elapsed = prefill_elapsed.unwrap();
        let decode_elapsed = decode_start.unwrap().elapsed();
        let prefill_seconds = prefill_elapsed.as_secs_f64();
        let decode_seconds = decode_elapsed.as_secs_f64();
        eprintln!("--- benchmark ---");
        eprintln!(
            "prefill: {} tokens in {:.3}ms -> {:.3} tok/s",
            prompt_len,
            prefill_seconds * 1000.0,
            prompt_len as f64 / prefill_seconds
        );
        eprintln!(
            "decode:  {} evals in {:.3}ms -> {:.3} eval/s",
            decode_evaluations,
            decode_seconds * 1000.0,
            decode_evaluations as f64 / decode_seconds
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
            let artifact = serde_json::json!({
                "schema_version": 1,
                "prefill": prefill_trace,
                "decode": aggregated,
            });
            write_json_file(path, &artifact).context("failed to write trace JSON")?;
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
    config: TensorDumpConfig<'_>,
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
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    validate_token_ids_for_model(&token_ids, model_vocab_size, "logit-dump prompt")?;
    let context_limit = config
        .max_seq_len
        .unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let logits = model.forward_last_logits_with_cache(backend, &token_ids, &mut cache, 0)?;
    validate_last_logits(backend, &logits, model_vocab_size)?;
    let shape = backend.shape(&logits);
    write_npy_2d(
        config.output_path,
        backend.data(&logits),
        &[shape[0], shape[1]],
    )?;
    let metadata_path = sidecar_path(config.output_path, "_metadata.json")?;
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
///   dtype:      little-endian f32
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
    config: TensorDumpConfig<'_>,
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
        anyhow::bail!("cannot dump layers for an empty prompt");
    }
    let model_vocab_size = model.vocab_size(backend);
    tokenizer.validate_model_vocab(model_vocab_size)?;
    validate_token_ids_for_model(&token_ids, model_vocab_size, "layer-dump prompt")?;
    let context_limit = config
        .max_seq_len
        .unwrap_or_else(|| model.max_seq_len(backend));
    ensure_sequence_fits(token_ids.len(), 0, context_limit)?;
    let mut cache = model.create_cache(backend, context_limit);
    let (layer_states, logits) =
        model.forward_last_logits_with_layer_dump(backend, &token_ids, &mut cache, 0)?;
    let embed_dim = model.config.embed_dim;
    let n_layers = layer_states.len();
    if n_layers != model.n_layers() {
        anyhow::bail!(
            "layer dump returned {n_layers} layers, expected {}",
            model.n_layers()
        );
    }
    for (layer, values) in layer_states.iter().enumerate() {
        if values.len() != embed_dim {
            anyhow::bail!(
                "layer {layer} dump has {} values, expected {embed_dim}",
                values.len()
            );
        }
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            anyhow::bail!(
                "layer {layer} dump contains non-finite value {value} at hidden index {index}"
            );
        }
    }
    validate_last_logits(backend, &logits, model_vocab_size)?;
    let flat = layer_states.into_iter().flatten().collect::<Vec<_>>();
    let expected_values = n_layers
        .checked_mul(embed_dim)
        .context("layer dump shape product overflow")?;
    if flat.len() != expected_values {
        anyhow::bail!(
            "layer dump payload has {} values, expected {expected_values}",
            flat.len()
        );
    }
    write_raw_f32_le_atomic(config.output_path, &flat)?;

    let metadata_path = sidecar_path(config.output_path, "_metadata.json")?;
    let metadata = serde_json::json!({
        "schema_version": 1,
        "model_path": config.model_path,
        "architecture": config.arch,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_sha256": config.run_metadata.tokenizer_sha256,
        "model_file_size_bytes": config.run_metadata.model_file_size_bytes,
        "model_sha256": config.run_metadata.model_sha256,
        "gguf_metadata": config.run_metadata.gguf_metadata,
        "run_manifest": config.run_metadata.run_manifest,
        "output_path": config.output_path,
        "format": "raw-f32",
        "dtype": "<f4",
        "byte_order": "little-endian",
        "shape": [n_layers, embed_dim],
        "selection": "last-prompt-token-after-layer",
        "context_limit": context_limit,
        "token_audit": token_audit_json(
            config.prompt,
            config.tokenizer_path,
            config.run_metadata.tokenizer_sha256.as_deref(),
            tokenizer.bos_token_id(),
            &token_ids,
            &offsets,
        ),
    });
    write_json_file(&metadata_path, &metadata)?;
    eprintln!(
        "saved {} layers × {} hidden = {} floats to {}",
        n_layers,
        embed_dim,
        flat.len(),
        config.output_path
    );
    eprintln!("saved layer-dump metadata to {metadata_path}");
    Ok(())
}

fn write_raw_f32_le_atomic(path: &str, values: &[f32]) -> anyhow::Result<()> {
    let final_path = Path::new(path);
    let filename = final_path
        .file_name()
        .context("raw output path must include a filename")?
        .to_string_lossy();
    let parent = final_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (temporary_path, file) = (0..128)
        .find_map(|_| {
            let sequence = RAW_DUMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary_path = parent.join(format!(
                ".{filename}.ember-tmp-{}-{sequence}",
                std::process::id()
            ));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary_path)
            {
                Ok(file) => Some(Ok((temporary_path, file))),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()
        .with_context(|| format!("failed to create staged raw output next to {path}"))?
        .context("could not allocate a unique staged raw output path")?;
    let mut cleanup = RemoveFileOnDrop(Some(temporary_path.clone()));
    let mut writer = io::BufWriter::new(file);
    let mut bytes = [0u8; 1024 * std::mem::size_of::<f32>()];
    for chunk in values.chunks(1024) {
        for (slot, value) in bytes
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(chunk)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..std::mem::size_of_val(chunk)])?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary_path, final_path).with_context(|| {
        format!(
            "failed to publish raw output '{}' from '{}'",
            final_path.display(),
            temporary_path.display()
        )
    })?;
    cleanup.0 = None;
    Ok(())
}

struct RemoveFileOnDrop(Option<PathBuf>);

impl Drop for RemoveFileOnDrop {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = fs::remove_file(path);
        }
    }
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
        let automatic = Args::try_parse_from(["ember"]).expect("default args should parse");
        assert_eq!(automatic.arch, "auto");
        assert!(automatic.tokenizer.is_none());

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
    fn trace_and_probe_specific_options_cannot_be_silent_noops() {
        assert!(Args::try_parse_from(["ember", "--trace-out", "trace.json"]).is_err());
        assert!(Args::try_parse_from(["ember", "--trace", "ops", "--probe",]).is_err());

        let trace_values = Args::try_parse_from(["ember", "--trace-values", "summary"]).unwrap();
        assert!(validate_experiment_options(&trace_values)
            .unwrap_err()
            .to_string()
            .contains("requires --trace"));

        let probe_option = Args::try_parse_from(["ember", "--probe-limit", "2"]).unwrap();
        assert!(validate_experiment_options(&probe_option)
            .unwrap_err()
            .to_string()
            .contains("require --probe"));
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
    fn activation_patch_cli_requires_targets_and_rejects_silent_noop_modes() {
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--activation-patch",
            "manifest.json"
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--patch-target",
            "1:after-mlp:prefill"
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "ember",
            "--arch",
            "qwen3",
            "--activation-patch",
            "manifest.json",
            "--patch-target",
            "1:after-mlp:prefill",
            "--dump-logits",
            "logits.npy"
        ])
        .is_err());
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

use anyhow::Context;
use clap::Parser;
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::loader::load_gguf;
use ember::model::Gpt2;
use ember::sampler::sample_token;
use std::io::{self, Write};
use std::time::Instant;

/// a lightweight, cpu-first llm inference engine.
#[derive(Parser)]
#[command(name = "ember", version)]
struct Args {
    /// path to gguf model file
    #[arg(short, long, default_value = "gpt2.Q8_0.gguf")]
    model: String,

    /// path to tokenizer.json
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: String,

    /// text prompt to complete
    #[arg(short, long, default_value = "The")]
    prompt: String,

    /// number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 20)]
    max_tokens: usize,

    /// sampling temperature (0 = greedy argmax)
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,

    /// top-k sampling: keep only the k highest logits
    #[arg(long)]
    top_k: Option<usize>,

    /// top-p (nucleus) sampling: keep smallest set of tokens with cumulative probability >= p
    #[arg(long)]
    top_p: Option<f32>,

    /// stay in an interactive read-eval-print loop after the first prompt
    #[arg(short, long)]
    interactive: bool,

    /// run a curated demo that showcases the project with deterministic output and timing
    #[arg(long, conflicts_with = "interactive")]
    demo: bool,

    /// print prefill/decode timing stats to stderr
    #[arg(long)]
    benchmark: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    // demo mode: suppress log noise for clean recordable output
    if args.demo {
        log::set_max_level(log::LevelFilter::Off);
    }

    let loader = load_gguf(&args.model)?;
    let n_tensors = loader.tensors.len();
    let model = Gpt2::from_loader(loader)?;

    let backend = CpuBackend;

    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&args.tokenizer)?;

    if args.demo {
        demo_mode(
            &backend,
            &model,
            &tokenizer,
            args.max_tokens,
            &args.model,
        )?;
    } else if args.interactive {
        log::info!("loading model from {}", args.model);
        log::info!("loaded {} tensors", n_tensors);
        log::info!("model built");
        log::debug!("wte shape: {:?}", backend.shape(&model.wte));
        log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());
        interactive_mode(
            &backend,
            &model,
            &tokenizer,
            &args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )?;
    } else {
        log::info!("loading model from {}", args.model);
        log::info!("loaded {} tensors", n_tensors);
        log::info!("model built");
        log::debug!("wte shape: {:?}", backend.shape(&model.wte));
        log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());

        let output = generate(
            &backend,
            &model,
            &tokenizer,
            &args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.benchmark,
        )?;
        println!("{}", output);
    }

    Ok(())
}

/// run a curated demo showcasing the project.
///
/// uses greedy sampling (temperature 0) for deterministic, repeatable output —
/// ideal for screen recordings, benchmarks, and project demonstrations.
/// runs through a fixed set of prompts, printing each one with its completion
/// and per-prompt timing, then a summary table.
fn demo_mode<B: Backend>(
    backend: &B,
    model: &Gpt2<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    max_tokens: usize,
    model_path: &str,
) -> anyhow::Result<()>
where
    B::Error: Send + Sync + 'static,
{
    let embed_dim = backend.shape(&model.wte)[1];
    let head_dim = embed_dim / model.n_heads;

    // ── header ──────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════╗");
    println!("║              ember  ·  llm inference              ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║ model     {:>38} ║", model_path);
    println!("║ layers    {:>38} ║", model.blocks.len());
    println!("║ heads     {:>38} ║", model.n_heads);
    println!("║ embed_dim {:>38} ║", embed_dim);
    println!("║ head_dim  {:>38} ║", head_dim);
    println!("║ vocab     {:>38} ║", tokenizer.vocab_size());
    println!("║ sampling  {:>38} ║", "greedy (temp=0)");
    println!("╚══════════════════════════════════════════════════╝");
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

    let mut total_prefill_ms = 0.0;
    let mut total_decode_ms = 0.0;
    let mut total_prompt_tokens = 0usize;
    let mut total_generated = 0usize;

    for (i, (prompt, category)) in prompts.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();
        let max_seq_len = prompt_len + max_tokens;

        // prefill
        let prefill_start = std::time::Instant::now();
        let mut cache = model.create_cache(backend, max_seq_len);
        let mut logits =
            model.forward_with_cache(backend, &prompt_tokens, &mut cache, 0)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // decode
        let decode_start = std::time::Instant::now();
        let mut all_tokens = prompt_tokens.clone();
        let mut generated = Vec::with_capacity(max_tokens);

        for step in 0..max_tokens {
            let logit_data = backend.data(&logits);
            let last_logits = if step == 0 {
                let last_offset = (all_tokens.len() - 1) * embed_dim;
                &logit_data[last_offset..last_offset + embed_dim]
            } else {
                &logit_data[..embed_dim]
            };

            let next = argmax_token(last_logits);

            if next == 50256 {
                break;
            }

            all_tokens.push(next as u32);
            generated.push(next as u32);

            logits = model.forward_with_cache(
                backend,
                &[next as u32],
                &mut cache,
                prompt_len + step + 1,
            )?;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let completion = tokenizer.decode(&generated)?;

        // ── per-prompt output ─────────────────────────────────────
        println!(
            "┌─ prompt {} ─ {} ───────────────────────┐",
            i + 1,
            category
        );
        println!("│");
        println!("│ prompt:    {}", prompt);
        println!("│ completion:{}", completion);
        println!("│");
        println!(
            "│ tokens:    {} prompt + {} generated = {} total",
            prompt_len,
            generated.len(),
            prompt_len + generated.len()
        );
        println!(
            "│ prefill:   {:.1} ms ({:.0} tok/s)",
            prefill_ms,
            prompt_len as f64 / (prefill_ms / 1000.0)
        );
        println!(
            "│ decode:    {:.1} ms ({:.0} tok/s)",
            decode_ms,
            generated.len() as f64 / (decode_ms / 1000.0)
        );
        println!("└────────────────────────────────────────────────┘");
        println!();

        total_prefill_ms += prefill_ms;
        total_decode_ms += decode_ms;
        total_prompt_tokens += prompt_len;
        total_generated += generated.len();
    }

    // ── summary ────────────────────────────────────────────────────
    let total_ms = total_prefill_ms + total_decode_ms;
    let total_tokens = total_prompt_tokens + total_generated;
    println!("══════════════════════════ summary ══════════════════════════");
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
        "  prefill avg:   {:.1} ms · {:.0} tok/s",
        total_prefill_ms / prompts.len() as f64,
        total_prompt_tokens as f64 / (total_prefill_ms / 1000.0)
    );
    println!(
        "  decode avg:    {:.1} ms · {:.0} tok/s",
        total_decode_ms / prompts.len() as f64,
        total_generated as f64 / (total_decode_ms / 1000.0)
    );
    println!();
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}

/// run the full autoregressive generation loop.
///
/// operates in two phases:
/// 1. **prefill** — feeds the entire prompt through the model in one forward pass,
///    populating the kv cache with key/value projections for all prompt tokens.
/// 2. **decode** — generates one token at a time: samples from the last position's
///    logits, appends it, and runs a single-token forward pass reusing the cached
///    k/v from all previous positions. stops when `max_tokens` is reached or the
///    eos token (50256) is predicted.
///
/// temperature 0.0 uses greedy argmax; any positive value enables temperature
/// scaling with optional top-k and top-p filtering via [`sample_token`].
#[allow(clippy::too_many_arguments)]
fn generate<B: Backend>(
    backend: &B,
    model: &Gpt2<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    benchmark: bool,
) -> anyhow::Result<String>
where
    B::Error: Send + Sync + 'static,
{
    let mut rng = rand::thread_rng();

    let mut all_tokens = tokenizer
        .encode(prompt)
        .context("failed to tokenize prompt")?;
    log::info!("prompt has {} tokens", all_tokens.len());

    let prompt_len = all_tokens.len();
    let max_seq_len = prompt_len + max_tokens;

    // ── 1. prefill: run full forward pass on the prompt and fill kv cache ──
    let prefill_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    log::info!("prefilling KV cache for {} tokens", prompt_len);
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_with_cache(backend, &all_tokens, &mut cache, 0)?;
    let prefill_elapsed = prefill_start.map(|s| s.elapsed());
    let embed_dim = backend.shape(&logits)[1];

    // ── 2. decode loop: one new token at a time ──────────────────────────
    let decode_start = if benchmark {
        Some(Instant::now())
    } else {
        None
    };
    let mut generated = Vec::with_capacity(max_tokens);
    let mut next_token: usize;

    for step in 0..max_tokens {
        let logit_data = backend.data(&logits);
        let last_logits = if step == 0 {
            // prefill step: pick the last position's logits
            let last_offset = (all_tokens.len() - 1) * embed_dim;
            &logit_data[last_offset..last_offset + embed_dim]
        } else {
            // decode step: only one token in the input, logits[0] is the output
            &logit_data[..embed_dim]
        };

        next_token = if temperature == 0.0 {
            argmax_token(last_logits)
        } else {
            sample_token(last_logits, temperature, top_k, top_p, &mut rng)
        };

        log::debug!("step {}: predicted token {}", step, next_token);

        if next_token == 50256 {
            log::info!("eos token reached after {} generated tokens", step);
            break;
        }

        all_tokens.push(next_token as u32);
        generated.push(next_token as u32);

        // decode step: forward with just the new token, using cached K/V
        logits = model.forward_with_cache(
            backend,
            &[next_token as u32],
            &mut cache,
            prompt_len + step + 1, // absolute position offset
        )?;
    }

    let output = tokenizer.decode(&generated)?;

    if benchmark {
        let prefill_ms = prefill_elapsed.unwrap().as_secs_f64() * 1000.0;
        let decode_ms = decode_start.unwrap().elapsed().as_secs_f64() * 1000.0;
        eprintln!("--- benchmark ---");
        eprintln!(
            "prefill: {} tokens in {:.1}ms → {:.0} tok/s",
            prompt_len,
            prefill_ms,
            prompt_len as f64 / prefill_elapsed.unwrap().as_secs_f64()
        );
        eprintln!(
            "decode:  {} tokens in {:.1}ms → {:.0} tok/s",
            generated.len(),
            decode_ms,
            generated.len() as f64 / decode_start.unwrap().elapsed().as_secs_f64()
        );
    }

    if log::log_enabled!(log::Level::Debug) {
        let decoded_prompt = tokenizer.decode(&all_tokens[..prompt_len])?;
        log::debug!("prompt: {:?}", decoded_prompt);
        log::debug!("generated: {:?}", output);
    }

    Ok(output)
}

/// greedy argmax: return the index of the largest logit value.
#[inline]
fn argmax_token(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn interactive_mode<B: Backend>(
    backend: &B,
    model: &Gpt2<B>,
    tokenizer: &ember::tokenizer::EmberTokenizer,
    _initial_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
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
                )?;
                println!("{}", output);
                print!("> ");
                io::stdout().flush()?;
            }
        }
    }

    Ok(())
}

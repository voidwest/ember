use anyhow::Context;
use clap::Parser;
use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::loader::load_gguf;
use ember::model::Gpt2;
use ember::sampler::sample_token;
use std::io::{self, Write};

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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    log::info!("loading model from {}", args.model);
    let loader = load_gguf(&args.model)?;
    log::info!("loaded {} tensors", loader.tensors.len());

    let model = Gpt2::from_loader(loader)?;
    log::info!("model built");

    let backend = CpuBackend;
    log::debug!("wte shape: {:?}", backend.shape(&model.wte));

    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&args.tokenizer)?;
    log::info!("tokenizer loaded, vocab size: {}", tokenizer.vocab_size());

    if args.interactive {
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
        let output = generate(
            &backend,
            &model,
            &tokenizer,
            &args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )?;
        println!("{}", output);
    }

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
    log::info!("prefilling KV cache for {} tokens", prompt_len);
    let mut cache = model.create_cache(backend, max_seq_len);
    let mut logits = model.forward_with_cache(backend, &all_tokens, &mut cache, 0)?;
    let embed_dim = backend.shape(&logits)[1];

    // ── 2. decode loop: one new token at a time ──────────────────────────
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
                )?;
                println!("{}", output);
                print!("> ");
                io::stdout().flush()?;
            }
        }
    }

    Ok(())
}

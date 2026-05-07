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
#[command(name = "ember")]
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
    let mut generated = Vec::with_capacity(max_tokens);

    for _step in 0..max_tokens {
        let logits = model.forward(backend, &all_tokens)?;
        let logit_data = backend.data(&logits);
        let embed_dim = backend.shape(&logits)[1];

        let last_offset = (all_tokens.len() - 1) * embed_dim;
        let last_logits = &logit_data[last_offset..last_offset + embed_dim];

        let next_token = if temperature == 0.0 {
            last_logits
                .iter()
                .enumerate()
                .max_by(|(_i1, a): &(usize, &f32), (_i2, b): &(usize, &f32)| {
                    a.partial_cmp(b).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            sample_token(last_logits, temperature, top_k, top_p, &mut rng)
        };

        log::debug!("step {}: predicted token {}", _step, next_token);

        if next_token == 50256 {
            log::info!("eos token reached");
            break;
        }

        all_tokens.push(next_token as u32);
        generated.push(next_token as u32);
    }

    let output = tokenizer.decode(&generated)?;

    if log::log_enabled!(log::Level::Debug) {
        let decoded_prompt = tokenizer.decode(&all_tokens[..prompt_len])?;
        log::debug!("prompt: {:?}", decoded_prompt);
        log::debug!("generated: {:?}", output);
    }

    Ok(output)
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

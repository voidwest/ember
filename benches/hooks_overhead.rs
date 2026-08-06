//! Hooks-overhead benchmark: planned single-token decode with and without a
//! hook-registered experiment runner (a noop experiment), on a real model.
//!
//! The Luminal review flagged this number explicitly ("a hook-registered run
//! vs bare run — a number reviewers will ask for"): the v0.5 contract (Gate H)
//! requires the experiment machinery not to contaminate ordinary execution,
//! and this benchmark quantifies the steady-state cost of firing the six
//! semantic hook sites when an experiment is attached.
//!
//! Run with a real GGUF, release profile:
//!
//!   cargo bench --bench hooks_overhead -- --model Llama-3.2-1B-Instruct.Q6_K.gguf
//!
//! or without a model (no-op, so plain `cargo bench` stays green):
//!
//!   cargo bench --bench hooks_overhead

use anyhow::bail;
use clap::Parser;
use ember::backend::CpuBackend;
use ember::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentRunner, ExperimentalForwardModel,
    ModelContext, ModelFamily, TracingState,
};
use ember::loader::{load_gguf_with_k_strategy, GgufLoader};
use ember::model::ForwardModel;
use ember::plan::ExecutionMode;
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

struct NoopExperiment;
impl Experiment for NoopExperiment {
    fn name(&self) -> &'static str {
        "noop"
    }
}

#[derive(Debug, Parser)]
#[command(about = "Measure planned-decode overhead when a (noop) experiment runner is attached")]
struct Args {
    /// Cargo passes this marker to custom benchmark harnesses.
    #[arg(long, hide = true)]
    bench: bool,

    /// GGUF model to decode with.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Prompt used for prefill.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// Greedy decode length (tokens after prefill).
    #[arg(long, default_value_t = 8)]
    tokens: usize,

    /// Warmup passes of each mode before sampling.
    #[arg(long, default_value_t = 2)]
    warmups: usize,

    /// Sampled runs of each mode (timed, best-of reported).
    #[arg(long, default_value_t = 5)]
    samples: usize,
}

struct RunStats {
    bare_ns: Vec<u128>,
    hooked_ns: Vec<u128>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let Some(model_path) = args.model.as_ref() else {
        return Ok(());
    };
    if args.samples == 0 {
        bail!("--samples must be greater than zero");
    }

    let loader: GgufLoader = load_gguf_with_k_strategy(model_path, KStrategy::Auto, false)?;
    let has_k_quant = loader
        .tensors
        .values()
        .any(|t| matches!(t, ember::loader::LoadedTensor::KQuant(_)));
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048))?;
    let tokenizer = EmberTokenizer::from_file("tokenizer.json")?;
    let backend = CpuBackend;
    model.set_execution_mode(ExecutionMode::Planned);

    if !has_k_quant {
        eprintln!(
            "note: model has no K-quant tensors; the Q8_0 fast path (contract D1) \
             short-circuits the planned interpreter, so this measures the generic \
             hooked path only"
        );
    }

    let ids = tokenizer.encode(&args.prompt)?;
    let ctx = ModelContext::new(
        ModelFamily::Llama,
        None,
        "llama",
        model.n_layers(),
        model.embed_dim(),
    );

    // Decode `args.tokens` tokens once; the reported unit is a whole decode run
    // (prefill + tokens) so hook-site firing and cache pressure are realistic.
    let decode_once = |hooked: bool| -> u128 {
        let mut cache = model.create_cache(&backend, 2048);
        let start = Instant::now();
        let mut position = 0usize;
        let mut current = ids.clone();
        let mut last = 0usize;
        for _ in 0..args.tokens {
            if hooked {
                let execution = ExecutionContext::new(
                    ctx,
                    if position == 0 {
                        ExecutionPhase::Prefill
                    } else {
                        ExecutionPhase::Decode
                    },
                    position,
                    current.len(),
                    TracingState::Disabled,
                );
                let mut runner = ExperimentRunner::new(NoopExperiment);
                let logits = model
                    .forward_last_logits_with_experiment(
                        &backend,
                        &current,
                        &mut cache,
                        position,
                        execution,
                        &mut runner,
                    )
                    .expect("hooked decode");
                last = ember::sampler::argmax_token(logits.data());
            } else {
                let logits = ForwardModel::forward_last_logits_with_cache(
                    &model, &backend, &current, &mut cache, position,
                )
                .expect("bare decode");
                last = ember::sampler::argmax_token(logits.data());
            }
            position += current.len();
            current = vec![u32::try_from(last).expect("token id fits u32")];
        }
        black_box(last);
        start.elapsed().as_nanos()
    };

    for _ in 0..args.warmups {
        decode_once(false);
        decode_once(true);
    }

    let mut stats = RunStats {
        bare_ns: Vec::new(),
        hooked_ns: Vec::new(),
    };
    for _ in 0..args.samples {
        stats.bare_ns.push(decode_once(false));
        stats.hooked_ns.push(decode_once(true));
    }
    let bare = *stats.bare_ns.iter().min().unwrap();
    let hooked = *stats.hooked_ns.iter().min().unwrap();
    let bare_ms = bare as f64 / 1e6;
    let hooked_ms = hooked as f64 / 1e6;
    let overhead_pct = (hooked as f64 / bare as f64 - 1.0) * 100.0;

    println!(
        "model: {}\nk-quant: {}\ntokens: {} (incl. prefill)\n\
         bare:   {bare_ms:8.2} ms/run\n\
         hooked: {hooked_ms:8.2} ms/run\n\
         hook overhead: {overhead_pct:+.1}%",
        model_path.display(),
        has_k_quant,
        args.tokens,
    );
    Ok(())
}

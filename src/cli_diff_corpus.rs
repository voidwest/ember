//! `ember diff-corpus`: scaled differential corpus runner (CLI surface).
//!
//! The engine, mutation operators, output schema, and frozen-tree guard live
//! in [`ember::diff_corpus`], shared with the optional Python binding.

use clap::Args as ClapArgs;
use ember::diff_corpus::{run_diff_corpus, CorpusMode, CorpusRequest};
use ember::diff_outcome::ExternalRuntime;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub(crate) struct DiffCorpusCommand {
    /// Number of mutations to generate and evaluate.
    #[arg(long)]
    pub n: usize,
    /// RNG seed (same operators/slots as diff_fuzz.py `--seed`, but the
    /// stream is NOT bit-identical: StdRng vs Python random.Random).
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Mutation mode: raw or construction.
    #[arg(long, value_enum, default_value_t = CorpusMode::Raw)]
    pub mode: CorpusMode,
    /// External runtimes to compare against (repeatable or comma-separated);
    /// the ember side is always evaluated.
    #[arg(long, value_delimiter = ',')]
    pub against: Vec<String>,
    /// Per-runtime deadline in seconds (externals; ember is in-process).
    #[arg(long, default_value_t = 8.0)]
    pub timeout_secs: f64,
    /// Parallel case workers (max 4: at most 4 concurrent child processes).
    #[arg(long, default_value_t = 4)]
    pub jobs: usize,
    /// Output directory (REFUSED when inside research/embersec/comparative/).
    #[arg(long)]
    pub out_dir: PathBuf,
    /// Explicit seed files (repeatable or comma-separated). Default mirrors
    /// diff_fuzz.py: all non-tokenizer corpus fixtures (raw) or the
    /// gguf-050/051/052 valid models (construction).
    #[arg(long, value_delimiter = ',')]
    pub seeds: Vec<String>,
}

pub(crate) fn run_diff_corpus_command(command: &DiffCorpusCommand) -> anyhow::Result<()> {
    anyhow::ensure!(command.n > 0, "--n must be positive");
    anyhow::ensure!(
        command.timeout_secs > 0.0,
        "--timeout-secs must be positive"
    );
    anyhow::ensure!(command.jobs > 0, "--jobs must be positive");
    if command.jobs > 4 {
        eprintln!(
            "diff-corpus: --jobs {} exceeds the 4-concurrent-child cap; clamping to 4",
            command.jobs
        );
    }
    let mut against = Vec::new();
    for name in &command.against {
        match ExternalRuntime::parse(name) {
            Some(runtime) if !against.contains(&runtime) => against.push(runtime),
            Some(_) => {}
            None => anyhow::bail!(
                "unknown runtime '{name}'; supported: llama.cpp, candle (see `ember diff runtimes`)"
            ),
        }
    }
    let request = CorpusRequest {
        n: command.n,
        seed: command.seed,
        mode: command.mode,
        against,
        timeout_secs: command.timeout_secs,
        jobs: command.jobs,
        out_dir: command.out_dir.clone(),
        seeds: command.seeds.clone(),
    };
    run_diff_corpus(&request, true)?;
    Ok(())
}

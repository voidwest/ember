use clap::{Args as ClapArgs, Subcommand};
use ember::diff_outcome::{
    evaluate_ember, evaluate_external, DiffOutcome, ExternalRuntime, SideReport,
};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;

/// Default per-runtime deadline. Matches the EmberSEC evaluation harness
/// (`run_eval.py --timeout` default): long enough for load+construct on
/// hostile inputs, short enough that a hung runtime cannot stall a corpus.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(ClapArgs)]
pub(crate) struct DiffCommand {
    /// File to evaluate (GGUF model or tokenizer.json).
    pub file: PathBuf,
    /// External runtimes to compare against (repeatable or comma-separated).
    #[arg(long, value_delimiter = ',', required = true)]
    pub against: Vec<String>,
    /// Per-runtime deadline in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECS)]
    pub timeout_secs: u64,
    /// Emit machine-readable JSON to stdout (default: human-readable).
    #[arg(long)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Option<DiffSubcommand>,
}

#[derive(Subcommand)]
pub(crate) enum DiffSubcommand {
    /// List the external runtimes this binary knows how to resolve.
    Runtimes,
}

#[derive(Debug, Clone, Serialize)]
struct DiffReport {
    schema: String,
    file: String,
    ember: SideReport,
    externals: Vec<SideReport>,
    agreement: Agreement,
}

#[derive(Debug, Clone, Serialize)]
struct Agreement {
    /// True when every evaluated side shares one outcome category.
    all_agree: bool,
    /// Distinct outcome tokens observed, in evaluation order.
    distinct_outcomes: Vec<String>,
    /// Human sentence describing the comparison.
    summary: String,
}

fn agreement(ember: &SideReport, externals: &[SideReport]) -> Agreement {
    let mut distinct = vec![ember.outcome.token().to_string()];
    for side in externals {
        let token = side.outcome.token().to_string();
        if !distinct.contains(&token) {
            distinct.push(token);
        }
    }
    let all_agree = distinct.len() == 1;
    let summary = if all_agree {
        format!("all runtimes agree: {}", distinct[0])
    } else {
        let parts = std::iter::once((&ember.runtime, &ember.outcome))
            .chain(externals.iter().map(|side| (&side.runtime, &side.outcome)))
            .map(|(runtime, outcome)| format!("{runtime}={}", outcome.token()))
            .collect::<Vec<_>>()
            .join(" ");
        format!("DISAGREE: {parts}")
    };
    Agreement {
        all_agree,
        distinct_outcomes: distinct,
        summary,
    }
}

fn render_human(report: &DiffReport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("diff: {}", report.file));
    lines.push(format!(
        "ember: {} ({})",
        report.ember.outcome.token(),
        report.ember.stderr_tail.lines().next().unwrap_or("")
    ));
    for side in &report.externals {
        let detail = if side.outcome == DiffOutcome::HarnessError {
            side.harness_detail.as_deref().unwrap_or("")
        } else {
            side.stderr_tail.lines().next().unwrap_or("")
        };
        let termination = side.termination.as_deref().unwrap_or("not-run");
        let wall = side
            .wall_ms
            .map(|ms| format!("{ms:.0}ms"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "{}: {} [{termination}] {wall} {detail}",
            side.runtime,
            side.outcome.token(),
            detail = detail.chars().take(120).collect::<String>(),
        ));
        if side.stdout_truncated || side.stderr_truncated {
            lines.push(format!(
                "  (output truncated: stdout={} stderr={})",
                side.stdout_truncated, side.stderr_truncated
            ));
        }
    }
    lines.push(report.agreement.summary.clone());
    lines.join("\n")
}

pub(crate) fn run_diff_command(command: &DiffCommand) -> anyhow::Result<()> {
    if let Some(DiffSubcommand::Runtimes) = &command.command {
        println!(
            "llama.cpp (env {} or PATH {})",
            ExternalRuntime::LlamaCpp.env_override(),
            ExternalRuntime::LlamaCpp.default_binary()
        );
        println!(
            "candle (env {} or PATH {})",
            ExternalRuntime::Candle.env_override(),
            ExternalRuntime::Candle.default_binary()
        );
        return Ok(());
    }
    anyhow::ensure!(command.timeout_secs > 0, "--timeout-secs must be positive");
    let mut runtimes = Vec::new();
    for name in &command.against {
        match ExternalRuntime::parse(name) {
            Some(runtime) if !runtimes.contains(&runtime) => runtimes.push(runtime),
            Some(_) => {}
            None => anyhow::bail!(
                "unknown runtime '{name}'; supported: llama.cpp, candle (see `ember diff <file> runtimes`)"
            ),
        }
    }
    let timeout = Duration::from_secs(command.timeout_secs);
    // Ember first (in-process, same trust boundary as `inspect`); external
    // runtimes after, evaluated CONCURRENTLY on scoped threads. Sequential
    // evaluation would stack per-runtime timeouts (2 runtimes x 30 s hangs
    // = 60 s per file; x62 corpus files = over an hour of pure waiting),
    // so parallelism is load-bearing for corpus use, not an optimization.
    // Each evaluation owns its pipes, timeout, and report — no shared
    // mutable state — so thread scope is sound; a panicking worker would
    // propagate via join (evaluators never panic by contract, but scope
    // makes even that a loud failure, not a silent hang).
    let ember = evaluate_ember(&command.file);
    let externals = std::thread::scope(|scope| {
        runtimes
            .iter()
            .map(|runtime| scope.spawn(move || evaluate_external(*runtime, &command.file, timeout)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("diff worker panicked; evaluators must not panic")
            })
            .collect::<Vec<_>>()
    });
    let report = DiffReport {
        schema: "ember.diff.v1".to_string(),
        file: command.file.display().to_string(),
        agreement: agreement(&ember, &externals),
        ember,
        externals,
    };
    if command.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", render_human(&report));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestDiffParser {
        #[command(flatten)]
        diff: DiffCommand,
    }

    #[test]
    fn diff_cli_parses_against_and_flags() {
        let parsed = TestDiffParser::try_parse_from([
            "test",
            "model.gguf",
            "--against",
            "llama.cpp,candle",
            "--timeout-secs",
            "10",
            "--json",
        ])
        .unwrap();
        assert_eq!(parsed.diff.file, PathBuf::from("model.gguf"));
        assert_eq!(parsed.diff.timeout_secs, 10);
        assert!(parsed.diff.json);
        // comma-separated single flag splits into two entries.
        assert_eq!(parsed.diff.against, vec!["llama.cpp", "candle"]);
    }

    #[test]
    fn diff_cli_accepts_repeatable_against() {
        let parsed = TestDiffParser::try_parse_from([
            "test",
            "model.gguf",
            "--against",
            "llama.cpp",
            "--against",
            "candle",
        ])
        .unwrap();
        assert_eq!(parsed.diff.against, vec!["llama.cpp", "candle"]);
    }

    #[test]
    fn agreement_all_same_is_trivially_true() {
        let ember = SideReport {
            runtime: "ember".to_string(),
            outcome: DiffOutcome::StructuredReject,
            termination: Some("in-process".to_string()),
            wall_ms: Some(1.0),
            stderr_tail: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            harness_detail: None,
        };
        let other = SideReport {
            runtime: "llama.cpp".to_string(),
            outcome: DiffOutcome::StructuredReject,
            ..ember.clone()
        };
        let agreement = agreement(&ember, &[other]);
        assert!(agreement.all_agree);
        assert_eq!(agreement.distinct_outcomes, vec!["STRUCTURED_REJECT"]);
    }

    #[test]
    fn agreement_detects_divergence() {
        let ember = SideReport {
            runtime: "ember".to_string(),
            outcome: DiffOutcome::Accept,
            termination: Some("in-process".to_string()),
            wall_ms: Some(1.0),
            stderr_tail: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            harness_detail: None,
        };
        let other = SideReport {
            runtime: "candle".to_string(),
            outcome: DiffOutcome::ProcessCrash,
            ..ember.clone()
        };
        let agreement = agreement(&ember, &[other]);
        assert!(!agreement.all_agree);
        assert!(agreement.summary.contains("DISAGREE"));
        assert!(agreement.summary.contains("ember=ACCEPT"));
        assert!(agreement.summary.contains("candle=PROCESS_CRASH"));
    }

    #[test]
    fn diff_rejects_zero_timeout() {
        let command = DiffCommand {
            file: PathBuf::from("x.gguf"),
            against: vec!["candle".to_string()],
            timeout_secs: 0,
            json: false,
            command: None,
        };
        assert!(run_diff_command(&command).is_err());
    }

    #[test]
    fn diff_rejects_unknown_runtime() {
        let command = DiffCommand {
            file: PathBuf::from("x.gguf"),
            against: vec!["vllm".to_string()],
            timeout_secs: 5,
            json: false,
            command: None,
        };
        let error = run_diff_command(&command).unwrap_err().to_string();
        assert!(error.contains("unknown runtime"));
    }

    #[test]
    fn externals_evaluate_concurrently_not_sequentially() {
        // Two 3-second sleeps as fake runtimes must complete in well under
        // 6 seconds wall time; sequential evaluation would stack them.
        // Uses the lib evaluators directly (no CLI, no real binaries).
        use ember::diff_outcome::ExternalRuntime;
        use std::time::Instant;
        let dir = std::env::temp_dir().join(format!("ember-diff-conc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sleeper = dir.join("sleeper.sh");
        std::fs::write(&sleeper, b"#!/bin/sh\nsleep 3\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sleeper, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        // SAFETY: same single-threaded-env reasoning as diff_outcome tests;
        // no other thread reads env here.
        unsafe {
            std::env::set_var(ExternalRuntime::LlamaCpp.env_override(), &sleeper);
            std::env::set_var(ExternalRuntime::Candle.env_override(), &sleeper);
        }
        let file = dir.join("junk.gguf");
        std::fs::write(&file, b"junk").unwrap();
        let start = Instant::now();
        let timeout = std::time::Duration::from_secs(20);
        let reports = std::thread::scope(|scope| {
            [ExternalRuntime::LlamaCpp, ExternalRuntime::Candle]
                .into_iter()
                .map(|runtime| {
                    let file = file.clone();
                    scope.spawn(move || {
                        ember::diff_outcome::evaluate_external(runtime, &file, timeout)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("worker panicked"))
                .collect::<Vec<_>>()
        });
        let wall = start.elapsed();
        for report in &reports {
            assert_eq!(report.outcome, ember::diff_outcome::DiffOutcome::Accept);
        }
        // Sequential would take >= 6 s; concurrent takes ~3 s. Generous
        // ceiling keeps loaded CI honest while still catching regression
        // to sequential evaluation.
        assert!(
            wall < std::time::Duration::from_secs(5),
            "externals ran sequentially: {wall:?}"
        );
        unsafe {
            std::env::remove_var(ExternalRuntime::LlamaCpp.env_override());
            std::env::remove_var(ExternalRuntime::Candle.env_override());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn diff_runs_end_to_end_without_external_binaries() {
        // No llama.cpp/candle installed in CI: both sides must report
        // HarnessError while Ember still evaluates the (junk) file.
        let dir = std::env::temp_dir().join(format!("ember-diff-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("junk.gguf");
        std::fs::write(&file, b"definitely not gguf").unwrap();
        let command = DiffCommand {
            file: file.clone(),
            against: vec!["llama.cpp".to_string(), "candle".to_string()],
            timeout_secs: 5,
            json: false,
            command: None,
        };
        // Must not error even though both externals are missing.
        run_diff_command(&command).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

use clap::{Args as ClapArgs, Subcommand};
use ember::diff_outcome::{evaluate_diff, DiffOutcome, DiffReport, ExternalRuntime};
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
    let report = evaluate_diff(
        &command.file,
        &runtimes,
        Duration::from_secs(command.timeout_secs),
    );
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
    use ember::diff_outcome::evaluate_external;

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

    #[cfg(unix)]
    #[test]
    fn externals_evaluate_concurrently_not_sequentially() {
        const CHILD_INPUT: &str = "EMBER_DIFF_CONCURRENCY_TEST_INPUT";
        let file = match std::env::var_os(CHILD_INPUT) {
            Some(file) => PathBuf::from(file),
            None => {
                // Other CLI tests change the same runtime overrides. Run
                // this test alone in a child with its own environment;
                // mutating the parallel test process races those tests.
                use std::os::unix::fs::PermissionsExt;
                let dir = std::env::temp_dir().join(format!(
                    "ember-diff-conc-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos(),
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let file = dir.join("junk.gguf");
                std::fs::write(&file, b"junk").unwrap();
                // Each runtime waits for BOTH start markers. Sequential
                // evaluation necessarily times out; concurrent evaluation
                // succeeds without a scheduler-sensitive wall-time bound.
                let script = b"#!/bin/sh\nset -eu\ntouch \"$1.$(basename \"$0\").started\"\nwhile [ ! -f \"$1.llama.sh.started\" ] || [ ! -f \"$1.candle.sh.started\" ]; do sleep 0.01; done\n";
                let llama = dir.join("llama.sh");
                let candle = dir.join("candle.sh");
                for path in [&llama, &candle] {
                    std::fs::write(path, script).unwrap();
                    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
                }
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "cli_diff::tests::externals_evaluate_concurrently_not_sequentially",
                        "--nocapture",
                    ])
                    .env(CHILD_INPUT, &file)
                    .env(ExternalRuntime::LlamaCpp.env_override(), &llama)
                    .env(ExternalRuntime::Candle.env_override(), &candle)
                    .output();
                let peers_started = ["llama.sh", "candle.sh"]
                    .into_iter()
                    .all(|name| dir.join(format!("junk.gguf.{name}.started")).is_file());
                std::fs::remove_dir_all(&dir).unwrap();
                let output = output.expect("failed to start isolated concurrency test");
                assert!(
                    output.status.success() && peers_started,
                    "isolated concurrency test failed (both peers started: {peers_started}): {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
                return;
            }
        };
        let timeout = Duration::from_secs(10);
        let reports = std::thread::scope(|scope| {
            [ExternalRuntime::LlamaCpp, ExternalRuntime::Candle]
                .into_iter()
                .map(|runtime| {
                    let file = file.clone();
                    scope.spawn(move || evaluate_external(runtime, &file, timeout))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("worker panicked"))
                .collect::<Vec<_>>()
        });
        for report in &reports {
            assert_eq!(report.outcome, DiffOutcome::Accept, "{report:#?}");
        }
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

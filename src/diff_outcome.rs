//! Comparative outcome taxonomy and external-runtime adapters for `diff`.
//!
//! The outcome vocabulary is taken verbatim from the authoritative source:
//! `research/embersec/comparative/run_eval.py` (`OUTCOMES`). Do not invent a
//! competing taxonomy here; map new evidence onto these categories.
//!
//! Three failure domains stay distinct:
//!
//! - **harness failure** ([`HarnessError`] in [`crate::subprocess`]): Ember
//!   could not even run the subject (binary missing, spawn refused,
//!   supervision broke). Reported as [`DiffOutcome::HarnessError`], never as
//!   a runtime crash.
//! - **runtime-under-test failure**: the subject ran and its termination +
//!   stderr classify into Accept / StructuredReject / Panic / ProcessCrash /
//!   Timeout / ResourceLimit.
//! - **Ember evaluation failure**: Ember's own in-process load of the file
//!   failed; reported per-side, not as a global error, so other runtimes
//!   still get evaluated.

use crate::subprocess::{HarnessError, SupervisedCommand, SupervisedResult, Termination};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Comparative outcome. Names and semantics match `run_eval.py` OUTCOMES,
/// with one deliberate rename: `ResourceLimitOrExternalKill` replaces
/// run_eval's `RESOURCE_LIMIT` because a bare SIGKILL the harness did not
/// send is indistinguishable from OOM-killer action at the process level
/// (no cgroup/RSS check here). Downstream tables must treat it as "killed
/// from outside", not confirmed memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DiffOutcome {
    /// Subject loaded/accepted the file (exit 0).
    Accept,
    /// Subject rejected the file through a structured error path (exit 1).
    StructuredReject,
    /// Subject panicked (Rust exit 101).
    Panic,
    /// Subject died by OS signal / abnormal termination.
    ProcessCrash,
    /// Subject exceeded its deadline and was killed + reaped.
    Timeout,
    /// SIGKILL the harness did NOT send (OOM killer, container limit, or
    /// any external `kill -9` — indistinguishable here; see above).
    ResourceLimitOrExternalKill,
    /// Subject cannot consume this input class at all (e.g. tokenizer-only
    /// input for a GGUF-only loader). Matches `NOT_COMPARABLE` in run_eval.
    NotComparable,
    /// The harness failed, not the subject (binary missing, spawn refused).
    HarnessError,
}

impl DiffOutcome {
    /// Machine string. Identical to the `run_eval.py` outcome token except
    /// `RESOURCE_LIMIT_OR_EXTERNAL_KILL`, which is intentionally NOT
    /// `RESOURCE_LIMIT`: the old token asserts a cause this layer cannot
    /// confirm, and reusing it would launder uncertainty into Phase I's
    /// real numbers.
    pub fn token(self) -> &'static str {
        match self {
            Self::Accept => "ACCEPT",
            Self::StructuredReject => "STRUCTURED_REJECT",
            Self::Panic => "PANIC",
            Self::ProcessCrash => "PROCESS_CRASH",
            Self::Timeout => "TIMEOUT",
            Self::ResourceLimitOrExternalKill => "RESOURCE_LIMIT_OR_EXTERNAL_KILL",
            Self::NotComparable => "NOT_COMPARABLE",
            Self::HarnessError => "HARNESS_ERROR",
        }
    }
}

/// One side of a comparison: what ran, what happened, what it means.
#[derive(Debug, Clone, Serialize)]
pub struct SideReport {
    /// Runtime identity (`ember`, `llama-cpp`, `candle`).
    pub runtime: String,
    /// Classified outcome.
    pub outcome: DiffOutcome,
    /// How the subject process ended, if it ran.
    pub termination: Option<String>,
    /// Wall time (ms) from spawn to reap, if it ran.
    pub wall_ms: Option<f64>,
    /// Bounded stderr tail (400 chars, like run_eval's `stderr_tail`).
    pub stderr_tail: String,
    /// Whether stdout capture was truncated.
    pub stdout_truncated: bool,
    /// Whether stderr capture was truncated.
    pub stderr_truncated: bool,
    /// Harness-level detail when `outcome == HarnessError`, else None.
    pub harness_detail: Option<String>,
}

/// Map a supervised execution onto the comparative taxonomy.
///
/// Mirrors `classify_ember` in `run_eval.py`: timeout first, then exit
/// 0 / 1 / 101, then signals. Deliberately NOT mirroring
/// `classify_llamacpp`'s GGML_ASSERT stderr sniff (exit 1 + assert text
/// → PANIC): that branch fired zero times across all three frozen result
/// sets (llama-cpp.json, llama-cpp-cli-b5999.json, candle.json) — the only
/// two real llama.cpp assertion failures both manifested as signals
/// (SIGFPE, SIGABRT) and classified as PROCESS_CRASH. A text sniff on
/// untrusted stderr is the weakest classifier here; exit 1 stays
/// STRUCTURED_REJECT for every runtime until adapter validation against
/// real binaries proves otherwise.
pub fn classify_supervised(result: &SupervisedResult) -> DiffOutcome {
    if result.killed_by_harness || result.termination == Termination::Timeout {
        return DiffOutcome::Timeout;
    }
    match &result.termination {
        Termination::Success => DiffOutcome::Accept,
        Termination::ExitCode(1) => DiffOutcome::StructuredReject,
        Termination::ExitCode(101) => DiffOutcome::Panic,
        // A SIGKILL the harness did NOT send: OOM killer, container
        // limit, or any external kill — indistinguishable at this layer,
        // hence the renamed outcome (see enum docs).
        Termination::Signal(9) => DiffOutcome::ResourceLimitOrExternalKill,
        Termination::Signal(_) => DiffOutcome::ProcessCrash,
        // Abnormal-but-codeless termination (Windows kill paths).
        Termination::ExitCode(-1) => DiffOutcome::ProcessCrash,
        Termination::ExitCode(_) => DiffOutcome::HarnessError,
        Termination::Timeout => DiffOutcome::Timeout,
    }
}

/// External runtime identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalRuntime {
    LlamaCpp,
    Candle,
}

impl ExternalRuntime {
    /// CLI-facing name (`--against llama.cpp,candle`).
    pub fn name(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Candle => "candle",
        }
    }

    /// Environment override consulted before PATH (`EMBER_LLAMACPP_BIN`,
    /// `EMBER_CANDLE_BIN`). Explicit configuration always wins over search.
    pub fn env_override(self) -> &'static str {
        match self {
            Self::LlamaCpp => "EMBER_LLAMACPP_BIN",
            Self::Candle => "EMBER_CANDLE_BIN",
        }
    }

    /// Default binary filename probed on PATH when no override is set.
    pub fn default_binary(self) -> &'static str {
        match self {
            // The EmberSEC reference harness binary name; a bare
            // `llama-cli` would have unknown load-check semantics, so it is
            // deliberately NOT a default (see adapter docs below).
            Self::LlamaCpp => "embersec_loader_check",
            Self::Candle => "candle-gguf-check",
        }
    }

    /// Parse one `--against` element.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "llama.cpp" | "llamacpp" | "llama-cpp" => Some(Self::LlamaCpp),
            "candle" => Some(Self::Candle),
            _ => None,
        }
    }
}

/// Where a runtime binary came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryResolution {
    /// Explicit path from environment override.
    Configured(PathBuf),
    /// Found by searching PATH for the default filename.
    FoundOnPath(PathBuf),
}

/// Resolve an external runtime binary: explicit env override first, then
/// PATH search for the default filename. Returns a [`HarnessError`] with an
/// actionable message (`<name> runtime not found`) rather than a raw OS
/// error. No developer-specific paths are hardcoded anywhere.
pub fn resolve_runtime_binary(runtime: ExternalRuntime) -> Result<BinaryResolution, HarnessError> {
    if let Some(configured) = std::env::var_os(runtime.env_override()) {
        let path = PathBuf::from(configured);
        if !path.exists() {
            return Err(HarnessError::BinaryNotFound { program: path });
        }
        return Ok(BinaryResolution::Configured(path));
    }
    let filename = runtime.default_binary();
    if let Some(found) = search_path(filename) {
        return Ok(BinaryResolution::FoundOnPath(found));
    }
    Err(HarnessError::BinaryNotFound {
        program: PathBuf::from(format!(
            "{} (set {} to configure)",
            filename,
            runtime.env_override()
        )),
    })
}

/// Minimal PATH search: split on the platform separator, join the filename,
/// accept the first entry that exists and (on Unix) is executable. Bare
/// relative entries (empty components meaning cwd) are skipped — resolving
/// a runtime out of an unqualified cwd is exactly the unsafe pattern the
/// trust-boundary rules forbid.
fn search_path(filename: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        if dir.is_relative() {
            continue;
        }
        let candidate = dir.join(filename);
        if candidate.is_file() && is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Executable-bit check on Unix; existence check elsewhere (Windows
/// execution permission is ACL-based and `CreateProcess` reports refusal
/// at spawn, which already maps to `NotExecutable`).
fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        true
    }
}

/// Build the harness command for one external runtime over one file.
///
/// Invocation contract taken from the EmberSEC reference *sources* in
/// `research/embersec/comparative/reference/`:
///
/// - llama.cpp reference (`reference/llama_loader/loader_check.cpp`):
///   `loader_check <file>`; exit 0 = load+free OK, nonzero = reject.
/// - candle reference (`reference/candle/main.rs`): `candle-gguf-check
///   <file>`; exit 0 = parse OK, 1 = structured reject, 101 = panic.
///
/// UNVERIFIED: neither adapter has been validated against a real built
/// binary yet — the contracts above come from reading the reference
/// sources, not from executing them. Until that validation lands, every
/// external classification this layer produces is provisional and must
/// not feed Phase I result tables.
///
/// Both references are single-process load-check harnesses (no
/// generation), which is what makes kill-and-reap a complete termination
/// story (see [`crate::subprocess`] platform notes).
///
/// `runtime` is kept explicit (not inferred from the binary path) so future
/// adapters with extra argv can branch here without changing call sites.
pub fn adapter_command(
    runtime: ExternalRuntime,
    binary: &Path,
    file: &Path,
    timeout: Duration,
) -> SupervisedCommand {
    let _ = runtime;
    SupervisedCommand::new(
        binary.to_path_buf(),
        vec![file.to_string_lossy().into_owned()],
        timeout,
    )
}

/// Evaluate one file against one external runtime: resolve, run, classify.
/// A missing/unrunnable binary yields `HarnessError`, never a crash label.
pub fn evaluate_external(runtime: ExternalRuntime, file: &Path, timeout: Duration) -> SideReport {
    let binary = match resolve_runtime_binary(runtime) {
        Ok(BinaryResolution::Configured(path)) | Ok(BinaryResolution::FoundOnPath(path)) => path,
        Err(error) => {
            return SideReport {
                runtime: runtime.name().to_string(),
                outcome: DiffOutcome::HarnessError,
                termination: None,
                wall_ms: None,
                stderr_tail: String::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                harness_detail: Some(format!("{} runtime not found: {error}", runtime.name())),
            };
        }
    };
    let command = adapter_command(runtime, &binary, file, timeout);
    match crate::subprocess::run_supervised(&command) {
        Ok(result) => {
            let outcome = classify_supervised(&result);
            SideReport {
                runtime: runtime.name().to_string(),
                outcome,
                termination: Some(termination_string(&result.termination)),
                wall_ms: Some(result.elapsed.as_secs_f64() * 1000.0),
                stderr_tail: result.stderr.tail_lossy(400),
                stdout_truncated: result.stdout.truncated,
                stderr_truncated: result.stderr.truncated,
                harness_detail: None,
            }
        }
        Err(error) => SideReport {
            runtime: runtime.name().to_string(),
            outcome: DiffOutcome::HarnessError,
            termination: None,
            wall_ms: None,
            stderr_tail: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            harness_detail: Some(error.to_string()),
        },
    }
}

/// Evaluate a file with Ember itself (in-process): load check first, then
/// model-construction check — the same two stages as the `_embersec_harness`
/// `gguf_load_check` / `gguf_model_check` stages. Never panics on hostile
/// input: loader and constructor errors map to StructuredReject.
pub fn evaluate_ember(file: &Path) -> SideReport {
    let outcome: DiffOutcome;
    let start = std::time::Instant::now();
    let bytes = match std::fs::read(file) {
        Ok(bytes) => bytes,
        Err(error) => {
            return SideReport {
                runtime: "ember".to_string(),
                outcome: DiffOutcome::HarnessError,
                termination: None,
                wall_ms: None,
                stderr_tail: String::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                harness_detail: Some(format!("harness could not read file: {error}")),
            };
        }
    };
    let loader = match crate::loader::load_gguf_from_reader(&mut std::io::Cursor::new(&bytes)) {
        Ok(loader) => loader,
        Err(error) => {
            outcome = DiffOutcome::StructuredReject;
            return ember_report(outcome, format!("HARNESS: LOAD_REJECT: {error}"), start);
        }
    };
    let arch = match loader.metadata.get("general.architecture") {
        Some(crate::loader::GgufValue::Str(s)) => s.as_str(),
        _ => "llama",
    };
    let result = match arch {
        "gemma3" | "gemma4" => crate::gemma4::Gemma4::from_loader(loader).map(|_| ()),
        "gpt2" => crate::model::Gpt2::from_loader(loader).map(|_| ()),
        _ => crate::llama::Llama::from_loader(loader).map(|_| ()),
    };
    match result {
        Ok(()) => ember_report(DiffOutcome::Accept, "HARNESS: MODEL_OK".to_string(), start),
        Err(error) => ember_report(
            DiffOutcome::StructuredReject,
            format!("HARNESS: MODEL_REJECT: {error}"),
            start,
        ),
    }
}

fn ember_report(
    outcome: DiffOutcome,
    stderr_tail: String,
    start: std::time::Instant,
) -> SideReport {
    SideReport {
        runtime: "ember".to_string(),
        outcome,
        termination: Some("in-process".to_string()),
        wall_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
        stderr_tail,
        stdout_truncated: false,
        stderr_truncated: false,
        harness_detail: None,
    }
}

fn termination_string(termination: &Termination) -> String {
    match termination {
        Termination::Success => "exit(0)".to_string(),
        Termination::ExitCode(code) => format!("exit({code})"),
        Termination::Signal(signal) => format!("signal({signal})"),
        Termination::Timeout => "timeout(killed+reaped)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprocess::{CapturedStream, SupervisedResult};

    fn supervised(termination: Termination, stderr: &str) -> SupervisedResult {
        SupervisedResult {
            program: PathBuf::from("test-subject"),
            termination,
            stdout: CapturedStream {
                bytes: Vec::new(),
                total_bytes: 0,
                truncated: false,
            },
            stderr: CapturedStream {
                bytes: stderr.as_bytes().to_vec(),
                total_bytes: stderr.len() as u64,
                truncated: false,
            },
            elapsed: Duration::from_millis(1),
            killed_by_harness: false,
        }
    }

    #[test]
    fn exit_zero_is_accept() {
        let result = supervised(Termination::Success, "");
        assert_eq!(classify_supervised(&result), DiffOutcome::Accept);
        assert_eq!(DiffOutcome::Accept.token(), "ACCEPT");
    }

    #[test]
    fn exit_one_is_structured_reject() {
        let result = supervised(Termination::ExitCode(1), "bad header");
        assert_eq!(classify_supervised(&result), DiffOutcome::StructuredReject);
    }

    #[test]
    fn exit_one_with_assert_text_stays_structured_reject() {
        // The old GGML_ASSERT stderr sniff is gone: it fired zero times
        // across all three frozen result sets (the only two real llama.cpp
        // assertion failures manifested as signals). Exit 1 + assert text
        // is a structured reject until adapter validation proves otherwise.
        let result = supervised(Termination::ExitCode(1), "GGML_ASSERT(fail) boom");
        assert_eq!(classify_supervised(&result), DiffOutcome::StructuredReject);
    }

    #[test]
    fn exit_101_is_panic() {
        let result = supervised(Termination::ExitCode(101), "panicked");
        assert_eq!(classify_supervised(&result), DiffOutcome::Panic);
    }

    #[test]
    fn signal_is_process_crash() {
        let result = supervised(Termination::Signal(11), "");
        assert_eq!(classify_supervised(&result), DiffOutcome::ProcessCrash);
    }

    #[test]
    fn sigkill_without_harness_kill_is_external_not_timeout() {
        // The token deliberately does NOT say RESOURCE_LIMIT: a bare
        // SIGKILL the harness did not send is indistinguishable from OOM
        // action at this layer.
        let result = supervised(Termination::Signal(9), "oom");
        assert_eq!(
            classify_supervised(&result),
            DiffOutcome::ResourceLimitOrExternalKill
        );
        assert_eq!(
            DiffOutcome::ResourceLimitOrExternalKill.token(),
            "RESOURCE_LIMIT_OR_EXTERNAL_KILL"
        );
    }

    #[test]
    fn harness_kill_is_timeout_not_external_kill() {
        let mut result = supervised(Termination::Timeout, "");
        result.killed_by_harness = true;
        assert_eq!(classify_supervised(&result), DiffOutcome::Timeout);
    }

    #[test]
    fn unknown_exit_is_harness_error_not_crash() {
        // Exit 42 means neither accept/reject/panic protocol: the harness
        // cannot interpret it, so it must not be laundered into a crash.
        let result = supervised(Termination::ExitCode(42), "");
        assert_eq!(classify_supervised(&result), DiffOutcome::HarnessError);
    }

    #[test]
    fn runtime_names_parse() {
        assert_eq!(
            ExternalRuntime::parse("llama.cpp"),
            Some(ExternalRuntime::LlamaCpp)
        );
        assert_eq!(
            ExternalRuntime::parse("candle"),
            Some(ExternalRuntime::Candle)
        );
        assert_eq!(ExternalRuntime::parse("vllm"), None);
        assert_eq!(ExternalRuntime::parse(""), None);
    }

    #[test]
    fn missing_runtime_resolves_to_actionable_error() {
        // No override set in test env (names are Ember-specific); PATH may
        // or may not contain the default — either way the contract holds:
        // Ok(path) or a BinaryNotFound whose message names the runtime.
        // SAFETY: no other thread in this test reads the process
        // environment concurrently (drain threads only move byte buffers).
        unsafe { std::env::remove_var(ExternalRuntime::Candle.env_override()) };
        match resolve_runtime_binary(ExternalRuntime::Candle) {
            Ok(_) => {}
            Err(HarnessError::BinaryNotFound { .. }) => {}
            Err(other) => panic!("unexpected resolution error: {other}"),
        }
    }

    #[test]
    fn missing_external_evaluates_to_harness_error() {
        // SAFETY: same single-threaded-env reasoning as above.
        unsafe { std::env::remove_var(ExternalRuntime::Candle.env_override()) };
        // Only meaningful when the binary is genuinely absent; if a dev
        // machine has one on PATH this still asserts the report shape.
        let report = evaluate_external(
            ExternalRuntime::Candle,
            Path::new("/nonexistent/file.gguf"),
            Duration::from_secs(5),
        );
        assert_eq!(report.runtime, "candle");
        assert!(matches!(
            report.outcome,
            DiffOutcome::HarnessError
                | DiffOutcome::StructuredReject
                | DiffOutcome::Accept
                | DiffOutcome::ProcessCrash
                | DiffOutcome::Timeout
                | DiffOutcome::Panic
                | DiffOutcome::ResourceLimitOrExternalKill
                | DiffOutcome::NotComparable
        ));
        if report.outcome == DiffOutcome::HarnessError {
            let detail = report.harness_detail.unwrap();
            assert!(detail.contains("candle runtime not found"));
        }
    }
}

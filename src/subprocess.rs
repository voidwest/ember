//! Supervised subprocess execution for differential testing.
//!
//! This module is the trust boundary between Ember and external runtimes
//! (llama.cpp, candle, or any future comparator). External binaries and the
//! input files they consume are **untrusted**: a malformed model may make a
//! runtime crash, hang, flood its pipes, or emit hostile bytes. The harness
//! itself must stay deterministic, bounded, and informative in all of those
//! cases.
//!
//! Design (mirrors `research/embersec/comparative/run_eval.py`, which is the
//! authoritative outcome vocabulary — see `OUTCOMES` there):
//!
//! - argv is passed directly to [`std::process::Command`]; there is no shell.
//! - stdin is always null; stdout/stderr are captured with hard byte caps.
//! - deadlines are enforced by kill + reap, never by merely "stop waiting".
//! - every failure mode has its own [`SupervisedOutcome`] variant; harness
//!   problems are never mislabeled as runtime crashes.
//!
//! Platform notes: Unix signal numbers are reported where
//! [`std::os::unix::process::ExitStatusExt`] is available. Windows has no
//! signals; abnormal termination surfaces as a nonzero exit code and maps to
//! `ProcessCrash` only when the status reports neither success nor a code.
//! Child-tree termination (grandchildren that outlive a kill) is **not**
//! attempted: killing a process group is not portable (and is dangerous when
//! Ember shares the group with the user's shell), so a runtime that
//! double-forks helpers may leave orphans. Adapter binaries must therefore
//! be single-process harnesses — the same constraint `run_eval.py` relies
//! on via `os.wait4` + `proc.kill()`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Hard cap on captured bytes per stream. Runtimes under test may flood
/// stdout/stderr on hostile input; the harness must not allocate without
/// bound. Matches the spirit of the loader's `MAX_NPY_BYTES`-style trust
/// boundaries: generous for legitimate diagnostics, fatal to floods.
pub const MAX_CAPTURE_BYTES_PER_STREAM: usize = 1 << 20;

/// How much head and tail to retain when a stream exceeds the cap. The
/// middle is discarded and [`CapturedStream::truncated`] records that fact.
pub const TRUNCATED_HEAD_BYTES: usize = 64 * 1024;
/// Tail counterpart of [`TRUNCATED_HEAD_BYTES`].
pub const TRUNCATED_TAIL_BYTES: usize = 64 * 1024;

/// What to invoke, with what, and for how long.
#[derive(Debug, Clone)]
pub struct SupervisedCommand {
    /// Executable path. Resolved by the caller (see `cli_diff` resolution);
    /// never searched implicitly here beyond what [`Command`] itself does
    /// with a bare name + PATH.
    pub program: PathBuf,
    /// Arguments passed verbatim (no shell, no joining, no quoting).
    pub argv: Vec<String>,
    /// Working directory for the child, if any.
    pub current_dir: Option<PathBuf>,
    /// Hard deadline for the whole execution (spawn to reap).
    pub timeout: Duration,
    /// Extra environment for the child. The parent environment is otherwise
    /// inherited unchanged.
    pub env_extra: Vec<(String, String)>,
}

impl SupervisedCommand {
    /// Build a command with a timeout; argv empty, no cwd override, no extras.
    pub fn new(program: impl Into<PathBuf>, argv: Vec<String>, timeout: Duration) -> Self {
        Self {
            program: program.into(),
            argv,
            current_dir: None,
            timeout,
            env_extra: Vec::new(),
        }
    }

    /// Override the child's working directory.
    pub fn with_current_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    /// Add one child-only environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_extra.push((key.into(), value.into()));
        self
    }
}

/// One captured byte stream with truncation metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedStream {
    /// Retained bytes (lossy-UTF8 decoded at the report layer, never here).
    pub bytes: Vec<u8>,
    /// Total bytes the child produced (post-cap), for diagnostics.
    pub total_bytes: u64,
    /// True when bytes were discarded (middle dropped, head+tail kept).
    pub truncated: bool,
}

impl CapturedStream {
    fn from_reader<R: Read>(mut reader: R) -> Self {
        let mut bytes = Vec::new();
        let mut total: u64 = 0;
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    total += n as u64;
                    if bytes.len() < MAX_CAPTURE_BYTES_PER_STREAM {
                        let room = MAX_CAPTURE_BYTES_PER_STREAM - bytes.len();
                        bytes.extend_from_slice(&chunk[..n.min(room)]);
                    }
                }
                Err(_) => break,
            }
        }
        if total <= MAX_CAPTURE_BYTES_PER_STREAM as u64 {
            Self {
                bytes,
                total_bytes: total,
                truncated: false,
            }
        } else {
            let head = bytes[..TRUNCATED_HEAD_BYTES.min(bytes.len())].to_vec();
            let tail_start = bytes.len().saturating_sub(TRUNCATED_TAIL_BYTES);
            let mut kept = head;
            kept.extend_from_slice(&bytes[tail_start..]);
            Self {
                bytes: kept,
                total_bytes: total,
                truncated: true,
            }
        }
    }

    /// Lossy text view for diagnostics. External bytes are never trusted as
    /// structured data; decode them lossily and only for display.
    pub fn text_lossy(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }

    /// Last up-to-`n` bytes as lossy text (stderr tails for reports).
    pub fn tail_lossy(&self, n: usize) -> String {
        let start = self.bytes.len().saturating_sub(n);
        String::from_utf8_lossy(&self.bytes[start..]).into_owned()
    }
}

/// How the child process ended, before outcome classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Termination {
    /// `exit(0)`.
    Success,
    /// Nonzero exit code.
    ExitCode(i32),
    /// Killed by a Unix signal (number where available).
    Signal(i32),
    /// Deadline expired; the child was killed and reaped by the harness.
    Timeout,
}

/// Structured result of one supervised execution.
#[derive(Debug, Clone)]
pub struct SupervisedResult {
    /// The command as invoked (echoed for provenance).
    pub program: PathBuf,
    /// How the child ended.
    pub termination: Termination,
    /// Captured stdout (bounded).
    pub stdout: CapturedStream,
    /// Captured stderr (bounded).
    pub stderr: CapturedStream,
    /// Wall time from spawn to reap.
    pub elapsed: Duration,
    /// True when the harness killed the child (timeout path).
    pub killed_by_harness: bool,
}

/// Why the harness itself could not produce a [`SupervisedResult`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessError {
    /// Executable does not exist at the resolved path.
    BinaryNotFound { program: PathBuf },
    /// Exists but the OS refused to execute it (permissions, bad format).
    NotExecutable { program: PathBuf, message: String },
    /// `spawn` failed for any other OS reason.
    SpawnFailed { program: PathBuf, message: String },
    /// Internal I/O failure while supervising (pipe error, clock error).
    SupervisionFailed { message: String },
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinaryNotFound { program } => {
                write!(f, "runtime binary not found: {}", program.display())
            }
            Self::NotExecutable { program, message } => {
                write!(
                    f,
                    "runtime binary cannot execute {}: {message}",
                    program.display()
                )
            }
            Self::SpawnFailed { program, message } => {
                write!(f, "failed to spawn {}: {message}", program.display())
            }
            Self::SupervisionFailed { message } => {
                write!(f, "subprocess supervision failed: {message}")
            }
        }
    }
}

impl std::error::Error for HarnessError {}

/// Classify an OS-level spawn failure without exposing raw errors.
/// Missing files and permission denials get their own variants so callers
/// can render actionable messages (`llama.cpp runtime not found`).
fn classify_spawn_error(program: &Path, error: std::io::Error) -> HarnessError {
    use std::io::ErrorKind;
    match error.kind() {
        ErrorKind::NotFound => HarnessError::BinaryNotFound {
            program: program.to_path_buf(),
        },
        ErrorKind::PermissionDenied => HarnessError::NotExecutable {
            program: program.to_path_buf(),
            message: error.to_string(),
        },
        _ => {
            // `ENOEXEC`/invalid-format surfaces as InvalidData or Other on
            // some platforms; treat a present-but-unrunnable file as
            // NotExecutable rather than a generic spawn failure.
            if program.exists() {
                HarnessError::NotExecutable {
                    program: program.to_path_buf(),
                    message: error.to_string(),
                }
            } else {
                HarnessError::SpawnFailed {
                    program: program.to_path_buf(),
                    message: error.to_string(),
                }
            }
        }
    }
}

/// Run a command under supervision: bounded capture, hard timeout with
/// kill + reap, no shell, null stdin.
///
/// Deadlock note: stdout and stderr are drained **concurrently** on two
/// threads before `wait` is called, so a child that floods one pipe while
/// blocked on the other cannot wedge the harness (the classic
/// `wait_with_output` single-thread hazard applies only when both pipes
/// share one reader; here each pipe has its own).
pub fn run_supervised(command: &SupervisedCommand) -> Result<SupervisedResult, HarnessError> {
    let mut child_cmd = Command::new(&command.program);
    child_cmd
        .args(&command.argv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = &command.current_dir {
        child_cmd.current_dir(dir);
    }
    for (key, value) in &command.env_extra {
        child_cmd.env(key, value);
    }
    let mut child = child_cmd
        .spawn()
        .map_err(|error| classify_spawn_error(&command.program, error))?;

    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| HarnessError::SupervisionFailed {
            message: "child stdout was not captured".to_string(),
        })?;
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| HarnessError::SupervisionFailed {
            message: "child stderr was not captured".to_string(),
        })?;

    // Drain each pipe on its own thread; the handles join below.
    let stdout_handle = std::thread::spawn(move || CapturedStream::from_reader(child_stdout));
    let stderr_handle = std::thread::spawn(move || CapturedStream::from_reader(child_stderr));

    let start = Instant::now();
    let (termination, killed_by_harness) = wait_bounded(&mut child, command.timeout)?;
    let elapsed = start.elapsed();

    let stdout = stdout_handle
        .join()
        .map_err(|_| HarnessError::SupervisionFailed {
            message: "stdout drain thread panicked".to_string(),
        })?;
    let stderr = stderr_handle
        .join()
        .map_err(|_| HarnessError::SupervisionFailed {
            message: "stderr drain thread panicked".to_string(),
        })?;

    Ok(SupervisedResult {
        program: command.program.clone(),
        termination,
        stdout,
        stderr,
        elapsed,
        killed_by_harness,
    })
}

/// Wait up to `timeout`, then kill + reap. Returns the termination and
/// whether the harness performed the kill.
fn wait_bounded(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(Termination, bool), HarnessError> {
    let start = Instant::now();
    loop {
        match child
            .try_wait()
            .map_err(|error| HarnessError::SupervisionFailed {
                message: format!("failed to poll child: {error}"),
            })? {
            Some(status) => return Ok((termination_of(status), false)),
            None => {
                if start.elapsed() >= timeout {
                    // Deadline expired: kill, then REAP. A kill without a
                    // blocking wait leaves a zombie; the blocking wait
                    // below is what reaps.
                    let _ = child.kill();
                    let status = child
                        .wait()
                        .map_err(|error| HarnessError::SupervisionFailed {
                            message: format!("failed to reap timed-out child: {error}"),
                        })?;
                    let _ = status;
                    return Ok((Termination::Timeout, true));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

/// Map an OS exit status onto [`Termination`]. Unix signals are preserved;
/// anything else abnormal becomes `ExitCode(-1)` so it still classifies as
/// a crash downstream (never as success, never as a harness error).
fn termination_of(status: std::process::ExitStatus) -> Termination {
    if status.success() {
        return Termination::Success;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return Termination::Signal(signal);
        }
    }
    match status.code() {
        Some(code) => Termination::ExitCode(code),
        // Abnormal termination with no code and no signal (Windows kill
        // paths): a crash, not a harness failure.
        None => Termination::ExitCode(-1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Platform-native no-op success: `true` on Unix, `cmd /c exit 0` on
    /// Windows. No shell is involved in the harness itself; the argv below
    /// targets the platform command directly.
    fn success_command() -> SupervisedCommand {
        #[cfg(unix)]
        {
            SupervisedCommand::new(PathBuf::from("/bin/true"), vec![], Duration::from_secs(10))
        }
        #[cfg(windows)]
        {
            SupervisedCommand::new(
                PathBuf::from("cmd"),
                vec!["/c".to_string(), "exit".to_string(), "0".to_string()],
                Duration::from_secs(10),
            )
        }
    }

    #[test]
    fn success_is_reported_with_empty_streams() {
        let result = run_supervised(&success_command()).unwrap();
        assert_eq!(result.termination, Termination::Success);
        assert!(!result.killed_by_harness);
        assert_eq!(result.stdout.total_bytes, 0);
        assert_eq!(result.stderr.total_bytes, 0);
        assert!(!result.stdout.truncated);
    }

    #[test]
    fn missing_binary_is_not_a_crash() {
        let command = SupervisedCommand::new(
            PathBuf::from("/nonexistent/ember-test-binary-xyz"),
            vec![],
            Duration::from_secs(10),
        );
        let error = run_supervised(&command).unwrap_err();
        assert!(matches!(error, HarnessError::BinaryNotFound { .. }));
        assert!(error.to_string().contains("not found"));
    }

    /// Nonzero exit with stderr: the harness reports, it does not fail.
    #[test]
    fn nonzero_exit_captures_stderr() {
        #[cfg(unix)]
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/sh"),
            vec!["-c".to_string(), "echo boom >&2; exit 3".to_string()],
            Duration::from_secs(10),
        );
        #[cfg(windows)]
        let command = SupervisedCommand::new(
            PathBuf::from("cmd"),
            vec!["/c".to_string(), "echo boom 1>&2 & exit 3".to_string()],
            Duration::from_secs(10),
        );
        // NOTE: /bin/sh IS invoked here, but as the supervised *subject*
        // under test (proving we survive odd children), never as a harness
        // mechanism. The harness itself never uses a shell.
        let result = run_supervised(&command).unwrap();
        assert_eq!(result.termination, Termination::ExitCode(3));
        assert!(!result.killed_by_harness);
        assert!(result.stderr.text_lossy().contains("boom"));
    }

    /// Stdout bytes round-trip exactly.
    #[test]
    fn stdout_is_captured_verbatim() {
        #[cfg(unix)]
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/echo"),
            vec!["hello-stdout".to_string()],
            Duration::from_secs(10),
        );
        #[cfg(windows)]
        let command = SupervisedCommand::new(
            PathBuf::from("cmd"),
            vec!["/c".to_string(), "echo hello-stdout".to_string()],
            Duration::from_secs(10),
        );
        let result = run_supervised(&command).unwrap();
        assert_eq!(result.termination, Termination::Success);
        assert!(result.stdout.text_lossy().contains("hello-stdout"));
    }

    /// argv with spaces and metacharacters must reach the child intact —
    /// proof there is no shell joining/quoting path in the harness.
    #[test]
    fn argv_with_spaces_and_metacharacters_is_exact() {
        let payload = "a b; rm -rf / | $(evil) `x` \"quoted\"";
        #[cfg(unix)]
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/echo"),
            vec![payload.to_string()],
            Duration::from_secs(10),
        );
        #[cfg(windows)]
        let command = SupervisedCommand::new(
            PathBuf::from("cmd"),
            vec!["/c".to_string(), "echo".to_string(), payload.to_string()],
            Duration::from_secs(10),
        );
        let result = run_supervised(&command).unwrap();
        assert_eq!(result.termination, Termination::Success);
        // If a shell had interpreted argv, `;`, `|`, `$()` or backticks
        // would have split or vanished the payload.
        assert!(result.stdout.text_lossy().contains(payload));
    }

    /// A hung child is killed, reaped (no zombie), and reported as Timeout.
    #[test]
    fn hang_is_killed_reaped_and_reported() {
        #[cfg(unix)]
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/sleep"),
            vec!["60".to_string()],
            Duration::from_millis(300),
        );
        #[cfg(windows)]
        let command = SupervisedCommand::new(
            PathBuf::from("cmd"),
            vec![
                "/c".to_string(),
                "timeout".to_string(),
                "/t".to_string(),
                "60".to_string(),
                "/nobreak".to_string(),
            ],
            Duration::from_millis(300),
        );
        let start = Instant::now();
        let result = run_supervised(&command).unwrap();
        let wall = start.elapsed();
        assert_eq!(result.termination, Termination::Timeout);
        assert!(result.killed_by_harness);
        // Bounded well past the deadline would mean kill-without-reap or a
        // reap that never returned; generous ceiling keeps CI honest.
        assert!(wall < Duration::from_secs(20));
        assert!(result.elapsed >= Duration::from_millis(300));
    }

    /// Floods are bounded: total recorded, head+tail kept, flagged.
    #[test]
    fn flood_is_bounded_and_flagged() {
        #[cfg(unix)]
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/sh"),
            vec!["-c".to_string(), "yes FLOOD | head -c 3000000".to_string()],
            Duration::from_secs(20),
        );
        #[cfg(windows)]
        let command = SupervisedCommand::new(
            PathBuf::from("powershell"),
            vec![
                "-NoProfile".to_string(),
                "-Command".to_string(),
                "$s='FLOOD '*500000; $s".to_string(),
            ],
            Duration::from_secs(20),
        );
        let result = run_supervised(&command).unwrap();
        assert!(result.stdout.total_bytes > MAX_CAPTURE_BYTES_PER_STREAM as u64);
        assert!(result.stdout.truncated);
        assert!(result.stdout.bytes.len() <= TRUNCATED_HEAD_BYTES + TRUNCATED_TAIL_BYTES + 1);
        assert!(result.stdout.text_lossy().contains("FLOOD"));
    }

    /// Sequential runs do not leak state (fds, threads, zombies).
    #[test]
    fn sequential_runs_are_independent() {
        for _ in 0..8 {
            let result = run_supervised(&success_command()).unwrap();
            assert_eq!(result.termination, Termination::Success);
        }
    }

    /// Parallel runs do not interfere (each pipe pair is per-child).
    #[test]
    fn parallel_runs_do_not_interfere() {
        let handles: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(|| run_supervised(&success_command()).unwrap()))
            .collect();
        for handle in handles {
            let result = handle.join().expect("worker panicked");
            assert_eq!(result.termination, Termination::Success);
        }
    }

    /// Unix signal death maps to Signal, not to a harness error.
    #[cfg(unix)]
    #[test]
    fn signal_death_is_classified() {
        let command = SupervisedCommand::new(
            PathBuf::from("/bin/sh"),
            vec!["-c".to_string(), "kill -SEGV $$".to_string()],
            Duration::from_secs(10),
        );
        let result = run_supervised(&command).unwrap();
        assert!(matches!(result.termination, Termination::Signal(_)));
        assert!(!result.killed_by_harness);
    }
}

//! Research tracing (Tracks J/K/P/Q): structured events, explicit order,
//! crash-tolerant JSONL persistence.
//!
//! A trace is the PRIMARY deliverable of an agent run, not debug logging.
//! Every event line carries
//!
//! ```json
//! {"schema":"ember.agent.trace.v1","run_id":"...","seq":12,
//!  "event_type":"tool_execution_finished","t_ms":841.3,
//!  "ts_epoch_ms":1776940000000,"step":"tool-0","phase":"execute",
//!  "data":{ ... }}
//! ```
//!
//! Ordering is the monotonically increasing `seq`, never wall-clock.
//! Lines are flushed after every append, so a crashed run leaves a
//! readable JSONL prefix (Track Q). Privacy controls (Track "privacy")
//! gate prompt/text/payload capture explicitly; nothing records full
//! content silently.
//!
//! Policy for write failures (Track X): opening the trace file fails the
//! run up front; a failure DURING the run is counted (`write_errors`)
//! and degrades to memory retention only — inference continues. This is
//! deliberate: losing a trace must not corrupt committed conversation
//! state mid-run.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::ids::short_hash;

/// Schema tag on every line.
pub const TRACE_SCHEMA: &str = "ember.agent.trace.v1";

// -- event type vocabulary (J1; snake_case) --------------------------------
pub mod ev {
    pub const RUN_STARTED: &str = "run_started";
    pub const PROVENANCE: &str = "provenance";
    pub const MODEL_CALL_STARTED: &str = "model_call_started";
    pub const MODEL_CALL_FINISHED: &str = "model_call_finished";
    pub const ASSISTANT_ACTION_PARSED: &str = "assistant_action_parsed";
    pub const TOOL_CALL_VALIDATED: &str = "tool_call_validated";
    pub const TOOL_CALL_REJECTED: &str = "tool_call_rejected";
    pub const TOOL_EXECUTION_STARTED: &str = "tool_execution_started";
    pub const TOOL_EXECUTION_FINISHED: &str = "tool_execution_finished";
    pub const TOOL_RESULT_COMMITTED: &str = "tool_result_committed";
    pub const TOOL_RESULT_UNCOMMITTED: &str = "tool_result_uncommitted";
    pub const MESSAGE_COMMITTED: &str = "message_committed";
    pub const SESSION_STATE_CHANGED: &str = "session_state_changed";
    pub const ARTIFACT_WRITTEN: &str = "artifact_written";
    pub const RUN_COMPLETED: &str = "run_completed";
    pub const RUN_FAILED: &str = "run_failed";
    pub const RUN_CANCELLED: &str = "run_cancelled";
    /// Clean stop because a hard limit fired.
    pub const RUN_TERMINATED: &str = "run_terminated";
    pub const GENERATION_TOKEN: &str = "generation_token";
}

/// How tool result payloads are recorded (privacy / volume control).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTraceMode {
    /// Complete payload text.
    Full,
    /// First N bytes plus total length.
    Summary(usize),
    /// SHA-256 digest only.
    Hash,
}

impl Default for ToolTraceMode {
    fn default() -> Self {
        // Documented default: summaries. Ember is a research instrument,
        // but unbounded payloads inside JSONL are a footgun; 2048 bytes
        // keeps results auditable while bounding file growth.
        ToolTraceMode::Summary(2048)
    }
}

/// Trace behavior knobs. Defaults are documented on each field.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// JSONL destination. `None` retains events in memory only (tests).
    pub output_path: Option<PathBuf>,
    /// Capture user/system prompts verbatim. Default true (research tool);
    /// when false only lengths + hashes are recorded.
    pub trace_prompts: bool,
    /// Capture generated assistant text verbatim. Default true.
    pub trace_generated_text: bool,
    /// Tool payload capture mode. Default: [`ToolTraceMode::Summary`] with a
    /// 2048-byte excerpt.
    pub tool_results: ToolTraceMode,
    /// Per-token trace events (id + fragment). Default false: this is the
    /// one knob that can produce very large files on long runs.
    pub token_events: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            output_path: None,
            trace_prompts: true,
            trace_generated_text: true,
            tool_results: ToolTraceMode::default(),
            token_events: false,
        }
    }
}

/// Append-only, crash-tolerant trace writer.
///
/// Single-owner by design (the agent session drives it sequentially);
/// no interior locking because runs are single-threaded like the rest
/// of ember's runtime.
pub struct TraceRecorder {
    config: TraceConfig,
    run_id: String,
    seq: u64,
    start: Instant,
    start_epoch_ms: u128,
    writer: Option<std::io::BufWriter<std::fs::File>>,
    memory: Vec<serde_json::Value>,
    write_errors: u64,
}

impl TraceRecorder {
    /// Open a recorder for `run_id`. Creating/append-opening the output
    /// file happens HERE and fails loudly; after that, writes degrade
    /// gracefully instead of killing the run.
    pub fn open(config: TraceConfig, run_id: &str) -> Result<Self> {
        let writer = match &config.output_path {
            Some(path) => {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("failed to create trace dir {}", parent.display())
                    })?;
                }
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("failed to open trace file {}", path.display()))?;
                Some(std::io::BufWriter::new(file))
            }
            None => None,
        };
        Ok(Self {
            start_epoch_ms: epoch_ms(),
            config,
            run_id: run_id.to_string(),
            seq: 0,
            start: Instant::now(),
            writer,
            memory: Vec::new(),
            write_errors: 0,
        })
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Re-key the recorder to the owning run's id (the session assigns
    /// fresh run ids per run; every emitted line must carry it).
    pub fn set_run_id(&mut self, run_id: &str) {
        self.run_id = run_id.to_string();
    }

    pub fn config(&self) -> &TraceConfig {
        &self.config
    }

    /// Elapsed milliseconds since the run started (monotonic clock).
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }

    pub fn events(&self) -> &[serde_json::Value] {
        &self.memory
    }

    pub fn write_errors(&self) -> u64 {
        self.write_errors
    }

    /// Emit one event. `step` is the owning step id ("user", "model-0",
    /// "tool-1", "finalize"); `phase` optionally narrows it
    /// ("prefill"/"decode"/"validate"/"execute").
    pub fn emit(
        &mut self,
        event_type: &str,
        step: &str,
        phase: Option<&str>,
        mut data: serde_json::Value,
    ) {
        if let Some(obj) = data.as_object_mut()
            && obj.is_empty()
        {
            data = serde_json::Value::Null;
        }
        let event = serde_json::json!({
            "schema": TRACE_SCHEMA,
            "run_id": self.run_id,
            "seq": self.seq,
            "event_type": event_type,
            "t_ms": (self.start.elapsed().as_secs_f64() * 1000.0),
            "ts_epoch_ms": self.start_epoch_ms.saturating_add(
                self.start.elapsed().as_millis()),
            "step": step,
            "phase": phase,
            "data": data,
        });
        self.seq += 1;
        self.memory.push(event.clone());
        if let Some(w) = self.writer.as_mut() {
            let line = match serde_json::to_string(&event) {
                Ok(l) => l,
                Err(_) => {
                    self.write_errors += 1;
                    return;
                }
            };
            if w.write_all(line.as_bytes()).is_err()
                || w.write_all(b"\n").is_err()
                || w.flush().is_err()
            {
                self.write_errors += 1;
            }
        }
    }

    // -- privacy-aware content emitters -----------------------------------

    /// Record prompt-like content honoring `trace_prompts`.
    pub fn emit_prompt(&mut self, event_type: &str, step: &str, role: &str, content: &str) {
        let data = if self.config.trace_prompts {
            serde_json::json!({ "role": role, "content": content })
        } else {
            serde_json::json!({
                "role": role,
                "content_omitted": true,
                "bytes": content.len(),
                "sha256": crate::extraction::sha256_bytes(content.as_bytes()),
            })
        };
        self.emit(event_type, step, None, data);
    }

    /// Record a tool payload honoring `tool_results`.
    pub fn tool_payload_data(&self, payload: &str) -> serde_json::Value {
        match self.config.tool_results {
            ToolTraceMode::Full => serde_json::json!({
                "payload": payload,
                "payload_bytes": payload.len(),
            }),
            ToolTraceMode::Summary(n) => {
                let excerpt: String = payload.chars().take(n).collect();
                serde_json::json!({
                    "payload_excerpt": excerpt,
                    "payload_bytes": payload.len(),
                    "truncated": payload.len() > excerpt.len(),
                })
            }
            ToolTraceMode::Hash => serde_json::json!({
                "payload_sha256": crate::extraction::sha256_bytes(payload.as_bytes()),
                "payload_bytes": payload.len(),
            }),
        }
    }

    /// Optional per-token event (off unless configured).
    pub fn emit_token(&mut self, step: &str, id: u32, piece: &str) {
        if self.config.token_events {
            self.emit(
                ev::GENERATION_TOKEN,
                step,
                Some("decode"),
                serde_json::json!({ "token_id": id, "piece": piece }),
            );
        }
    }
}

/// Flush-on-drop safety net (recorder already flushes per event; this
/// covers buffered OS state on abnormal exits that still run destructors).
impl Drop for TraceRecorder {
    fn drop(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.flush();
        }
    }
}

fn epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Content identity helper reused by sessions (hash of committed text).
pub fn content_digest(text: &str) -> String {
    short_hash(text.as_bytes())
}

/// Parse a trace JSONL file tolerantly (Track Q/S): complete lines are
/// returned as parsed values; a torn trailing line (crash mid-write) is
/// reported separately instead of failing the whole file.
pub fn parse_trace_file(path: &Path) -> Result<(Vec<serde_json::Value>, Vec<String>)> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read trace {}", path.display()))?;
    let mut events = Vec::new();
    let mut skipped = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => events.push(v),
            Err(e) => skipped.push(format!("line {}: {e}", i + 1)),
        }
    }
    Ok((events, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ember-agent-trace-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn emits_monotonic_sequence_and_flushes_per_event() {
        let path = tmp_path("seq.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut rec = TraceRecorder::open(
            TraceConfig {
                output_path: Some(path.clone()),
                ..Default::default()
            },
            "run-t",
        )
        .unwrap();
        for i in 0..25 {
            rec.emit(
                ev::MESSAGE_COMMITTED,
                "user",
                None,
                serde_json::json!({ "i": i }),
            );
        }
        drop(rec);
        let (events, skipped) = parse_trace_file(&path).unwrap();
        assert!(skipped.is_empty());
        assert_eq!(events.len(), 25);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e["seq"], i as u64);
            assert_eq!(e["run_id"], "run-t");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn torn_trailing_line_still_parses_prefix() {
        let path = tmp_path("torn.jsonl");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let good1 = r#"{"schema":"x","seq":0,"event_type":"a"}"#;
        let good2 = r#"{"schema":"x","seq":1,"event_type":"b"}"#;
        let mut raw = format!("{good1}\n{good2}\n{}", r#"{"schema":"x","seq":2,"ev"#);
        // simulate a crash: no trailing newline, truncated JSON
        raw.truncate(raw.len());
        std::fs::write(&path, raw).unwrap();
        let (events, skipped) = parse_trace_file(&path).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(skipped.len(), 1);
        assert_eq!(events[1]["seq"], 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn privacy_modes_redact_content() {
        let mut rec = TraceRecorder::open(
            TraceConfig {
                output_path: None,
                trace_prompts: false,
                trace_generated_text: false,
                tool_results: ToolTraceMode::Hash,
                ..Default::default()
            },
            "run-p",
        )
        .unwrap();
        rec.emit_prompt("m", "user", "user", "secret prompt");
        rec.tool_payload_data("secret payload");
        let data = &rec.events()[0]["data"];
        assert_eq!(data["content_omitted"], true);
        assert!(data.get("content").is_none());
        assert_eq!(
            data["sha256"],
            crate::extraction::sha256_bytes(b"secret prompt")
        );
    }

    #[test]
    fn summary_mode_excerpts_and_reports_truncation() {
        let rec = TraceRecorder::open(
            TraceConfig {
                tool_results: ToolTraceMode::Summary(4),
                ..Default::default()
            },
            "run-s",
        )
        .unwrap();
        let v = rec.tool_payload_data("abcdefghij");
        assert_eq!(v["payload_excerpt"], "abcd");
        assert_eq!(v["payload_bytes"], 10);
        assert_eq!(v["truncated"], true);
    }

    #[test]
    fn token_events_default_off() {
        let mut rec = TraceRecorder::open(TraceConfig::default(), "run-k").unwrap();
        rec.emit_token("model-0", 7, "x");
        assert!(rec
            .events()
            .iter()
            .all(|e| e["event_type"] != ev::GENERATION_TOKEN));
    }
}

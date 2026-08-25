//! Trace inspection (Track R): tolerant parsing + compact timeline
//! summaries over JSONL traces. Shared by `ember trace inspect` and the
//! trace-validation tests (Track S).

use std::path::Path;

use anyhow::{Context, Result};

use super::trace::{parse_trace_file, TRACE_SCHEMA};

/// Aggregates of one run's trace, derived only from ordered events.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TraceSummary {
    pub run_id: String,
    pub protocol_id: Option<String>,
    pub model_path: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub status: Option<String>,
    pub final_text: Option<String>,
    pub error: Option<String>,
    pub model_steps: usize,
    pub tool_calls: usize,
    pub rejected_calls: usize,
    pub total_model_ms: f64,
    pub total_tool_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prompt_tokens_committed: u64,
    pub output_tokens: u64,
    pub tok_per_s: Option<f64>,
    pub artifacts: Vec<String>,
    pub event_counts: std::collections::BTreeMap<String, usize>,
    pub first_seq: Option<u64>,
    pub last_seq: Option<u64>,
    pub first_t_ms: f64,
    pub last_t_ms: f64,
}

/// One human-readable timeline row.
#[derive(Debug, Clone)]
pub struct TimelineRow {
    pub t_ms: f64,
    pub label: String,
}

/// Parse + summarize a trace file. Tolerates torn trailing lines.
pub fn inspect_file(path: &Path) -> Result<(TraceSummary, Vec<TimelineRow>, Vec<String>)> {
    let (events, skipped) = parse_trace_file(path)
        .with_context(|| format!("failed to read trace {}", path.display()))?;
    Ok((summarize(&events), timeline(&events), skipped))
}

fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Derive a summary from ordered events (no wall-clock reliance).
pub fn summarize(events: &[serde_json::Value]) -> TraceSummary {
    let mut s = TraceSummary::default();
    let mut decode_evals = 0u64;
    let mut tok_per_s_sum = 0.0;
    let mut tok_per_s_n = 0usize;
    for e in events {
        if e["schema"] != TRACE_SCHEMA {
            continue;
        }
        let et = e["event_type"].as_str().unwrap_or("");
        let data = &e["data"];
        *s.event_counts.entry(et.to_string()).or_insert(0) += 1;
        s.last_seq = e["seq"].as_u64();
        if s.first_seq.is_none() {
            s.first_seq = e["seq"].as_u64();
            s.first_t_ms = f(&e["t_ms"]);
        }
        s.last_t_ms = f(&e["t_ms"]);
        if s.run_id.is_empty() {
            s.run_id = e["run_id"].as_str().unwrap_or("").to_string();
        }
        match et {
            "provenance" => {
                s.protocol_id = data["protocol_id"].as_str().map(String::from);
                s.model_path = data["model"]["model_path"].as_str().map(String::from);
                s.architecture = data["model"]["architecture"].as_str().map(String::from);
                s.quantization = data["model"]["quantization"].as_str().map(String::from);
            }
            "model_call_finished" => {
                s.model_steps += 1;
                s.total_model_ms += f(&data["wall_ms"]);
                s.prefill_ms += f(&data["prefill_ms"]);
                s.decode_ms += f(&data["decode_ms"]);
                s.output_tokens += data["output_tokens_committed"].as_u64().unwrap_or(0);
                decode_evals += data["decode_evaluations"].as_u64().unwrap_or(0);
                if let Some(t) = data["tok_per_s"].as_f64()
                    && t > 0.0
                {
                    tok_per_s_sum += t;
                    tok_per_s_n += 1;
                }
            }
            "tool_execution_finished" => {
                s.tool_calls += 1;
                s.total_tool_ms += f(&data["duration_ms"]);
            }
            "tool_call_rejected" => {
                s.rejected_calls += 1;
            }
            "run_completed" => {
                s.status = Some("completed".to_string());
                s.final_text = data["final_text"].as_str().map(String::from);
            }
            "run_cancelled" => {
                s.status = Some("cancelled".to_string());
            }
            "run_terminated" => {
                s.status = Some(format!(
                    "limit:{}",
                    data["limit_hit"].as_str().unwrap_or("?")
                ));
            }
            "run_failed" => {
                s.status = Some("failed".to_string());
                s.error = data["error"].as_str().map(String::from);
            }
            "artifact_written" => {
                if let Some(id) = data["artifact_id"].as_str() {
                    s.artifacts.push(id.to_string());
                }
            }
            _ => {}
        }
    }
    // prompt tokens come from message/session events; approximate from
    // spans if present
    for e in events {
        let et = e["event_type"].as_str().unwrap_or("");
        if (et == "message_committed" || et == "session_state_changed")
            && let Some(span) = e["data"]["span"].as_array()
            && let (Some(a), Some(b)) = (
                span.first().and_then(|v| v.as_u64()),
                span.get(1).and_then(|v| v.as_u64()),
            )
        {
            s.prompt_tokens_committed += b.saturating_sub(a);
        }
    }
    let _ = decode_evals;
    if tok_per_s_n > 0 {
        s.tok_per_s = Some(tok_per_s_sum / tok_per_s_n as f64);
    }
    s
}

/// Compact timeline rows (the CLI prints these one per line).
pub fn timeline(events: &[serde_json::Value]) -> Vec<TimelineRow> {
    let mut rows = Vec::new();
    for e in events {
        let et = e["event_type"].as_str().unwrap_or("");
        let data = &e["data"];
        let step = e["step"].as_str().unwrap_or("");
        let t = f(&e["t_ms"]);
        match et {
            "run_started" => rows.push(TimelineRow {
                t_ms: t,
                label: "run start".to_string(),
            }),
            "provenance" => {}
            "model_call_started" => rows.push(TimelineRow {
                t_ms: t,
                label: format!("{step} start"),
            }),
            "assistant_action_parsed" => match data["action"].as_str().unwrap_or("") {
                "final_text" => {}
                "malformed_tool_call" => rows.push(TimelineRow {
                    t_ms: t,
                    label: format!("{step} MALFORMED tool call"),
                }),
                _ => rows.push(TimelineRow {
                    t_ms: t,
                    label: format!(
                        "{step} requested tool `{}`",
                        e["data"]["name"].as_str().unwrap_or("?")
                    ),
                }),
            },
            "tool_call_rejected" => rows.push(TimelineRow {
                t_ms: t,
                label: format!(
                    "{step} REJECTED {} [{}]",
                    data["tool"].as_str().unwrap_or("?"),
                    data["kind"].as_str().unwrap_or("?")
                ),
            }),
            "tool_execution_started" => rows.push(TimelineRow {
                t_ms: t,
                label: format!(
                    "{} `{}` execute",
                    step,
                    data["tool"].as_str().unwrap_or("?")
                ),
            }),
            "tool_execution_finished" => rows.push(TimelineRow {
                t_ms: t,
                label: format!(
                    "{} `{}` {} ({:.1}ms)",
                    step,
                    data["tool"].as_str().unwrap_or("?"),
                    if data["ok"] == true { "ok" } else { "failed" },
                    f(&data["duration_ms"])
                ),
            }),
            "tool_result_uncommitted" => rows.push(TimelineRow {
                t_ms: t,
                label: format!("{step} result UNCOMMITTED (cancelled)"),
            }),
            "run_completed" => rows.push(TimelineRow {
                t_ms: t,
                label: "final answer".to_string(),
            }),
            "run_failed" => rows.push(TimelineRow {
                t_ms: t,
                label: "RUN FAILED".to_string(),
            }),
            "run_cancelled" => rows.push(TimelineRow {
                t_ms: t,
                label: "run cancelled".to_string(),
            }),
            _ => {}
        }
    }
    rows
}

/// Structural validation shared by Track S tests: RunStarted first,
/// terminal event last, sequence monotonic, tool starts matched by
/// finishes, model spans balanced. Returns a list of violations (empty =
/// valid).
pub fn validate_trace_invariants(events: &[serde_json::Value]) -> Vec<String> {
    let mut problems = Vec::new();
    let owned: Vec<&serde_json::Value> = events.iter().collect();
    if owned.is_empty() {
        return vec!["empty trace".to_string()];
    }
    let first_type = owned[0]["event_type"].as_str().unwrap_or("");
    if first_type != super::trace::ev::RUN_STARTED {
        problems.push(format!("first event is {first_type}, expected run_started"));
    }
    let last_type = owned[owned.len() - 1]["event_type"].as_str().unwrap_or("");
    if !matches!(
        last_type,
        "run_completed" | "run_failed" | "run_cancelled" | "run_terminated"
    ) {
        problems.push(format!(
            "last event is {last_type}, expected a terminal event"
        ));
    }
    let mut prev_seq: Option<u64> = None;
    for e in &owned {
        let seq = e["seq"].as_u64().unwrap_or(u64::MAX);
        if let Some(p) = prev_seq
            && seq <= p
        {
            problems.push(format!("sequence not monotonic at {seq} after {p}"));
        }
        prev_seq = Some(seq);
    }
    let starts = owned
        .iter()
        .filter(|e| e["event_type"] == "tool_execution_started")
        .count();
    let finishes = owned
        .iter()
        .filter(|e| e["event_type"] == "tool_execution_finished")
        .count();
    if starts != finishes {
        problems.push(format!(
            "tool starts ({starts}) != tool finishes ({finishes})"
        ));
    }
    let m_starts = owned
        .iter()
        .filter(|e| e["event_type"] == "model_call_started")
        .count();
    let m_finishes = owned
        .iter()
        .filter(|e| e["event_type"] == "model_call_finished")
        .count();
    // cancelled generations legitimately have a start without a finish
    if m_starts != m_finishes && last_type == "run_completed" {
        problems.push(format!(
            "model spans unbalanced: starts={m_starts} finished={m_finishes}"
        ));
    }
    problems
}

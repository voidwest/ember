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

// ---------------------------------------------------------------------------
// Phase 2: trace diff, deterministic replay, HTML report
// ---------------------------------------------------------------------------

use crate::agent::schema::{canonical_json, ValidatedArguments};
use crate::agent::tool::ToolRegistry;

/// Field-level differences between two run summaries.
#[derive(Debug, Default)]
pub struct TraceDiff {
    pub differences: Vec<String>,
}

impl TraceDiff {
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }
}

/// Compare two runs' summaries plus their event-type skeletons. The
/// skeleton check catches structural divergence (a tool call appearing in
/// one run and not the other) even when totals happen to match.
pub fn diff(
    a: &[serde_json::Value],
    b: &[serde_json::Value],
) -> (TraceSummary, TraceSummary, TraceDiff) {
    let sa = summarize(a);
    let sb = summarize(b);
    let mut diff = TraceDiff::default();
    let field = |diff: &mut TraceDiff, name: &str, va: String, vb: String| {
        if va != vb {
            diff.differences.push(format!("{name}: {va} vs {vb}"));
        }
    };
    field(
        &mut diff,
        "status",
        sa.status.clone().unwrap_or_default(),
        sb.status.clone().unwrap_or_default(),
    );
    field(
        &mut diff,
        "model_steps",
        sa.model_steps.to_string(),
        sb.model_steps.to_string(),
    );
    field(
        &mut diff,
        "tool_calls",
        sa.tool_calls.to_string(),
        sb.tool_calls.to_string(),
    );
    field(
        &mut diff,
        "rejected_calls",
        sa.rejected_calls.to_string(),
        sb.rejected_calls.to_string(),
    );
    field(
        &mut diff,
        "artifacts",
        sa.artifacts.len().to_string(),
        sb.artifacts.len().to_string(),
    );
    field(
        &mut diff,
        "output_tokens",
        sa.output_tokens.to_string(),
        sb.output_tokens.to_string(),
    );
    field(
        &mut diff,
        "final_text",
        sha_digest(sa.final_text.as_deref().unwrap_or("")),
        sha_digest(sb.final_text.as_deref().unwrap_or("")),
    );

    let skeleton_a: Vec<&str> = a
        .iter()
        .map(|e| e["event_type"].as_str().unwrap_or(""))
        .filter(|t| {
            !matches!(
                *t,
                "provenance" | "message_committed" | "session_state_changed"
            )
        })
        .collect();
    let skeleton_b: Vec<&str> = b
        .iter()
        .map(|e| e["event_type"].as_str().unwrap_or(""))
        .filter(|t| {
            !matches!(
                *t,
                "provenance" | "message_committed" | "session_state_changed"
            )
        })
        .collect();
    if skeleton_a != skeleton_b
        && let Some(i) = skeleton_a
            .iter()
            .zip(skeleton_b.iter())
            .position(|(x, y)| x != y)
    {
        diff.differences.push(format!(
            "event skeleton diverges at position {i}: {} vs {} (lengths {}/{} )",
            skeleton_a[i],
            skeleton_b.get(i).copied().unwrap_or("<end>"),
            skeleton_a.len(),
            skeleton_b.len()
        ));
    }
    (sa, sb, diff)
}

use crate::extraction::sha256_bytes;

fn sha_digest(text: &str) -> String {
    if text.is_empty() {
        "<none>".to_string()
    } else {
        crate::extraction::sha256_bytes(text.as_bytes())[..12].to_string()
    }
}

/// Result of replaying one recorded call.
#[derive(Debug)]
pub struct ReplayOutcome {
    pub seq: u64,
    pub tool: String,
    pub ok: bool,
    pub matches: Option<bool>,
    /// Empty digest when the recorded event predates replay digests or
    /// the call failed.
    pub recorded_replay_sha256: String,
    pub computed_replay_sha256: String,
}

#[derive(Debug, Default)]
pub struct ReplayReport {
    pub outcomes: Vec<ReplayOutcome>,
}

impl ReplayReport {
    pub fn all_matched(&self) -> bool {
        self.outcomes.iter().all(|o| o.matches.unwrap_or(false))
    }

    pub fn skipped(&self) -> usize {
        self.outcomes.iter().filter(|o| o.matches.is_none()).count()
    }
}

/// Re-execute every successful recorded tool call against `registry` and
/// verify the stable payload digest. Requires deterministic tools; the
/// recorded `replay_sha256` deliberately excludes volatile identity
/// fields (`path`, `artifact_id`) so artifact-producing tools verify too.
pub fn replay(
    events: &[serde_json::Value],
    registry: &ToolRegistry,
) -> anyhow::Result<ReplayReport> {
    use crate::agent::tool::{execute_tool, ToolInvocation};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    let mut report = ReplayReport::default();
    // validated arguments strictly precede their execution event
    let mut pending_args: std::collections::VecDeque<serde_json::Value> =
        std::collections::VecDeque::new();
    let dir = std::env::temp_dir().join(format!(
        "ember-replay-{}-{}",
        std::process::id(),
        std::time::Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&dir)?;
    let artifacts = Arc::new(Mutex::new(crate::agent::ArtifactStore::open(
        &dir, "replay",
    )?));
    let cancel = crate::agent::CancelFlag::new();

    for e in events {
        let et = e["event_type"].as_str().unwrap_or("");
        let step = e["step"].as_str().unwrap_or("");
        match et {
            "tool_call_validated" => {
                if let Some(args) = e["data"]["arguments"].as_object() {
                    pending_args.push_back(e["data"]["arguments"].clone());
                    let _ = args;
                }
            }
            "tool_execution_finished" => {
                let recorded = pending_args.pop_front();
                let tool_name = e["data"]["tool"].as_str().unwrap_or("").to_string();
                let ok = e["data"]["ok"] == true;
                let expected = e["data"]["replay_sha256"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();

                if !ok || expected.is_empty() || recorded.is_none() {
                    report.outcomes.push(ReplayOutcome {
                        seq: e["seq"].as_u64().unwrap_or(0),
                        tool: tool_name,
                        ok,
                        matches: None,
                        recorded_replay_sha256: expected,
                        computed_replay_sha256: String::new(),
                    });
                    continue;
                }
                let Some(tool) = registry.get(&tool_name) else {
                    anyhow::bail!("registry lacks recorded tool `{tool_name}`");
                };
                let schema = tool.schema();
                let args_text = canonical_json(&recorded.unwrap());
                let validated = ValidatedArguments::parse(&schema, &args_text)?;
                let invocation = ToolInvocation {
                    tool: Arc::clone(tool),
                    args: validated,
                    run_id: "replay".to_string(),
                    step_id: step.to_string(),
                    call_seq: 0,
                    deadline: Instant::now() + Duration::from_secs(60),
                    cancel: cancel.clone(),
                    artifacts: Arc::clone(&artifacts),
                };
                let outcome = execute_tool(&invocation);
                let (ok_now, stable_now) = match outcome {
                    Ok(out) => {
                        let mut value: serde_json::Value = serde_json::from_str(
                            &out.payload.serialize_compact(),
                        )
                        .unwrap_or(serde_json::Value::String(out.payload.serialize_compact()));
                        if let Some(obj) = value.as_object_mut() {
                            obj.remove("path");
                            obj.remove("artifact_id");
                        }
                        (true, sha256_bytes(canonical_json(&value).as_bytes()))
                    }
                    Err(failure) => (false, failure.message),
                };
                report.outcomes.push(ReplayOutcome {
                    seq: e["seq"].as_u64().unwrap_or(0),
                    tool: tool_name,
                    ok: ok && ok_now,
                    matches: Some(ok_now && stable_now == expected),
                    recorded_replay_sha256: expected,
                    computed_replay_sha256: stable_now,
                });
            }
            _ => {}
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    Ok(report)
}

/// Self-contained HTML report (inline CSS, no JS, no external assets).
pub fn render_html(events: &[serde_json::Value], summary: &TraceSummary) -> String {
    fn esc(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }
    let mut body = String::new();
    body.push_str("<!doctype html>\n<html><head><meta charset=\"utf-8\">\n<title>ember trace ");
    body.push_str(&esc(&summary.run_id));
    body.push_str("</title>\n<style>\nbody{font-family:ui-monospace,Menlo,Consolas,monospace;margin:2rem;background:#111;color:#ddd}\n");
    body.push_str(".card{border:1px solid #333;border-radius:6px;padding:.8rem 1rem;margin-bottom:1rem;background:#181818}\n");
    body.push_str("h1{font-size:1.1rem}h2{font-size:.9rem;color:#9ab}\n");
    body.push_str("table{border-collapse:collapse;width:100%;font-size:.78rem}\ntd,th{border-bottom:1px solid #262626;padding:.25rem .5rem;text-align:left;white-space:pre-wrap}\nth{color:#9ab}\n");
    body.push_str(".bar{height:10px;border-radius:5px;display:inline-block;vertical-align:middle}\n.model{background:#4a7dbd}.tool{background:#57a05a}.gap{background:#333}\n.ok{color:#8c8}.err{color:#c77}\n");
    body.push_str("</style></head><body>\n");

    body.push_str("<div class=\"card\"><h1>run ");
    body.push_str(&esc(&summary.run_id));
    body.push_str("</h1><p>");
    body.push_str(&format!(
        "{} ({:?}, {}) &middot; protocol {}<br>status <span class=\"{}\">{}</span> &middot; steps {} &middot; tools {} &middot; rejected {} &middot; artifacts {}<br>",
        esc(summary.model_path.as_deref().unwrap_or("?")),
        summary.quantization.as_deref().unwrap_or("?"),
        esc(summary.architecture.as_deref().unwrap_or("?")),
        esc(summary.protocol_id.as_deref().unwrap_or("?")),
        if summary.status.as_deref() == Some("completed") { "ok" } else { "err" },
        esc(summary.status.as_deref().unwrap_or("(incomplete)")),
        summary.model_steps,
        summary.tool_calls,
        summary.rejected_calls,
        summary.artifacts.len(),
    ));
    body.push_str(&format!(
        "model {:.0}ms (prefill {:.0} / decode {:.0}) &middot; tools {:.1}ms &middot; tokens out {}</p></div>",
        summary.total_model_ms, summary.prefill_ms, summary.decode_ms, summary.total_tool_ms, summary.output_tokens,
    ));

    // timeline: contiguous spans from model/tool start->finish events
    body.push_str("<div class=\"card\"><h2>timeline</h2>\n<table><tr><th>step</th><th>kind</th><th>span</th><th>ms</th><th width=\"40%\"></th></tr>\n");
    let total_ms = summary.last_t_ms.max(1.0);
    let starts: Vec<(String, String, f64)> = events
        .iter()
        .filter_map(|e| {
            let t = e["t_ms"].as_f64()?;
            match e["event_type"].as_str()? {
                "model_call_started" => {
                    Some((e["step"].as_str()?.to_string(), "model".to_string(), t))
                }
                "tool_execution_started" => {
                    Some((e["step"].as_str()?.to_string(), "tool".to_string(), t))
                }
                _ => None,
            }
        })
        .collect();
    for (i, (step, kind, t0)) in starts.iter().enumerate() {
        let end = starts
            .get(i + 1)
            .map(|(_, _, t)| *t)
            .unwrap_or(summary.last_t_ms);
        let dur = (end - t0).max(0.0);
        let pct = (dur / total_ms * 100.0).clamp(0.15, 100.0);
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.1} - {:.1}</td><td>{dur:.1}</td><td>\
             <span class=\"bar {kind}\" style=\"width:{pct:.2}%\"></span></td></tr>\n",
            esc(step),
            kind,
            t0,
            end
        ));
    }
    body.push_str("</table></div>\n");

    // artifacts
    if !summary.artifacts.is_empty() {
        body.push_str("<div class=\"card\"><h2>artifacts</h2><ul>");
        for id in &summary.artifacts {
            body.push_str(&format!("<li>{}</li>", esc(id)));
        }
        body.push_str("</ul></div>");
    }

    // full event table
    body.push_str("<div class=\"card\"><h2>events</h2><table><tr><th>seq</th><th>t_ms</th><th>type</th><th>step</th><th>data</th></tr>\n");
    for e in events {
        body.push_str(&format!(
            "<tr><td>{}</td><td>{:.1}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            e["seq"].as_u64().unwrap_or(0),
            e["t_ms"].as_f64().unwrap_or(0.0),
            esc(e["event_type"].as_str().unwrap_or("")),
            esc(e["step"].as_str().unwrap_or("")),
            esc(&canonical_json(&e["data"])),
        ));
    }
    body.push_str("</table></div>\n</body></html>\n");
    body
}

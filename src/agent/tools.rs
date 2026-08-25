//! Deterministic built-in tools (Tracks I/H).
//!
//! Everything here is local, reproducible, and side-effect-bounded:
//! arithmetic over f64, fixture lookups over an injected map, artifact
//! writes through the run's artifact store (`crate::agent::ArtifactStore`),
//! and read-only text-file
//! access strictly rooted at a configured directory (traversal fails
//! closed). There is deliberately NO shell/network/delete tool in Phase 1
//! — this phase proves orchestration, not general-purpose execution.
//!
//! Risk model: every schema carries a [`ToolEffect`]; built-ins are
//! `ReadOnly` or `LocalWrite` only.

use std::collections::BTreeMap;
use std::io::Read;
use std::time::{Duration, Instant};

use super::schema::{JsonType, ParamSchema, ToolEffect, ToolSchema, ValidatedArguments};
use super::tool::{
    Tool, ToolContext, ToolFailure, ToolFailureKind, ToolOutcome, ToolOutput, ToolPayload,
};

fn fail(kind: ToolFailureKind, message: impl Into<String>) -> ToolFailure {
    ToolFailure {
        kind,
        message: message.into(),
    }
}

// ---------------------------------------------------------------------------
// calculator
// ---------------------------------------------------------------------------

/// Deterministic arithmetic on f64 inputs. `divide` rejects zero divisors
/// as a structured tool failure (fed back to the model, not a runtime
/// error).
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "calculate",
            "Evaluate a two-operand arithmetic operation deterministically.",
        )
        .param(
            ParamSchema::new("operation", JsonType::String)
                .required()
                .one_of(&["add", "subtract", "multiply", "divide", "power"])
                .describe("the operation to perform"),
        )
        .param(ParamSchema::new("a", JsonType::Number).required())
        .param(ParamSchema::new("b", JsonType::Number).required())
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let op = args.get("operation").and_then(|v| v.as_str()).unwrap_or("");
        let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        let result = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(fail(
                        ToolFailureKind::Execution,
                        "division by zero is undefined",
                    ));
                }
                a / b
            }
            "power" => a.powf(b),
            _ => return Err(fail(ToolFailureKind::Execution, "unknown operation")),
        };
        if !result.is_finite() {
            return Err(fail(
                ToolFailureKind::Execution,
                format!("operation produced a non-finite result ({result})"),
            ));
        }
        Ok(ToolOutput::json(serde_json::json!({
            "operation": op,
            "a": a,
            "b": b,
            "result": result,
        })))
    }
}

// ---------------------------------------------------------------------------
// lookup fixture
// ---------------------------------------------------------------------------

/// `lookup(key) -> value` over an injected map. The canonical way to force
/// real tool usage in tests and demos without network dependencies.
pub struct LookupFixtureTool {
    fixtures: BTreeMap<String, String>,
}

impl LookupFixtureTool {
    pub fn new(pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Self {
        Self {
            fixtures: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    pub fn from_map(map: BTreeMap<String, String>) -> Self {
        Self { fixtures: map }
    }
}

impl Tool for LookupFixtureTool {
    fn schema(&self) -> ToolSchema {
        let keys: Vec<String> = self.fixtures.keys().cloned().collect();
        ToolSchema::new(
            "lookup",
            format!(
                "Look up a key in the local fixture table. Available keys: {:?}.",
                keys
            ),
        )
        .param(
            ParamSchema::new("key", JsonType::String)
                .required()
                .describe("exact key to look up"),
        )
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
        match self.fixtures.get(key) {
            Some(value) => Ok(ToolOutput::json(serde_json::json!({
                "key": key,
                "value": value,
            }))),
            None => Err(fail(
                ToolFailureKind::Execution,
                format!("no fixture for key `{key}`"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// write_artifact
// ---------------------------------------------------------------------------

/// Persist content as a hashed artifact of the current run.
pub struct WriteArtifactTool;

impl WriteArtifactTool {
    /// Media type inferred from extension when the model omits one.
    pub fn default_media_type(name: &str) -> String {
        match name.rsplit('.').next() {
            Some("json") => "application/json".to_string(),
            Some("csv") => "text/csv".to_string(),
            Some("txt") | Some("text") => "text/plain".to_string(),
            _ => "text/markdown".to_string(),
        }
    }
}

impl Tool for WriteArtifactTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "write_artifact",
            "Write text content to a named file artifact of this run; returns its id and SHA-256.",
        )
        .param(
            ParamSchema::new("name", JsonType::String)
                .required()
                .describe("file name ([A-Za-z0-9._-] only, e.g. results-note.md)"),
        )
        .param(
            ParamSchema::new("content", JsonType::String)
                .required()
                .describe("full text content to write"),
        )
        .param(
            ParamSchema::new("media_type", JsonType::String)
                .describe("RFC 2046 media type; defaults from the file extension"),
        )
        .effect(ToolEffect::LocalWrite)
    }

    fn execute(&self, args: &ValidatedArguments, ctx: &ToolContext<'_>) -> ToolOutcome {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let media_type = args
            .get("media_type")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| Self::default_media_type(name));
        if content.is_empty() {
            return Err(fail(
                ToolFailureKind::Execution,
                "refusing to write an empty artifact",
            ));
        }
        let record = {
            let mut store = ctx
                .artifacts
                .lock()
                .map_err(|_| fail(ToolFailureKind::Execution, "artifact store poisoned"))?;
            store.write(
                name,
                &media_type,
                content.as_bytes(),
                "write_artifact",
                ctx.step_id,
            )
        };
        match record {
            Ok(record) => Ok(ToolOutput {
                payload: ToolPayload::Json(serde_json::json!({
                    "artifact_id": record.artifact_id,
                    "path": record.path.display().to_string(),
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "media_type": record.media_type,
                })),
                artifact_ids: vec![record.artifact_id],
            }),
            Err(e) => Err(fail(
                ToolFailureKind::Execution,
                format!("artifact write failed: {e:#}"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// controlled failure / echo / slow
// ---------------------------------------------------------------------------

/// Always fails with the requested message — the loop's failure-path probe.
pub struct FailTool;

impl Tool for FailTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "fail",
            "Always fails; used to exercise structured error handling.",
        )
        .param(
            ParamSchema::new("message", JsonType::String)
                .required()
                .describe("the failure message to report"),
        )
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("<no message>");
        Err(fail(ToolFailureKind::Execution, message))
    }
}

/// Returns its input verbatim (protocol reinjection checks).
pub struct EchoTool;

impl Tool for EchoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("echo", "Return the given text unchanged.")
            .param(
                ParamSchema::new("text", JsonType::String)
                    .required()
                    .describe("text to echo"),
            )
            .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolOutput::text(text))
    }
}

/// Sleeps in small increments until `milliseconds` elapses or the
/// invocation deadline/cancellation fires — the cooperative-timeout probe.
pub struct SlowTool {
    /// Sleep slice per checkpoint (deadline/cancel granularity).
    pub slice_ms: u64,
}

impl Default for SlowTool {
    fn default() -> Self {
        Self { slice_ms: 25 }
    }
}

impl SlowTool {
    pub fn new(slice_ms: u64) -> Self {
        Self { slice_ms }
    }
}

impl Tool for SlowTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "slow",
            "Sleeps for the given duration; used to exercise timeouts.",
        )
        .param(
            ParamSchema::new("milliseconds", JsonType::Integer)
                .required()
                .describe("how long to sleep"),
        )
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, ctx: &ToolContext<'_>) -> ToolOutcome {
        let ms = args
            .get("milliseconds")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .max(0) as u64;
        let started = Instant::now();
        loop {
            let elapsed = started.elapsed().as_millis() as u64;
            if elapsed >= ms {
                break;
            }
            if ctx.cancel.is_cancelled() {
                return Err(fail(ToolFailureKind::Execution, "cancelled by request"));
            }
            if Instant::now() >= ctx.deadline {
                return Err(fail(
                    ToolFailureKind::Timeout,
                    format!("hit its deadline after {elapsed}ms"),
                ));
            }
            let remaining = ms.saturating_sub(elapsed).max(1);
            std::thread::sleep(Duration::from_millis(self.slice_ms.min(remaining)));
        }
        Ok(ToolOutput::json(serde_json::json!({ "slept_ms": ms })))
    }
}

// ---------------------------------------------------------------------------
// read-only local research tools (sandboxed)
// ---------------------------------------------------------------------------

const MAX_READ_BYTES: u64 = 1024 * 1024;

/// Resolve `rel` under a fixed root; absolute paths and traversal fail
/// closed. Shared by both file tools.
fn resolve_under_root(
    root: &std::path::Path,
    rel: &str,
) -> Result<std::path::PathBuf, ToolFailure> {
    let rel_path = std::path::Path::new(rel);
    if rel_path.is_absolute() {
        return Err(fail(
            ToolFailureKind::Execution,
            "absolute paths are not allowed; use a path relative to the sandbox root",
        ));
    }
    if rel.is_empty()
        || rel.split(['/', '\\']).any(|seg| seg == "..")
        || !rel_path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
    {
        return Err(fail(
            ToolFailureKind::Execution,
            "path traversal is not allowed (only simple relative paths)",
        ));
    }
    Ok(root.join(rel_path))
}

/// Read a UTF-8 text file under a fixed root (1 MiB cap). ReadOnly.
pub struct ReadTextFileTool {
    root: std::path::PathBuf,
}

impl ReadTextFileTool {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Tool for ReadTextFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "read_text_file",
            "Read a UTF-8 text file from the local experiment directory.",
        )
        .param(
            ParamSchema::new("path", JsonType::String)
                .required()
                .describe("path relative to the sandbox root"),
        )
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let rel = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let path = resolve_under_root(&self.root, rel)?;
        let meta = std::fs::metadata(&path).map_err(|e| {
            fail(
                ToolFailureKind::Execution,
                format!("cannot stat `{rel}`: {e}"),
            )
        })?;
        if meta.len() > MAX_READ_BYTES {
            return Err(fail(
                ToolFailureKind::Execution,
                format!("`{rel}` exceeds the 1 MiB read cap ({} bytes)", meta.len()),
            ));
        }
        let mut text = String::new();
        std::fs::File::open(&path)
            .and_then(|mut f| f.read_to_string(&mut text))
            .map_err(|e| {
                fail(
                    ToolFailureKind::Execution,
                    format!("cannot read `{rel}`: {e}"),
                )
            })?;
        Ok(ToolOutput::json(serde_json::json!({
            "path": rel,
            "bytes": meta.len(),
            "content": text,
        })))
    }
}

/// Literal substring search over one text file (deterministic; no regex).
/// Reports match count plus the first 50 matching 1-based line numbers.
pub struct SearchTextTool {
    root: std::path::PathBuf,
}

impl SearchTextTool {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Tool for SearchTextTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "search_text",
            "Search a UTF-8 text file for a literal substring; returns the match count and up to 50 line numbers.",
        )
        .param(
            ParamSchema::new("path", JsonType::String)
                .required()
                .describe("path relative to the sandbox root"),
        )
        .param(
            ParamSchema::new("pattern", JsonType::String)
                .required()
                .describe("literal substring to find (case-sensitive)"),
        )
        .effect(ToolEffect::ReadOnly)
    }

    fn execute(&self, args: &ValidatedArguments, _ctx: &ToolContext<'_>) -> ToolOutcome {
        let rel = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if pattern.is_empty() {
            return Err(fail(
                ToolFailureKind::Execution,
                "empty search pattern is not allowed",
            ));
        }
        let path = resolve_under_root(&self.root, rel)?;
        let mut text = String::new();
        std::fs::File::open(&path)
            .and_then(|mut f| f.read_to_string(&mut text))
            .map_err(|e| {
                fail(
                    ToolFailureKind::Execution,
                    format!("cannot read `{rel}`: {e}"),
                )
            })?;

        let mut total = 0usize;
        let mut lines: Vec<u64> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let hits = line.matches(pattern).count();
            if hits > 0 {
                total += hits;
                if lines.len() < 50 {
                    lines.push(i as u64 + 1);
                }
            }
        }
        Ok(ToolOutput::json(serde_json::json!({
            "path": rel,
            "pattern": pattern,
            "matches": total,
            "lines": lines,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::artifact::ArtifactStore;
    use crate::agent::CancelFlag;
    use std::sync::Mutex;

    fn ctx<'a>(artifacts: &'a Mutex<ArtifactStore>, cancel: &'a CancelFlag) -> ToolContext<'a> {
        ToolContext {
            run_id: "run-test",
            step_id: "tool-0",
            call_seq: 1,
            deadline: Instant::now() + Duration::from_secs(3600),
            cancel,
            artifacts,
        }
    }

    fn args_for(schema: &ToolSchema, raw: &str) -> ValidatedArguments {
        ValidatedArguments::parse(schema, raw).expect("valid args")
    }

    #[test]
    fn calculator_is_deterministic_and_structured() {
        let store = Mutex::new(
            ArtifactStore::open(std::env::temp_dir().join("ember-calc-t"), "r").unwrap(),
        );
        let cancel = CancelFlag::new();
        let c = ctx(&store, &cancel);
        let tool = CalculatorTool;
        let schema = tool.schema();
        let out = tool
            .execute(
                &args_for(&schema, r#"{"operation":"multiply","a":6,"b":7}"#),
                &c,
            )
            .unwrap();
        match out.payload {
            ToolPayload::Json(v) => assert_eq!(v["result"], 42.0),
            other => panic!("expected json payload, got {other:?}"),
        }
        let err = tool
            .execute(
                &args_for(&schema, r#"{"operation":"divide","a":1,"b":0}"#),
                &c,
            )
            .unwrap_err();
        assert_eq!(err.kind, ToolFailureKind::Execution);
        assert!(err.message.contains("division by zero"));
    }

    #[test]
    fn lookup_reports_missing_keys_as_tool_failures() {
        let store = Mutex::new(
            ArtifactStore::open(std::env::temp_dir().join("ember-lookup-t"), "r").unwrap(),
        );
        let cancel = CancelFlag::new();
        let c = ctx(&store, &cancel);
        let tool = LookupFixtureTool::new([("alpha", "42"), ("beta", "43")]);
        let schema = tool.schema();
        let out = tool
            .execute(&args_for(&schema, r#"{"key":"alpha"}"#), &c)
            .unwrap();
        match out.payload {
            ToolPayload::Json(v) => assert_eq!(v["value"], "42"),
            other => panic!("expected json, got {other:?}"),
        }
        let err = tool
            .execute(&args_for(&schema, r#"{"key":"gamma"}"#), &c)
            .unwrap_err();
        assert!(err.message.contains("no fixture"));
    }

    #[test]
    fn write_artifact_hashes_content_and_lists_ids() {
        let dir = std::env::temp_dir().join(format!(
            "ember-wa-{}",
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let store = Mutex::new(ArtifactStore::open(&dir, "run-wa").unwrap());
        let cancel = CancelFlag::new();
        let c = ctx(&store, &cancel);
        let tool = WriteArtifactTool;
        let schema = tool.schema();
        let out = tool
            .execute(
                &args_for(&schema, "{\"name\":\"note.md\",\"content\":\"# hi\"}"),
                &c,
            )
            .unwrap();
        assert_eq!(out.artifact_ids.len(), 1);
        match out.payload {
            ToolPayload::Json(v) => {
                assert_eq!(v["sha256"], crate::extraction::sha256_bytes(b"# hi"));
                assert_eq!(v["media_type"], "text/markdown");
            }
            other => panic!("expected json, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fail_tool_always_fails_with_the_message() {
        let store = Mutex::new(
            ArtifactStore::open(std::env::temp_dir().join("ember-fail-t"), "r").unwrap(),
        );
        let cancel = CancelFlag::new();
        let c = ctx(&store, &cancel);
        let err = FailTool
            .execute(&args_for(&FailTool.schema(), r#"{"message":"boom"}"#), &c)
            .unwrap_err();
        assert_eq!(err.message, "boom");
    }

    #[test]
    fn file_tools_reject_traversal_and_absolute_paths() {
        let root = std::env::temp_dir().join(format!(
            "ember-sb-{}",
            std::time::Instant::now().elapsed().as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("data.txt"), "hello world\nsecond line\n").unwrap();
        let store = Mutex::new(ArtifactStore::open(&root, "r").unwrap());
        let cancel = CancelFlag::new();
        let c = ctx(&store, &cancel);
        let tool = ReadTextFileTool::new(&root);
        let schema = tool.schema();

        let ok = tool
            .execute(&args_for(&schema, r#"{"path":"data.txt"}"#), &c)
            .unwrap();
        match ok.payload {
            ToolPayload::Json(v) => assert_eq!(v["content"], "hello world\nsecond line\n"),
            other => panic!("expected json, got {other:?}"),
        }
        for bad in ["../secret", "/etc/passwd", "./x", ""] {
            assert!(
                tool.execute(&args_for(&schema, &format!(r#"{{"path":"{bad}"}}"#)), &c)
                    .is_err(),
                "expected rejection of {bad:?}"
            );
        }

        let search = SearchTextTool::new(&root);
        let sschema = search.schema();
        let hits = search
            .execute(
                &args_for(&sschema, r#"{"path":"data.txt","pattern":"line"}"#),
                &c,
            )
            .unwrap();
        match hits.payload {
            ToolPayload::Json(v) => {
                assert_eq!(v["matches"], 1);
                assert_eq!(v["lines"], serde_json::json!([2]));
            }
            other => panic!("expected json, got {other:?}"),
        }
        std::fs::remove_dir_all(&root).ok();
    }
}

// ---------------------------------------------------------------------------
// multimodal tool-result fixture (Track W)
// ---------------------------------------------------------------------------

/// Generates a tiny deterministic PNG as a tool result. Proves the
/// artifact path can carry binary media with a real media type; future
/// renderers attach such payloads to `ContentPart::Image` through the
/// existing multimodal architecture. The image content is a pure function
/// of the arguments, so runs stay reproducible.
pub struct ImageFixtureTool;

impl ImageFixtureTool {
    /// Deterministic RGB pixels from `width` x `height` + seed; same
    /// arguments, same bytes.
    fn render(width: u32, height: u32, seed: u64) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            let h = (x as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((y as u64).wrapping_mul(1442695040888963407))
                .wrapping_add(seed)
                .swap_bytes();
            let channel = |shift: u32| ((h >> shift) & 0xFF) as u8;
            image::Rgb([channel(56), channel(40), channel(24)])
        });
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("png encode of an in-memory raster cannot fail");
        png.into_inner()
    }
}

impl Tool for ImageFixtureTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "image_fixture",
            "Produce a small deterministic PNG test pattern as an artifact of this run.",
        )
        .param(
            ParamSchema::new("name", JsonType::String)
                .required()
                .describe("artifact file name, e.g. pattern.png"),
        )
        .param(
            ParamSchema::new("seed", JsonType::Integer)
                .describe("pattern seed (default 0); same seed, same bytes"),
        )
        .effect(ToolEffect::LocalWrite)
    }

    fn execute(&self, args: &ValidatedArguments, ctx: &ToolContext<'_>) -> ToolOutcome {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let seed = args
            .get("seed")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .max(0) as u64;
        let png = Self::render(8, 8, seed);
        if !name.ends_with(".png") {
            return Err(fail(
                ToolFailureKind::Execution,
                "image_fixture requires a `.png` file name",
            ));
        }
        let record = {
            let mut store = ctx
                .artifacts
                .lock()
                .map_err(|_| fail(ToolFailureKind::Execution, "artifact store poisoned"))?;
            store.write(name, "image/png", &png, "image_fixture", ctx.step_id)
        };
        match record {
            Ok(record) => Ok(ToolOutput {
                payload: ToolPayload::Json(serde_json::json!({
                    "kind": "image",
                    "media_type": record.media_type,
                    "artifact_id": record.artifact_id,
                    "path": record.path.display().to_string(),
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                })),
                artifact_ids: vec![record.artifact_id],
            }),
            Err(e) => Err(fail(
                ToolFailureKind::Execution,
                format!("artifact write failed: {e:#}"),
            )),
        }
    }
}

#[cfg(test)]
mod image_fixture_tests {
    use super::*;
    use crate::agent::artifact::ArtifactStore;
    use crate::agent::CancelFlag;
    use std::sync::Mutex;

    #[test]
    fn deterministic_png_artifact_with_real_media_type() {
        let dir = std::env::temp_dir().join(format!(
            "ember-imgfx-{}",
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let store = Mutex::new(ArtifactStore::open(&dir, "run-img").unwrap());
        let cancel = CancelFlag::new();
        let ctx = ToolContext {
            run_id: "run-img",
            step_id: "tool-0",
            call_seq: 1,
            deadline: Instant::now() + Duration::from_secs(60),
            cancel: &cancel,
            artifacts: &store,
        };
        let tool = ImageFixtureTool;
        let schema = tool.schema();
        let a = tool
            .execute(
                &ValidatedArguments::parse(&schema, r#"{"name":"p.png","seed":7}"#).unwrap(),
                &ctx,
            )
            .unwrap();
        let b = tool
            .execute(
                &ValidatedArguments::parse(&schema, r#"{"name":"q.png","seed":7}"#).unwrap(),
                &ctx,
            )
            .unwrap();
        // same seed -> identical bytes (reproducible fixture)
        match (&a.payload, &b.payload) {
            (ToolPayload::Json(x), ToolPayload::Json(y)) => assert_eq!(x["sha256"], y["sha256"]),
            other => panic!("expected json payloads, got {other:?}"),
        }
        // the artifact is a real decodable PNG
        match &a.payload {
            ToolPayload::Json(v) => {
                let path = v["path"].as_str().unwrap();
                let reader = image::ImageReader::open(path)
                    .unwrap()
                    .with_guessed_format()
                    .unwrap();
                let img = reader.decode().unwrap();
                assert_eq!((img.width(), img.height()), (8, 8));
                assert_eq!(v["media_type"], "image/png");
            }
            other => panic!("expected json payload, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_non_png_names() {
        let dir = std::env::temp_dir().join(format!(
            "ember-imgfx-bad-{}",
            std::time::Instant::now().elapsed().as_nanos()
        ));
        let store = Mutex::new(ArtifactStore::open(&dir, "run").unwrap());
        let cancel = CancelFlag::new();
        let ctx = ToolContext {
            run_id: "run",
            step_id: "tool-0",
            call_seq: 1,
            deadline: Instant::now() + Duration::from_secs(60),
            cancel: &cancel,
            artifacts: &store,
        };
        let tool = ImageFixtureTool;
        let err = tool
            .execute(
                &ValidatedArguments::parse(&tool.schema(), r#"{"name":"x.txt"}"#).unwrap(),
                &ctx,
            )
            .unwrap_err();
        assert_eq!(err.kind, ToolFailureKind::Execution);
        std::fs::remove_dir_all(&dir).ok();
    }
}

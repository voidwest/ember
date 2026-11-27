use super::*;

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

/// Entry point for `ember web-gui`.
pub(crate) fn run_gui_command(
    gui: &GuiArgs,
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> anyhow::Result<()> {
    let session = Arc::new(Mutex::new(GuiSession::new(k_strategy, k_allow_fallback)));
    let server = tiny_http::Server::http((gui.host.as_str(), gui.port)).map_err(|error| {
        anyhow::anyhow!(
            "cannot bind the GUI server on {}:{}: {error}",
            gui.host,
            gui.port
        )
    })?;
    let url = format!("http://{}/", server.server_addr());
    // Per-session bearer token: the browser page carries it in a meta tag
    // and sends it on every API request. This blocks cross-site request
    // forgery from arbitrary websites (which cannot read the token), and
    // the Host-header allowlist below blocks DNS rebinding (which could
    // otherwise read the token from the served page).
    let token = console_token();
    let strict_origin = gui.host == "127.0.0.1" || gui.host == "localhost" || gui.host == "::1";
    if !strict_origin {
        eprintln!(
            "  WARNING: binding to {} exposes the console to the network: anyone who can\n  reach this address can load the page and use it. The API still requires the\n  console token and same-origin POSTs, but the loopback Host allowlist (the\n  DNS-rebinding defence) cannot apply in this mode.",
            gui.host
        );
    }
    eprintln!(
        "EMBER experiment console v{} — {url}",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  console token: {token}");
    eprintln!("  press Ctrl-C to stop; the GUI is fully offline.");
    if !gui.no_open {
        open_browser(&url);
    }
    let limiter = Arc::new(SlotLimiter::new());
    for request in server.incoming_requests() {
        let session = Arc::clone(&session);
        let token = token.clone();
        limiter.spawn(move || {
            if let Err(error) = handle_request(request, &session, &token, strict_origin) {
                log::error!("gui request failed: {error:#}");
            }
        });
    }
    Ok(())
}

/// Fresh 128-bit bearer token per server run.
fn console_token() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| format!("{:x}", rng.gen_range(0u8..16)))
        .collect()
}

/// Loopback hosts accepted in the Host header (defeats DNS rebinding, which
/// would otherwise let a remote origin read the console page and its token).
pub(super) fn host_is_loopback(host: &str) -> bool {
    let host = host.trim();
    let host = if let Some(bracketed) = host.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or("")
    } else if host.matches(':').count() == 1 {
        host.split(':').next().unwrap_or(host)
    } else {
        host
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Origin header hostname is loopback (or absent — allow header-less
/// clients like curl; browsers always send Origin on cross-site POSTs).
pub(super) fn origin_is_loopback_or_absent(origin: Option<&str>) -> bool {
    let Some(origin) = origin else {
        return true;
    };
    let authority = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .unwrap_or(origin);
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or("")
    } else {
        authority.split(['/', ':']).next().unwrap_or(authority)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Origin header names the same authority as the Host header (or is absent,
/// for header-less clients). This is the network-bind counterpart of
/// [`origin_is_loopback_or_absent`]: the served address is not known in
/// advance there, so a cross-site POST is recognised by its Origin differing
/// from the host it was sent to.
pub(super) fn origin_matches_host_or_absent(origin: Option<&str>, host: Option<&str>) -> bool {
    let Some(origin) = origin else {
        return true;
    };
    let Some(host) = host else {
        return false;
    };
    let Some(authority) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        // `null` and non-HTTP origins never match a served host.
        return false;
    };
    authority.eq_ignore_ascii_case(host.trim())
}

/// The headers the security gate reads, separated from `tiny_http` so the
/// decision is a pure function.
pub(super) struct GateInput<'a> {
    pub(super) is_root: bool,
    pub(super) is_post: bool,
    pub(super) host: Option<&'a str>,
    pub(super) origin: Option<&'a str>,
    pub(super) token: Option<&'a str>,
}

/// Decide whether a request may proceed; `Err` carries the 403 body.
///
/// The bearer token guards the JSON API in every mode. A loopback bind also
/// pins the Host header (anti-DNS-rebinding) and requires a loopback Origin on
/// POSTs. A network bind cannot allowlist hosts, so it requires POSTs to be
/// same-origin instead.
pub(super) fn security_gate(
    strict_origin: bool,
    expected_token: &str,
    input: &GateInput<'_>,
) -> Result<(), &'static str> {
    if strict_origin && !input.host.is_some_and(host_is_loopback) {
        return Err("forbidden: bad Host header");
    }
    if input.is_root {
        return Ok(());
    }
    if input.token != Some(expected_token) {
        return Err("forbidden: missing or invalid token");
    }
    if input.is_post {
        let origin_ok = if strict_origin {
            origin_is_loopback_or_absent(input.origin)
        } else {
            origin_matches_host_or_absent(input.origin, input.host)
        };
        if !origin_ok {
            return Err("forbidden: cross-origin request");
        }
    }
    Ok(())
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/c", "start", "", url])
        .spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

fn handle_request(
    mut request: tiny_http::Request,
    session: &Arc<Mutex<GuiSession>>,
    token: &str,
    strict_origin: bool,
) -> anyhow::Result<()> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or(&url);
    let method = request.method().clone();
    let is_root = method == tiny_http::Method::Get && path == "/";

    // Security gate: see `security_gate` for what each bind mode enforces.
    let header_value = |name: &'static str| {
        request
            .headers()
            .iter()
            .find(|h| h.field.equiv(name))
            .map(|h| h.value.as_str())
    };
    let verdict = security_gate(
        strict_origin,
        token,
        &GateInput {
            is_root,
            is_post: method == tiny_http::Method::Post,
            host: header_value("Host"),
            origin: header_value("Origin"),
            token: header_value("X-Ember-Token"),
        },
    );
    if let Err(message) = verdict {
        let response = plain_response(403, message);
        return Ok(request.respond(response)?);
    }

    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_response(is_root, &method, path, &mut request, session, token)
    }));
    let response = match built {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            log::error!("gui request failed: {error:#}");
            json_response(&ApiEnvelope::err(format!("internal error: {error}")))?
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            log::error!("gui request panicked: {message}");
            json_response(&ApiEnvelope::err(
                "internal error: request handler panicked",
            ))?
        }
    };
    request.respond(response)?;
    Ok(())
}

fn build_response(
    is_root: bool,
    method: &tiny_http::Method,
    path: &str,
    request: &mut tiny_http::Request,
    session: &Arc<Mutex<GuiSession>>,
    token: &str,
) -> anyhow::Result<tiny_http::Response<std::io::Cursor<Vec<u8>>>> {
    if is_root {
        // inject the session token into the served page
        let page = PAGE.replace("__EMBER_TOKEN__", token);
        return Ok(tiny_http::Response::from_string(page)
            .with_header(header("Content-Type", "text/html; charset=utf-8"))
            .with_header(header("Cache-Control", "no-store"))
            .with_header(header("X-Content-Type-Options", "nosniff")));
    }
    let envelope = match (method, path) {
        (tiny_http::Method::Get, "/api/state") => state_payload(session),
        (tiny_http::Method::Post, "/api/prepare") => match read_json::<PrepareRequest>(request) {
            Ok(req) => {
                let mut session = lock_session(session);
                match session.ensure_prepared(&req.model_path) {
                    Ok(()) => {
                        let info = session_info_payload(&session);
                        ApiEnvelope::ok(info.unwrap_or_else(|| serde_json::json!({})))
                    }
                    Err(error) => {
                        session.load_error = Some(error.clone());
                        ApiEnvelope::err(error)
                    }
                }
            }
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        (tiny_http::Method::Post, "/api/run") => match read_json::<RunRequest>(request) {
            Ok(req) => run_experiment(session, req),
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        (tiny_http::Method::Post, "/api/restore") => match read_json::<RestoreRequest>(request) {
            Ok(req) => run_restore(session, req),
            Err(error) => ApiEnvelope::err(format!("malformed request: {error}")),
        },
        _ => {
            let response = plain_response(404, "not found");
            return Ok(response);
        }
    };
    json_response(&envelope)
}

/// Simple non-JSON status response (errors, auth failures).
fn plain_response(status: u16, message: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(message.to_string())
        .with_status_code(status)
        .with_header(header("Content-Type", "text/plain; charset=utf-8"))
        .with_header(header("Cache-Control", "no-store"))
        .with_header(header("X-Content-Type-Options", "nosniff"))
}

fn header(name: &str, value: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes())
        .expect("static header bytes are valid")
}

/// Request bodies are bounded: the console serves localhost only, but an
/// unbounded read lets any local process exhaust memory through the server.
const MAX_REQUEST_BODY_BYTES: u64 = 1 << 20;

fn read_json<T: for<'de> Deserialize<'de>>(request: &mut tiny_http::Request) -> anyhow::Result<T> {
    let mut body = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut total: u64 = 0;
    loop {
        let n = request
            .as_reader()
            .read(&mut chunk)
            .context("cannot read request body")?;
        if n == 0 {
            break;
        }
        total += n as u64;
        anyhow::ensure!(
            total <= MAX_REQUEST_BODY_BYTES,
            "request body exceeds the {MAX_REQUEST_BODY_BYTES}-byte limit"
        );
        body.extend_from_slice(&chunk[..n]);
    }
    serde_json::from_slice(&body).context("malformed JSON request body")
}

fn json_response(
    value: &ApiEnvelope,
) -> anyhow::Result<tiny_http::Response<std::io::Cursor<Vec<u8>>>> {
    let body = serde_json::to_vec(value).context("cannot serialize JSON response")?;
    Ok(tiny_http::Response::from_data(body)
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        .with_header(header("Cache-Control", "no-store"))
        .with_header(header("X-Content-Type-Options", "nosniff")))
}

/// Lock the GUI session, recovering from poisoning so a single panicking
/// request cannot permanently brick the console.
fn lock_session(session: &Arc<Mutex<GuiSession>>) -> std::sync::MutexGuard<'_, GuiSession> {
    session.lock().unwrap_or_else(|poisoned| {
        log::warn!("gui session mutex was poisoned; recovering guarded state");
        poisoned.into_inner()
    })
}

/// Bounds concurrent request-handler threads so a slow or malicious client
/// cannot exhaust the process with unbounded thread-per-request spawning.
pub(super) struct SlotLimiter {
    pub(super) active: Mutex<usize>,
    released: Condvar,
}

impl SlotLimiter {
    pub(super) fn new() -> Self {
        Self {
            active: Mutex::new(0),
            released: Condvar::new(),
        }
    }

    pub(super) fn acquire(self: &Arc<Self>) -> SlotGuard {
        const MAX_HANDLER_THREADS: usize = 8;
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *active >= MAX_HANDLER_THREADS {
            active = self
                .released
                .wait(active)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *active += 1;
        SlotGuard {
            limiter: Arc::clone(self),
        }
    }

    pub(super) fn spawn(
        self: &Arc<Self>,
        work: impl FnOnce() + Send + 'static,
    ) -> std::thread::JoinHandle<()> {
        // Reserve on the accept loop, so excess requests cannot create
        // unbounded worker threads that only wait for a slot.
        let slot = self.acquire();
        std::thread::spawn(move || {
            let _slot = slot;
            work();
        })
    }
}

pub(super) struct SlotGuard {
    limiter: Arc<SlotLimiter>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *active = active.saturating_sub(1);
        self.limiter.released.notify_one();
    }
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

fn state_payload(session: &Arc<Mutex<GuiSession>>) -> ApiEnvelope {
    let session = lock_session(session);
    ApiEnvelope::ok(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": ember::extraction::build_commit().unwrap_or("unknown"),
        "models": discover_models(),
        "hook_stages": hook_stages(),
        "per_layer_stages": per_layer_hook_stages(),
        "session": session_info_payload(&session),
    }))
}

fn session_info_payload(session: &GuiSession) -> Option<serde_json::Value> {
    let prepared = session.prepared.as_ref()?;
    Some(serde_json::json!({
        "model_path": prepared.model_path.display().to_string(),
        "model_name": model_basename(&prepared.model_path),
        "architecture": prepared.architecture,
        "n_layers": prepared.n_layers,
        "embed_dim": prepared.embed_dim,
        "vocab_size": prepared.model.config.vocab_size,
        "model_sha": prepared.model_sha,
        "tokenizer_sha": prepared.tokenizer_sha,
        "load_ms": session.load_ms,
        "load_error": session.load_error,
    }))
}

pub(super) fn model_basename(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn verification_json(report: &VerificationReport) -> serde_json::Value {
    serde_json::json!({
        "ok": report.ok,
        "checks": report.checks.iter().map(|check| serde_json::json!({
            "name": check.name,
            "ok": check.ok,
            "detail": check.detail,
        })).collect::<Vec<_>>(),
        "warnings": report.warnings,
    })
}

fn run_experiment(session: &Arc<Mutex<GuiSession>>, req: RunRequest) -> ApiEnvelope {
    let cfg = match parse_run_request(&req) {
        Ok(cfg) => cfg,
        Err(error) => return ApiEnvelope::err(error),
    };
    let mut session = lock_session(session);
    match session.run_baseline_intervention(&cfg) {
        Ok(bundle) => ApiEnvelope::ok(serde_json::json!({
            "baseline": bundle.baseline,
            "intervention": bundle.intervention,
            "verification": verification_json(&bundle.verification),
            "elapsed_ms_total": bundle.elapsed_ms_total,
            "elapsed_ms_baseline": bundle.elapsed_ms_baseline,
            "config": serde_json::json!({
                "site": cfg.site.stage_id(),
                "layer": cfg.layer,
                "operation": cfg.operation.label(cfg.factor, cfg.alpha),
                "max_new_tokens": cfg.max_new_tokens,
                "execution": cfg.execution.name(),
            }),
        })),
        Err(error) => ApiEnvelope::err(error),
    }
}

fn run_restore(session: &Arc<Mutex<GuiSession>>, req: RestoreRequest) -> ApiEnvelope {
    // The restore run reuses the full run configuration; only the
    // operation differs (restore-original instead of the user intervention).
    let run_req = RunRequest {
        model_path: req.model_path,
        prompt: req.prompt.clone(),
        max_new_tokens: req.max_new_tokens,
        execution: req.execution,
        site: req.site,
        layer: req.layer,
        operation: "restore-original".to_string(),
        factor: None,
        alpha: None,
        source: "capture".to_string(),
        source_layer: None,
        token_kind: req.token_kind,
        span_text: req.span_text,
    };
    let cfg = match parse_run_request(&run_req) {
        Ok(cfg) => cfg,
        Err(error) => return ApiEnvelope::err(error),
    };
    let mut session = lock_session(session);
    match session.run_restore_leg(&cfg) {
        Ok(bundle) => ApiEnvelope::ok(serde_json::json!({
            "restore": bundle.output,
            "verification": verification_json(&bundle.verification),
            "restore_matches_baseline": bundle.matches_baseline,
            "baseline_comparable": bundle.baseline_comparable,
            "baseline_text": bundle.baseline_text,
            "restore_text": bundle.output.text,
        })),
        Err(error) => ApiEnvelope::err(error),
    }
}

// ---------------------------------------------------------------------------
// model discovery
// ---------------------------------------------------------------------------

/// Discover GGUF files near the working directory (depth-limited, skipping
/// build/vendor trees) so the page can offer a model picker.
pub(crate) fn discover_models() -> Vec<String> {
    discover_models_in(&std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Find `.gguf` files under `root`, at most four directories deep.
///
/// Hidden directories, build output and dependency trees are skipped, and so
/// is `Library`: the macOS app bundle falls back to starting in `$HOME`, and
/// walking `~/Library` is slow and never where a model lives.
pub(crate) fn discover_models_in(root: &Path) -> Vec<String> {
    let mut found = BTreeSet::new();
    let mut stack = vec![(root.to_path_buf(), 0)];
    let skipped = [
        "target",
        "node_modules",
        "Library",
        "runs",
        "logs",
        "paper",
        "data",
    ];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 4 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            // `file_type` does not follow links: a symlinked directory (a
            // link to `.` or `/`) would otherwise multiply or widen the walk.
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                // Hidden directories (.git, .venv, .cache, ...) included.
                if name.starts_with('.') || skipped.contains(&name.as_str()) {
                    continue;
                }
                stack.push((path, depth + 1));
            } else if path.extension().is_some_and(|ext| ext == "gguf") {
                if let Ok(absolute) = std::fs::canonicalize(&path) {
                    found.insert(absolute.display().to_string());
                } else {
                    found.insert(path.display().to_string());
                }
            }
        }
    }
    found.into_iter().collect()
}

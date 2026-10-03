//! The model worker: owns the resident model session and runs experiments
//! off the UI thread.

use super::*;
use ember::cancel::CancelToken;

pub(super) enum WorkerMsg {
    Prepare(String),
    /// A baseline + intervention pair, stoppable through its token.
    Run(RunConfig, CancelToken),
    /// One point of a layer sweep, with the layers the sweep still plans
    /// (this one first); answered like `Run`.
    SweepPoint(RunConfig, Vec<usize>, CancelToken),
    /// The sweep is over: drop its shared pass (answered with `SweepEnded`).
    EndSweep,
    Restore(RunConfig, CancelToken),
}

#[derive(Debug, Clone)]
pub(super) enum WorkerReply {
    Prepared(Box<Result<SessionInfo, String>>),
    RunDone(Box<Result<RunBundle, String>>),
    RestoreDone(Box<Result<RestoreBundle, String>>),
    SweepEnded,
    /// The run's token fired. Nothing it produced was kept: bundle
    /// directories it had written are removed, and the model stays loaded.
    Cancelled,
    /// The request panicked; whatever was in flight is over.
    Failed(String),
}

fn handle_worker_msg(session: &mut crate::gui::GuiSession, msg: WorkerMsg) -> WorkerReply {
    match msg {
        WorkerMsg::Prepare(path) => {
            let result = session.ensure_prepared(&path).and_then(|_| {
                session
                    .info()
                    .ok_or_else(|| "model session is not prepared".to_string())
            });
            WorkerReply::Prepared(Box::new(result))
        }
        // Runs name their model. Preparing it here (a no-op when it is already
        // resident) means a run can never execute on whichever model happened
        // to finish loading last.
        WorkerMsg::Run(cfg, cancel) => {
            // Cancelled while queued: never start it.
            if cancel.is_cancelled() {
                return WorkerReply::Cancelled;
            }
            session.set_cancel(Some(cancel.clone()));
            let result = session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_baseline_intervention(&cfg));
            finish_pair(session, &cancel, result)
        }
        WorkerMsg::SweepPoint(cfg, planned, cancel) => {
            if cancel.is_cancelled() {
                return WorkerReply::Cancelled;
            }
            session.set_cancel(Some(cancel.clone()));
            let result = session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_sweep_point(&cfg, &planned));
            finish_pair(session, &cancel, result)
        }
        WorkerMsg::EndSweep => {
            session.end_sweep();
            WorkerReply::SweepEnded
        }
        WorkerMsg::Restore(cfg, cancel) => {
            if cancel.is_cancelled() {
                return WorkerReply::Cancelled;
            }
            session.set_cancel(Some(cancel.clone()));
            let result = session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_restore_leg(&cfg));
            session.set_cancel(None);
            if cancel.is_cancelled() {
                if let Ok(bundle) = &result {
                    remove_bundle_dir(&bundle.output.bundle_dir);
                }
                return WorkerReply::Cancelled;
            }
            WorkerReply::RestoreDone(Box::new(result))
        }
    }
}

/// The reply for a finished (or cancelled) baseline + intervention pair.
fn finish_pair(
    session: &mut crate::gui::GuiSession,
    cancel: &CancelToken,
    result: Result<RunBundle, String>,
) -> WorkerReply {
    session.set_cancel(None);
    if cancel.is_cancelled() {
        // A pair that finished just as Cancel was pressed is discarded
        // too: the user asked for it not to count.
        if let Ok(bundle) = &result {
            discard_run_bundles(bundle);
        }
        session.end_sweep();
        return WorkerReply::Cancelled;
    }
    WorkerReply::RunDone(Box::new(result))
}

/// Remove both bundles of a pair that is being discarded.
pub(super) fn discard_run_bundles(bundle: &RunBundle) {
    remove_bundle_dir(&bundle.baseline.bundle_dir);
    remove_bundle_dir(&bundle.intervention.bundle_dir);
}

/// Remove one bundle directory the console wrote. Only real directories are
/// touched: fixtures and reopened runs carry placeholder paths.
fn remove_bundle_dir(dir: &str) {
    let path = std::path::Path::new(dir);
    if path.is_dir() {
        let _ = std::fs::remove_dir_all(path);
    }
}

pub(super) fn spawn_worker(
    k_strategy: KStrategy,
    k_allow_fallback: bool,
) -> (
    mpsc::Sender<WorkerMsg>,
    Arc<Mutex<mpsc::Receiver<WorkerReply>>>,
) {
    let (tx, rx) = mpsc::channel();
    let (reply_tx, reply_rx) = mpsc::channel();
    let reply_rx = Arc::new(Mutex::new(reply_rx));
    std::thread::spawn(move || {
        let mut session = crate::gui::GuiSession::new(k_strategy, k_allow_fallback);
        while let Ok(msg) = rx.recv() {
            // A panic in a loader or kernel must come back as an error reply:
            // an unwinding worker would leave the console waiting forever on a
            // reply that never arrives. The session may be half-updated after a
            // panic, so it is rebuilt and the next request reloads the model.
            let reply = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle_worker_msg(&mut session, msg)
            }))
            .unwrap_or_else(|payload| {
                session = crate::gui::GuiSession::new(k_strategy, k_allow_fallback);
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                WorkerReply::Failed(format!("the model worker crashed: {message}"))
            });
            if reply_tx.send(reply).is_err() {
                break;
            }
        }
    });
    (tx, reply_rx)
}

#[cfg(test)]
mod tests {
    use super::super::{parse_run_request, RunBundle};
    use super::{
        discard_run_bundles, handle_worker_msg, remove_bundle_dir, WorkerMsg, WorkerReply,
    };
    use ember::cancel::CancelToken;
    use ember::quant_k::KStrategy;

    #[test]
    fn a_run_cancelled_before_it_starts_never_touches_the_model() {
        let mut session = crate::gui::GuiSession::new(KStrategy::Auto, false);
        let mut values = super::super::tests::form();
        values.model_path = "/nonexistent/never-loaded.gguf".into();
        let request = values.build_run_request().unwrap();
        let cfg = parse_run_request(&request).unwrap();
        let token = CancelToken::new();
        token.cancel();
        // Loading the nonexistent model would fail with an error; a
        // cancelled run must come back as Cancelled before that.
        assert!(matches!(
            handle_worker_msg(&mut session, WorkerMsg::Run(cfg.clone(), token.clone())),
            WorkerReply::Cancelled
        ));
        assert!(matches!(
            handle_worker_msg(&mut session, WorkerMsg::Restore(cfg, token)),
            WorkerReply::Cancelled
        ));
    }

    #[test]
    fn discarding_a_pair_removes_both_bundle_dirs_and_ignores_placeholders() {
        let root = std::env::temp_dir().join(format!("ember-cancel-{}", std::process::id()));
        let baseline = root.join("baseline");
        let intervention = root.join("intervention");
        std::fs::create_dir_all(&baseline).unwrap();
        std::fs::create_dir_all(&intervention).unwrap();
        std::fs::write(baseline.join("manifest.json"), b"{}").unwrap();
        let (mut left, mut right, comparison, _) = super::super::sample_result();
        left.bundle_dir = baseline.display().to_string();
        right.bundle_dir = intervention.display().to_string();
        let bundle = RunBundle {
            baseline: left,
            intervention: right,
            comparison,
            verification: ember::v05::verify::VerificationReport {
                bundle_schema: String::new(),
                ok: true,
                semantic_hash: String::new(),
                payload_hash: String::new(),
                checks: Vec::new(),
                warnings: Vec::new(),
                timestamp: String::new(),
            },
            elapsed_ms_total: 0.0,
            elapsed_ms_baseline: 0.0,
            baseline_key: String::new(),
        };
        discard_run_bundles(&bundle);
        assert!(!baseline.exists() && !intervention.exists());
        // "history" and "sample" are not directories and must not error.
        remove_bundle_dir("history");
        let _ = std::fs::remove_dir_all(&root);
    }
}

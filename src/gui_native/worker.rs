//! The model worker: owns the resident model session and runs experiments
//! off the UI thread.

use super::*;

pub(super) enum WorkerMsg {
    Prepare(String),
    Run(RunConfig),
    /// One point of a layer sweep, with the layers the sweep still plans
    /// (this one first); answered with `RunDone` like `Run`.
    SweepPoint(RunConfig, Vec<usize>),
    Restore(RunConfig),
}

#[derive(Debug, Clone)]
pub(super) enum WorkerReply {
    Prepared(Box<Result<SessionInfo, String>>),
    RunDone(Box<Result<RunBundle, String>>),
    RestoreDone(Box<Result<RestoreBundle, String>>),
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
        WorkerMsg::Run(cfg) => WorkerReply::RunDone(Box::new(
            session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_baseline_intervention(&cfg)),
        )),
        WorkerMsg::SweepPoint(cfg, planned) => WorkerReply::RunDone(Box::new(
            session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_sweep_point(&cfg, &planned)),
        )),
        WorkerMsg::Restore(cfg) => WorkerReply::RestoreDone(Box::new(
            session
                .ensure_prepared(&cfg.model_path)
                .and_then(|_| session.run_restore_leg(&cfg)),
        )),
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

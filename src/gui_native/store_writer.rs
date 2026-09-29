//! Run-history writes, off the UI thread.
//!
//! A save locks the store, re-reads it, merges and writes it atomically
//! ([`AppStore::save_merged`]); on a slow or network home directory that is
//! long enough to drop frames, and it used to run inside click handlers. One
//! background thread owns every write instead. Saves queued while it is busy
//! coalesce -- each is a full snapshot, so only the latest matters -- and each
//! write's outcome comes back to the console, which adopts the merged store and
//! keeps the status bar honest when a write fails.

use ember::app_store::{AppStore, Merged};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::JoinHandle;

/// What one write did. `sent` is the snapshot that was written, so the console
/// can tell whether its store has moved on since.
pub(super) struct SaveOutcome {
    pub(super) sent: AppStore,
    pub(super) result: Result<Merged, String>,
}

enum Job {
    Save(Box<AppStore>),
    Flush(mpsc::Sender<()>),
}

pub(super) struct StoreWriter {
    jobs: Option<mpsc::Sender<Job>>,
    outcomes: mpsc::Receiver<SaveOutcome>,
    thread: Option<JoinHandle<()>>,
}

impl StoreWriter {
    pub(super) fn spawn(path: PathBuf) -> Self {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (outcome_tx, outcomes) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("ember-store-writer".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let mut latest = None;
                    let mut flushes = Vec::new();
                    let mut take = |job| match job {
                        Job::Save(store) => latest = Some(store),
                        Job::Flush(done) => flushes.push(done),
                    };
                    take(job);
                    // Everything already queued: only the newest snapshot is
                    // written, and every flush waiting on it is answered after.
                    while let Ok(job) = job_rx.try_recv() {
                        take(job);
                    }
                    if let Some(store) = latest {
                        let result = store.save_merged(&path).map_err(|error| error.to_string());
                        let _ = outcome_tx.send(SaveOutcome {
                            sent: *store,
                            result,
                        });
                    }
                    for done in flushes {
                        let _ = done.send(());
                    }
                }
            })
            .expect("spawn the store writer thread");
        Self {
            jobs: Some(jobs),
            outcomes,
            thread: Some(thread),
        }
    }

    /// Queue a snapshot to be written. Returns at once.
    pub(super) fn save(&self, store: AppStore) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(Job::Save(Box::new(store)));
        }
    }

    /// Block until every snapshot queued so far is on disk (or has failed).
    pub(super) fn flush(&self) {
        let Some(jobs) = &self.jobs else {
            return;
        };
        let (done, wait) = mpsc::channel();
        if jobs.send(Job::Flush(done)).is_ok() {
            let _ = wait.recv();
        }
    }

    /// Outcomes of writes finished since the last call.
    pub(super) fn finished(&self) -> Vec<SaveOutcome> {
        self.outcomes.try_iter().collect()
    }
}

impl Drop for StoreWriter {
    /// Closing the queue lets the thread write what is pending and exit.
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ember::app_store::{self, Draft};

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ember-store-writer-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn with_draft(revision: u64) -> AppStore {
        let mut store = AppStore::default();
        store.draft = Some(Draft {
            revision,
            prompt: format!("draft {revision}"),
            model_path: String::new(),
            fields: Default::default(),
            step: "prompt".into(),
            updated_at: 0,
        });
        store
    }

    #[test]
    fn queued_saves_coalesce_and_flush_waits_for_the_latest() {
        let dir = scratch("coalesce");
        let path = dir.join("app-state.v2.json");
        let writer = StoreWriter::spawn(path.clone());
        for revision in 1..=50 {
            writer.save(with_draft(revision));
        }
        writer.flush();
        // Flush returned, so the newest snapshot is already on disk.
        let on_disk = app_store::load(&path).unwrap();
        assert_eq!(on_disk.draft.unwrap().revision, 50);
        let outcomes = writer.finished();
        assert!(
            !outcomes.is_empty() && outcomes.len() < 50,
            "saves coalesced"
        );
        let last = outcomes.last().unwrap();
        assert_eq!(last.sent, with_draft(50));
        assert!(last.result.is_ok());
        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_is_reported_back() {
        let dir = scratch("failure");
        // The store's parent is a file, so the directory cannot be created.
        let blocker = dir.join("not-a-directory");
        std::fs::write(&blocker, b"").unwrap();
        let writer = StoreWriter::spawn(blocker.join("app-state.v2.json"));
        writer.save(AppStore::default());
        writer.flush();
        let outcomes = writer.finished();
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].result.is_err());
        drop(writer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dropping_the_writer_finishes_pending_saves() {
        let dir = scratch("drop");
        let path = dir.join("app-state.v2.json");
        let writer = StoreWriter::spawn(path.clone());
        writer.save(with_draft(7));
        drop(writer);
        assert_eq!(app_store::load(&path).unwrap().draft.unwrap().revision, 7);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

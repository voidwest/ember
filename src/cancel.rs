//! Cooperative cancellation for long-running loops.
//!
//! A [`CancelToken`] is a shared flag that generation and agent loops poll at
//! documented check points. It carries no scheduling or signal handling of its
//! own: the CLI installs a SIGINT handler that fires the token, agent sessions
//! poll it per step, and tests can fire it directly.
//!
//! Check-point contract (see `docs/cancellation.md`):
//!
//! - the CLI generation loop checks before prefill and at the top of every
//!   decode step;
//! - prefill is a single forward pass over the prompt and is not interruptible
//!   mid-pass — a cancel arriving during prefill is honored before the first
//!   decode step;
//! - a cancelled CLI generation exits with code 4.
//!
//! Cancellation leaves no partial state behind: the KV cache is owned by the
//! caller and is dropped or truncated by the caller's own policy.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cloneable cancellation flag. Clones share one flag.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// Create a fresh, uncancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fire the token. Idempotent and safe to call from any thread
    /// (signal-handler threads included).
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// True once [`CancelToken::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// `Some(())` when cancelled; convenient for `if token.when().is_some()`.
    pub fn when(&self) -> Option<()> {
        self.is_cancelled().then_some(())
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// Error returned by cancellable operations when their token fires.
///
/// Callers that need to distinguish cancellation from other failures (for
/// example the CLI, which exits with code 4) can downcast through
/// `anyhow::Error::is::<Cancelled>()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_starts_uncancelled_and_clones_share_state() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        assert!(token.when().is_none());
        let clone = token.clone();
        token.cancel();
        assert!(clone.is_cancelled());
        assert!(token.when().is_some());
    }

    #[test]
    fn cancel_is_visible_across_threads() {
        let token = CancelToken::new();
        let other = token.clone();
        std::thread::spawn(move || other.cancel()).join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancelled_error_is_downcastable() {
        let error = anyhow::Error::new(Cancelled);
        assert!(error.is::<Cancelled>());
        assert_eq!(error.to_string(), "cancelled");
    }
}

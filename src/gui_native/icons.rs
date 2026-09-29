//! Asset source for the native console.
//!
//! The console is typographic: navigation, buttons and status indicators are
//! text and dots, so the embedded icon set is gone. This source remains so
//! kit-owned controls (select chevrons, sort arrows, clear buttons) can still
//! resolve their own artwork through the kit asset fallback.

use gpui_kit::{AssetSource, Result, SharedString};
use std::borrow::Cow;

pub(super) struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        gpui_kit::assets::Assets.list(path)
    }
}

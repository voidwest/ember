//! Native-console appearance and semantic design tokens.

use gpui::{rgb, rgba, Rgba, WindowAppearance};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AppearanceMode {
    System,
    Dark,
    Light,
}

impl AppearanceMode {
    pub(super) fn load() -> Self {
        let Ok(value) = std::fs::read_to_string(settings_path()) else {
            return Self::System;
        };
        match value.trim() {
            "dark" => Self::Dark,
            "light" => Self::Light,
            _ => Self::System,
        }
    }

    pub(super) fn next(self) -> Self {
        match self {
            Self::System => Self::Dark,
            Self::Dark => Self::Light,
            Self::Light => Self::System,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::System => "SYSTEM",
            Self::Dark => "DARK",
            Self::Light => "LIGHT",
        }
    }

    pub(super) fn is_dark(self, system_dark: bool) -> bool {
        match self {
            Self::System => system_dark,
            Self::Dark => true,
            Self::Light => false,
        }
    }

    pub(super) fn persist(self) {
        let path = settings_path();
        let Some(parent) = path.parent() else {
            return;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
        let _ = std::fs::write(path, self.label().to_ascii_lowercase());
    }
}

pub(super) fn system_is_dark(appearance: WindowAppearance) -> bool {
    matches!(
        appearance,
        WindowAppearance::Dark | WindowAppearance::VibrantDark
    )
}

fn settings_path() -> PathBuf {
    if let Some(root) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(root).join("ember/native-console-theme");
    }
    if let Some(root) = std::env::var_os("HOME") {
        return PathBuf::from(root).join(".config/ember/native-console-theme");
    }
    std::env::temp_dir().join("ember-native-console-theme")
}

/// Semantic colors. Layout code names the role it needs instead of selecting
/// an arbitrary shade, so light mode can be designed independently.
#[derive(Clone, Copy)]
pub(super) struct Colors {
    pub canvas: Rgba,
    pub sidebar: Rgba,
    pub surface: Rgba,
    pub surface_raised: Rgba,
    pub overlay: Rgba,
    pub text: Rgba,
    pub text_muted: Rgba,
    pub text_faint: Rgba,
    pub border: Rgba,
    pub border_strong: Rgba,
    pub hover: Rgba,
    pub selected: Rgba,
    pub focus_ring: Rgba,
    pub accent: Rgba,
    pub accent_hover: Rgba,
    pub accent_pressed: Rgba,
    pub accent_soft: Rgba,
    pub ok: Rgba,
    pub err: Rgba,
    pub warn: Rgba,
    pub busy: Rgba,
    pub err_box_bg: Rgba,
    pub err_box_border: Rgba,
    pub warn_box_bg: Rgba,
    pub warn_box_border: Rgba,
    pub selection: Rgba,
    pub caret: Rgba,
}

pub(super) fn light() -> Colors {
    Colors {
        canvas: rgb(0xf7f7f8),
        sidebar: rgb(0xf0f1f3),
        surface: rgb(0xffffff),
        surface_raised: rgb(0xf5f5f6),
        overlay: rgb(0xffffff),
        text: rgb(0x202126),
        text_muted: rgb(0x555b66),
        text_faint: rgb(0x777f8c),
        border: rgb(0xdfe2e7),
        border_strong: rgb(0xc7ccd4),
        hover: rgb(0xe9ebef),
        selected: rgb(0xffe5d8),
        focus_ring: rgba(0xdc5c2066),
        accent: rgb(0xd65318),
        accent_hover: rgb(0xe86122),
        accent_pressed: rgb(0xb94412),
        accent_soft: rgba(0xdc5c201f),
        ok: rgb(0x197a48),
        err: rgb(0xc43e36),
        warn: rgb(0x9a6810),
        busy: rgb(0xb5477b),
        err_box_bg: rgb(0xfceceb),
        err_box_border: rgb(0xe5b8b4),
        warn_box_bg: rgb(0xfaf3df),
        warn_box_border: rgb(0xe0c98f),
        selection: rgba(0xdc5c2033),
        caret: rgb(0xd65318),
    }
}

pub(super) fn dark() -> Colors {
    Colors {
        canvas: rgb(0x090a0d),
        sidebar: rgb(0x101216),
        surface: rgb(0x15181e),
        surface_raised: rgb(0x1c2028),
        overlay: rgb(0x20242d),
        text: rgb(0xe9eaed),
        text_muted: rgb(0xa3a9b3),
        text_faint: rgb(0x747c89),
        border: rgb(0x282d37),
        border_strong: rgb(0x3a414e),
        hover: rgb(0x242933),
        selected: rgb(0x35231d),
        focus_ring: rgba(0xf06b2f80),
        accent: rgb(0xf06b2f),
        accent_hover: rgb(0xff7f45),
        accent_pressed: rgb(0xd95b24),
        accent_soft: rgba(0xf06b2f26),
        ok: rgb(0x4cc38a),
        err: rgb(0xf0685c),
        warn: rgb(0xe8b34b),
        busy: rgb(0xe06b9f),
        err_box_bg: rgb(0x2a1a18),
        err_box_border: rgb(0x5c332e),
        warn_box_bg: rgb(0x2a2418),
        warn_box_border: rgb(0x5c4a2a),
        selection: rgba(0xf06b2f40),
        caret: rgb(0xff8b55),
    }
}

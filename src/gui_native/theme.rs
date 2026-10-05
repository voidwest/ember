//! Native-console appearance and semantic design tokens.

use gpui_kit::{rgb, rgba, Rgba, WindowAppearance};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AppearanceMode {
    System,
    Dark,
    Light,
}

impl AppearanceMode {
    pub(super) fn load() -> Self {
        let Some(path) = settings_path() else {
            return Self::System;
        };
        let Some(value) = read_setting(&path) else {
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
            Self::System => "System",
            Self::Dark => "Dark",
            Self::Light => "Light",
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
        let Some(path) = settings_path() else {
            return;
        };
        write_setting(&path, &self.label().to_ascii_lowercase());
    }
}

pub(super) fn system_is_dark(appearance: WindowAppearance) -> bool {
    matches!(
        appearance,
        WindowAppearance::Dark | WindowAppearance::VibrantDark
    )
}

/// The appearance setting, beside the run history. `None` under unit tests,
/// which drive the real console and must neither read the developer's settings
/// (results would depend on them) nor write them.
fn settings_path() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    Some(ember::app_store::config_dir()?.join("native-console-theme"))
}

/// A one-word persisted workspace flag, one file per key, beside the theme
/// setting. The inspector and sidebar remember their state across launches --
/// a workspace that resets itself is a demo, not a tool.
fn flag_path(key: &str) -> Option<PathBuf> {
    let mut path = settings_path()?;
    path.set_file_name(format!("native-console-{key}"));
    Some(path)
}

pub(super) fn load_flag(key: &str) -> Option<bool> {
    read_setting(&flag_path(key)?).map(|value| value.trim() == "true")
}

/// A one-word setting: a regular file of at most a few bytes (see
/// `app_store::read_config_file`), anything else reads as unset.
fn read_setting(path: &std::path::Path) -> Option<String> {
    let bytes = ember::app_store::read_config_file(path, 64).ok()??;
    String::from_utf8(bytes).ok()
}

/// Replace a setting file by rename, so a symlink planted in its place is
/// replaced rather than followed and its target truncated.
fn write_setting(path: &std::path::Path, value: &str) {
    if let Some(parent) = path.parent()
        && std::fs::create_dir_all(parent).is_ok()
    {
        let _ = ember::atomic_file::atomic_write(path, value.as_bytes());
    }
}

pub(super) fn persist_flag(key: &str, value: bool) {
    let Some(path) = flag_path(key) else {
        return;
    };
    write_setting(&path, if value { "true" } else { "false" });
}

/// Semantic colors. Layout code names the role it needs instead of selecting
/// an arbitrary shade, so light mode can be designed independently.
#[derive(Clone, Copy)]
pub(super) struct Colors {
    pub canvas: Rgba,
    pub sidebar: Rgba,
    pub surface: Rgba,
    pub surface_raised: Rgba,
    /// One luminance step above `surface`: the hover state for quiet rows,
    /// tabs and toolbar controls. Never used for resting surfaces.
    pub surface_hover: Rgba,
    /// The selected-row fill for navigation, tabs and segmented choices.
    /// Deliberately neutral -- selection is position, not emphasis, so the
    /// accent stays reserved for the intervention and the primary action.
    pub selection: Rgba,
    pub text: Rgba,
    pub text_muted: Rgba,
    pub text_faint: Rgba,
    pub border: Rgba,
    pub border_strong: Rgba,
    pub accent: Rgba,
    pub accent_soft: Rgba,
    pub ok: Rgba,
    pub err: Rgba,
    pub warn: Rgba,
    pub busy: Rgba,
    pub err_box_bg: Rgba,
}

pub(super) fn light() -> Colors {
    Colors {
        // Warm neutrals. The previous palette was blue-tinted grey, which
        // reads as a code-hosting site; these sit in the Obsidian/Notion
        // family where the canvas is a warm off-white and separation comes
        // from a barely-there hairline rather than a cool line.
        canvas: rgb(0xfbfaf9),
        sidebar: rgb(0xf5f3f1),
        surface: rgb(0xfdfdfc),
        surface_raised: rgb(0xffffff),
        surface_hover: rgb(0xf1efeb),
        selection: rgb(0xe9e5df),
        text: rgb(0x2e2c29),
        text_muted: rgb(0x6b6862),
        // Faint is the smallest metadata text in the UI, so it is held to the
        // 4.5:1 body-text floor against every surface it can land on rather
        // than being dialled down for looks. The old light value sat at 3.8:1.
        text_faint: rgb(0x6e6961),
        border: rgb(0xe8e4de),
        border_strong: rgb(0xd5cfc6),
        accent: rgb(0xb8501a),
        accent_soft: rgba(0xb8501a1f),
        ok: rgb(0x2f7d4f),
        err: rgb(0xb23c33),
        warn: rgb(0x8a6410),
        busy: rgb(0x9c3f70),
        err_box_bg: rgb(0xfbeceb),
    }
}

pub(super) fn dark() -> Colors {
    Colors {
        // A canvas at #1a1918 is mid-grey, not "very dark charcoal", and every
        // surface had to announce itself with a border to be seen at all. The
        // darkest plane is now the content canvas and each step up is lighter,
        // so hierarchy is carried by luminance and the hairlines can recede.
        // Warm-neutral rather than blue-black, and deliberately not pure black.
        canvas: rgb(0x0d0c0b),
        sidebar: rgb(0x141312),
        surface: rgb(0x1a1917),
        surface_raised: rgb(0x232120),
        surface_hover: rgb(0x211f1d),
        selection: rgb(0x262320),
        text: rgb(0xe6e2dc),
        text_muted: rgb(0xa8a29a),
        // Held to 4.5:1 on the raised surface too, which is where the old
        // value failed.
        text_faint: rgb(0x9c968c),
        border: rgba(0xffffff1e),
        border_strong: rgba(0xffffff33),
        accent: rgb(0xef8c48),
        accent_soft: rgba(0xef8c4826),
        ok: rgb(0x5cb88a),
        err: rgb(0xe2695e),
        warn: rgb(0xd8a94a),
        busy: rgb(0xdb6a9c),
        err_box_bg: rgb(0x2b1d1a),
    }
}

/// Type scale. One place to change a size, so hierarchy cannot drift.
///
/// The console had no scale: of roughly fifty text calls, thirty-nine sat
/// between 8 and 11px and nothing exceeded 20px, which is why the interface
/// read as small, dense and technical no matter how the surfaces were
/// retuned. Sizes are a token, not an argument to be chosen per call site.
pub(super) struct Type;

impl Type {
    /// Section title within a page.
    pub(super) const SECTION: f32 = 19.0;
    /// A result value read from across a room: landmark values, verdicts.
    pub(super) const VALUE: f32 = 18.0;
    /// Generated model output -- the text the demo is about.
    pub(super) const OUTPUT: f32 = 16.0;
    /// Sub-heading / card title.
    pub(super) const SUBSECTION: f32 = 16.0;
    /// Body copy and input text.
    pub(super) const BODY: f32 = 15.0;
    /// Field labels and secondary copy. Sentence case, never uppercase.
    pub(super) const LABEL: f32 = 14.0;
    /// Metadata: model names, hook ids, seeds, token counts.
    pub(super) const META: f32 = 13.0;
    /// Micro labels and dense secondary rows.
    pub(super) const MICRO: f32 = 12.5;
}

/// Presentation mode's text scale. Text goes through [`scaled`] at the few
/// primitives every string is drawn by, so one factor moves the whole console
/// and no call site has to know about it.
static UI_SCALE_MILLI: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1000);

pub(super) const PRESENTATION_SCALE: f32 = 1.18;

pub(super) fn set_ui_scale(scale: f32) {
    UI_SCALE_MILLI.store(
        (scale * 1000.0).round() as u32,
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub(super) fn scaled(size: f32) -> f32 {
    size * UI_SCALE_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0
}

/// Spacing scale. 4/8/12/16/24/32, so rhythm is a token rather than a guess.
pub(super) struct Space;

impl Space {
    pub(super) const XS: f32 = 4.0;
    pub(super) const SM: f32 = 8.0;
    pub(super) const MD: f32 = 12.0;
    pub(super) const LG: f32 = 16.0;
    pub(super) const XL: f32 = 24.0;
    pub(super) const XXL: f32 = 32.0;
}

/// Corner radii. Very few values; full pills are for status chips only.
pub(super) struct Radius;

impl Radius {
    pub(super) const SM: f32 = 6.0;
    pub(super) const MD: f32 = 8.0;
    pub(super) const LG: f32 = 10.0;
}

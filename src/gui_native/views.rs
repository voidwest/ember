//! View builders: the shell, the pages, the experiment steps and the result
//! panels, split by concern under `views/`.

mod controls;
mod pages;
mod results;
mod shell;

use super::*;

const SIDEBAR_MIN_WINDOW: f32 = 1240.0;

impl Render for Console {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_spec_view(cx);
        // Narrow windows give up the sidebar so the two workspace panes keep
        // the room they need. The stored flag is untouched: widening the
        // window brings it back exactly as the user left it.
        let width = f32::from(window.viewport_size().width);
        let show_sidebar = self.sidebar_open && width >= SIDEBAR_MIN_WINDOW;
        let colors = self.colors();
        let topbar = self.topbar(&colors, cx);
        let statusbar = self.statusbar(&colors, cx);

        // The content region depends on the destination. Only the experiment
        // view carries the workflow stepper and the contextual inspector; the
        // other views are plain pages.
        let content: AnyElement = match self.view {
            View::Home => self.home_view(&colors, cx).into_any_element(),
            View::Models => self.models_view(&colors, cx).into_any_element(),
            View::Runs => self.runs_view(&colors, window, cx),
            View::Settings => self.settings_view(&colors, cx).into_any_element(),
            // The workspace: setup and results side by side. It is the only
            // page with two panes of its own, so it fills the content region.
            View::Experiment => div()
                .flex()
                .flex_col()
                .flex_1()
                .min_h(px(0.0))
                // The row must be allowed to shrink below its content or a
                // wide results page pushes the window's edge out.
                .min_w(px(0.0))
                .child(self.main_panel(&colors, cx))
                .into_any_element(),
        };

        let body = div()
            .flex()
            .flex_row()
            .w_full()
            .flex_1()
            .min_h(px(0.0))
            .when(show_sidebar, |row| row.child(self.nav_rail(&colors, cx)))
            .child(content);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(colors.canvas)
            .relative()
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|console, event: &KeyDownEvent, window, cx| {
                console.picker_key(event, window, cx);
            }))
            .child(topbar)
            .child(body)
            // The status line appears only when it has something to say: a
            // run in flight, or history that could not be saved. A permanent
            // 'Ready' strip is chrome, not information.
            .when(self.statusbar_needed(), |shell| {
                shell.child(statusbar.flex_none())
            })
            .children(self.toast_overlay(&colors))
            .when(self.palette_open, |shell| {
                shell.child(self.palette_overlay(&colors, cx))
            })
    }
}

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

fn short_id(hash: &str) -> String {
    if hash.len() > 6 {
        format!("{}\u{2026}", &hash[..6])
    } else {
        hash.to_string()
    }
}

/// The shortcut hint a palette command carries, when it has one. Parsed from
/// the same spelling the keybindings use, so the hint cannot drift from the
/// binding it advertises.
fn palette_shortcut(command: Command) -> Option<Kbd> {
    // `secondary` is Cmd on macOS and Ctrl elsewhere, the key the handler
    // answers to; `cmd` would read "Win" on Linux.
    let stroke = match command {
        Command::NewExperiment => "secondary-n",
        Command::RerunExperiment => "secondary-r",
        Command::RunExperiment => "secondary-enter",
        Command::CancelRun => "escape",
        Command::GoSettings => "secondary-,",
        Command::GoToRun => "secondary-p",
        Command::ShowOverview => "secondary-1",
        Command::ShowLayers => "secondary-2",
        Command::ShowTokens => "secondary-3",
        Command::ShowTrace => "secondary-4",
        Command::ToggleSidebar => "secondary-b",
        _ => return None,
    };
    Some(Kbd::new(Keystroke::parse(stroke).ok()?))
}

fn fmt_ms(ms: f64) -> String {
    format!("{:.2} s", ms / 1000.0)
}

fn fmt_tps(tps: Option<f64>) -> String {
    match tps {
        Some(tps) => format!("{tps:.1} tok/s"),
        None => "\u{2014}".to_string(),
    }
}

/// Show a file in the platform file manager, selected.
///
/// One command per platform, spawned detached: the file manager is not ours
/// to wait on, and a failure is a banner, not a crash.
pub(super) fn reveal_in_finder(path: &str) -> Result<(), String> {
    // A stored path that starts with `-` must not reach the opener as an
    // option.
    let anchored;
    let path = if path.starts_with('-') {
        anchored = format!("./{path}");
        anchored.as_str()
    } else {
        path
    };
    let result = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("explorer")
            .arg(format!("/select,{path}"))
            .spawn()
    } else {
        let parent = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("/"));
        std::process::Command::new("xdg-open").arg(parent).spawn()
    };
    result.map(|_| ()).map_err(|error| error.to_string())
}

fn isolate_bidi(text: &str) -> String {
    format!("\u{2068}{text}\u{2069}")
}

/// The byte offset in `text` where it and `other` stop agreeing.
///
/// When the two part inside a word, the offset moves back to the start of
/// that word. A mark gives its text a separate run, and on macOS each run is
/// shaped apart, so a mark inside a word would break Arabic letter joining or
/// split an accent or an emoji from its base. A whole changed word also reads
/// better than a changed suffix.
pub(super) fn divergence_offset(text: &str, other: &str) -> usize {
    let shared = text
        .char_indices()
        .zip(other.chars())
        .find(|((_, a), b)| a != b)
        .map_or(text.len().min(other.len()), |((index, _), _)| index);
    let in_word = |c: char| !parts_words(c);
    let continues_word = |side: &str| side[shared..].chars().next().is_some_and(in_word);
    if !(continues_word(text) || continues_word(other)) {
        return shared;
    }
    text[..shared]
        .char_indices()
        .rev()
        .take_while(|(_, c)| in_word(*c))
        .last()
        .map_or(shared, |(index, _)| index)
}

/// Whether two outputs may part next to `c` without splitting what a script
/// shapes as one unit: whitespace, and Latin, CJK and Arabic punctuation.
/// Zero-width joiners and the other format characters are not on the list,
/// so an emoji sequence or a joined word stays whole.
fn parts_words(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_punctuation()
        || matches!(
            c as u32,
            // Dashes, quotes, the ellipsis and the other general punctuation.
            0x2010..=0x2027 | 0x2030..=0x205E
            // CJK symbols and punctuation, and the full-width forms.
            | 0x3000..=0x303F | 0xFF01..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40
            | 0xFF5B..=0xFF65
            // Arabic comma, semicolon, question mark and full stop.
            | 0x060C | 0x061B | 0x061F | 0x06D4
        )
}

/// The output text as its card draws it, and the byte range of that to mark:
/// the text from `from` to its end.
///
/// The text is a bidi isolate (see [`isolate_bidi`]). gpui paints a run's
/// background in glyph order, and right-to-left glyphs come in reverse order,
/// so a mark that starts partway through a line with right-to-left text would
/// tint the whole line. On such a line the changed text starts a line of its
/// own. The mark also takes in the isolate characters at the ends of its
/// lines, so that every line it touches has one style from end to end.
pub(super) fn marked_display(text: &str, from: usize) -> (String, Option<std::ops::Range<usize>>) {
    const FSI: &str = "\u{2068}";
    const PDI: &str = "\u{2069}";
    let line_end = |at: usize| text[at..].find('\n').map_or(text.len(), |index| at + index);
    let mut from = from.min(text.len());
    // A mark that holds nothing on its first line but the line ending starts
    // on the next line, so the split below never draws an empty line.
    if text[from..line_end(from)].trim_end_matches('\r').is_empty() {
        from = line_end(from) + 1;
    }
    if from >= text.len() {
        return (isolate_bidi(text), None);
    }
    let line_start = text[..from].rfind('\n').map_or(0, |index| index + 1);
    if from > line_start && text[line_start..line_end(from)].chars().any(is_rtl) {
        let display = format!("{FSI}{}{PDI}\n{FSI}{}{PDI}", &text[..from], &text[from..]);
        let start = FSI.len() + from + PDI.len() + 1;
        let end = display.len();
        return (display, Some(start..end));
    }
    let display = isolate_bidi(text);
    let start = if from == 0 { 0 } else { from + FSI.len() };
    let end = display.len();
    (display, Some(start..end))
}

/// A character of a right-to-left script: Hebrew, Arabic, Syriac, Thaana,
/// N'Ko and their extensions and presentation forms.
fn is_rtl(c: char) -> bool {
    matches!(
        c as u32,
        0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFC | 0x10800..=0x10FFF | 0x1E800..=0x1EFFF
    )
}

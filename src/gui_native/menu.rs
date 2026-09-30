//! The menu bar and the routing of its commands to the console.

use super::*;

/// The menu bar. The commands mirror the palette and the key handler, so a
/// menu item and its shortcut always do the same thing; Edit uses the kit's
/// own text actions so Cut, Copy, Paste and Undo reach the focused field.
pub(super) fn app_menus() -> Vec<Menu> {
    use gpui_kit::component::input as text;
    vec![
        Menu::new("Ember").items([
            MenuItem::action("Settings\u{2026}", OpenSettings),
            MenuItem::separator(),
            MenuItem::action("Quit Ember", Quit),
        ]),
        Menu::new("Edit").items([
            MenuItem::os_action("Undo", text::Undo, OsAction::Undo),
            MenuItem::os_action("Redo", text::Redo, OsAction::Redo),
            MenuItem::separator(),
            MenuItem::os_action("Cut", text::Cut, OsAction::Cut),
            MenuItem::os_action("Copy", text::Copy, OsAction::Copy),
            MenuItem::os_action("Paste", text::Paste, OsAction::Paste),
            MenuItem::os_action("Select All", text::SelectAll, OsAction::SelectAll),
        ]),
        Menu::new("Experiment").items([
            MenuItem::action("New Experiment", StartExperiment),
            MenuItem::action("Rerun Last Run", ReplayLastRun),
            MenuItem::separator(),
            MenuItem::action("Open Sample Result", OpenSampleResult),
        ]),
        Menu::new("View").items([
            MenuItem::action("Command Palette", OpenPalette),
            MenuItem::separator(),
            MenuItem::action("Show or Hide Sidebar", HideShowSidebar),
            MenuItem::action("Presentation Mode", EnterPresentation),
        ]),
        Menu::new("Help").items([
            MenuItem::action("Keyboard Shortcuts", ShowShortcuts),
            MenuItem::action("Ember on GitHub", OpenRepository),
        ]),
    ]
}

/// Route menu commands to the console. Registered per window, holding only a
/// weak handle, so a closed window never keeps the console alive.
pub(super) fn register_menu_actions(
    console: WeakEntity<Console>,
    window: AnyWindowHandle,
    cx: &mut App,
) {
    fn to_console(
        console: &WeakEntity<Console>,
        cx: &mut App,
        f: impl FnOnce(&mut Console, &mut Context<Console>),
    ) {
        let _ = console.update(cx, f);
    }
    macro_rules! route {
        ($action:ty, |$c:ident, $cx:ident| $body:expr) => {{
            let console = console.clone();
            cx.on_action(move |_: &$action, cx: &mut App| {
                to_console(&console, cx, |$c, $cx| $body);
            });
        }};
    }
    route!(OpenSettings, |c, cx| c.goto(View::Settings, cx));
    route!(ShowShortcuts, |c, cx| c.goto(View::Settings, cx));
    route!(HideShowSidebar, |c, cx| c.toggle_sidebar(cx));
    route!(EnterPresentation, |c, cx| c.toggle_presentation(cx));
    route!(OpenSampleResult, |c, cx| c.show_sample(cx));
    route!(StartExperiment, |c, cx| {
        c.goto(View::Experiment, cx);
        c.step = WorkspaceStep::Prompt;
        cx.notify();
    });
    route!(ReplayLastRun, |c, cx| {
        if c.result_context.is_some() {
            c.goto(View::Experiment, cx);
            c.step = WorkspaceStep::Review;
            c.rerun(cx);
            cx.notify();
        }
    });
    cx.on_action(|_: &OpenRepository, cx: &mut App| cx.open_url(REPOSITORY_URL));
    // The palette needs the window to focus its field. A menu action runs
    // while that window is already being updated, and a nested update is
    // refused, so the work is deferred to just after the dispatch.
    let palette_console = console.clone();
    cx.on_action(move |_: &OpenPalette, cx: &mut App| {
        let console = palette_console.clone();
        cx.defer(move |cx| {
            let _ = window.update(cx, |_, window, cx| {
                let _ = console.update(cx, |console, cx| console.toggle_palette(window, cx));
            });
        });
    });
}

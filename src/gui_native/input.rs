//! Domain adapter for GPUI Kit input controls. Editing, IME, selection,
//! clipboard, undo, accessibility and keyboard navigation belong to the kit.
use super::theme::Colors;
use gpui_kit::component::input::{
    Input, InputEvent as KitEvent, InputState, Textarea, TextareaState,
};
use gpui_kit::{prelude::*, *};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputId {
    ModelPath,
    Layer,
    Value,
    SourceLayer,
    Span,
    MaxTokens,
    Prompt,
    PaletteQuery,
    /// The spec view's TOML editor.
    Spec,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputKind {
    Text,
    Multiline,
    /// Multi-line, in the monospace face: the spec view's TOML.
    Code,
    Integer,
    Decimal,
}
#[derive(Debug, Clone)]
pub(super) struct InputEvent {
    pub id: InputId,
    pub value: String,
}

enum Control {
    Line(Entity<InputState>),
    Paragraph(Entity<TextareaState>),
}
pub(super) struct TextInput {
    control: Control,
    // Programmatic changes can arrive from worker replies without a Window.
    // Apply them at the next render; Kit set_value deliberately emits no Change.
    pending: Option<SharedString>,
    pending_placeholder: Option<&'static str>,
    /// Drawn in the monospace face (`InputKind::Code`).
    code: bool,
}
impl EventEmitter<InputEvent> for TextInput {}
impl TextInput {
    pub(super) fn new(
        id: InputId,
        kind: InputKind,
        value: String,
        placeholder: &'static str,
        _colors: &Colors,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let control = if matches!(kind, InputKind::Multiline | InputKind::Code) {
            let state = cx.new(|cx| {
                let mut state = TextareaState::new(window, cx).placeholder(placeholder);
                state.set_value(value, window, cx);
                state
            });
            cx.subscribe(&state, move |_, state, event, cx| {
                if matches!(event, KitEvent::Change) {
                    cx.emit(InputEvent {
                        id,
                        value: state.read(cx).value().to_string(),
                    });
                }
            })
            .detach();
            Control::Paragraph(state)
        } else {
            let state = cx.new(|cx| {
                let mut state = InputState::new(window, cx).placeholder(placeholder);
                state.set_value(value, window, cx);
                state
            });
            cx.subscribe(&state, move |_, state, event, cx| {
                if matches!(event, KitEvent::Change) {
                    cx.emit(InputEvent {
                        id,
                        value: state.read(cx).value().to_string(),
                    });
                }
            })
            .detach();
            Control::Line(state)
        };
        Self {
            control,
            pending: None,
            pending_placeholder: None,
            code: kind == InputKind::Code,
        }
    }
    #[cfg(all(test, feature = "gui-tests"))]
    pub(super) fn focus_for_test(&self, window: &mut Window, cx: &mut App) {
        self.focus(window, cx);
    }

    /// Give this control keyboard focus. Used when the command palette opens
    /// so typing lands in its query field immediately.
    pub(super) fn focus(&self, window: &mut Window, cx: &mut App) {
        let handle = match &self.control {
            Control::Line(state) => state.read(cx).focus_handle(cx),
            Control::Paragraph(state) => state.read(cx).focus_handle(cx),
        };
        handle.focus(window, cx);
    }

    /// The text the control holds: the value set for the next render, if
    /// there is one, else the text on screen.
    pub(super) fn value(&self, cx: &App) -> String {
        if let Some(value) = &self.pending {
            return value.to_string();
        }
        match &self.control {
            Control::Line(state) => state.read(cx).value().to_string(),
            Control::Paragraph(state) => state.read(cx).value().to_string(),
        }
    }

    pub(super) fn set_value(&mut self, value: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.pending = Some(value.into());
        cx.notify();
    }

    /// Replace the hint shown while the field is empty. Applied at the next
    /// render, as [`Self::set_value`] is, so it needs no Window.
    pub(super) fn set_placeholder(&mut self, placeholder: &'static str, cx: &mut Context<Self>) {
        self.pending_placeholder = Some(placeholder);
        cx.notify();
    }
}
impl Render for TextInput {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        match &self.control {
            Control::Line(state) => {
                if let Some(value) = self.pending.take() {
                    state.update(cx, |state, cx| state.set_value(value, window, cx));
                }
                if let Some(placeholder) = self.pending_placeholder.take() {
                    state.update(cx, |state, cx| {
                        state.set_placeholder(placeholder, window, cx)
                    });
                }
                Input::new(state).w_full().into_any_element()
            }
            Control::Paragraph(state) => {
                if let Some(value) = self.pending.take() {
                    state.update(cx, |state, cx| state.set_value(value, window, cx));
                }
                if let Some(placeholder) = self.pending_placeholder.take() {
                    state.update(cx, |state, cx| {
                        state.set_placeholder(placeholder, window, cx)
                    });
                }
                let (font, size, line) = if self.code {
                    (super::FONT_MONO_NAME, 13.0, 20.0)
                } else {
                    (super::FONT_ARABIC_NAME, 18.0, 30.0)
                };
                Textarea::new(state)
                    .h_full()
                    .w_full()
                    .font_family(font)
                    .text_size(px(super::theme::scaled(size)))
                    .line_height(px(super::theme::scaled(line)))
                    .into_any_element()
            }
        }
    }
}

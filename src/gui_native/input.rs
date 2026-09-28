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
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputKind {
    Text,
    Multiline,
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
        let control = if kind == InputKind::Multiline {
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
        }
    }
    #[cfg(all(test, feature = "gui-tests"))]
    pub(super) fn focus_for_test(&self, window: &mut Window, cx: &mut App) {
        let handle = match &self.control {
            Control::Line(state) => state.read(cx).focus_handle(cx),
            Control::Paragraph(state) => state.read(cx).focus_handle(cx),
        };
        handle.focus(window, cx);
    }

    pub(super) fn set_value(&mut self, value: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.pending = Some(value.into());
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
                Input::new(state).w_full().into_any_element()
            }
            Control::Paragraph(state) => {
                if let Some(value) = self.pending.take() {
                    state.update(cx, |state, cx| state.set_value(value, window, cx));
                }
                Textarea::new(state)
                    .h_full()
                    .w_full()
                    .font_family(super::FONT_ARABIC_NAME)
                    .text_size(px(15.0))
                    .line_height(px(28.0))
                    .into_any_element()
            }
        }
    }
}

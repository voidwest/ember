//! Small, reusable presentation primitives for the native console.

use super::input::TextInput;
use super::theme::Colors;
use gpui_kit::component::{
    button::{Button, ButtonVariants},
    Disableable, Icon, Sizable,
};
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::time::Duration;

pub(super) fn label(content: impl Into<SharedString>, size: f32, color: Rgba) -> Div {
    div()
        .child(content.into())
        .text_size(px(size))
        .text_color(color)
}

pub(super) fn mono(content: impl Into<SharedString>, size: f32, color: Rgba) -> Div {
    div()
        .child(content.into())
        .font_family(super::FONT_MONO_NAME)
        .text_size(px(size))
        .text_color(color)
}

pub(super) fn multiline(content: &str, size: f32, color: Rgba, font: &'static str) -> Div {
    div().flex_col().children(
        content
            .split('\n')
            .map(|line| {
                div()
                    .child(line.to_string())
                    .font_family(font)
                    .text_size(px(size))
                    .line_height(px(size * 1.8))
                    .text_color(color)
                    .into_any_element()
            })
            .collect::<Vec<_>>(),
    )
}

pub(super) fn section_label(colors: &Colors, label_text: &'static str) -> Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .w(px(2.0))
                .h(px(10.0))
                .bg(colors.border_strong)
                .rounded_full(),
        )
        .child(label(label_text, 10.0, colors.text_faint))
}

pub(super) fn field(colors: &Colors, title: &'static str, control: impl IntoElement) -> Div {
    div()
        .flex_col()
        .gap_1()
        .w_full()
        .child(label(title, 10.0, colors.text_faint))
        .child(control)
}

pub(super) fn panel(colors: &Colors, content: impl IntoElement) -> Div {
    div()
        .w_full()
        .p_4()
        .bg(colors.surface)
        .rounded(px(10.0))
        .child(content)
}

pub(super) fn rule_h(colors: &Colors) -> Div {
    div().w_full().h(px(1.0)).bg(colors.border)
}

pub(super) fn chip(label_text: &str, color: Rgba) -> Div {
    let hsla = Hsla::from(color);
    div()
        .px_2()
        .py_1()
        .bg(hsla.opacity(0.13))
        .border_1()
        .border_color(hsla.opacity(0.40))
        .rounded_full()
        .child(label(label_text.to_string(), 10.0, color))
}

pub(super) fn status_dot(color: Rgba, busy: bool) -> AnyElement {
    let hsla = Hsla::from(color);
    let dot = div()
        .size(px(8.0))
        .bg(color)
        .border_1()
        .border_color(hsla.opacity(0.45))
        .rounded_full();
    if busy {
        dot.with_animation(
            "busy-status-pulse",
            Animation::new(Duration::from_millis(900)).repeat(),
            |dot, delta| dot.opacity(0.45 + 0.55 * (1.0 - (2.0 * delta - 1.0).abs())),
        )
        .into_any_element()
    } else {
        dot.into_any_element()
    }
}

pub(super) fn icon_button(
    _colors: &Colors,
    icon_path: &'static str,
    accessible_label: &'static str,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> Button {
    Button::new(SharedString::from(format!(
        "icon-button:{accessible_label}"
    )))
    .ghost()
    .small()
    .icon(Icon::default().path(icon_path))
    .tooltip(accessible_label)
    .on_click(on_click)
}

pub(super) fn btn_primary(
    _colors: &Colors,
    icon_path: &'static str,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Button {
    let button = Button::new(SharedString::from(format!("btn:{label_text}")))
        .primary()
        .w_full()
        .icon(Icon::default().path(icon_path))
        .label(label_text.to_string())
        .disabled(on_click.is_none());
    match on_click {
        Some(callback) => button.on_click(callback),
        None => button,
    }
}

pub(super) fn btn_secondary(
    _colors: &Colors,
    icon_path: &'static str,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Button {
    let button = Button::new(SharedString::from(format!("btn:{label_text}")))
        .w_full()
        .icon(Icon::default().path(icon_path))
        .label(label_text.to_string())
        .disabled(on_click.is_none());
    match on_click {
        Some(callback) => button.on_click(callback),
        None => button,
    }
}

pub(super) fn text_input(
    colors: &Colors,
    input: Entity<TextInput>,
    font: &'static str,
    size: f32,
    height: Option<f32>,
    cx: &App,
) -> Div {
    let _ = (colors, cx);
    div()
        .w_full()
        .h(px(height.unwrap_or(32.0)))
        .font_family(font)
        .text_size(px(size))
        .child(input)
}

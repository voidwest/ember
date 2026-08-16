//! Small, reusable presentation primitives for the native console.

use super::icons;
use super::input::TextInput;
use super::theme::Colors;
use gpui::prelude::*;
use gpui::*;
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
    colors: &Colors,
    icon_path: &'static str,
    accessible_label: &'static str,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    div()
        .id(ElementId::Name(SharedString::from(format!(
            "icon-button:{accessible_label}"
        ))))
        .size(px(26.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded_md()
        .cursor_pointer()
        .hover(|style| style.bg(colors.hover))
        .active(|style| style.bg(colors.selected))
        .on_click(on_click)
        .child(
            icons::icon(icon_path)
                .size(px(14.0))
                .text_color(colors.text_muted),
        )
}

pub(super) fn btn_primary(
    colors: &Colors,
    icon_path: &'static str,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Stateful<Div> {
    let enabled = on_click.is_some();
    let mut button = div()
        .id(ElementId::Name(SharedString::from(format!(
            "btn:{label_text}"
        ))))
        .w_full()
        .h(px(32.0))
        .px_4()
        .flex()
        .items_center()
        .justify_center()
        .gap_2()
        .bg(if enabled {
            colors.accent
        } else {
            colors.accent_soft
        })
        .rounded_md()
        .when(enabled, |button| {
            button
                .cursor_pointer()
                .hover(|style| style.bg(colors.accent_hover))
                .active(|style| style.bg(colors.accent_pressed))
        })
        .child(
            icons::icon(icon_path)
                .size(px(14.0))
                .text_color(if enabled {
                    rgb(0xffffff)
                } else {
                    colors.text_faint
                }),
        )
        .child(label(label_text.to_string(), 12.0, rgb(0xffffff)));
    if let Some(on_click) = on_click {
        button = button.on_click(on_click);
    }
    button
}

pub(super) fn btn_secondary(
    colors: &Colors,
    icon_path: &'static str,
    label_text: &str,
    on_click: Option<impl Fn(&ClickEvent, &mut Window, &mut App) + 'static>,
) -> Stateful<Div> {
    let enabled = on_click.is_some();
    let mut button = div()
        .id(ElementId::Name(SharedString::from(format!(
            "btn:{label_text}"
        ))))
        .w_full()
        .h(px(32.0))
        .px_4()
        .flex()
        .items_center()
        .justify_center()
        .gap_2()
        .bg(colors.surface)
        .border_1()
        .border_color(colors.border)
        .rounded_md()
        .when(enabled, |button| {
            button
                .cursor_pointer()
                .hover(|style| style.bg(colors.surface_raised).border_color(colors.accent))
        })
        .when(!enabled, |button| button.opacity(0.55))
        .child(
            icons::icon(icon_path)
                .size(px(14.0))
                .text_color(colors.text_muted),
        )
        .child(label(label_text.to_string(), 12.0, colors.text));
    if let Some(on_click) = on_click {
        button = button.on_click(on_click);
    }
    button
}

pub(super) fn text_input(
    colors: &Colors,
    input: Entity<TextInput>,
    font: &'static str,
    size: f32,
    height: Option<f32>,
    cx: &App,
) -> Div {
    let focus = input.read(cx).handle();
    let mut field = div()
        .w_full()
        .px_2()
        .py_1()
        .bg(colors.surface_raised)
        .border_1()
        .border_color(colors.border)
        .rounded_md()
        .track_focus(&focus)
        .focus(|style| style.border_color(colors.focus_ring).bg(colors.surface))
        .cursor_text()
        .font_family(font)
        .text_size(px(size))
        .line_height(px(size + 7.0))
        .text_color(colors.text)
        .child(input);
    if let Some(height) = height {
        field = field.h(px(height)).overflow_hidden();
    } else {
        field = field.h(px(32.0));
    }
    field
}

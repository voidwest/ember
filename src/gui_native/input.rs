//! Reusable native text input with selection, clipboard, undo/redo and IME.
//!
//! This follows gpui's official `examples/input.rs` element/input-handler seam,
//! extended for multi-line prompts and Ember's semantic field identities.

use super::theme::Colors;
use gpui::{
    actions, div, fill, point, prelude::*, px, relative, size, App, Bounds, ClipboardItem, Context,
    CursorStyle, Element, ElementId, ElementInputHandler, Entity, EntityInputHandler, EventEmitter,
    FocusHandle, Focusable, GlobalElementId, InspectorElementId, IntoElement, KeyBinding, LayoutId,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PaintQuad, Pixels, Point,
    ShapedLine, SharedString, Style, TextRun, UTF16Selection, UnderlineStyle, Window,
};
use std::ops::Range;
use unicode_segmentation::UnicodeSegmentation;

actions!(
    ember_text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        Paste,
        Cut,
        Copy,
        Undo,
        Redo,
        Newline,
    ]
);

pub(super) fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, Some("EmberTextInput")),
        KeyBinding::new("delete", Delete, Some("EmberTextInput")),
        KeyBinding::new("left", Left, Some("EmberTextInput")),
        KeyBinding::new("right", Right, Some("EmberTextInput")),
        KeyBinding::new("shift-left", SelectLeft, Some("EmberTextInput")),
        KeyBinding::new("shift-right", SelectRight, Some("EmberTextInput")),
        KeyBinding::new("cmd-a", SelectAll, Some("EmberTextInput")),
        KeyBinding::new("ctrl-a", SelectAll, Some("EmberTextInput")),
        KeyBinding::new("cmd-v", Paste, Some("EmberTextInput")),
        KeyBinding::new("ctrl-v", Paste, Some("EmberTextInput")),
        KeyBinding::new("cmd-c", Copy, Some("EmberTextInput")),
        KeyBinding::new("ctrl-c", Copy, Some("EmberTextInput")),
        KeyBinding::new("cmd-x", Cut, Some("EmberTextInput")),
        KeyBinding::new("ctrl-x", Cut, Some("EmberTextInput")),
        KeyBinding::new("cmd-z", Undo, Some("EmberTextInput")),
        KeyBinding::new("ctrl-z", Undo, Some("EmberTextInput")),
        KeyBinding::new("cmd-shift-z", Redo, Some("EmberTextInput")),
        KeyBinding::new("ctrl-shift-z", Redo, Some("EmberTextInput")),
        KeyBinding::new("home", Home, Some("EmberTextInput")),
        KeyBinding::new("end", End, Some("EmberTextInput")),
        KeyBinding::new("enter", Newline, Some("EmberTextInput")),
    ]);
}

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

#[derive(Clone)]
struct Snapshot {
    content: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
}

#[derive(Clone)]
struct LineLayout {
    range: Range<usize>,
    shaped: ShapedLine,
    bounds: Bounds<Pixels>,
}

fn utf16_to_utf8(content: &str, offset: usize) -> usize {
    let mut utf8 = 0;
    let mut utf16 = 0;
    for ch in content.chars() {
        if utf16 >= offset {
            break;
        }
        utf16 += ch.len_utf16();
        utf8 += ch.len_utf8();
    }
    utf8
}

fn utf8_to_utf16(content: &str, offset: usize) -> usize {
    let mut utf8 = 0;
    let mut utf16 = 0;
    for ch in content.chars() {
        if utf8 >= offset {
            break;
        }
        utf8 += ch.len_utf8();
        utf16 += ch.len_utf16();
    }
    utf16
}

fn previous_grapheme_boundary(content: &str, offset: usize) -> usize {
    content
        .grapheme_indices(true)
        .rev()
        .find_map(|(index, _)| (index < offset).then_some(index))
        .unwrap_or(0)
}

fn next_grapheme_boundary(content: &str, offset: usize) -> usize {
    content
        .grapheme_indices(true)
        .find_map(|(index, _)| (index > offset).then_some(index))
        .unwrap_or(content.len())
}

fn sanitize_text(kind: InputKind, text: &str) -> String {
    match kind {
        InputKind::Text => text.replace(['\r', '\n'], " "),
        InputKind::Multiline => text.replace("\r\n", "\n").replace('\r', "\n"),
        InputKind::Integer => text.chars().filter(char::is_ascii_digit).collect(),
        InputKind::Decimal => {
            let mut dot = false;
            text.chars()
                .filter(|character| {
                    if character.is_ascii_digit() || *character == '-' {
                        true
                    } else if *character == '.' && !dot {
                        dot = true;
                        true
                    } else {
                        false
                    }
                })
                .collect()
        }
    }
}

pub(super) struct TextInput {
    id: InputId,
    kind: InputKind,
    focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_lines: Vec<LineLayout>,
    last_bounds: Option<Bounds<Pixels>>,
    is_selecting: bool,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    placeholder_color: gpui::Rgba,
    selection_color: gpui::Rgba,
    caret_color: gpui::Rgba,
}

impl TextInput {
    pub(super) fn new(
        id: InputId,
        kind: InputKind,
        value: impl Into<SharedString>,
        placeholder: impl Into<SharedString>,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Self {
        let content = value.into();
        let end = content.len();
        Self {
            id,
            kind,
            focus_handle: cx.focus_handle(),
            content,
            placeholder: placeholder.into(),
            selected_range: end..end,
            selection_reversed: false,
            marked_range: None,
            last_lines: Vec::new(),
            last_bounds: None,
            is_selecting: false,
            undo: Vec::new(),
            redo: Vec::new(),
            placeholder_color: colors.text_faint,
            selection_color: colors.selection,
            caret_color: colors.caret,
        }
    }

    pub(super) fn handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    pub(super) fn set_value(&mut self, value: impl Into<SharedString>, cx: &mut Context<Self>) {
        let value = value.into();
        if self.content == value {
            return;
        }
        self.content = value;
        let end = self.content.len();
        self.selected_range = end..end;
        self.selection_reversed = false;
        self.marked_range = None;
        self.undo.clear();
        self.redo.clear();
        cx.notify();
    }

    pub(super) fn set_palette(&mut self, colors: &Colors, cx: &mut Context<Self>) {
        self.placeholder_color = colors.text_faint;
        self.selection_color = colors.selection;
        self.caret_color = colors.caret;
        cx.notify();
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            content: self.content.clone(),
            selected_range: self.selected_range.clone(),
            selection_reversed: self.selection_reversed,
        }
    }

    fn restore(&mut self, snapshot: Snapshot, cx: &mut Context<Self>) {
        self.content = snapshot.content;
        self.selected_range = snapshot.selected_range;
        self.selection_reversed = snapshot.selection_reversed;
        self.marked_range = None;
        self.emit_changed(cx);
    }

    fn record_edit(&mut self) {
        self.undo.push(self.snapshot());
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn undo(&mut self, _: &Undo, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(snapshot) = self.undo.pop() {
            self.redo.push(self.snapshot());
            self.restore(snapshot, cx);
        }
    }

    fn redo(&mut self, _: &Redo, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(snapshot) = self.redo.pop() {
            self.undo.push(self.snapshot());
            self.restore(snapshot, cx);
        }
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        let offset = if self.selected_range.is_empty() {
            self.previous_boundary(self.cursor_offset())
        } else {
            self.selected_range.start
        };
        self.move_to(offset, cx);
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        let offset = if self.selected_range.is_empty() {
            self.next_boundary(self.cursor_offset())
        } else {
            self.selected_range.end
        };
        self.move_to(offset, cx);
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.selected_range = 0..self.content.len();
        self.selection_reversed = false;
        cx.notify();
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        let cursor = self.cursor_offset();
        let start = self.content[..cursor].rfind('\n').map_or(0, |ix| ix + 1);
        self.move_to(start, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        let cursor = self.cursor_offset();
        let end = self.content[cursor..]
            .find('\n')
            .map_or(self.content.len(), |ix| cursor + ix);
        self.move_to(end, cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn newline(&mut self, _: &Newline, window: &mut Window, cx: &mut Context<Self>) {
        if self.kind == InputKind::Multiline {
            self.replace_text_in_range(None, "\n", window, cx);
        }
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.focus_handle.focus(window);
        self.is_selecting = true;
        let index = self.index_for_mouse_position(event.position);
        if event.modifiers.shift {
            self.select_to(index, cx);
        } else {
            self.move_to(index, cx);
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        }
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            let text = if self.kind == InputKind::Multiline {
                text
            } else {
                text.replace(['\r', '\n'], " ")
            };
            self.replace_text_in_range(None, &text, window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            self.copy(&Copy, window, cx);
            self.replace_text_in_range(None, "", window, cx);
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify();
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        let Some(line) = self
            .last_lines
            .iter()
            .find(|line| position.y >= line.bounds.top() && position.y <= line.bounds.bottom())
        else {
            return if self
                .last_bounds
                .is_some_and(|bounds| position.y < bounds.top())
            {
                0
            } else {
                self.content.len()
            };
        };
        line.range.start
            + line
                .shaped
                .closest_index_for_x(position.x - line.bounds.left())
                .min(line.range.len())
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset;
        } else {
            self.selected_range.end = offset;
        }
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify();
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        utf16_to_utf8(&self.content, offset)
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        utf8_to_utf16(&self.content, offset)
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range.start)..self.offset_from_utf16(range.end)
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        previous_grapheme_boundary(&self.content, offset)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        next_grapheme_boundary(&self.content, offset)
    }

    fn sanitize(&self, text: &str) -> String {
        sanitize_text(self.kind, text)
    }

    fn emit_changed(&self, cx: &mut Context<Self>) {
        cx.emit(InputEvent {
            id: self.id,
            value: self.content.to_string(),
        });
        cx.notify();
    }
}

impl EventEmitter<InputEvent> for TextInput {}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        range: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.content.get(range)?.to_string())
    }

    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        new_text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let new_text = self.sanitize(new_text);
        if self.marked_range.is_none() {
            self.record_edit();
        }
        self.content =
            (self.content[..range.start].to_owned() + &new_text + &self.content[range.end..])
                .into();
        let cursor = range.start + new_text.len();
        self.selected_range = cursor..cursor;
        self.selection_reversed = false;
        self.marked_range = None;
        self.emit_changed(cx);
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        if self.marked_range.is_none() {
            self.record_edit();
        }
        let new_text = self.sanitize(new_text);
        self.content =
            (self.content[..range.start].to_owned() + &new_text + &self.content[range.end..])
                .into();
        self.marked_range =
            (!new_text.is_empty()).then_some(range.start..range.start + new_text.len());
        self.selected_range = new_selected_range
            .as_ref()
            .map(|selection| self.range_from_utf16(selection))
            .map(|selection| range.start + selection.start..range.start + selection.end)
            .unwrap_or_else(|| range.start + new_text.len()..range.start + new_text.len());
        self.emit_changed(cx);
    }

    fn bounds_for_range(
        &mut self,
        range: Range<usize>,
        _: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let range = self.range_from_utf16(&range);
        let line = self
            .last_lines
            .iter()
            .find(|line| range.start >= line.range.start && range.start <= line.range.end)?;
        Some(Bounds::from_corners(
            point(
                line.bounds.left() + line.shaped.x_for_index(range.start - line.range.start),
                line.bounds.top(),
            ),
            point(
                line.bounds.left()
                    + line
                        .shaped
                        .x_for_index(range.end.min(line.range.end) - line.range.start),
                line.bounds.bottom(),
            ),
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        Some(self.offset_to_utf16(self.index_for_mouse_position(point)))
    }
}

struct TextElement {
    input: Entity<TextInput>,
}

struct PrepaintState {
    lines: Vec<LineLayout>,
    cursor: Option<PaintQuad>,
    selections: Vec<PaintQuad>,
}

impl IntoElement for TextElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextElement {
    type RequestLayoutState = ();
    type PrepaintState = PrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.0).into();
        style.size.height = relative(1.0).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> PrepaintState {
        let input = self.input.read(cx);
        let style = window.text_style();
        let line_height = window.line_height();
        let display: SharedString = if input.content.is_empty() {
            input.placeholder.clone()
        } else {
            input.content.clone()
        };
        let text_color = if input.content.is_empty() {
            input.placeholder_color
        } else {
            style.color.into()
        };
        let font_size = style.font_size.to_pixels(window.rem_size());
        let mut lines = Vec::new();
        let mut start = 0;
        for (line_index, text) in display.split('\n').enumerate() {
            let end = start + text.len();
            let run = TextRun {
                len: text.len(),
                font: style.font(),
                color: text_color.into(),
                background_color: None,
                underline: input.marked_range.as_ref().and_then(|marked| {
                    (marked.start < end && marked.end > start).then_some(UnderlineStyle {
                        color: Some(text_color.into()),
                        thickness: px(1.0),
                        wavy: false,
                    })
                }),
                strikethrough: None,
            };
            let runs = if text.is_empty() {
                Vec::new()
            } else {
                vec![run]
            };
            let shaped = window.text_system().shape_line(
                SharedString::from(text.to_string()),
                font_size,
                &runs,
                None,
            );
            let top = bounds.top() + line_height * line_index;
            lines.push(LineLayout {
                range: start..end,
                shaped,
                bounds: Bounds::new(
                    point(bounds.left(), top),
                    size(bounds.size.width, line_height),
                ),
            });
            start = end + 1;
        }

        let mut selections = Vec::new();
        let mut cursor = None;
        if input.selected_range.is_empty() {
            let caret = input.cursor_offset();
            if let Some(line) = lines
                .iter()
                .find(|line| caret >= line.range.start && caret <= line.range.end)
                .or_else(|| lines.last())
            {
                let local = caret.saturating_sub(line.range.start).min(line.range.len());
                cursor = Some(fill(
                    Bounds::new(
                        point(
                            line.bounds.left() + line.shaped.x_for_index(local),
                            line.bounds.top(),
                        ),
                        size(px(1.5), line.bounds.size.height),
                    ),
                    input.caret_color,
                ));
            }
        } else {
            for line in &lines {
                let start = input.selected_range.start.max(line.range.start);
                let end = input.selected_range.end.min(line.range.end);
                if start < end {
                    selections.push(fill(
                        Bounds::from_corners(
                            point(
                                line.bounds.left()
                                    + line.shaped.x_for_index(start - line.range.start),
                                line.bounds.top(),
                            ),
                            point(
                                line.bounds.left()
                                    + line.shaped.x_for_index(end - line.range.start),
                                line.bounds.bottom(),
                            ),
                        ),
                        input.selection_color,
                    ));
                }
            }
        }
        PrepaintState {
            lines,
            cursor,
            selections,
        }
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        state: &mut PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus = self.input.read(cx).focus_handle.clone();
        window.handle_input(
            &focus,
            ElementInputHandler::new(bounds, self.input.clone()),
            cx,
        );
        for selection in state.selections.drain(..) {
            window.paint_quad(selection);
        }
        for line in &state.lines {
            let _ = line
                .shaped
                .paint(line.bounds.origin, line.bounds.size.height, window, cx);
        }
        if focus.is_focused(window)
            && let Some(cursor) = state.cursor.take()
        {
            window.paint_quad(cursor);
        }
        self.input.update(cx, |input, _| {
            input.last_lines = state.lines.clone();
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("EmberTextInput")
            .track_focus(&self.focus_handle)
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::undo))
            .on_action(cx.listener(Self::redo))
            .on_action(cx.listener(Self::newline))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .size_full()
            .child(TextElement { input: cx.entity() })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        next_grapheme_boundary, previous_grapheme_boundary, sanitize_text, utf16_to_utf8,
        utf8_to_utf16, InputKind,
    };

    #[test]
    fn cursor_boundaries_preserve_grapheme_clusters() {
        let content = "أ🔥";
        assert_eq!(previous_grapheme_boundary(content, content.len()), 4);
        assert_eq!(previous_grapheme_boundary(content, 4), 0);
        assert_eq!(next_grapheme_boundary(content, 0), 4);
        assert_eq!(next_grapheme_boundary(content, 4), content.len());
    }

    #[test]
    fn utf16_offsets_round_trip_arabic_and_astral_text() {
        let content = "أ🔥ب";
        for utf8 in [0, 2, 6, 8] {
            assert_eq!(utf16_to_utf8(content, utf8_to_utf16(content, utf8)), utf8);
        }
    }

    #[test]
    fn field_kinds_sanitize_inserted_text() {
        assert_eq!(sanitize_text(InputKind::Integer, "L12x"), "12");
        assert_eq!(sanitize_text(InputKind::Text, "a\nb"), "a b");
        assert_eq!(sanitize_text(InputKind::Multiline, "a\r\nb"), "a\nb");
    }
}

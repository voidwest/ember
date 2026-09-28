//! Searchable, keyboard-accessible GPUI Kit selectors for domain values.
use super::{combo_value_label, ComboId};
use gpui_kit::component::select::{SearchableVec, Select, SelectEvent, SelectItem, SelectState};
use gpui_kit::{prelude::*, *};

#[derive(Clone)]
struct Item {
    value: String,
    title: SharedString,
}
impl SelectItem for Item {
    type Value = String;
    fn title(&self) -> SharedString {
        self.title.clone()
    }
    fn render(&self, _: &mut Window, _: &mut App) -> impl IntoElement {
        use gpui_kit::TestSupportExt;
        div()
            .id(SharedString::from(format!("choice:{}", self.value)))
            .test_support()
            .aria_label(self.title.clone())
            .child(self.title.clone())
    }
    fn value(&self) -> &String {
        &self.value
    }
    fn matches(&self, query: &str) -> bool {
        self.title.to_lowercase().contains(&query.to_lowercase())
            || self.value.to_lowercase().contains(&query.to_lowercase())
    }
}
pub(super) struct Picked(pub String);
pub(super) struct Picker {
    combo: ComboId,
    state: Entity<SelectState<SearchableVec<Item>>>,
    options: Vec<String>,
    selected: String,
    pending: bool,
}
impl EventEmitter<Picked> for Picker {}
impl Picker {
    pub(super) fn new(
        combo: ComboId,
        options: Vec<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let state = cx.new(|cx| {
            SelectState::new(
                SearchableVec::from(
                    options
                        .iter()
                        .map(|value| Item {
                            value: value.clone(),
                            title: combo_value_label(combo, value).into(),
                        })
                        .collect::<Vec<_>>(),
                ),
                None,
                window,
                cx,
            )
            .searchable(true)
        });
        cx.subscribe(&state, |this, _, event, cx| {
            if let SelectEvent::Confirm(Some(value)) = event {
                this.selected = value.clone();
                cx.emit(Picked(value.clone()));
            }
        })
        .detach();
        Self {
            combo,
            state,
            options,
            selected: String::new(),
            pending: false,
        }
    }
    pub(super) fn sync(&mut self, options: &[String], selected: &str) {
        if self.options != options || self.selected != selected {
            self.options = options.to_vec();
            self.selected = selected.to_string();
            self.pending = true;
        }
    }
}
impl Render for Picker {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.pending {
            let items = self
                .options
                .iter()
                .map(|value| Item {
                    value: value.clone(),
                    title: combo_value_label(self.combo, value).into(),
                })
                .collect::<Vec<_>>();
            self.state.update(cx, |state, cx| {
                state.set_items(items.into(), window, cx);
                state.set_selected_value(&self.selected, window, cx);
            });
            self.pending = false;
        }
        Select::new(&self.state)
            .id(SharedString::from(format!("picker:{:?}", self.combo)))
            .w_full()
    }
}

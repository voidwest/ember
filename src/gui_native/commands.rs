use super::*;

impl Console {
    // -- command palette -----------------------------------------------------

    /// Open or close the command list.
    pub(super) fn toggle_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.open_palette(PaletteMode::Commands, window, cx);
    }

    /// Open the palette in `mode`, or close it when it is already open in
    /// that mode. Opening clears the previous query and puts focus in the
    /// field, so Cmd+K, type, Enter just works; the other mode's key switches
    /// an open palette over instead of closing it.
    pub(super) fn open_palette(
        &mut self,
        mode: PaletteMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.palette_open = !(self.palette_open && self.palette_mode == mode);
        if self.palette_open {
            self.palette_mode = mode;
            self.palette_query.clear();
            self.palette_index = 0;
            self.palette_input.update(cx, |input, cx| {
                input.set_value("", cx);
                input.set_placeholder(mode.placeholder(), cx);
                input.focus(window, cx);
            });
        } else {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    /// Whether the palette lists saved runs: Go to run, until the query
    /// starts with `>`.
    pub(super) fn palette_lists_runs(&self) -> bool {
        self.palette_mode == PaletteMode::Runs && !self.palette_query.trim_start().starts_with('>')
    }

    /// The palette's command list: the catalog ranked by a fuzzy match on the
    /// label, then any command whose hint holds the query, in catalog order.
    /// A leading `>` is the editor's command prefix and is not matched.
    pub(super) fn palette_candidates(&self) -> Vec<Command> {
        let query = self.palette_query.trim().trim_start_matches('>').trim();
        if query.is_empty() {
            return Command::ALL.to_vec();
        }
        let lowered = query.to_lowercase();
        let mut scored: Vec<(u32, Command)> = Command::ALL
            .into_iter()
            .filter_map(|command| {
                palette::fuzzy_score(query, command.label())
                    .or_else(|| {
                        command
                            .hint()
                            .to_lowercase()
                            .contains(&lowered)
                            .then_some(0)
                    })
                    .map(|score| (score, command))
            })
            .collect();
        // Stable, so equal scores keep the catalog order.
        scored.sort_by_key(|&(score, _)| std::cmp::Reverse(score));
        scored.into_iter().map(|(_, command)| command).collect()
    }

    /// Go to run's list: the saved runs that can be reopened, ranked by
    /// [`palette::run_score`], newest first among equals. At most
    /// [`GO_TO_RUN_LIMIT`] rows, as the Runs table also draws only a window
    /// of a long history; the query narrows the rest.
    pub(super) fn run_candidates(&self) -> Vec<u64> {
        let query = self.palette_query.trim();
        let mut scored: Vec<(u32, u64)> = self
            .store
            .runs
            .iter()
            .filter(|run| run.result.is_some() && run.config.is_some())
            .filter_map(|run| {
                let prompt = run.prompt.split_whitespace().collect::<Vec<_>>().join(" ");
                let detail = format!("{} {}", run.intervention, run.model);
                palette::run_score(query, run.number, &prompt, &detail)
                    .map(|score| (score, run.number))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        scored.truncate(GO_TO_RUN_LIMIT);
        scored.into_iter().map(|(_, number)| number).collect()
    }

    pub(super) fn palette_move(&mut self, delta: isize, cx: &mut Context<Self>) {
        let count = if self.palette_lists_runs() {
            self.run_candidates().len()
        } else {
            self.palette_candidates().len()
        };
        if count == 0 {
            return;
        }
        let index = self.palette_index as isize + delta;
        self.palette_index = index.rem_euclid(count as isize) as usize;
        cx.notify();
    }

    pub(super) fn palette_execute(&mut self, cx: &mut Context<Self>) {
        if self.palette_lists_runs() {
            if let Some(number) = self.run_candidates().get(self.palette_index).copied() {
                self.palette_open = false;
                self.open_run(number, cx);
            }
            return;
        }
        let candidates = self.palette_candidates();
        let Some(command) = candidates.get(self.palette_index).copied() else {
            return;
        };
        self.palette_open = false;
        match command {
            Command::NewExperiment => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Prompt;
            }
            Command::LoadModel => {
                self.goto(View::Experiment, cx);
                self.load();
            }
            Command::RunExperiment => {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Review;
                self.run();
            }
            Command::DuplicateExperiment => self.duplicate_experiment(cx),
            Command::RerunExperiment => {
                if self.result_context.is_some() {
                    self.goto(View::Experiment, cx);
                    self.step = WorkspaceStep::Review;
                    self.rerun(cx);
                }
            }
            Command::GoHome => self.goto(View::Home, cx),
            Command::GoExperiments => self.goto(View::Experiment, cx),
            Command::GoModels => self.goto(View::Models, cx),
            Command::GoRuns => self.goto(View::Runs, cx),
            Command::GoSettings | Command::KeyboardShortcuts => self.goto(View::Settings, cx),
            Command::GoToRun => {
                // The field keeps focus, so the palette stays open and
                // switches lists, as Cmd+P would.
                self.palette_open = true;
                self.palette_mode = PaletteMode::Runs;
                self.palette_query.clear();
                self.palette_index = 0;
                self.palette_input.update(cx, |input, cx| {
                    input.set_value("", cx);
                    input.set_placeholder(PaletteMode::Runs.placeholder(), cx);
                });
            }
            Command::ShowOverview => self.show_result_view(ResultView::Overview, cx),
            Command::ShowLayers => self.show_result_view(ResultView::Layers, cx),
            Command::ShowTokens => self.show_result_view(ResultView::Tokens, cx),
            Command::ShowTrace => self.show_result_view(ResultView::Trace, cx),
            Command::ToggleSpec => {
                // From another page, the command shows the view.
                if self.view != View::Experiment {
                    self.goto(View::Experiment, cx);
                    if !self.spec_open {
                        self.toggle_spec(cx);
                    }
                } else {
                    self.toggle_spec(cx);
                }
            }
            Command::ToggleSidebar => self.toggle_sidebar(cx),
            Command::ToggleTheme => self.cycle_appearance(cx),
            Command::TogglePresentation => self.toggle_presentation(cx),
            Command::CancelRun => self.cancel_run(cx),
        }
        cx.notify();
    }

    /// The result tabs on offer: the four views, and Sweep once a sweep has
    /// finished.
    pub(super) fn result_views(&self) -> Vec<ResultView> {
        let mut views = ResultView::ALL.to_vec();
        if self.sweep.as_ref().is_some_and(|sweep| sweep.finished) {
            views.push(ResultView::Sweep);
        }
        views
    }

    /// Show one result tab, when there is a result to show it for.
    pub(super) fn show_result_view(&mut self, view: ResultView, cx: &mut Context<Self>) {
        if self.comparison.is_none() || !self.result_views().contains(&view) {
            return;
        }
        if self.view != View::Experiment {
            self.goto(View::Experiment, cx);
        }
        self.result_view = view;
        cx.notify();
    }

    /// The next result tab, or with `back` the previous one, wrapping round.
    pub(super) fn cycle_result_view(&mut self, back: bool, cx: &mut Context<Self>) {
        let views = self.result_views();
        let current = views
            .iter()
            .position(|view| *view == self.result_view)
            .unwrap_or(0);
        let step = if back { views.len() - 1 } else { 1 };
        self.result_view = views[(current + step) % views.len()];
        cx.notify();
    }

    pub(super) fn system_appearance_changed(&mut self, dark: bool, cx: &mut Context<Self>) {
        if self.system_dark == dark {
            return;
        }
        self.system_dark = dark;
        if self.appearance == AppearanceMode::System {
            self.sync_kit_theme(cx);
            cx.notify();
        }
    }

    pub(super) fn input_changed(&mut self, event: &InputEvent, cx: &mut Context<Self>) {
        match event.id {
            InputId::ModelPath => {
                if self.model_path != event.value {
                    self.session = None;
                }
                self.model_path.clone_from(&event.value);
            }
            InputId::Layer => {
                self.layer.clone_from(&event.value);
                self.clamp_source_layer(cx);
            }
            InputId::Value => self.value.clone_from(&event.value),
            InputId::SourceLayer => self.source_layer.clone_from(&event.value),
            InputId::Span => self.span.clone_from(&event.value),
            InputId::MaxTokens => self.max_tokens.clone_from(&event.value),
            InputId::Prompt => self.prompt.clone_from(&event.value),
            InputId::Spec => {
                self.spec_edited(&event.value, cx);
                return;
            }
            InputId::PaletteQuery => {
                // A new query means a new candidate list; the cursor restarts
                // so Enter always runs what the eye is resting on.
                self.palette_query.clone_from(&event.value);
                self.palette_index = 0;
            }
        }
        cx.notify();
    }

    pub(super) fn select_combo(&mut self, combo: ComboId, value: &str, cx: &mut Context<Self>) {
        match combo {
            ComboId::Model => {
                self.model_path = value.to_string();
                self.session = None;
                self.inputs
                    .model
                    .update(cx, |input, cx| input.set_value(value.to_string(), cx));
            }
            ComboId::Site => self.site = value.to_string(),
            ComboId::Op => self.op = value.to_string(),
            ComboId::Source => self.source = value.to_string(),
            ComboId::Token => self.token = value.to_string(),
            ComboId::Execution => self.execution = value.to_string(),
        }
        // Selecting a non-per-layer site drops the layer fields.
        if combo == ComboId::Site && !per_layer(&self.site) {
            self.layer = "0".to_string();
            self.source_layer = "0".to_string();
            self.inputs
                .layer
                .update(cx, |input, cx| input.set_value("0", cx));
            self.inputs
                .source_layer
                .update(cx, |input, cx| input.set_value("0", cx));
        }
        cx.notify();
    }

    /// Keep the source layer at or above the target layer (the capture must
    /// fire before the intervention in the same pass).
    pub(super) fn clamp_source_layer(&mut self, cx: &mut Context<Self>) {
        if let (Ok(target), Ok(source)) =
            (self.layer.parse::<i64>(), self.source_layer.parse::<i64>())
            && source > target
        {
            self.source_layer = target.saturating_sub(1).max(0).to_string();
            self.inputs.source_layer.update(cx, |input, cx| {
                input.set_value(self.source_layer.clone(), cx)
            });
        }
    }

    pub(super) fn set_input_value(
        &mut self,
        input: Entity<TextInput>,
        value: String,
        cx: &mut Context<Self>,
    ) {
        input.update(cx, |input, cx| input.set_value(value, cx));
    }

    pub(super) fn set_max_tokens(&mut self, value: usize, cx: &mut Context<Self>) {
        self.max_tokens = value.to_string();
        self.set_input_value(self.inputs.max_tokens.clone(), self.max_tokens.clone(), cx);
        cx.notify();
    }

    pub(super) fn adjust_layer(&mut self, delta: isize, cx: &mut Context<Self>) {
        let max = self
            .session
            .as_ref()
            .map(|session| session.n_layers.saturating_sub(1))
            .unwrap_or(63);
        let current = self.layer.parse::<usize>().unwrap_or(0);
        let next = current.saturating_add_signed(delta).min(max);
        self.layer = next.to_string();
        self.set_input_value(self.inputs.layer.clone(), self.layer.clone(), cx);
        self.clamp_source_layer(cx);
        cx.notify();
    }

    pub(super) fn apply_preset(&mut self, preset: Preset, cx: &mut Context<Self>) {
        let layers = self.session.as_ref().map(|session| session.n_layers);
        match preset {
            Preset::ZeroMiddle => {
                self.op = "zero".to_string();
                self.site = "after-mlp".to_string();
                self.layer = layers.map_or(8, |count| count / 2).to_string();
            }
            Preset::SilenceEarly => {
                // Found by probing a real 16-layer model: silencing an early
                // layer's output at the last prompt token makes the answer
                // drift completely ("Paris..." becomes talk of fog and quiet
                // streets), where zeroing a middle MLP changes nothing at all.
                // The prompt is set too, so the example reads the same on a
                // machine whose form still holds the Arabic default.
                self.op = "scale".to_string();
                self.site = "after-layer".to_string();
                self.value = "0.0".to_string();
                self.token = "prompt-final".to_string();
                self.prompt = "The capital of France is".to_string();
                // Short: the effect shows within a couple of dozen tokens,
                // and a live demo should not wait on a long generation.
                self.max_tokens = "24".to_string();
                self.layer = layers.map_or(6, |count| count * 3 / 8).to_string();
            }
            Preset::ScaleLate => {
                self.op = "scale".to_string();
                self.site = "after-mlp".to_string();
                self.value = "0.5".to_string();
                self.layer = layers
                    .map_or(14, |count| count.saturating_sub(2))
                    .to_string();
            }
            Preset::CopyEarlier => {
                self.op = "replace".to_string();
                self.site = "after-layer".to_string();
                self.source = "capture".to_string();
                let target = layers.map_or(12, |count| count.saturating_sub(2));
                self.layer = target.to_string();
                self.source_layer = target.saturating_sub(4).to_string();
            }
            Preset::ArabicMorphology => {
                self.op = "scale".to_string();
                self.site = "after-mlp".to_string();
                self.value = "0.5".to_string();
                self.token = "matched-span".to_string();
                self.span = "المدينة".to_string();
                self.prompt = "اكتب جملة قصيرة عن المدينة المنورة".to_string();
                self.layer = layers.map_or(8, |count| count / 2).to_string();
            }
        }
        for (input, value) in [
            (self.inputs.layer.clone(), self.layer.clone()),
            (self.inputs.value.clone(), self.value.clone()),
            (self.inputs.source_layer.clone(), self.source_layer.clone()),
            (self.inputs.span.clone(), self.span.clone()),
            (self.inputs.prompt.clone(), self.prompt.clone()),
        ] {
            self.set_input_value(input, value, cx);
        }
        self.step = WorkspaceStep::Intervention;
        self.error = None;
        cx.notify();
    }

    /// Poll the worker reply channel every 80 ms while the app lives. The
    /// worker thread is unchanged from the iced implementation; only the
    /// foreground subscription is replaced by gpui's async executor.
    /// Launch straight into the workspace for someone who has used Ember
    /// before: their unfinished experiment is restored into the setup, and
    /// Home -- a page about getting started -- is not the first thing they
    /// see. A first launch has no history and no draft, and still opens Home.
    pub(super) fn open_where_you_left_off(&mut self, cx: &mut Context<Self>) {
        if self.store.draft.is_some() {
            self.restore_draft(cx);
        } else if !self.store.runs.is_empty() {
            self.view = View::Experiment;
        }
    }

    pub(super) fn spawn_poll(&mut self, cx: &mut Context<Self>) {
        self.poll_task(cx).detach();
    }

    /// The polling loop behind [`Console::spawn_poll`]. It ends when the
    /// console is released: a loop that outlived its entity would keep
    /// waking every 250 ms for nothing, forever.
    pub(super) fn poll_task(&mut self, cx: &mut Context<Self>) -> Task<()> {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                loop {
                    let Ok(delay) =
                        this.read_with(&cx, |console, _| if console.busy() { 50 } else { 250 })
                    else {
                        break;
                    };
                    cx.background_executor()
                        .timer(Duration::from_millis(delay))
                        .await;
                    let updated = this.update(&mut cx, |console, cx| {
                        let replies = console.drain_replies(cx);
                        if console.drain_store_writes() || replies {
                            cx.notify();
                        }
                    });
                    if updated.is_err() {
                        break;
                    }
                }
            }
        })
    }

    pub(super) fn picker_key(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let key = event.keystroke.key.as_str();
        let cmd = event.keystroke.modifiers.control || event.keystroke.modifiers.platform;
        let shift = event.keystroke.modifiers.shift;

        // Shell shortcuts. Both panes are workspace state, so the toggles go
        // through the persisting methods rather than flipping the fields.
        if cmd && !shift && key == "b" {
            self.toggle_sidebar(cx);
            return;
        }
        if cmd && !shift && key == "n" {
            self.goto(View::Experiment, cx);
            self.step = WorkspaceStep::Prompt;
            cx.notify();
            return;
        }
        if cmd && !shift && key == "r" {
            if self.result_context.is_some() {
                self.goto(View::Experiment, cx);
                self.step = WorkspaceStep::Review;
                self.rerun(cx);
            }
            cx.notify();
            return;
        }
        if cmd && !shift && key == "," {
            self.goto(View::Settings, cx);
            cx.notify();
            return;
        }

        // The palette is a modal: while it is open it owns the keyboard.
        // Cmd+K and Cmd+Shift+P (VS Code's spelling) open the command list;
        // Cmd+P opens Go to run, the editor's quick open.
        let p = key.eq_ignore_ascii_case("p");
        if cmd && ((!shift && key == "k") || (shift && p)) {
            self.open_palette(PaletteMode::Commands, window, cx);
            return;
        }
        if cmd && !shift && p {
            self.open_palette(PaletteMode::Runs, window, cx);
            return;
        }
        if self.palette_open {
            match key {
                "escape" => {
                    self.palette_open = false;
                    cx.notify();
                }
                "up" => self.palette_move(-1, cx),
                "down" => self.palette_move(1, cx),
                "enter" | "return" if !cmd => self.palette_execute(cx),
                _ => {}
            }
            // A closed palette hands the keyboard back to the console, so the
            // next shortcut works without a click first.
            if !self.palette_open {
                self.focus_handle.focus(window, cx);
            }
            return;
        }
        // Esc stops a run in flight. Only then: elsewhere it is left to the
        // focused control (closing a picker, say).
        if key == "escape" && self.can_cancel() {
            self.cancel_run(cx);
            return;
        }

        // Ctrl+Tab and Ctrl+Shift+Tab step through the result tabs, as they
        // step through an editor's tabs.
        if event.keystroke.modifiers.control
            && key == "tab"
            && self.view == View::Experiment
            && self.comparison.is_some()
        {
            self.cycle_result_view(shift, cx);
            return;
        }
        if cmd {
            let result_view = match key {
                "1" => Some(ResultView::Overview),
                "2" => Some(ResultView::Layers),
                "3" => Some(ResultView::Tokens),
                "4" => Some(ResultView::Trace),
                _ => None,
            };
            // With a finished result on screen the number keys address its
            // views: Overview, Layers, Tokens, Raw trace.
            if let Some(view) = result_view
                && self.view == View::Experiment
                && self.comparison.is_some()
            {
                self.result_view = view;
                cx.notify();
                return;
            }
        }
        if matches!(key, "enter" | "return") && cmd {
            self.advance_or_run();
            cx.notify();
        }
    }

    /// Navigate to a destination, checkpointing an in-progress experiment on
    /// the way out. One switch point, so every path out of the workspace
    /// saves the draft and none of them can forget to.
    pub(super) fn goto(&mut self, view: View, cx: &mut Context<Self>) {
        if self.view == View::Experiment && view != View::Experiment {
            self.save_draft();
            self.persist();
        }
        self.view = view;
        cx.notify();
    }

    // -- spec view ---------------------------------------------------------

    /// Open or close the spec view. When it opens, it shows the form's spec.
    pub(super) fn toggle_spec(&mut self, cx: &mut Context<Self>) {
        self.spec_open = !self.spec_open;
        self.revert_spec(cx);
    }

    /// Drop a held edit: the view shows the form's spec again.
    pub(super) fn revert_spec(&mut self, cx: &mut Context<Self>) {
        self.spec_held = false;
        self.spec_note = None;
        self.spec_shown.clear();
        cx.notify();
    }

    /// Keep the spec view in step with the form, unless it holds an edit
    /// that the form could not take. Runs before each render.
    pub(super) fn sync_spec_view(&mut self, cx: &mut Context<Self>) {
        if !self.spec_open || self.spec_held {
            return;
        }
        match self.form_values().spec_text() {
            Ok(text) => {
                self.spec_note = None;
                if text != self.spec_shown {
                    self.spec_shown = text.clone();
                    self.set_input_value(self.spec_input.clone(), text, cx);
                }
            }
            Err(error) => self.spec_note = Some(format!("The form is not complete: {error}")),
        }
    }

    /// An edit in the spec view: the form takes it when it can, and the
    /// view keeps the text as typed either way.
    pub(super) fn spec_edited(&mut self, text: &str, cx: &mut Context<Self>) {
        if !self.spec_open {
            return;
        }
        if text == self.spec_shown {
            // The text is the form's spec again: there is no edit to hold.
            if self.spec_held {
                self.spec_held = false;
                self.spec_note = None;
                cx.notify();
            }
            return;
        }
        match self.form_values().with_spec(text) {
            Ok(values) => {
                self.apply_form_values(values, cx);
                self.spec_held = false;
                self.spec_note = None;
                // The form's own text can put the same spec in a different
                // layout. The view keeps the typed text.
                self.spec_shown = self.form_values().spec_text().unwrap_or_default();
            }
            Err(error) => {
                self.spec_held = true;
                self.spec_note = Some(error);
            }
        }
        cx.notify();
    }

    /// Put a captured form configuration back into the console. The resident
    /// session only resets when the model actually changed, so duplicating a
    /// run on the same model does not pay for a reload.
    pub(super) fn apply_form_values(&mut self, values: FormValues, cx: &mut Context<Self>) {
        if values.model_path != self.model_path {
            self.session = None;
        }
        self.model_path = values.model_path;
        self.prompt = values.prompt;
        self.max_tokens = values.max_tokens;
        self.execution = values.execution;
        self.site = values.site;
        self.layer = values.layer;
        self.op = values.op;
        self.value = values.value;
        self.source = values.source;
        self.source_layer = values.source_layer;
        self.token = values.token;
        self.span = values.span;
        self.clamp_source_layer(cx);
        for (input, value) in [
            (self.inputs.model.clone(), self.model_path.clone()),
            (self.inputs.prompt.clone(), self.prompt.clone()),
            (self.inputs.max_tokens.clone(), self.max_tokens.clone()),
            (self.inputs.layer.clone(), self.layer.clone()),
            (self.inputs.value.clone(), self.value.clone()),
            (self.inputs.source_layer.clone(), self.source_layer.clone()),
            (self.inputs.span.clone(), self.span.clone()),
        ] {
            self.set_input_value(input, value, cx);
        }
    }

    /// Replay the completed experiment exactly: the form returns to the state
    /// that produced the last run, then the run starts. The bottom bar's
    /// "Run experiment again" runs the current form instead, which is the
    /// change-one-variable path.
    pub(super) fn rerun(&mut self, cx: &mut Context<Self>) {
        if self.busy() {
            return;
        }
        // "Rerun" replays a completed run. With none, it must do nothing
        // rather than quietly start whatever happens to be in the form.
        let Some(context) = self.result_context.clone() else {
            return;
        };
        self.apply_form_values(context, cx);
        self.run();
    }

    /// Branch from the completed run: the form returns to the state that
    /// produced it, and the user changes one thing. The original run stays in
    /// the history; the next run appends a new record.
    pub(super) fn duplicate_experiment(&mut self, cx: &mut Context<Self>) {
        let Some(context) = self.result_context.clone() else {
            return;
        };
        self.apply_form_values(context, cx);
        self.save_draft();
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Prompt;
        cx.notify();
    }

    /// Switch whichever family of tabs `prefix` names. One decision point, so
    /// the two tab bars cannot drift apart the way two implementations did.
    pub(super) fn select_tab(&mut self, prefix: &str, key: &str) {
        if prefix == "step" {
            if let Some(step) = WorkspaceStep::ALL.iter().find(|step| step.key() == key) {
                self.step = *step;
            }
        } else if let Some(view) = ResultView::ALL.iter().find(|view| view.key() == key) {
            self.result_view = *view;
        }
        self.copied = false;
    }
}

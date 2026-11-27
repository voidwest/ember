use super::*;

impl Console {
    /// Scan for models on a background thread and fill the model list when
    /// the scan finishes.
    pub(super) fn start_model_discovery(&mut self, cx: &mut Context<Self>) {
        cx.spawn(|this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                let found = cx
                    .background_executor()
                    .spawn(async { scan_models() })
                    .await;
                let _ = this.update(&mut cx, |console, cx| console.models_discovered(found, cx));
            }
        })
        .detach();
    }

    /// Adopt a finished model scan. The first model becomes the selection only
    /// if the user has not already typed or picked one.
    pub(super) fn models_discovered(
        &mut self,
        found: Vec<(String, Option<u64>)>,
        cx: &mut Context<Self>,
    ) {
        self.model_sizes = found.iter().cloned().collect();
        self.model_options = found.into_iter().map(|(path, _)| path).collect();
        if self.model_path.trim().is_empty()
            && let Some(first) = self.model_options.first().cloned()
        {
            self.model_path = first.clone();
            self.set_input_value(self.inputs.model.clone(), first, cx);
        }
        cx.notify();
    }

    pub(super) fn busy(&self) -> bool {
        self.status != Status::Idle
    }

    /// The active semantic palette, resolved from the persisted appearance mode.
    pub(super) fn colors(&self) -> Colors {
        if self.appearance.is_dark(self.system_dark) {
            theme::dark()
        } else {
            theme::light()
        }
    }

    pub(super) fn form_values(&self) -> FormValues {
        FormValues {
            model_path: self.model_path.clone(),
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens.clone(),
            execution: self.execution.clone(),
            site: self.site.clone(),
            layer: self.layer.clone(),
            op: self.op.clone(),
            value: self.value.clone(),
            source: self.source.clone(),
            source_layer: self.source_layer.clone(),
            token: self.token.clone(),
            span: self.span.clone(),
        }
    }

    /// Whether the primary workspace action may proceed.
    ///
    /// Single source of truth for the gate. It used to be a local in `render`
    /// that only the button consulted, while the Ctrl+Enter handler advanced
    /// the workflow with no gate at all -- so a disabled button could still be
    /// driven by the keyboard.
    pub(super) fn action_enabled(&self) -> bool {
        !self.busy() && self.validation_error().is_none()
    }

    /// The primary action, shared by the button and the keyboard shortcut.
    ///
    /// Both paths must go through here so they cannot disagree about when the
    /// action is allowed.
    pub(super) fn advance_or_run(&mut self) {
        if !self.action_enabled() {
            return;
        }
        self.run();
    }

    /// Run with the settings on screen. Shared by the setup pane's button.
    pub(super) fn run_now(&mut self) {
        self.advance_or_run();
    }

    pub(super) fn validation_error(&self) -> Option<String> {
        self.build_run_request()
            .and_then(|request| parse_run_request(&request).map(|_| ()))
            .err()
    }

    /// Build the v0.5 request from the current form fields; the shared
    /// `parse_run_request` gate validates it exactly like the web console.
    pub(super) fn build_run_request(&self) -> Result<RunRequest, String> {
        self.form_values().build_run_request()
    }

    pub(super) fn send_run(&mut self, cfg: RunConfig) {
        self.sample = false;
        self.saved_run = None;
        self.opened_note = None;
        self.status = Status::Running;
        self.step = WorkspaceStep::Review;
        self.pending_context = Some(self.form_values());
        self.error = None;
        self.cancelled = false;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self.worker_tx.send(WorkerMsg::Run(cfg, token));
    }

    /// `send_run` for one point of a sweep: `planned` is this layer and the
    /// layers still queued, so the worker can compute them in one pass.
    pub(super) fn send_sweep_point(&mut self, cfg: RunConfig, planned: Vec<usize>) {
        self.sample = false;
        self.saved_run = None;
        self.opened_note = None;
        self.status = Status::Running;
        self.step = WorkspaceStep::Review;
        self.pending_context = Some(self.form_values());
        self.error = None;
        self.cancelled = false;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self
            .worker_tx
            .send(WorkerMsg::SweepPoint(cfg, planned, token));
    }

    pub(super) fn send_restore(&mut self, cfg: RunConfig) {
        self.status = Status::Restoring;
        self.error = None;
        self.cancelled = false;
        let token = ember::cancel::CancelToken::new();
        self.run_cancel = Some(token.clone());
        let _ = self.worker_tx.send(WorkerMsg::Restore(cfg, token));
    }

    /// Whether Cancel has something to stop: a run or restore in flight, or
    /// a model load that a run is waiting on.
    pub(super) fn can_cancel(&self) -> bool {
        match self.status {
            Status::Running | Status::Restoring => true,
            Status::Preparing => self.pending_run || self.pending_sweep,
            Status::Idle | Status::Cancelling => false,
        }
    }

    /// Stop the run in flight. Shared by the button, Esc, the palette and
    /// the menu.
    ///
    /// The token is checked at every decode step, so the worker stops within
    /// one token; the console waits in `Cancelling` for it to confirm, then
    /// returns to idle with nothing recorded. A model load cannot be
    /// interrupted, but a run waiting on one is dropped and the model is kept
    /// for next time. A running sweep stops too; its finished points stay.
    pub(super) fn cancel_run(&mut self, cx: &mut Context<Self>) {
        if !self.can_cancel() {
            return;
        }
        if let Some(sweep) = self.sweep.as_mut() {
            sweep.stop = true;
        }
        if self.status == Status::Preparing {
            self.pending_run = false;
            self.pending_sweep = false;
            self.pending_context = None;
            self.cancelled = true;
        } else {
            if let Some(token) = &self.run_cancel {
                token.cancel();
            }
            self.status = Status::Cancelling;
        }
        cx.notify();
    }

    /// The worker confirmed a cancellation: back to idle, nothing kept.
    pub(super) fn run_cancelled(&mut self, cx: &mut Context<Self>) {
        if let Some(number) = self.rebundle.take() {
            self.export_note = Some(format!(
                "Re-run of run #{number} cancelled; its record is unchanged."
            ));
        }
        self.run_cancel = None;
        self.pending_context = None;
        self.pending_run = false;
        self.error = None;
        self.cancelled = true;
        self.status = Status::Idle;
        if self.sweep_running() {
            self.stop_sweep(cx);
            self.advance_sweep(cx);
        }
    }

    /// Drain the worker reply channel; returns true when anything changed.
    pub(super) fn drain_replies(&mut self, cx: &mut Context<Self>) -> bool {
        let (mut replies, disconnected) = {
            let rx = self.reply_rx.lock().expect("reply receiver lock");
            let mut replies = Vec::new();
            let disconnected = loop {
                match rx.try_recv() {
                    Ok(reply) => replies.push(reply),
                    Err(mpsc::TryRecvError::Empty) => break false,
                    Err(mpsc::TryRecvError::Disconnected) => break true,
                }
            };
            (replies, disconnected)
        };
        // Report a dead worker once, when something was still waiting on it.
        if disconnected && self.busy() {
            replies.push(WorkerReply::Failed(
                "the model worker stopped; restart Ember to run again".to_string(),
            ));
        }
        if replies.is_empty() {
            return false;
        }
        for reply in replies {
            // A reply for a run the user already cancelled is discarded: it
            // finished before the worker reached a check point, and it must
            // not land in history or on screen.
            if self.status == Status::Cancelling {
                match reply {
                    WorkerReply::RunDone(result) => {
                        if let Ok(bundle) = result.as_ref() {
                            worker::discard_run_bundles(bundle);
                        }
                        self.run_cancelled(cx);
                        continue;
                    }
                    WorkerReply::RestoreDone(_) | WorkerReply::Cancelled => {
                        self.run_cancelled(cx);
                        continue;
                    }
                    _ => {}
                }
            }
            match reply {
                WorkerReply::SweepEnded => {}
                WorkerReply::Cancelled => self.run_cancelled(cx),
                WorkerReply::Failed(error) => {
                    self.run_cancel = None;
                    self.rebundle = None;
                    self.pending_run = false;
                    self.pending_context = None;
                    self.session = None;
                    self.error = Some(error);
                    self.status = Status::Idle;
                }
                // A load that finished for a model the user has since moved
                // away from: load the current one instead of claiming it.
                WorkerReply::Prepared(result) if matches!(&*result, Ok(info) if info.model_path != self.model_path.trim()) =>
                {
                    let pending_run = self.pending_run;
                    self.status = Status::Idle;
                    self.load();
                    self.pending_run = pending_run && self.status == Status::Preparing;
                }
                WorkerReply::Prepared(result) => match *result {
                    Ok(info) => {
                        self.session = Some(info);
                        self.model_open = false;
                        let n = self.session.as_ref().map(|s| s.n_layers).unwrap_or(1);
                        let target = self
                            .layer
                            .parse::<usize>()
                            .unwrap_or(0)
                            .min(n.saturating_sub(1));
                        self.layer = target.to_string();
                        self.source_layer = self
                            .source_layer
                            .parse::<usize>()
                            .unwrap_or(0)
                            .min(target.saturating_sub(1))
                            .to_string();
                        self.inputs
                            .layer
                            .update(cx, |input, cx| input.set_value(self.layer.clone(), cx));
                        self.inputs.source_layer.update(cx, |input, cx| {
                            input.set_value(self.source_layer.clone(), cx)
                        });
                        self.status = Status::Idle;
                        if self.pending_run {
                            self.pending_run = false;
                            self.run();
                        }
                        if self.pending_sweep {
                            self.start_sweep(cx);
                        }
                    }
                    Err(error) => {
                        self.pending_run = false;
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
                WorkerReply::RunDone(result) => match *result {
                    // A re-run for History's export: its bundles go to the
                    // record it re-ran, and nothing on screen changes.
                    Ok(bundle) if self.rebundle.is_some() => {
                        self.run_cancel = None;
                        self.status = Status::Idle;
                        if let Some(number) = self.rebundle.take() {
                            self.rebundle_done(number, &bundle);
                        }
                    }
                    Err(error) if self.rebundle.is_some() => {
                        self.run_cancel = None;
                        self.status = Status::Idle;
                        if let Some(number) = self.rebundle.take() {
                            self.export_note =
                                Some(format!("Re-run of run #{number} failed: {error}"));
                        }
                    }
                    Ok(bundle) => {
                        self.run_cancel = None;
                        self.sample = false;
                        self.saved_run = None;
                        self.opened_note = None;
                        self.copied = false;
                        if !self.sweep_running() {
                            self.result_epoch += 1;
                        }
                        self.result_context = self.pending_context.take();
                        self.baseline = Some(bundle.baseline.clone());
                        self.intervention = Some(bundle.intervention.clone());
                        self.layer_series = Arc::from(bundle.comparison.layers.clone());
                        self.comparison = Some(bundle.comparison.clone());
                        self.result_view = ResultView::Overview;
                        self.hovered_layer = None;
                        self.selected_layer = bundle
                            .comparison
                            .layers
                            .iter()
                            .filter(|metric| metric.relative_l2_difference.is_some())
                            .max_by(|left, right| {
                                left.relative_l2_difference
                                    .unwrap_or(0.0)
                                    .total_cmp(&right.relative_l2_difference.unwrap_or(0.0))
                            })
                            .map(|metric| metric.layer);
                        self.verification =
                            Some(VerificationView::from_report(&bundle.verification));
                        self.restore = None;
                        self.last_config = Some(bundle.baseline_key.clone());
                        self.last_metrics = Some((
                            bundle.intervention.semantic_hash.clone(),
                            bundle.elapsed_ms_total,
                            bundle.intervention.decode_tps,
                        ));
                        // Record the form as it was sent, not as it is now: it
                        // stays editable while a run is in flight.
                        let sent = self
                            .result_context
                            .clone()
                            .unwrap_or_else(|| self.form_values());
                        // A sweep's runs are kept on the sweep, not in history:
                        // sixteen near-identical rows would bury the runs the
                        // user chose to make.
                        let in_sweep = self.sweep_running();
                        if !in_sweep {
                            let number = self.store.next_run_number();
                            let now = unix_now();
                            self.store.push_run(RunRecord {
                                number,
                                finished_at: now,
                                model: model_display_name(&sent.model_path),
                                intervention: operation_label(&sent.op).to_string(),
                                hook: site_label(&sent.site).to_string(),
                                layer: per_layer(&sent.site)
                                    .then(|| sent.layer.parse::<u32>().ok())
                                    .flatten(),
                                duration_ms: Some(bundle.elapsed_ms_total.max(0.0).round() as u64),
                                baseline_tokens: Some(bundle.baseline.generated_tokens as u32),
                                intervention_tokens: Some(
                                    bundle.intervention.generated_tokens as u32,
                                ),
                                diverged_at_step: bundle
                                    .comparison
                                    .first_token_divergence
                                    .map(|step| step as u32),
                                outputs_equal: bundle.comparison.generated_text_equal,
                                verified: bundle.verification.ok,
                                pinned: false,
                                prompt: sent.prompt.clone(),
                                config: Some(sent.record_config()),
                                result: Some(record_result(&bundle)),
                                bundles: Some(record_bundles(&bundle)),
                            });
                            self.store.touch_model(&sent.model_path, now);
                            // The state that produced this run is the resume point.
                            self.save_draft();
                            self.persist();
                        }
                        self.status = Status::Idle;
                        if in_sweep {
                            self.sweep_point_done(cx);
                        }
                    }
                    Err(error) => {
                        self.run_cancel = None;
                        self.pending_context = None;
                        self.error = Some(error);
                        self.status = Status::Idle;
                        if self.sweep_running() {
                            self.stop_sweep(cx);
                            self.advance_sweep(cx);
                        }
                    }
                },
                WorkerReply::RestoreDone(result) => match *result {
                    Ok(bundle) => {
                        self.run_cancel = None;
                        self.restore = Some(RestoreView {
                            matches: bundle.matches_baseline,
                            comparable: bundle.baseline_comparable,
                        });
                        self.verification =
                            Some(VerificationView::from_report(&bundle.verification));
                        if self.last_metrics.is_some() {
                            self.last_metrics = Some((
                                bundle.output.semantic_hash.clone(),
                                bundle.output.wall_ms,
                                bundle.output.decode_tps,
                            ));
                        }
                        self.status = Status::Idle;
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.status = Status::Idle;
                    }
                },
            }
        }
        true
    }

    pub(super) fn load(&mut self) {
        if self.busy() {
            return;
        }
        let path = self.model_path.trim().to_string();
        if path.is_empty() {
            self.error = Some("model path must not be empty".to_string());
            return;
        }
        self.status = Status::Preparing;
        self.pending_run = false;
        self.error = None;
        let _ = self.worker_tx.send(WorkerMsg::Prepare(path));
    }

    pub(super) fn run(&mut self) {
        if self.busy() {
            return;
        }
        match self.build_run_request() {
            Ok(req) => match parse_run_request(&req) {
                Ok(_cfg) if self.session.is_none() => {
                    self.pending_run = true;
                    self.status = Status::Preparing;
                    self.error = None;
                    self.cancelled = false;
                    let _ = self
                        .worker_tx
                        .send(WorkerMsg::Prepare(self.model_path.trim().to_string()));
                }
                Ok(cfg) => self.send_run(cfg),
                Err(error) => self.error = Some(error),
            },
            Err(error) => self.error = Some(error),
        }
    }

    pub(super) fn restore(&mut self) {
        if self.busy() {
            return;
        }
        if self.last_config.is_none() {
            self.error =
                Some("run an experiment first; restore verifies against its baseline".to_string());
            return;
        }
        match self.build_run_request() {
            Ok(mut req) => {
                req.operation = "restore-original".to_string();
                req.factor = None;
                req.alpha = None;
                req.source = "capture".to_string();
                req.source_layer = None;
                match parse_run_request(&req) {
                    Ok(cfg) => self.send_restore(cfg),
                    Err(error) => self.error = Some(error),
                }
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub(super) fn sync_kit_theme(&self, cx: &mut App) {
        use gpui_kit::component::{Theme, ThemeMode};
        Theme::change(
            if self.appearance.is_dark(self.system_dark) {
                ThemeMode::Dark
            } else {
                ThemeMode::Light
            },
            None,
            cx,
        );
        let colors = self.colors();
        let theme = Theme::global_mut(cx);
        theme.font_family = FONT_SANS_NAME.into();
        // The top bar is drawn with the kit's TitleBar; flatten its default
        // gradient to the canvas so it reads as part of the window.
        theme.title_bar = colors.canvas.into();
        theme.title_bar_border = colors.border.into();
        theme.font_size = px(theme::scaled(14.0));
        theme.mono_font_family = FONT_MONO_NAME.into();
        theme.mono_font_size = px(theme::scaled(13.0));
        // Kit-owned geometry and hairlines follow the console's scale, so an
        // Input, a Select or a table cell does not disagree with a hand-laid
        // surface beside it.
        theme.radius = px(Radius::MD);
        theme.radius_lg = px(Radius::LG);
        theme.colors.border = colors.border.into();
        // Ghost buttons, tabs, nav rows and tiles hover to the kit's accent
        // token. Left at its default the shift was a single shade -- present
        // but not noticeable -- so it is set from the console's own palette:
        // a clear step above the selected-row fill, still neutral (the orange
        // stays reserved for the intervention and the primary action).
        let hover: Hsla = if self.appearance.is_dark(self.system_dark) {
            rgb(0x34302c).into()
        } else {
            rgb(0xddd7cd).into()
        };
        theme.colors.accent = hover;
        theme.colors.accent_foreground = colors.text.into();
        theme.primary = colors.accent.into();
        theme.primary_hover = colors.accent.into();
        theme.primary_active = colors.accent.into();
        theme.primary_foreground = rgb(0xffffff).into();
        theme.ring = colors.accent.into();
        theme.button_primary = colors.accent.into();
        theme.button_primary_hover = colors.accent.into();
        theme.button_primary_active = colors.accent.into();
        theme.button_primary_foreground = rgb(0xffffff).into();
        theme.tokens.button_primary = Hsla::from(colors.accent).into();
        theme.tokens.button_primary_hover = Hsla::from(colors.accent).into();
        theme.tokens.button_primary_active = Hsla::from(colors.accent).into();
        // Ghost controls and selectable rows take their hover and selected
        // fills from these tokens: navigation, tabs and segmented choices all
        // land on the same neutral step, which is what makes selection read as
        // position rather than emphasis.
        theme.tokens.secondary_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.secondary_active = Hsla::from(colors.selection).into();
        theme.tokens.button_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.list_hover = Hsla::from(colors.surface_hover).into();
        theme.tokens.list_active = Hsla::from(colors.selection).into();
        Theme::sync_base(cx);
    }

    pub(super) fn cycle_appearance(&mut self, cx: &mut Context<Self>) {
        self.appearance = self.appearance.next();
        self.appearance.persist();
        self.sync_kit_theme(cx);
        cx.notify();
    }

    pub(super) fn toggle_sidebar(&mut self, cx: &mut Context<Self>) {
        self.sidebar_open = !self.sidebar_open;
        if self.presentation.is_none() {
            theme::persist_flag("sidebar", self.sidebar_open);
        }
        cx.notify();
    }

    /// The finished comparison as Markdown, for a slide, a lab notebook or a
    /// message: what was asked, what was changed, what happened, and the
    /// per-layer numbers. Everything is read from what is on screen, so the
    /// copy can never disagree with the page.
    pub(super) fn result_markdown(&self) -> Option<String> {
        let (baseline, intervention, comparison, context) = (
            self.baseline.as_ref()?,
            self.intervention.as_ref()?,
            self.comparison.as_ref()?,
            self.result_context.as_ref()?,
        );
        let note = if let Some(number) = self.saved_run {
            Some(format!("Run #{number}, reopened from history."))
        } else if self.sample {
            Some("Sample result: illustrative data, not a measurement.".to_string())
        } else {
            None
        };
        Some(export::experiment_markdown(
            note.as_deref(),
            baseline,
            intervention,
            comparison,
            context,
            &self.layer_series,
        ))
    }

    /// Open Review on the built-in sample: a finished comparison a newcomer
    /// (or a demo with no model on the machine) can read without running
    /// anything. It is labelled as illustrative on the page, and the first
    /// real run replaces it.
    pub(super) fn show_sample(&mut self, cx: &mut Context<Self>) {
        let (baseline, intervention, comparison, mut values) = sample_result();
        values.model_path = self.model_path.clone();
        self.show_result(baseline, intervention, comparison, values, None, cx);
    }

    /// Put a finished comparison on the Review page without running anything.
    /// Shared by the sample and by History; `saved_run` says which it is.
    /// Whether a bundle directory exists, checked once per path (see
    /// `bundle_dir_check`).
    pub(super) fn bundle_dir_on_disk(&self, path: &str) -> bool {
        let mut check = self.bundle_dir_check.borrow_mut();
        if let Some((checked, on_disk)) = check.as_ref()
            && checked == path
        {
            return *on_disk;
        }
        let on_disk = std::path::Path::new(path).is_dir();
        *check = Some((path.to_string(), on_disk));
        on_disk
    }

    pub(super) fn show_result(
        &mut self,
        baseline: RunOutput,
        intervention: RunOutput,
        comparison: ExperimentComparison,
        values: FormValues,
        saved_run: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        self.result_epoch += 1;
        self.apply_form_values(values.clone(), cx);
        self.layer_series = Arc::from(comparison.layers.clone());
        let selected = comparison
            .landmarks
            .peak_layer
            .or(comparison.landmarks.first_layer_divergence);
        self.baseline = Some(baseline);
        self.intervention = Some(intervention);
        self.comparison = Some(comparison);
        self.result_context = Some(values);
        self.verification = None;
        self.restore = None;
        self.selected_layer = selected;
        self.result_view = ResultView::Overview;
        self.sample = true;
        self.saved_run = saved_run;
        self.copied = false;
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Review;
        cx.notify();
    }

    /// Presentation mode: a larger text scale with the sidebar
    /// out of the way. Nothing else changes, and leaving restores the panes
    /// exactly as they were. The panes are hidden without touching the
    /// persisted flags, so a crash mid-talk does not lose the workspace.
    pub(super) fn toggle_presentation(&mut self, cx: &mut Context<Self>) {
        match self.presentation.take() {
            Some(sidebar) => {
                self.sidebar_open = sidebar;
                theme::set_ui_scale(1.0);
            }
            None => {
                self.presentation = Some(self.sidebar_open);
                self.sidebar_open = false;
                theme::set_ui_scale(theme::PRESENTATION_SCALE);
            }
        }
        self.sync_kit_theme(cx);
        cx.notify();
    }
}

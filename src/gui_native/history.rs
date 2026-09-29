//! Run history and the draft: persisting the store, resuming, reopening and
//! reusing past runs.

use super::*;

impl Console {
    /// Reopen a run from History. Only records that kept their result can be
    /// opened; the Runs table shows Open on exactly those.
    pub(super) fn open_run(&mut self, number: u64, cx: &mut Context<Self>) {
        let Some(record) = self
            .store
            .runs
            .iter()
            .find(|run| run.number == number)
            .cloned()
        else {
            return;
        };
        let (Some(result), Some(config)) = (record.result.clone(), record.config.clone()) else {
            return;
        };
        let output = |text: String, tokens: Option<u32>| RunOutput {
            text,
            generated_token_ids: (1..=tokens.unwrap_or(0)).collect(),
            generated_token_texts: Vec::new(),
            prompt_tokens: 0,
            generated_tokens: tokens.unwrap_or(0) as usize,
            bundle_dir: "history".to_string(),
            semantic_hash: "0000000000000000".to_string(),
            payload_hash: "00000000".to_string(),
            // History keeps one total for the pair, not per-side timings, so
            // none is shown rather than a made-up split (see output_panel).
            wall_ms: 0.0,
            decode_tps: None,
            events: Vec::new(),
        };
        let baseline = output(result.baseline_text.clone(), record.baseline_tokens);
        let intervention = output(result.intervention_text.clone(), record.intervention_tokens);
        let comparison = ExperimentComparison {
            layers: result
                .layers
                .iter()
                .map(|layer| crate::gui::LayerMetric {
                    layer: layer.layer,
                    relative_l2_difference: layer.relative_l2,
                    cosine_distance: layer.cosine,
                    maximum_absolute_difference: None,
                    exact: layer.relative_l2 == Some(0.0),
                })
                .collect(),
            tokens: result
                .tokens
                .iter()
                .map(|token| crate::gui::TokenMetric {
                    position: token.position,
                    baseline_token_id: None,
                    intervention_token_id: None,
                    baseline_text: token.baseline.clone(),
                    intervention_text: token.intervention.clone(),
                    differs: token.differs,
                })
                .collect(),
            first_token_divergence: record.diverged_at_step.map(|step| step as usize),
            generated_tokens_equal: result.tokens_equal,
            generated_text_equal: record.outputs_equal,
            landmarks: crate::gui::DivergenceLandmarks {
                first_layer_divergence: result.first_layer_divergence,
                peak_layer: result.peak_layer,
                peak_relative_l2: result.peak_relative_l2,
                stable_token_tail_step: None,
            },
            layer_token_grid: None,
        };
        let values = FormValues {
            model_path: config.model_path,
            prompt: record.prompt,
            max_tokens: config.max_tokens,
            execution: config.execution,
            site: config.site,
            layer: config.layer,
            op: config.op,
            value: config.value,
            source: config.source,
            source_layer: config.source_layer,
            token: config.token,
            span: config.span,
        };
        self.show_result(baseline, intervention, comparison, values, Some(number), cx);
    }

    /// Write the store, surfacing a failure instead of dropping the record.
    ///
    /// A run that completed and was never written is the one loss this app
    /// cannot explain to the user, so the error is shown rather than swallowed.
    /// Queue the store to be written. The write -- lock, re-read, merge,
    /// replace -- happens on the writer thread; its outcome comes back through
    /// [`Console::drain_store_writes`].
    pub(super) fn persist(&mut self) {
        let Some(path) = &self.store_path else {
            return;
        };
        self.store_writer
            .get_or_insert_with(|| store_writer::StoreWriter::spawn(path.clone()))
            .save(self.store.clone());
    }

    /// Adopt finished writes. The merged store replaces the in-memory one when
    /// nothing changed since the snapshot was taken; otherwise only the run
    /// renumbering is followed, and the next save merges the rest.
    pub(super) fn drain_store_writes(&mut self) -> bool {
        let Some(writer) = &self.store_writer else {
            return false;
        };
        let outcomes = writer.finished();
        let changed = !outcomes.is_empty();
        for outcome in outcomes {
            match outcome.result {
                Ok(merged) => {
                    self.store_error = None;
                    for change in &merged.renumbered {
                        if self.saved_run == Some(change.from) {
                            self.saved_run = Some(change.to);
                        }
                    }
                    if self.store == outcome.sent {
                        self.store = merged.store;
                    } else {
                        self.store.apply_renumbering(&merged.renumbered);
                    }
                }
                Err(error) => {
                    self.store_error = Some(format!("could not save run history: {error}"));
                }
            }
        }
        changed
    }

    /// Block until queued history writes are on disk. Called on quit.
    pub(super) fn flush_store(&self) {
        if let Some(writer) = &self.store_writer {
            writer.flush();
        }
    }

    /// Snapshot the current form as the resume point.
    pub(super) fn save_draft(&mut self) {
        // The sample rewrites the form; it must not become a draft to resume.
        if self.sample {
            return;
        }
        let values = self.form_values();
        // A form identical to the run that just finished has nothing left to
        // resume -- the run record already holds it, and Reuse reopens it.
        // Keeping a draft anyway made Home offer "Continue where you left
        // off" for an experiment the user had just completed.
        if self.result_context.as_ref() == Some(&values) {
            self.store.clear_draft();
            return;
        }
        let revision = self.store.next_draft_revision();
        let fields = [
            ("max_tokens", values.max_tokens),
            ("execution", values.execution),
            ("site", values.site),
            ("layer", values.layer),
            ("op", values.op),
            ("value", values.value),
            ("source", values.source),
            ("source_layer", values.source_layer),
            ("token", values.token),
            ("span", values.span),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
        self.store.draft = Some(app_store::Draft {
            revision,
            prompt: values.prompt,
            model_path: values.model_path,
            fields,
            step: self.step.key().to_string(),
            updated_at: unix_now(),
        });
    }

    /// Restore the saved draft, if one exists. Form values ride the same
    /// `set_value` path as the pickers so the controls cannot disagree with
    /// the model behind them.
    pub(super) fn restore_draft(&mut self, cx: &mut Context<Self>) {
        let Some(draft) = self.store.draft.clone() else {
            return;
        };
        let field = |name: &str| draft.fields.get(name).cloned().unwrap_or_default();
        self.apply_form_values(
            FormValues {
                model_path: draft.model_path,
                prompt: draft.prompt,
                max_tokens: field("max_tokens"),
                execution: field("execution"),
                site: field("site"),
                layer: field("layer"),
                op: field("op"),
                value: field("value"),
                source: field("source"),
                source_layer: field("source_layer"),
                token: field("token"),
                span: field("span"),
            },
            cx,
        );
        self.step = WorkspaceStep::ALL
            .iter()
            .find(|step| step.key() == draft.step)
            .copied()
            .unwrap_or(WorkspaceStep::Prompt);
        self.view = View::Experiment;
        cx.notify();
    }

    /// Reuse from the Runs table, which holds only row numbers: look the
    /// record up and load its configuration.
    pub(super) fn reuse_run(&mut self, number: u64, cx: &mut Context<Self>) {
        let Some((config, prompt)) = self
            .store
            .runs
            .iter()
            .find(|run| run.number == number)
            .and_then(|run| Some((run.config.clone()?, run.prompt.clone())))
        else {
            return;
        };
        self.reuse_record(&config, &prompt, cx);
    }

    /// The Runs-page path into the same loop: load a stored run's recorded
    /// configuration into the form. Records written before configurations
    /// were stored have nothing to load, so their rows do not offer it.
    pub(super) fn reuse_record(
        &mut self,
        config: &app_store::RecordConfig,
        prompt: &str,
        cx: &mut Context<Self>,
    ) {
        self.apply_form_values(
            FormValues {
                model_path: config.model_path.clone(),
                prompt: prompt.to_string(),
                max_tokens: config.max_tokens.clone(),
                execution: config.execution.clone(),
                site: config.site.clone(),
                layer: config.layer.clone(),
                op: config.op.clone(),
                value: config.value.clone(),
                source: config.source.clone(),
                source_layer: config.source_layer.clone(),
                token: config.token.clone(),
                span: config.span.clone(),
            },
            cx,
        );
        self.save_draft();
        self.goto(View::Experiment, cx);
        self.step = WorkspaceStep::Prompt;
        cx.notify();
    }
}

/// Load the run history, and decide whether this session may write it back.
///
/// `path` is the store this build owns; `legacy` is the file older builds
/// wrote, which is read on every open and its new runs imported, but never
/// written ([`app_store::open`]).
///
/// A file that exists but cannot be read -- damaged, or written by a newer
/// build -- still holds the user's history. The session then runs on an empty
/// in-memory store and never writes, so the file survives for a build that can
/// read it; the status bar says so instead of pretending nothing was ever run.
/// An unreadable legacy file is only reported: this build never writes it, so
/// using the store alongside it cannot hurt it.
pub(super) fn open_store(
    path: std::path::PathBuf,
    legacy: &std::path::Path,
) -> (AppStore, Option<String>, Option<std::path::PathBuf>) {
    match app_store::open(&path, legacy) {
        // Readable, but written by a newer build: its extra fields would be
        // dropped by a write from this one, so the session is read-only.
        Ok(opened) if opened.store.written_by_newer_build() => (
            opened.store,
            Some(format!(
                "{} was written by a newer Ember -- history is read-only; runs this session will not be saved",
                path.display()
            )),
            None,
        ),
        Ok(opened) => {
            let warning = opened.legacy_error.map(|error| {
                format!("runs from an older Ember could not be imported: {error} -- left untouched")
            });
            (opened.store, warning, Some(path))
        }
        Err(error) => (
            AppStore::default(),
            Some(format!(
                "{error} -- left untouched; runs this session will not be saved"
            )),
            None,
        ),
    }
}

/// What a finished run showed, in the shape History stores.
pub(super) fn record_result(bundle: &crate::gui::RunBundle) -> app_store::RecordResult {
    let comparison = &bundle.comparison;
    app_store::RecordResult {
        baseline_text: bundle.baseline.text.clone(),
        intervention_text: bundle.intervention.text.clone(),
        layers: comparison
            .layers
            .iter()
            .map(|metric| app_store::RecordLayer {
                layer: metric.layer,
                relative_l2: metric.relative_l2_difference,
                cosine: metric.cosine_distance,
            })
            .collect(),
        tokens: comparison
            .tokens
            .iter()
            .map(|token| app_store::RecordToken {
                position: token.position,
                baseline: token.baseline_text.clone(),
                intervention: token.intervention_text.clone(),
                differs: token.differs,
            })
            .collect(),
        first_layer_divergence: comparison.landmarks.first_layer_divergence,
        peak_layer: comparison.landmarks.peak_layer,
        peak_relative_l2: comparison.landmarks.peak_relative_l2,
        tokens_equal: comparison.generated_tokens_equal,
    }
}

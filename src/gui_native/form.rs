//! The experiment form: its input entities and the values they hold.

use super::*;

#[derive(Clone)]
pub(super) struct Inputs {
    pub(super) model: Entity<TextInput>,
    pub(super) layer: Entity<TextInput>,
    pub(super) value: Entity<TextInput>,
    pub(super) source_layer: Entity<TextInput>,
    pub(super) span: Entity<TextInput>,
    pub(super) max_tokens: Entity<TextInput>,
    pub(super) prompt: Entity<TextInput>,
}

impl Inputs {
    pub(super) fn all(&self) -> [Entity<TextInput>; 7] {
        [
            self.model.clone(),
            self.layer.clone(),
            self.value.clone(),
            self.source_layer.clone(),
            self.span.clone(),
            self.max_tokens.clone(),
            self.prompt.clone(),
        ]
    }
}

#[derive(Clone, PartialEq)]
pub(super) struct FormValues {
    pub(super) model_path: String,
    pub(super) prompt: String,
    pub(super) max_tokens: String,
    pub(super) execution: String,
    pub(super) site: String,
    pub(super) layer: String,
    pub(super) op: String,
    pub(super) value: String,
    pub(super) source: String,
    pub(super) source_layer: String,
    pub(super) token: String,
    pub(super) span: String,
}

impl FormValues {
    /// The configuration part of a history record.
    pub(super) fn record_config(&self) -> app_store::RecordConfig {
        app_store::RecordConfig {
            model_path: self.model_path.clone(),
            execution: self.execution.clone(),
            site: self.site.clone(),
            layer: self.layer.clone(),
            op: self.op.clone(),
            value: self.value.clone(),
            source: self.source.clone(),
            source_layer: self.source_layer.clone(),
            token: self.token.clone(),
            span: self.span.clone(),
            max_tokens: self.max_tokens.clone(),
        }
    }

    pub(super) fn build_run_request(&self) -> Result<RunRequest, String> {
        let layer = if per_layer(&self.site) {
            Some(
                self.layer
                    .parse::<usize>()
                    .map_err(|_| "layer must be an integer".to_string())?,
            )
        } else {
            None
        };
        let source_layer = if per_layer(&self.site) && self.source == "capture" {
            let target = layer.expect("layer checked above");
            Some(
                self.source_layer
                    .parse::<usize>()
                    .map_err(|_| "source layer must be an integer".to_string())?
                    .min(target.saturating_sub(1)),
            )
        } else {
            None
        };
        Ok(RunRequest {
            model_path: self.model_path.trim().to_string(),
            prompt: self.prompt.clone(),
            max_new_tokens: self
                .max_tokens
                .parse::<usize>()
                .map_err(|_| "max new tokens must be an integer".to_string())?,
            execution: self.execution.clone(),
            site: self.site.clone(),
            layer,
            operation: self.op.clone(),
            factor: if self.op == "scale" {
                Some(
                    self.value
                        .parse::<f32>()
                        .map_err(|_| "scale factor must be a number".to_string())?,
                )
            } else {
                None
            },
            alpha: if self.op == "interpolate" {
                Some(
                    self.value
                        .parse::<f32>()
                        .map_err(|_| "interpolate alpha must be a number".to_string())?,
                )
            } else {
                None
            },
            source: self.source.clone(),
            source_layer,
            token_kind: self.token.clone(),
            span_text: (self.token == "matched-span").then(|| self.span.clone()),
        })
    }
}

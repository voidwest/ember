//! Built-in research experiments and the v0.2 activation-capture writer.
//!
//! The hook framework they plug into (runner, hook traits, contexts) lives in
//! `ember_core::experiments` and is re-exported here, so every
//! `ember::experiments::*` path is unchanged. This API is intentionally
//! narrow and **pre-1.0**; see `docs/api-stability.md`.

pub use ember_core::experiments::*;

mod activation_patch;
mod activation_stats;
mod capture;

pub use activation_patch::{ActivationPatch, PatchTarget};
pub use activation_stats::ActivationStats;
pub use capture::CaptureSink;

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::artifact::{DispatchObservation, DispatchPath, ManifestExperiment};

    /// Model SHA-256 recorded by [`CaptureArtifactBuilder`] artifacts.
    pub(crate) const CAPTURE_MODEL_SHA256: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /// A small activation-capture artifact written through the real
    /// `CaptureSink`: a Qwen3 context with 4 layers of width 8, capturing
    /// `stages` at layer 1 in both phases. Prefill rows are recorded with
    /// the generic dispatch path, decode rows with the fast one.
    pub(crate) struct CaptureArtifactBuilder {
        sink: CaptureSink,
        model: ModelContext<'static>,
        input_ids: Vec<u32>,
    }

    impl CaptureArtifactBuilder {
        /// Input token ids are `1..=prompt_len`.
        pub(crate) fn new(
            tag: &str,
            stages: &[&str],
            prompt_len: usize,
            tokenizer_sha256: Option<&str>,
        ) -> Self {
            let dir = crate::v05::testutil::temp_root(tag);
            std::fs::create_dir_all(&dir).unwrap();
            let config_path = dir.join("capture.toml");
            std::fs::write(
                &config_path,
                format!(
                    "schema_version = 1\noutput_dir = {:?}\nlayers = [1]\nstages = {stages:?}\nphase = \"both\"\n",
                    dir.to_str().unwrap()
                ),
            )
            .unwrap();
            let mut sink = CaptureSink::from_toml_path(
                config_path.to_str().unwrap(),
                "capture test prompt",
                1,
                serde_json::json!({}),
                Some(CAPTURE_MODEL_SHA256.to_string()),
                tokenizer_sha256.map(str::to_string),
                serde_json::json!({}),
            )
            .unwrap();
            let model = ModelContext::new(ModelFamily::Qwen3, Some("tiny.gguf"), "qwen3", 4, 8);
            sink.on_model_loaded(&model).unwrap();
            Self {
                sink,
                model,
                input_ids: (1..=prompt_len as u32).collect(),
            }
        }

        pub(crate) fn input_ids(&self) -> &[u32] {
            &self.input_ids
        }

        pub(crate) fn record_prefill(&mut self, mut values: Vec<f32>) {
            let seq = self.input_ids.len();
            let execution = ExecutionContext::new(
                self.model,
                ExecutionPhase::Prefill,
                0,
                seq,
                TracingState::Disabled,
            );
            let tensor = TensorAccess::new(seq, 8, &mut values);
            self.sink
                .after_mlp(&execution, 1, &tensor, DispatchPath::Generic)
                .unwrap();
        }

        pub(crate) fn record_decode(&mut self, position: usize, mut values: Vec<f32>) {
            let execution = ExecutionContext::new(
                self.model,
                ExecutionPhase::Decode,
                position,
                1,
                TracingState::Disabled,
            );
            let tensor = TensorAccess::new(1, 8, &mut values);
            self.sink
                .after_mlp(&execution, 1, &tensor, DispatchPath::Fast)
                .unwrap();
        }

        /// Finalize with `generated_ids` (one decode evaluation each) and
        /// return the manifest path.
        pub(crate) fn finalize(
            mut self,
            generated_ids: &[u32],
            dispatch: Vec<DispatchObservation>,
        ) -> std::path::PathBuf {
            let generation = GenerationContext::new(
                self.model,
                self.input_ids.len(),
                generated_ids.len(),
                generated_ids.len(),
                TracingState::Disabled,
                &self.input_ids,
                generated_ids,
            );
            self.sink
                .finalize(
                    &generation,
                    ManifestExperiment {
                        name: "none".to_string(),
                        arguments: serde_json::Value::Null,
                    },
                    dispatch,
                )
                .unwrap()
        }
    }
}

//! Independent observer of the lower-level hooks, compared to real v1 bundles.
//! Run explicitly with EMBER_SIX_SITE_BASELINE pointing at the capture matrix.
use ember::artifact::ActivationStage;
use ember::backend::CpuBackend;
use ember::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, ExperimentRunner,
    ExperimentalForwardModel, LayerContext, ModelContext, ModelFamily, TensorAccess, TracingState,
};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::ExecutionMode;
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
use ember::v05::{manifest::sha256_hex, safetensors, verify::CaptureIndexEntry};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Snapshot {
    shape: [usize; 2],
    values: Vec<f32>,
}
type Observations = Arc<Mutex<BTreeMap<String, Snapshot>>>;
struct Observer {
    prompt_len: usize,
    observations: Observations,
}
impl Observer {
    fn record(&self, site: &str, ctx: &ExecutionContext<'_>, tensor: &TensorAccess<'_>) {
        let phase = match ctx.phase {
            ExecutionPhase::Prefill => "prefill",
            ExecutionPhase::Decode if ctx.start_position == self.prompt_len => "decode",
            _ => return,
        };
        assert!(
            self.observations
                .lock()
                .unwrap()
                .insert(
                    format!("{site}-{phase}"),
                    Snapshot {
                        shape: *tensor.shape(),
                        values: tensor.values().to_vec()
                    },
                )
                .is_none(),
            "duplicate callback at {site}-{phase}"
        );
    }
}
macro_rules! layer_hook {
    ($method:ident, $site:literal) => {
        fn $method(
            &mut self,
            ctx: &LayerContext<'_>,
            tensor: &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError> {
            if ctx.layer_index == 0 {
                self.record($site, &ctx.execution, tensor);
            }
            Ok(())
        }
    };
}
impl Experiment for Observer {
    fn name(&self) -> &'static str {
        "independent-six-site-observer"
    }
    fn uses_activation_site(
        &self,
        _: ActivationStage,
        layer: Option<usize>,
        _: ExecutionPhase,
    ) -> bool {
        layer.is_none_or(|index| index == 0)
    }
    layer_hook!(before_layer, "residual-pre-attention");
    layer_hook!(after_attention, "attention-output");
    layer_hook!(after_mlp, "mlp-output");
    layer_hook!(after_layer, "residual-post-mlp");
    fn before_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("final-norm-output", ctx, tensor);
        Ok(())
    }
    fn after_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.record("logits", ctx, tensor);
        Ok(())
    }
}

#[test]
#[ignore = "requires the explicit, real-model six-site capture matrix"]
fn direct_observer_matches_v1_exports_after_scratch_reuse() {
    let root = PathBuf::from(
        std::env::var("EMBER_SIX_SITE_BASELINE").expect("set EMBER_SIX_SITE_BASELINE"),
    );
    let identities: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("identities.json")).unwrap()).unwrap();
    for name in ["model", "tokenizer", "binary"] {
        let path = identities[name]["path"].as_str().unwrap();
        assert_eq!(
            sha256_hex(&std::fs::read(path).unwrap()),
            identities[name]["sha256"].as_str().unwrap(),
            "{name} identity"
        );
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();
    let loader = load_gguf_with_k_strategy(
        identities["model"]["path"].as_str().unwrap(),
        KStrategy::Auto,
        false,
    )
    .unwrap();
    assert!(
        !loader.k_decisions.is_empty(),
        "requires a K-quant model to exercise planned execution"
    );
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048)).unwrap();
    let backend = CpuBackend;
    let tokenizer =
        EmberTokenizer::from_file(identities["tokenizer"]["path"].as_str().unwrap()).unwrap();
    let ids = tokenizer.encode("The Arabic word كتاب means").unwrap();
    let context = ModelContext::new(
        ModelFamily::Llama,
        None,
        "llama",
        model.n_layers(),
        model.embed_dim(),
    );
    for (name, mode) in [
        ("reference", ExecutionMode::Reference),
        ("planned", ExecutionMode::Planned),
        ("planned-fused", ExecutionMode::PlannedFused),
    ] {
        model.set_execution_mode(mode);
        let observations = Observations::default();
        let mut runner = ExperimentRunner::new(Observer {
            prompt_len: ids.len(),
            observations: observations.clone(),
        });
        let mut cache = model.create_cache(&backend, 2048);
        let mut current = ids.clone();
        let mut position = 0;
        let mut tokens = Vec::new();
        // Eight evaluations reuse model/cache scratch after the selected decode
        // step. Compare only after all evaluations have finished.
        for _ in 0..8 {
            let phase = if position == 0 {
                ExecutionPhase::Prefill
            } else {
                ExecutionPhase::Decode
            };
            let execution = ExecutionContext::new(
                context,
                phase,
                position,
                current.len(),
                TracingState::Disabled,
            );
            let logits = model
                .forward_last_logits_with_experiment(
                    &backend,
                    &current,
                    &mut cache,
                    position,
                    execution,
                    &mut runner,
                )
                .unwrap();
            let token = ember::sampler::argmax_token(logits.data()) as u32;
            tokens.push(token);
            position += current.len();
            current = vec![token];
        }
        let bundle = root.join(name);
        let payload = std::fs::read(bundle.join("captures/tensors.safetensors")).unwrap();
        let tensors: BTreeMap<_, _> = safetensors::deserialize(&payload)
            .unwrap()
            .into_iter()
            .collect();
        let observed = observations.lock().unwrap();
        assert_eq!(observed.len(), 12);
        let mut count = 0;
        for line in std::fs::read_to_string(bundle.join("captures/index.jsonl"))
            .unwrap()
            .lines()
        {
            let entry: CaptureIndexEntry = serde_json::from_str(line).unwrap();
            let (key, full) = if let Some(key) = entry.capture_id.strip_suffix("-full-tensor") {
                (key, true)
            } else {
                (
                    entry.capture_id.strip_suffix("-selected-rows").unwrap(),
                    false,
                )
            };
            let snapshot = &observed[key];
            let expected = if full {
                snapshot.values.as_slice()
            } else {
                &snapshot.values[(snapshot.shape[0] - 1) * snapshot.shape[1]..]
            };
            let actual = safetensors::tensor_f32(&payload, &tensors[&entry.tensor_name]).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}: {}",
                entry.capture_id
            );
            let expected_position = if key.ends_with("-decode") {
                ids.len()
            } else {
                ids.len() - 1
            };
            assert_eq!(*entry.positions.last().unwrap(), expected_position);
            count += 1;
        }
        assert_eq!(count, 24);
        let output: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(bundle.join("outputs.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(
            serde_json::json!(&tokens[..3]),
            output["generated_token_ids"]
        );
        eprintln!(
            "{name}: 24 exported captures bit-exact with direct observer after eight evaluations"
        );
    }
}

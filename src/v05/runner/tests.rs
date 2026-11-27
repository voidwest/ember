use super::*;
use crate::experiments::{LayerContext, ModelFamily, TracingState};

/// Pinned against the original per-float `update` + per-byte `format!`
/// implementation (and independently against Python's hashlib over the
/// same little-endian bytes); 1027 values straddle the internal chunk.
#[test]
fn checksum_f32_digest_is_pinned() {
    let values: Vec<f32> = (0u32..1027)
        .map(|i| f32::from_bits(i.wrapping_mul(2_654_435_761)))
        .collect();
    assert_eq!(
        checksum_f32(&values),
        "e95474f423de87316fd007b2eb42bfa1dbbc13016eed90e41a7b85fae0d4c4ad"
    );
    assert_eq!(
        checksum_f32(&[]),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}
use crate::v05::spec::RawExperimentSpec;

/// A resolvable v0.5 spec exercising both captures (selected-rows and
/// summary-only) and three interventions (zero, scale, replace with an
/// inline source) across two layers, two inputs.
fn test_spec() -> ExperimentSpecV1 {
    let text = r#"
schema = "ember.experiment.v1"

[experiment]
name = "runner-test"
description = "unit test of the v0.5 runner"
seed = 42

[model]
path = "/models/tiny.gguf"
expected_sha256 = "aa"

[execution]
mode = "planned"
threads = 1
deterministic = true

[generation]
max_new_tokens = 2
temperature = 0.0

[[inputs]]
id = "i1"
text = "hello world"

[[inputs]]
id = "i2"
text = "second prompt"

[[captures]]
id = "cap-attn"
site = "attention-output"
layers = [0]

[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "cap-mlp"
site = "mlp-output"
layers = [0, 1]
storage = "summary-only"

[captures.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-zero"
site = "mlp-output"
layers = [1]
operation = { kind = "zero" }

[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-scale"
site = "mlp-output"
layers = [1]
operation = { kind = "scale", factor = 0.5 }

[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "iv-replace"
site = "attention-output"
layers = [0]
operation = { kind = "replace" }
source = { kind = "inline-vector", values = [7.0, 8.0, 9.0, 10.0] }

[interventions.tokens]
kind = "prompt-final"

[output]
directory = "runs/runner-test"
"#;
    RawExperimentSpec::from_toml_str(text)
        .expect("spec parses")
        .resolve()
        .expect("spec resolves")
}

fn model_ctx() -> ModelContext<'static> {
    ModelContext::new(ModelFamily::Llama, Some("tiny.gguf"), "llama", 2, 4)
}

fn new_experiment(spec: &ExperimentSpecV1, input_index: usize) -> V05Experiment {
    let mut experiment = V05Experiment::new(
        spec.clone(),
        input_index,
        ModelFacts {
            n_layers: 2,
            embed_dim: 4,
            vocab_size: 100,
            n_heads: 2,
            head_dim: 2,
        },
        Some("model-sha".into()),
        Some("tokenizer-sha".into()),
    );
    // Prompt-final selectors resolve against the tokenization, so inject
    // a synthetic one ("hello world" -> 3 tokens).
    experiment.inject_tokenization(TokenizationInfo {
        text: "hello world".into(),
        normalized_text: "hello world".into(),
        token_ids: vec![1, 2, 3],
        pieces: vec!["<s>".into(), "hello".into(), "world".into()],
        byte_offsets: vec![(0, 0), (0, 5), (6, 11)],
    });
    experiment
}

fn exec_ctx(
    model: ModelContext<'static>,
    phase: ExecutionPhase,
    position: usize,
    token_count: usize,
) -> ExecutionContext<'static> {
    ExecutionContext::new(model, phase, position, token_count, TracingState::Disabled)
}

#[test]
fn prefill_head_row_is_prompt_final_and_captured_before_intervention() {
    use crate::v05::capture::LayerSelector;
    for site in [SemanticHookSite::FinalNormOutput, SemanticHookSite::Logits] {
        for storage in [
            CaptureStorage::SelectedRows,
            CaptureStorage::FullTensor,
            CaptureStorage::SummaryOnly,
        ] {
            let mut spec = test_spec();
            spec.captures.truncate(1);
            spec.captures[0].site = site;
            spec.captures[0].layers = LayerSelector::All("all".into());
            spec.captures[0].storage = storage;
            spec.interventions.truncate(1);
            spec.interventions[0].site = site;
            spec.interventions[0].layers = LayerSelector::All("all".into());
            let mut experiment = new_experiment(&spec, 0);
            let context = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
            experiment.before_prefill(&context).unwrap();
            let mut values = vec![2.0f32; 4];
            experiment
                .fire_site(&context, 0, site, &mut TensorAccess::new(1, 4, &mut values))
                .unwrap();
            assert_eq!(
                values,
                vec![0.0; 4],
                "prompt-final intervention must reach head"
            );
            let result = experiment.result.as_ref().unwrap();
            assert_eq!(result.events[0].positions, vec![2]);
            if storage == CaptureStorage::SummaryOnly {
                assert_eq!(result.summaries[0].positions, vec![2]);
            } else {
                assert_eq!(result.captures[0].positions, vec![2]);
                assert_eq!(result.captures[0].rows, vec![2.0; 4]);
            }
        }
    }
}

/// A spec over the 2-head (`head_dim` 2) test facts at `attention-heads`
/// layer 0, prompt-final, with `body` as its captures and interventions.
fn heads_spec(body: &str) -> Result<ExperimentSpecV1, crate::v05::spec::SpecError> {
    let text = format!(
        r#"
schema = "ember.experiment.v1"

[experiment]
name = "heads-test"

[model]
path = "/models/tiny.gguf"

[[inputs]]
id = "i1"
text = "hello world"
{body}
[output]
directory = "runs/heads-test"
"#
    );
    RawExperimentSpec::from_toml_str(&text)?.resolve()
}

fn fire_heads(
    spec: &ExperimentSpecV1,
    values: &mut [f32],
) -> Result<V05Experiment, ExperimentError> {
    let mut experiment = new_experiment(spec, 0);
    let context = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    experiment.before_prefill(&context)?;
    let mut tensor = TensorAccess::new(3, 4, values);
    experiment.before_o_proj(&LayerContext::new(context, 0), &mut tensor)?;
    Ok(experiment)
}

const HEADS_TOKENS: &str = "[captures.tokens]\nkind = \"prompt-final\"\n";

#[test]
fn heads_select_columns_for_captures_and_operations() {
    let spec = heads_spec(&format!(
        r#"
[[captures]]
id = "all"
site = "attention-heads"
layers = [0]
{HEADS_TOKENS}
[[captures]]
id = "second"
site = "attention-heads"
layers = [0]
heads = [1]
{HEADS_TOKENS}
[[interventions]]
id = "zero-first"
site = "attention-heads"
layers = [0]
heads = [0]
operation = {{ kind = "zero" }}
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "set-second"
site = "attention-heads"
layers = [0]
heads = [1]
operation = {{ kind = "replace" }}
source = {{ kind = "inline-vector", values = [7.0, 8.0] }}
[interventions.tokens]
kind = "prompt-final"
"#
    ))
    .unwrap();
    let mut values: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let experiment = fire_heads(&spec, &mut values).unwrap();
    // Rows 0 and 1 are untouched; the prompt-final row's head 0 is zeroed
    // and head 1 replaced.
    assert_eq!(&values[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(&values[8..], &[0.0, 0.0, 7.0, 8.0]);
    let result = experiment.result.as_ref().unwrap();
    // Captures record the values before the interventions.
    let all = &result.captures[0];
    assert_eq!(
        (all.columns, all.rows.clone()),
        (4, vec![9.0, 10.0, 11.0, 12.0])
    );
    let second = &result.captures[1];
    assert_eq!((second.columns, second.rows.clone()), (2, vec![11.0, 12.0]));
    assert_eq!(second.bytes, 8);
    assert!(result
        .events
        .iter()
        .all(|event| event.applied && event.positions == [2]));
}

#[test]
fn restore_original_with_heads_restores_only_those_heads() {
    let spec = heads_spec(
        r#"
[[interventions]]
id = "double"
site = "attention-heads"
layers = [0]
operation = { kind = "scale", factor = 2.0 }
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "restore-first"
site = "attention-heads"
layers = [0]
heads = [0]
operation = { kind = "restore-original" }
[interventions.tokens]
kind = "prompt-final"
"#,
    )
    .unwrap();
    let mut values: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    fire_heads(&spec, &mut values).unwrap();
    assert_eq!(&values[8..], &[9.0, 10.0, 22.0, 24.0]);
}

/// A full-width capture source gives each intervention its own heads:
/// head 1 from the source's head 1, head 0 from its head 0, even when
/// both read the same capture in one fire.
#[test]
fn full_width_sources_give_each_intervention_its_heads() {
    let spec = heads_spec(&format!(
        r#"
[[captures]]
id = "src"
site = "attention-heads"
layers = [0]
{HEADS_TOKENS}
[[interventions]]
id = "scale"
site = "attention-heads"
layers = [0]
operation = {{ kind = "scale", factor = 10.0 }}
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "patch-second"
site = "attention-heads"
layers = [0]
heads = [1]
operation = {{ kind = "replace" }}
source = {{ kind = "capture-from-current-run", capture_id = "src" }}
[interventions.tokens]
kind = "prompt-final"

[[interventions]]
id = "patch-first"
site = "attention-heads"
layers = [0]
heads = [0]
operation = {{ kind = "replace" }}
source = {{ kind = "capture-from-current-run", capture_id = "src" }}
[interventions.tokens]
kind = "prompt-final"
"#
    ))
    .unwrap();
    let mut values: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    fire_heads(&spec, &mut values).unwrap();
    // Scaled, then both heads patched back from the capture.
    assert_eq!(&values[8..], &[9.0, 10.0, 11.0, 12.0]);
    assert_eq!(&values[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn heads_sources_must_have_the_selected_width() {
    let spec = heads_spec(
        r#"
[[interventions]]
id = "wide"
site = "attention-heads"
layers = [0]
heads = [1]
operation = { kind = "replace" }
source = { kind = "inline-vector", values = [1.0, 2.0, 3.0, 4.0] }
[interventions.tokens]
kind = "prompt-final"
"#,
    )
    .unwrap();
    let mut values = vec![1.0f32; 12];
    let error = fire_heads(&spec, &mut values).err().unwrap();
    assert!(error.message().contains("target has 2 columns"), "{error}");
    assert_eq!(values, vec![1.0; 12], "nothing changed before the error");
}

#[test]
fn heads_beyond_the_model_fail_before_anything_runs() {
    let spec = heads_spec(&format!(
        r#"
[[captures]]
id = "third"
site = "attention-heads"
layers = [0]
heads = [2]
{HEADS_TOKENS}"#
    ))
    .unwrap();
    let mut experiment = new_experiment(&spec, 0);
    let error = experiment
        .before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("head 2 is out of range for a 2-head model"),
        "{error}"
    );
}

#[test]
fn heads_are_valid_only_at_attention_heads_in_ascending_order() {
    let capture = |site: &str, heads: &str| {
        heads_spec(&format!(
            "[[captures]]\nid = \"c\"\nsite = \"{site}\"\nlayers = [0]\n{heads}\n{HEADS_TOKENS}"
        ))
    };
    let message = |result: Result<ExperimentSpecV1, crate::v05::spec::SpecError>| {
        result.unwrap_err().to_string()
    };
    assert!(capture("attention-heads", "heads = [0, 1]").is_ok());
    assert!(
        message(capture("attention-output", "heads = [0]")).contains("use site attention-heads")
    );
    assert!(message(capture("attention-heads", "heads = []")).contains("empty"));
    assert!(message(capture("attention-heads", "heads = [1, 0]")).contains("ascending"));
    assert!(message(capture("attention-heads", "heads = [1, 1]")).contains("ascending"));
}

#[test]
fn the_per_head_stage_routes_to_attention_heads_only() {
    let spec = heads_spec(&format!(
        r#"
[[captures]]
id = "all"
site = "attention-heads"
layers = [1]
{HEADS_TOKENS}"#
    ))
    .unwrap();
    let experiment = new_experiment(&spec, 0);
    assert!(experiment.uses_activation_site(
        ActivationStage::BeforeOProj,
        Some(1),
        ExecutionPhase::Prefill,
    ));
    assert!(!experiment.uses_activation_site(
        ActivationStage::BeforeOProj,
        Some(0),
        ExecutionPhase::Prefill,
    ));
    assert!(!experiment.uses_activation_site(
        ActivationStage::AfterAttention,
        Some(1),
        ExecutionPhase::Prefill,
    ));
    // A spec without the site never asks for the stage.
    let other = new_experiment(&test_spec(), 0);
    assert!((0..2).all(|layer| !other.uses_activation_site(
        ActivationStage::BeforeOProj,
        Some(layer),
        ExecutionPhase::Prefill,
    )));
}

#[test]
fn plan_site_routing_is_phase_and_layer_exact() {
    let experiment = new_experiment(&test_spec(), 0);
    assert!(experiment.uses_activation_site(
        ActivationStage::AfterAttention,
        Some(0),
        ExecutionPhase::Prefill,
    ));
    assert!(!experiment.uses_activation_site(
        ActivationStage::AfterAttention,
        Some(1),
        ExecutionPhase::Prefill,
    ));
    assert!(!experiment.uses_activation_site(
        ActivationStage::AfterAttention,
        Some(0),
        ExecutionPhase::Decode,
    ));
    assert!(experiment.uses_activation_site(
        ActivationStage::AfterMlp,
        Some(1),
        ExecutionPhase::Prefill,
    ));
    assert!(!experiment.uses_activation_site(
        ActivationStage::BeforeLogits,
        None,
        ExecutionPhase::Prefill,
    ));
}

#[test]
fn before_prefill_resolves_captures_and_interventions_per_input() {
    let spec = test_spec();
    let mut e1 = new_experiment(&spec, 0);
    e1.before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
        .expect("prepare i1");
    assert_eq!(e1.captures.len(), 2, "i1 targets both captures");
    assert_eq!(
        e1.interventions.len(),
        3,
        "i1 targets all three interventions"
    );
    assert_eq!(e1.prompt_len, 3);
    // capture layer resolution
    assert_eq!(e1.captures[0].layers, vec![0]);
    assert_eq!(e1.captures[1].layers, vec![0, 1]);
    // selection predicates
    assert!(e1.site_has_captures(SemanticHookSite::AttentionOutput, 0));
    assert!(!e1.site_has_captures(SemanticHookSite::AttentionOutput, 1));
    assert!(e1.site_has_captures(SemanticHookSite::MlpOutput, 1));
    assert!(e1.site_has_interventions(SemanticHookSite::MlpOutput, 1));
    assert!(!e1.site_has_interventions(SemanticHookSite::MlpOutput, 0));
    // result initialized with selection records for static selectors
    let result = e1.result.as_ref().expect("result initialized");
    assert_eq!(result.selection_records.len(), 2);
    assert_eq!(result.input.id, "i1");

    // i2 does not restrict anything: same resolutions
    let mut e2 = new_experiment(&spec, 1);
    e2.before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
        .expect("prepare i2");
    assert_eq!(e2.captures.len(), 2);
    assert_eq!(e2.result.as_ref().unwrap().input.id, "i2");
}

#[test]
fn fire_site_captures_prompt_final_row() {
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    // 3 rows x 4 cols; prompt-final selects the LAST row.
    let mut values = vec![1.0f32; 12];
    values[8..12].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
        .expect("fire capture");

    let result = e.result.as_ref().unwrap();
    assert_eq!(result.captures.len(), 1, "attention-output capture");
    let captured = &result.captures[0];
    assert_eq!(captured.capture_id, "cap-attn");
    assert_eq!(captured.site, SemanticHookSite::AttentionOutput);
    assert_eq!(
        captured.positions,
        vec![2],
        "prompt-final selects last position"
    );
    assert_eq!(captured.columns, 4);
    assert_eq!(captured.rows, vec![10.0, 20.0, 30.0, 40.0]);
    assert!(!captured.full_tensor);
}

#[test]
fn fire_site_captures_summary_only_stats() {
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    let mut values = vec![0.0f32; 12];
    values[8..12].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    // cap-mlp targets layers [0, 1] at residual-post-mlp
    e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
        .expect("fire summary capture");

    let result = e.result.as_ref().unwrap();
    assert_eq!(
        result.captures.len(),
        0,
        "summary-only produces no tensor capture"
    );
    assert_eq!(result.summaries.len(), 1);
    let summary = &result.summaries[0];
    assert_eq!(summary.capture_id, "cap-mlp");
    assert_eq!(summary.layer, 1);
    assert_eq!(summary.shape, [1, 4]);
    assert_eq!(summary.minimum, 1.0);
    assert_eq!(summary.maximum, 4.0);
    assert!((summary.mean - 2.5).abs() < 1e-6);
    assert_eq!(summary.finite_count, 4);
}

#[test]
fn fire_site_zero_intervention_zeros_target_row() {
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    let mut values = vec![5.0f32; 12];
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
        .expect("fire zero intervention");

    // iv-zero targets (residual-post-mlp, layer 1), prompt-final -> row 2.
    assert_eq!(tensor.values()[8..12], vec![0.0; 4]);
    // other rows untouched
    assert_eq!(tensor.values()[0..8], vec![5.0; 8]);

    let result = e.result.as_ref().unwrap();
    let event = result
        .events
        .iter()
        .find(|ev| ev.intervention_id == "iv-zero")
        .expect("zero event recorded");
    assert!(event.applied);
    assert_eq!(event.positions, vec![2]);
    assert!(
        event.snapshot_checksum.is_some(),
        "snapshot taken before zeroing"
    );
}

#[test]
fn fire_site_scale_intervention_scales_row() {
    let mut spec = test_spec();
    // Isolate: without iv-zero firing first (declaration order), the row
    // is scaled but not zeroed.
    spec.interventions.retain(|i| i.id == "iv-scale");
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    let mut values = vec![4.0f32; 12];
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_mlp(&LayerContext::new(ctx, 1), &mut tensor)
        .expect("fire scale intervention");

    // iv-scale: factor 0.5 on the prompt-final row (index 2).
    assert_eq!(tensor.values()[8..12], vec![2.0; 4]);
    assert_eq!(tensor.values()[0..8], vec![4.0; 8]);
}

#[test]
fn fire_site_replace_uses_inline_source_and_checks_columns() {
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    let mut values = vec![1.0f32; 12];
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
        .expect("fire replace");
    assert_eq!(
        tensor.values()[8..12],
        vec![7.0, 8.0, 9.0, 10.0],
        "inline source replaces the prompt-final row"
    );
    assert_eq!(tensor.values()[0..8], vec![1.0; 8]);

    // Column mismatch must be a clean error, not a panic.
    let mut e2 = new_experiment(&spec, 0);
    e2.before_prefill(&ctx).expect("prepare");
    let mut bad_values = vec![1.0f32; 6]; // 3 rows x 2 cols
    let mut bad_tensor = TensorAccess::new(3, 2, &mut bad_values);
    let err = e2
        .after_attention(&LayerContext::new(ctx, 0), &mut bad_tensor)
        .expect_err("inline source width mismatch is an error");
    assert!(err.message().contains("inline source"), "{err:?}");
}

#[test]
fn current_run_source_uses_the_target_layer_not_the_first_capture() {
    use crate::v05::capture::LayerSelector;
    for source_has_target in [false, true] {
        let mut spec = test_spec();
        spec.captures.truncate(1);
        spec.captures[0].layers = LayerSelector::List(if source_has_target {
            vec![0, 1]
        } else {
            vec![0]
        });
        spec.interventions.retain(|i| i.id == "iv-replace");
        spec.interventions[0].layers = LayerSelector::List(vec![1]);
        spec.interventions[0].source = Some(InterventionSource::CaptureFromCurrentRun {
            capture_id: "cap-attn".into(),
        });
        let mut experiment = new_experiment(&spec, 0);
        let context = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        experiment.before_prefill(&context).unwrap();
        let mut first = vec![1.0; 12];
        experiment
            .after_attention(
                &LayerContext::new(context, 0),
                &mut TensorAccess::new(3, 4, &mut first),
            )
            .unwrap();
        let mut second = vec![2.0; 12];
        let outcome = experiment.after_attention(
            &LayerContext::new(context, 1),
            &mut TensorAccess::new(3, 4, &mut second),
        );
        if source_has_target {
            outcome.unwrap();
        } else {
            assert!(outcome
                .unwrap_err()
                .message()
                .contains("site/layer mismatch"));
        }
        assert_eq!(
            second,
            vec![2.0; 12],
            "must never copy layer zero into layer one"
        );
    }
}

#[test]
fn bundle_source_layer_must_match_all_target_layers_before_loading() {
    use crate::v05::capture::LayerSelector;
    let mut intervention = test_spec().interventions.remove(2);
    intervention.layers = LayerSelector::List(vec![1]);
    let source = InterventionSource::CaptureFromBundle {
        bundle_path: "does-not-exist".into(),
        capture_id: "capture".into(),
        input_id: "i1".into(),
        layer: 0,
        semantic_hash: None,
    };
    let error = load_bundle_source(&intervention, &source, "model", "tokenizer", 2).unwrap_err();
    assert!(error.contains("does not match target layers"), "{error}");
    intervention.layers = LayerSelector::List(vec![0, 1]);
    let error = load_bundle_source(&intervention, &source, "model", "tokenizer", 2).unwrap_err();
    assert!(error.contains("does not match target layers"), "{error}");
}

#[test]
fn cross_bundle_sources_enforce_identity_site_and_verified_payload() {
    use crate::v05::capture::LayerSelector;
    use crate::v05::testutil::{
        temp_root, write_test_bundle, FIXTURE_MODEL_SHA, FIXTURE_TOKENIZER_SHA,
    };
    let root = temp_root("source-compatibility");
    let rows = vec![1.0, 2.0, 3.0, 4.0];
    write_test_bundle(&root, &rows, &[0]);
    let source = InterventionSource::CaptureFromBundle {
        bundle_path: root.clone(),
        capture_id: "cap-1".into(),
        input_id: "i1".into(),
        layer: 0,
        semantic_hash: None,
    };
    let mut intervention = test_spec().interventions.remove(2);
    intervention.site = SemanticHookSite::ResidualPostMlp;
    intervention.layers = LayerSelector::List(vec![0]);
    let load =
        |target: &InterventionSpec, source: &InterventionSource, model: &str, tokenizer: &str| {
            load_bundle_source(target, source, model, tokenizer, 2)
        };
    let valid = load(
        &intervention,
        &source,
        FIXTURE_MODEL_SHA,
        FIXTURE_TOKENIZER_SHA,
    )
    .unwrap();
    assert_eq!(valid.rows, rows);
    assert_eq!(valid.columns, 4);
    let error = load(
        &intervention,
        &source,
        "different-model",
        FIXTURE_TOKENIZER_SHA,
    )
    .unwrap_err();
    assert!(error.contains("model SHA"), "{error}");
    let error = load(
        &intervention,
        &source,
        FIXTURE_MODEL_SHA,
        "different-tokenizer",
    )
    .unwrap_err();
    assert!(error.contains("tokenizer SHA"), "{error}");
    // Overrides apply to the named identity only; they cannot bypass the
    // other identity, semantic site, or source bundle integrity.
    // Unpinned: the resolved source is recorded.
    assert_eq!(valid.warnings.len(), 1, "{:?}", valid.warnings);
    assert!(valid.warnings[0].contains("not pinned"));
    // A pinned semantic hash must match the bundle at the path.
    let pinned = |hash: &str| {
        let mut pinned = source.clone();
        if let InterventionSource::CaptureFromBundle { semantic_hash, .. } = &mut pinned {
            *semantic_hash = Some(hash.to_string());
        }
        pinned
    };
    let actual = crate::v05::verify::load_bundle_for_source(&root)
        .unwrap()
        .semantic_hash;
    let pinned_ok = load(
        &intervention,
        &pinned(&actual),
        FIXTURE_MODEL_SHA,
        FIXTURE_TOKENIZER_SHA,
    )
    .unwrap();
    assert!(pinned_ok.warnings.is_empty(), "{:?}", pinned_ok.warnings);
    let error = load(
        &intervention,
        &pinned(&"0".repeat(64)),
        FIXTURE_MODEL_SHA,
        FIXTURE_TOKENIZER_SHA,
    )
    .unwrap_err();
    assert!(error.contains("is pinned"), "{error}");
    intervention.compatibility.allow_model_mismatch = true;
    let overridden = load(
        &intervention,
        &source,
        "different-model",
        FIXTURE_TOKENIZER_SHA,
    )
    .unwrap();
    assert_eq!(overridden.warnings.len(), 2);
    assert!(overridden.warnings[1].contains("allow_model_mismatch used"));
    assert!(load(
        &intervention,
        &source,
        "different-model",
        "different-tokenizer"
    )
    .is_err());
    intervention.compatibility.allow_tokenizer_mismatch = true;
    assert!(load(
        &intervention,
        &source,
        "different-model",
        "different-tokenizer"
    )
    .is_ok());
    intervention.site = SemanticHookSite::AttentionOutput;
    let error = load(
        &intervention,
        &source,
        "different-model",
        "different-tokenizer",
    )
    .unwrap_err();
    assert!(error.contains("source capture site"), "{error}");
    intervention.site = SemanticHookSite::ResidualPostMlp;
    let mut missing = source.clone();
    if let InterventionSource::CaptureFromBundle { input_id, .. } = &mut missing {
        *input_id = "absent-input".into();
    }
    let error = load(
        &intervention,
        &missing,
        "different-model",
        "different-tokenizer",
    )
    .unwrap_err();
    assert!(error.contains("has no capture"), "{error}");
    let payload = root.join("captures/tensors.safetensors");
    let mut bytes = std::fs::read(&payload).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&payload, bytes).unwrap();
    assert!(
        load(
            &intervention,
            &source,
            "different-model",
            "different-tokenizer"
        )
        .is_err(),
        "identity overrides must not permit a corrupted source"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn captured_source_row_count_mismatch_fails_before_mutation() {
    for operation in [
        InterventionOperation::Replace,
        InterventionOperation::Interpolate { alpha: 0.5 },
        InterventionOperation::AddDelta,
    ] {
        let mut spec = test_spec();
        spec.captures.truncate(1);
        spec.captures[0].storage = CaptureStorage::FullTensor;
        spec.interventions.retain(|i| i.id == "iv-replace");
        spec.interventions[0].operation = operation;
        spec.interventions[0].source = Some(InterventionSource::CaptureFromCurrentRun {
            capture_id: "cap-attn".into(),
        });
        let mut experiment = new_experiment(&spec, 0);
        let context = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
        experiment.before_prefill(&context).unwrap();
        let mut values = vec![1.0f32; 12];
        let error = experiment
            .after_attention(
                &LayerContext::new(context, 0),
                &mut TensorAccess::new(3, 4, &mut values),
            )
            .expect_err("three source rows cannot fit one target row");
        assert!(error.message().contains("selected rows"), "{error:?}");
        assert_eq!(values, vec![1.0; 12]);
        assert!(experiment.result.as_ref().unwrap().events.is_empty());
    }
}

#[test]
fn captured_sources_apply_pairwise_or_broadcast_for_consuming_operations() {
    for operation in [
        InterventionOperation::Replace,
        InterventionOperation::Interpolate { alpha: 0.5 },
        InterventionOperation::AddDelta,
    ] {
        for broadcast in [false, true] {
            let mut spec = test_spec();
            spec.captures.truncate(1);
            spec.captures[0].storage = if broadcast {
                CaptureStorage::SelectedRows
            } else {
                CaptureStorage::FullTensor
            };
            spec.interventions.retain(|i| i.id == "iv-replace");
            spec.interventions[0].operation = operation;
            spec.interventions[0].source = Some(InterventionSource::CaptureFromCurrentRun {
                capture_id: "cap-attn".into(),
            });
            let mut experiment = new_experiment(&spec, 0);
            let context = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
            experiment.before_prefill(&context).unwrap();
            // Supply an already resolved three-row selector to isolate the
            // runtime source/target mapping from tokenizer span behavior.
            experiment.interventions[0]
                .static_record
                .as_mut()
                .unwrap()
                .selected_indices = vec![0, 1, 2];
            let original = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let mut values = original.clone();
            experiment
                .after_attention(
                    &LayerContext::new(context, 0),
                    &mut TensorAccess::new(3, 4, &mut values),
                )
                .unwrap();
            for index in 0..values.len() {
                let source = if broadcast {
                    original[8 + index % 4]
                } else {
                    original[index]
                };
                let expected = match operation {
                    InterventionOperation::Replace => source,
                    InterventionOperation::Interpolate { .. } => {
                        0.5 * original[index] + 0.5 * source
                    }
                    InterventionOperation::AddDelta => original[index] + source,
                    _ => unreachable!(),
                };
                assert_eq!(
                    values[index], expected,
                    "{operation:?}, broadcast={broadcast}, element={index}"
                );
            }
        }
    }
}

#[test]
fn restore_original_restores_pre_intervention_snapshot() {
    // Two interventions at the same site in declaration order: replace
    // (mutates), then restore-original (writes the snapshot back).
    let mut spec = test_spec();
    // build a derived spec adding restore-original after the replace
    let mut interventions = spec.interventions.clone();
    let mut restore = interventions[2].clone(); // iv-replace
    restore.id = "iv-restore".into();
    restore.operation = InterventionOperation::RestoreOriginal;
    restore.source = None;
    interventions.push(restore);
    spec.interventions = interventions;

    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");

    let mut values = vec![3.0f32; 12];
    values[8..12].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
        .expect("fire replace+restore");

    // Replace writes 7..10, then restore-original writes the snapshot
    // (the pre-intervention row) back -> original 1.0s.
    assert_eq!(tensor.values()[8..12], vec![1.0, 1.0, 1.0, 1.0]);
    let result = e.result.as_ref().unwrap();
    let events: Vec<_> = result
        .events
        .iter()
        .filter(|ev| ev.intervention_id == "iv-restore")
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].applied);
    assert!(events[0].snapshot_checksum.is_some());
}

#[test]
fn decode_phase_generated_step_intervention_applies_to_decode_row() {
    // One input; iv-zero (mlp-output, layer 1, zero) retargeted to
    // generated step 1 (the first decode token, absolute position =
    // prompt_len + 0).
    let mut spec = test_spec();
    spec.inputs.truncate(1);
    spec.captures.clear();
    spec.interventions.truncate(1);
    spec.interventions[0].id = "iv-gen".into();
    spec.interventions[0].tokens = TokenSelector::GeneratedStep { step: 1 };
    let mut e = new_experiment(&spec, 0);
    let prefill = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&prefill).expect("prepare");

    let mut prompt = vec![2.0f32; 12];
    e.after_mlp(
        &LayerContext::new(prefill, 1),
        &mut TensorAccess::new(3, 4, &mut prompt),
    )
    .expect("prefill must not intervene");
    assert_eq!(prompt, vec![2.0; 12]);
    assert!(e.snapshots.is_empty());
    assert!(e.result.as_ref().unwrap().events.is_empty());

    // Decode step 1 = absolute position 3 (prompt_len 3 + step 1 - 1).
    let decode = exec_ctx(model_ctx(), ExecutionPhase::Decode, 3, 1);
    let mut values = vec![2.0f32; 4];
    let mut tensor = TensorAccess::new(1, 4, &mut values);
    e.after_mlp(&LayerContext::new(decode, 1), &mut tensor)
        .expect("fire decode intervention");

    assert_eq!(
        tensor.values(),
        vec![0.0; 4],
        "zero applies to the decode row"
    );
    let event = e
        .result
        .as_ref()
        .unwrap()
        .events
        .iter()
        .find(|ev| ev.intervention_id == "iv-gen")
        .expect("event");
    assert_eq!(event.positions, vec![3], "absolute position recorded");
}

#[test]
fn generated_step_capture_fires_only_on_requested_step() {
    // One input; an mlp-output layer-0 selected-rows capture of
    // generated step 1, no interventions.
    let mut spec = test_spec();
    spec.inputs.truncate(1);
    spec.interventions.clear();
    spec.captures.remove(0);
    spec.captures[0].id = "cap-gen".into();
    spec.captures[0].layers = LayerSelector::List(vec![0]);
    spec.captures[0].storage = CaptureStorage::SelectedRows;
    spec.captures[0].tokens = TokenSelector::GeneratedStep { step: 1 };
    let mut e = new_experiment(&spec, 0);
    let prefill = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&prefill).expect("prepare");

    let mut prompt_values = vec![8.0f32; 12];
    e.after_mlp(
        &LayerContext::new(exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3), 0),
        &mut TensorAccess::new(3, 4, &mut prompt_values),
    )
    .expect("prefill hook");
    assert!(
        e.result.as_ref().unwrap().captures.is_empty(),
        "generated step must not capture prompt row zero"
    );
    let mut values = vec![9.0f32; 4];
    let mut tensor = TensorAccess::new(1, 4, &mut values);
    // decode step 1 = position 3 (1-based: prompt_len 3 + step 1 - 1):
    // the requested step, so the capture fires here.
    e.after_mlp(
        &LayerContext::new(exec_ctx(model_ctx(), ExecutionPhase::Decode, 3, 1), 0),
        &mut tensor,
    )
    .expect("fire");
    let captures = &e.result.as_ref().unwrap().captures;
    assert_eq!(captures.len(), 1, "generated step 1 fires at position 3");
    assert_eq!(captures[0].capture_id, "cap-gen");
    assert_eq!(captures[0].positions, vec![3]);
    assert_eq!(captures[0].rows, vec![9.0; 4]);
    // decode step 2 (position 4): not requested -> no additional capture
    e.after_mlp(
        &LayerContext::new(exec_ctx(model_ctx(), ExecutionPhase::Decode, 4, 1), 0),
        &mut tensor,
    )
    .expect("fire");
    assert_eq!(e.result.as_ref().unwrap().captures.len(), 1);

    // finalize_generated appends the generated-step selection record.
    e.finalize_generated(&[7, 8]).expect("finalize");
    let result = e.result.as_ref().unwrap();
    assert!(result.generated_token_ids.is_empty()); // not set by finalize
    let record = result
        .selection_records
        .iter()
        .find(|r| r.selector.rule_id() == "generated-step")
        .expect("generated-step selection record");
    assert_eq!(
        record.selected_indices,
        vec![3],
        "prompt_len 3 + (step 1 - 1)"
    );
}

#[test]
fn fire_site_before_prefill_is_a_noop() {
    // Without before_prefill there are no resolved captures/interventions,
    // so hook fires are harmless no-ops (the "result not initialized"
    // error is only reachable if prepare_input succeeded but before_prefill
    // failed midway — a defensive path).
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    let mut values = vec![1.0f32; 4];
    let mut tensor = TensorAccess::new(1, 4, &mut values);
    e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
        .expect("noop fire succeeds");
    assert_eq!(tensor.values(), vec![1.0; 4]);
    assert!(e.result.is_none());
}

#[test]
fn into_result_orders_merged_rows_without_changing_their_bits() {
    let spec = test_spec();
    let mut experiment = new_experiment(&spec, 0);
    experiment
        .before_prefill(&exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3))
        .unwrap();
    let capture = CapturedTensor {
        capture_id: "cap-attn".into(),
        input_id: "i1".into(),
        site: SemanticHookSite::AttentionOutput,
        layer: 0,
        positions: vec![7, 3],
        rows: vec![70.0, -0.0, 30.0, 31.0],
        columns: 2,
        full_tensor: false,
        bytes: 16,
        dtype: crate::v05::capture::CaptureDType::F32,
    };
    let later_buffer = CapturedTensor {
        positions: vec![5],
        rows: vec![50.0, 51.0],
        bytes: 8,
        ..capture.clone()
    };
    experiment.result.as_mut().unwrap().captures = vec![capture, later_buffer];
    let result = experiment.into_result().unwrap();
    assert_eq!(result.captures.len(), 1);
    let merged = &result.captures[0];
    assert_eq!(merged.positions, [3, 5, 7]);
    assert_eq!(merged.columns, 2);
    assert_eq!(merged.bytes, 24);
    let expected = [30.0_f32, 31.0, 50.0, 51.0, 70.0, -0.0];
    assert_eq!(
        merged.rows.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn into_result_assembles_complete_input_result() {
    let spec = test_spec();
    let mut e = new_experiment(&spec, 0);
    let ctx = exec_ctx(model_ctx(), ExecutionPhase::Prefill, 0, 3);
    e.before_prefill(&ctx).expect("prepare");
    let mut values = vec![1.0f32; 12];
    let mut tensor = TensorAccess::new(3, 4, &mut values);
    e.after_attention(&LayerContext::new(ctx, 0), &mut tensor)
        .expect("fire");
    e.finalize_generated(&[5]).expect("finalize");
    e.set_generated_text("hello".into());

    let result = e.into_result().expect("into_result");
    assert_eq!(result.input.id, "i1");
    assert_eq!(result.generated_text, "hello");
    assert_eq!(result.captures.len(), 1);
    assert_eq!(result.selection_records.len(), 2);
    assert_eq!(
        result.events.len(),
        1,
        "iv-replace applied at attention-output"
    );
    assert!(result.final_top1.is_none());
}

use super::*;
use crate::cli_commands::validate_experiment_options;
use crate::cli_support::{build_run_manifest, default_tokenizer_for_arch};
use crate::{Commands, LifecycleModeArg, PackedSelectionArg};
use clap::Parser;

use ember::{backend::CpuError, kv_cache::KVCache, tensor::CpuTensor};

/// A one-layer, four-token model whose forwards log which entry point
/// ran: prefill/allocating forwards rank token 1 first, reusing forwards
/// token 2.
#[derive(Default)]
struct DispatchModel(std::cell::RefCell<Vec<&'static str>>);
impl ForwardModel<CpuBackend> for DispatchModel {
    fn create_cache(&self, _: &CpuBackend, capacity: usize) -> KVCache {
        KVCache::new(1, 1, 1, capacity)
    }
    fn max_seq_len(&self, _: &CpuBackend) -> usize {
        64
    }
    fn n_layers(&self) -> usize {
        1
    }
    fn embed_dim(&self) -> usize {
        1
    }
    fn vocab_size(&self, _: &CpuBackend) -> usize {
        4
    }
    fn forward_with_cache(
        &self,
        _: &CpuBackend,
        _: &[u32],
        _: &mut KVCache,
        _: usize,
    ) -> Result<CpuTensor, CpuError> {
        panic!("generation should request last-position logits")
    }
    fn forward_last_logits_with_cache(
        &self,
        _: &CpuBackend,
        ids: &[u32],
        cache: &mut KVCache,
        start: usize,
    ) -> Result<CpuTensor, CpuError> {
        self.0.borrow_mut().push("allocated");
        cache.validate_start_pos(start);
        for _ in ids {
            cache.advance_cursor();
        }
        Ok(CpuTensor::from_data(vec![1, 4], vec![0.0, 2.0, 1.0, -10.0]))
    }
    fn forward_last_logits_with_cache_reusing(
        &self,
        _: &CpuBackend,
        ids: &[u32],
        cache: &mut KVCache,
        start: usize,
        output: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        self.0.borrow_mut().push("reused");
        cache.validate_start_pos(start);
        for _ in ids {
            cache.advance_cursor();
        }
        output.data_mut().copy_from_slice(&[0.0, 1.0, 2.0, -10.0]);
        Ok(())
    }
    fn forward_with_activations(
        &self,
        _: &CpuBackend,
        _: &[u32],
    ) -> Result<(Vec<Vec<f32>>, CpuTensor), CpuError> {
        panic!("generation should not capture activations")
    }
}

fn word_tokenizer() -> ember::tokenizer::EmberTokenizer {
    ember::tokenizer::EmberTokenizer::from_bytes(
        r#"{
        "version":"1.0", "truncation":null, "padding":null,
        "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
        "post_processor":null, "decoder":null,
        "model":{"type":"WordLevel","vocab":{"p":0,"a":1,"b":2,"[UNK]":3},"unk_token":"[UNK]"}
    }"#,
    )
    .unwrap()
}

/// Three tokens from prompt "p" with top-k=1, which makes the run
/// deterministic without consulting the optional fused-greedy
/// environment setting or mutating process-global state.
fn generate<E: GenerationExecution<CpuBackend, DispatchModel>>(
    model: &DispatchModel,
    execution: &mut E,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<String> {
    generate_with_execution(
        &CpuBackend,
        model,
        execution,
        &word_tokenizer(),
        "p",
        3,
        1.0,
        Some(1),
        None,
        false,
        false,
        None,
        false,
        false,
        1,
        64,
        Some(1),
        cancel,
    )
}

#[test]
fn generation_reuses_standard_logits_without_bypassing_custom_execution() {
    #[derive(Default)]
    struct CustomExecution(Vec<ExecutionPhase>);
    impl GenerationExecution<CpuBackend, DispatchModel> for CustomExecution {
        fn before_prefill(&mut self, _: &[u32]) -> anyhow::Result<()> {
            Ok(())
        }
        fn forward_last_logits(
            &mut self,
            backend: &CpuBackend,
            model: &DispatchModel,
            ids: &[u32],
            cache: &mut KVCache,
            start: usize,
            phase: ExecutionPhase,
        ) -> Result<CpuTensor, CpuError> {
            self.0.push(phase);
            let mut output = model.forward_last_logits_with_cache(backend, ids, cache, start)?;
            // This models a custom execution hook that changes the logits.
            output.data_mut().copy_from_slice(&[0.0, 1.0, 2.0, -10.0]);
            Ok(output)
        }
        fn generation_complete(
            &mut self,
            prompt: usize,
            generated: usize,
            evaluations: usize,
            _: &[u32],
            _: &[u32],
        ) -> anyhow::Result<()> {
            assert_eq!((prompt, generated, evaluations), (1, 3, 2));
            Ok(())
        }
    }

    let model = DispatchModel::default();
    let output = generate(&model, &mut StandardGeneration, None).unwrap();
    assert_eq!(output, "a b b");
    assert_eq!(*model.0.borrow(), ["allocated", "reused", "reused"]);
    model.0.borrow_mut().clear();
    let mut custom = CustomExecution::default();
    let output = generate(&model, &mut custom, None).unwrap();
    assert_eq!(output, "b b b");
    assert_eq!(*model.0.borrow(), ["allocated", "allocated", "allocated"]);
    assert_eq!(
        custom.0,
        [
            ExecutionPhase::Prefill,
            ExecutionPhase::Decode,
            ExecutionPhase::Decode
        ]
    );
}

#[test]
fn cancellation_stops_before_prefill_and_at_step_boundaries() {
    // (a) a pre-cancelled token stops before any model work happens.
    let model = DispatchModel::default();
    let token = CancelToken::new();
    token.cancel();
    let error = generate(&model, &mut StandardGeneration, Some(&token)).unwrap_err();
    assert!(error.is::<Cancelled>());
    assert!(
        model.0.borrow().is_empty(),
        "no forward should run after a pre-cancel"
    );

    // (b) a token fired during the first decode forward is honored at the
    // next step boundary: one prefill + one decode, never the second.
    struct CancelDuringDecode {
        token: CancelToken,
        phases: Vec<ExecutionPhase>,
    }
    impl GenerationExecution<CpuBackend, DispatchModel> for CancelDuringDecode {
        fn before_prefill(&mut self, _: &[u32]) -> anyhow::Result<()> {
            Ok(())
        }
        fn forward_last_logits(
            &mut self,
            _: &CpuBackend,
            _: &DispatchModel,
            ids: &[u32],
            cache: &mut KVCache,
            start: usize,
            phase: ExecutionPhase,
        ) -> Result<CpuTensor, CpuError> {
            self.phases.push(phase);
            if phase == ExecutionPhase::Decode {
                self.token.cancel();
            }
            cache.validate_start_pos(start);
            for _ in ids {
                cache.advance_cursor();
            }
            Ok(CpuTensor::from_data(vec![1, 4], vec![0.0, 2.0, 1.0, -10.0]))
        }
        fn generation_complete(
            &mut self,
            _: usize,
            _: usize,
            _: usize,
            _: &[u32],
            _: &[u32],
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }
    let model = DispatchModel::default();
    let token = CancelToken::new();
    let mut custom = CancelDuringDecode {
        token: token.clone(),
        phases: Vec::new(),
    };
    let error = generate(&model, &mut custom, Some(&token)).unwrap_err();
    assert!(error.is::<Cancelled>());
    assert_eq!(
        custom.phases,
        [ExecutionPhase::Prefill, ExecutionPhase::Decode],
        "the loop must stop at the step boundary after cancellation"
    );
}

#[test]
fn final_requested_token_needs_no_followup_evaluation() {
    assert!(!has_next_decode_evaluation(0, 1));
    assert!(has_next_decode_evaluation(0, 2));
    assert!(!has_next_decode_evaluation(1, 2));
}

#[test]
fn default_tokenizer_tracks_architecture() {
    let automatic = Args::try_parse_from(["ember"]).expect("default args should parse");
    assert_eq!(automatic.arch, "auto");
    assert!(automatic.tokenizer.is_none());

    for (arch, expected) in [
        ("llama", "tokenizer.json"),
        ("gemma4", "tokenizer-gemma4.json"),
        ("qwen3", "tokenizer-qwen3.json"),
    ] {
        let args = Args::try_parse_from(["ember", "--arch", arch])
            .unwrap_or_else(|error| panic!("{arch} args should parse: {error}"));
        assert_eq!(
            args.tokenizer
                .as_deref()
                .unwrap_or_else(|| default_tokenizer_for_arch(&args.arch)),
            expected
        );
    }
}

#[test]
fn cli_rejects_invalid_sampling_args() {
    assert!(Args::try_parse_from(["ember", "--temperature", "-0.1"]).is_err());
    assert!(Args::try_parse_from(["ember", "--top-k", "0"]).is_err());
    assert!(Args::try_parse_from(["ember", "--top-p", "0"]).is_err());
    assert!(Args::try_parse_from(["ember", "--top-p", "1.1"]).is_err());
    assert!(Args::try_parse_from(["ember", "--max-seq-len", "0"]).is_err());
}

#[test]
fn trace_and_probe_specific_options_cannot_be_silent_noops() {
    assert!(Args::try_parse_from(["ember", "--trace-out", "trace.json"]).is_err());
    assert!(Args::try_parse_from(["ember", "--trace", "ops", "--probe",]).is_err());

    let trace_values = Args::try_parse_from(["ember", "--trace-values", "summary"]).unwrap();
    assert!(validate_experiment_options(&trace_values)
        .unwrap_err()
        .to_string()
        .contains("requires --trace"));

    let probe_option = Args::try_parse_from(["ember", "--probe-limit", "2"]).unwrap();
    assert!(validate_experiment_options(&probe_option)
        .unwrap_err()
        .to_string()
        .contains("require --probe"));
}

#[test]
fn zero_layer_output_cli_parses_typed_spec() {
    let args = Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--zero-layer-output",
        "4:attention",
    ])
    .unwrap();
    let spec = args.zero_layer_output.unwrap();
    assert_eq!(spec.layer(), 4);
    assert_eq!(
        spec.stage(),
        ember::experiments::ZeroLayerOutputStage::Attention
    );
    validate_experiment_options(&args).unwrap();
}

#[test]
fn zero_layer_output_cli_rejects_malformed_and_incompatible_uses() {
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "llama",
        "--zero-layer-output",
        "4:residual",
    ])
    .is_err());
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "gemma4",
        "--zero-layer-output",
        "4:attention",
        "--dump-layers",
        "layers.bin",
    ])
    .is_err());

    let gpt2 = Args::try_parse_from(["ember", "--arch", "gpt2", "--zero-layer-output", "0:layer"])
        .unwrap();
    assert!(validate_experiment_options(&gpt2)
        .unwrap_err()
        .to_string()
        .contains("not gpt2"));

    let subcommand = Args::try_parse_from([
        "ember",
        "--zero-layer-output",
        "0:layer",
        "bench-decode",
        "--model",
        "model.gguf",
        "--arch",
        "llama",
    ])
    .unwrap();
    assert!(validate_experiment_options(&subcommand)
        .unwrap_err()
        .to_string()
        .contains("normal generation"));
}

#[test]
fn activation_stats_cli_parses_and_rejects_incompatible_uses() {
    let args = Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--activation-stats",
        "activation-stats.json",
    ])
    .unwrap();
    assert_eq!(
        args.activation_stats.as_deref(),
        Some("activation-stats.json")
    );
    validate_experiment_options(&args).unwrap();

    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--activation-stats",
        "stats.json",
        "--zero-layer-output",
        "1:mlp",
    ])
    .is_err());
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "gemma4",
        "--activation-stats",
        "stats.json",
        "--dump-layers",
        "layers.bin",
    ])
    .is_err());

    let gpt2 = Args::try_parse_from([
        "ember",
        "--arch",
        "gpt2",
        "--activation-stats",
        "stats.json",
    ])
    .unwrap();
    assert!(validate_experiment_options(&gpt2)
        .unwrap_err()
        .to_string()
        .contains("not gpt2"));
}

#[test]
fn activation_patch_cli_requires_targets_and_rejects_silent_noop_modes() {
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--activation-patch",
        "manifest.json"
    ])
    .is_err());
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--patch-target",
        "1:after-mlp:prefill"
    ])
    .is_err());
    assert!(Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--activation-patch",
        "manifest.json",
        "--patch-target",
        "1:after-mlp:prefill",
        "--dump-logits",
        "logits.npy"
    ])
    .is_err());
}

#[test]
fn run_manifest_omits_disabled_experiment_and_records_active_one() {
    let normal = Args::try_parse_from(["ember", "--arch", "llama"]).unwrap();
    let normal_manifest = build_run_manifest(
        &normal,
        "tokenizer.json",
        None,
        None,
        &serde_json::json!({}),
        "test prompt",
    );
    assert!(normal_manifest["execution"].get("experiment").is_none());

    let active =
        Args::try_parse_from(["ember", "--arch", "gemma4", "--zero-layer-output", "2:mlp"])
            .unwrap();
    let active_manifest = build_run_manifest(
        &active,
        "tokenizer.json",
        None,
        None,
        &serde_json::json!({}),
        "test prompt",
    );
    assert_eq!(
        active_manifest["execution"]["experiment"],
        serde_json::json!({
            "name": "zero-layer-output",
            "layer": 2,
            "stage": "mlp",
            "modifies_execution": true,
        })
    );

    let observation = Args::try_parse_from([
        "ember",
        "--arch",
        "qwen3",
        "--activation-stats",
        "stats.json",
    ])
    .unwrap();
    let observation_manifest = build_run_manifest(
        &observation,
        "tokenizer-qwen3.json",
        None,
        None,
        &serde_json::json!({}),
        "test prompt",
    );
    assert_eq!(
        observation_manifest["execution"]["experiment"],
        serde_json::json!({
            "name": "activation-stats",
            "output": "stats.json",
            "modifies_execution": false,
        })
    );
}

#[test]
fn bench_decode_subcommand_parses_matched_timing_options() {
    let args = Args::try_parse_from([
        "ember",
        "bench-decode",
        "--model",
        "model.gguf",
        "--arch",
        "gemma4",
        "--tokens",
        "16",
        "--warmups",
        "1",
        "--repetitions",
        "3",
    ])
    .unwrap();
    match args.command {
        Some(Commands::BenchDecode(command)) => {
            assert_eq!(command.tokens, 16);
            assert_eq!(command.warmups, 1);
            assert_eq!(command.repetitions, 3);
        }
        _ => panic!("expected bench-decode command"),
    }
}

#[test]
fn bench_lifecycle_subcommand_parses_explicit_experiment_modes() {
    let args = Args::try_parse_from([
        "ember",
        "bench-lifecycle",
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.json",
        "--lifecycle",
        "pack-before-prefill-reevict",
        "--selection",
        "attention-gate-up",
        "--tokens",
        "32",
    ])
    .unwrap();
    match args.command {
        Some(Commands::BenchLifecycle(command)) => {
            assert!(matches!(
                command.lifecycle,
                LifecycleModeArg::PackBeforePrefillReevict
            ));
            assert!(matches!(
                command.selection,
                PackedSelectionArg::AttentionGateUp
            ));
            assert_eq!(command.tokens, 32);
            assert!(!command.timing_only);
        }
        _ => panic!("expected bench-lifecycle command"),
    }
}

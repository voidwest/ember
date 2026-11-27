use super::*;

fn test_config() -> ExtractionConfig {
    ExtractionConfig {
        run_id: None,
        model_path: "model.gguf".to_string(),
        architecture: Some("llama".to_string()),
        tokenizer_path: None,
        backend: ExecutionBackendName::Native,
        prompt_template: "word: {word}".to_string(),
        input_jsonl_path: "input.jsonl".to_string(),
        output_dir: "out".to_string(),
        layers: vec![0, 2],
        token_position: TokenPositionMode::PromptFinal,
        word_field: "word".to_string(),
        sample_id_field: "id".to_string(),
        batch_size: 1,
        dtype: ArtifactDType::F32,
        output_format: ArtifactOutputFormat::Npy,
        prompt_hashes_only: false,
        write_logits: false,
        resume: false,
        max_seq_len: None,
        record_model_sha256: false,
        llama_cpp_binary: None,
        run_metadata: Value::Null,
    }
}

#[test]
fn config_defaults_validate() {
    let config = test_config();
    config.validate().expect("valid extraction config");
    assert_eq!(config.effective_layers(4).unwrap(), vec![0, 2]);
}

#[test]
fn config_rejects_ignored_execution_flags() {
    let mut config = test_config();
    config.batch_size = 2;
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("requires batch_size=1"));

    config.batch_size = 1;
    config.resume = true;
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("resume=true is unsupported"));
}

#[test]
fn render_prompt_replaces_single_and_double_braces() {
    let mut object = Map::new();
    object.insert("word".to_string(), Value::String("kataba".to_string()));
    let rendered = render_prompt("{word} / {{word}}", &object).unwrap();
    assert_eq!(rendered, "kataba / kataba");
    assert!(render_prompt("{missing}", &object).is_err());
    assert!(render_prompt("{word", &object).is_err());
}

#[test]
fn arabic_source_spans_convert_from_utf8_bytes_to_tokenizer_characters() {
    let prompt = "قل كتب الآن";
    let byte_span = unique_substring_byte_span(prompt, "كتب").unwrap();
    assert_eq!(&prompt[byte_span[0]..byte_span[1]], "كتب");
    assert_eq!(
        byte_span_to_character_span(prompt, byte_span).unwrap(),
        [3, 6]
    );
    assert!(unique_substring_byte_span("كتب ثم كتب", "كتب").is_err());
}

#[test]
fn run_directory_transaction_publishes_only_on_commit() {
    let parent = crate::v05::testutil::temp_root("run-transaction");
    fs::create_dir_all(&parent).unwrap();

    let abandoned = parent.join("abandoned");
    {
        let transaction = RunDirectoryTransaction::begin(&abandoned).unwrap();
        fs::write(transaction.staging_path().join("partial"), b"partial").unwrap();
        assert!(!abandoned.exists());
    }
    assert!(!abandoned.exists());

    let published = parent.join("published");
    let transaction = RunDirectoryTransaction::begin(&published).unwrap();
    fs::write(transaction.staging_path().join("complete"), b"complete").unwrap();
    transaction.commit().unwrap();
    assert_eq!(fs::read(published.join("complete")).unwrap(), b"complete");
    assert!(RunDirectoryTransaction::begin(&published).is_err());
    fs::remove_dir_all(parent).ok();
}

#[test]
fn contract_names_layers_and_pooling_stably() {
    assert_eq!(ARTIFACT_CONTRACT_VERSION, 2);
    assert_eq!(ARTIFACT_LAYOUT, "ember.layer_sharded_npy.v1");
    assert_eq!(layer_name(4), "layer_0004");
    assert_eq!(layer_relative_path(4), "layers/layer_0004.npy");
    assert_eq!(pooling_for_mode(TokenPositionMode::PromptFinal), "single");
    assert_eq!(pooling_for_mode(TokenPositionMode::WordMean), "mean");

    let order_a = sample_order_hash(&[
        ("a".to_string(), "fnv1a64:1111".to_string()),
        ("b".to_string(), "fnv1a64:2222".to_string()),
    ]);
    let order_b = sample_order_hash(&[
        ("b".to_string(), "fnv1a64:2222".to_string()),
        ("a".to_string(), "fnv1a64:1111".to_string()),
    ]);
    assert_ne!(order_a, order_b);
}

#[test]
fn file_hash_spanning_several_chunks_matches_the_bytes_hash() {
    let path = crate::v05::testutil::temp_root("sha256-chunks");
    // Not a multiple of the chunk size, so the last read is short.
    let bytes: Vec<u8> = (0..2 * SHA256_FILE_CHUNK + 1234)
        .map(|index| (index * 31 % 251) as u8)
        .collect();
    fs::write(&path, &bytes).unwrap();
    assert_eq!(sha256_file_result(&path).unwrap(), sha256_bytes(&bytes));
    fs::write(&path, b"").unwrap();
    assert_eq!(sha256_file_result(&path).unwrap(), sha256_bytes(b""));
    let abandoned =
        sha256_file_result_unless(&path, &std::sync::atomic::AtomicBool::new(true)).unwrap_err();
    assert!(
        abandoned.to_string().starts_with("abandoned hashing "),
        "{abandoned:#}"
    );
    fs::remove_file(&path).unwrap();
    let missing = sha256_file_result(&path).unwrap_err();
    assert!(
        missing
            .to_string()
            .starts_with("failed to open file for SHA-256: "),
        "{missing:#}"
    );
}

#[test]
fn prompt_final_skips_zero_width_offsets() {
    let mut config = test_config();
    config.architecture = None;
    config.prompt_template = "x".to_string();
    config.layers.clear();
    let selected =
        select_token_positions(&[1, 2, 3], &[(0, 0), (0, 1), (1, 3)], &config, None).unwrap();
    assert_eq!(selected, vec![2]);
}

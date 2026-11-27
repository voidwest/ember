use super::*;

pub fn validate_artifact_contract(
    run_dir: impl AsRef<Path>,
    allow_missing_layers: bool,
) -> Result<ArtifactValidationSummary> {
    let run_dir = run_dir.as_ref();
    if !run_dir.is_dir() {
        anyhow::bail!(
            "artifact run directory does not exist: {}",
            run_dir.display()
        );
    }
    let canonical_run_dir = fs::canonicalize(run_dir).with_context(|| {
        format!(
            "failed to canonicalize run directory: {}",
            run_dir.display()
        )
    })?;
    let manifest_path = run_dir.join(MANIFEST_FILENAME);
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read manifest: {}", manifest_path.display()))?;
    let manifest: ArtifactManifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse manifest: {}", manifest_path.display()))?;

    if manifest.schema_version != ARTIFACT_CONTRACT_VERSION {
        anyhow::bail!(
            "manifest schema_version {} does not match expected {}",
            manifest.schema_version,
            ARTIFACT_CONTRACT_VERSION
        );
    }
    if manifest.layout != ARTIFACT_LAYOUT {
        anyhow::bail!(
            "manifest layout '{}' does not match expected '{}'",
            manifest.layout,
            ARTIFACT_LAYOUT
        );
    }
    if manifest.artifact_kind != "ember_hidden_states" {
        anyhow::bail!(
            "manifest artifact_kind '{}' does not match 'ember_hidden_states'",
            manifest.artifact_kind
        );
    }
    manifest.extraction_config.validate()?;
    if manifest.model.path != manifest.extraction_config.model_path {
        anyhow::bail!("manifest model path does not match extraction_config.model_path");
    }
    if manifest.backend.name != manifest.extraction_config.backend.as_str() {
        anyhow::bail!(
            "manifest backend '{}' does not match extraction config '{}'",
            manifest.backend.name,
            manifest.extraction_config.backend.as_str()
        );
    }
    if manifest.sample_count == 0 {
        anyhow::bail!("manifest sample_count must be greater than zero");
    }
    validate_stable_hash(&manifest.sample_order_hash, "manifest sample_order_hash")?;
    validate_stable_hash(&manifest.config_hash, "manifest config_hash")?;
    if manifest.dtype != "f32" || manifest.output_format != "npy" {
        anyhow::bail!(
            "manifest requires dtype=f32 and output_format=npy, got dtype='{}' output_format='{}'",
            manifest.dtype,
            manifest.output_format
        );
    }
    if manifest.backend.name.trim().is_empty() {
        anyhow::bail!("manifest backend.name is empty");
    }
    if let Some(sha256) = &manifest.model.sha256 {
        validate_sha256(sha256, "manifest model.sha256")?;
    }
    verify_file_identity(
        Path::new(&manifest.model.path),
        manifest.model.file_size_bytes,
        manifest.model.sha256.as_ref(),
        "model",
    )?;
    if let Some(tokenizer) = &manifest.tokenizer {
        require_non_empty(&tokenizer.path, "manifest tokenizer.path")?;
        validate_sha256(&tokenizer.sha256, "manifest tokenizer.sha256")?;
        verify_file_identity(
            Path::new(&tokenizer.path),
            Some(tokenizer.file_size_bytes),
            Some(&tokenizer.sha256),
            "tokenizer",
        )?;
    }
    if manifest.tensor_contract.storage != "layer-sharded-npy"
        || manifest.tensor_contract.dtype != "f32"
        || manifest.tensor_contract.byte_order != "little-endian"
        || manifest.tensor_contract.sample_axis != 0
        || manifest.tensor_contract.hidden_axis != 1
    {
        anyhow::bail!("manifest tensor_contract does not match the layer-sharded f32 contract");
    }
    if manifest.config_path != CONFIG_FILENAME
        || manifest.samples_path != SAMPLES_FILENAME
        || manifest.tokenization_path != TOKENIZATION_FILENAME
        || manifest.positions_path != POSITIONS_FILENAME
        || manifest.checksums_path != CHECKSUMS_FILENAME
        || manifest.report_path != REPORT_FILENAME
    {
        anyhow::bail!("manifest core paths do not match the canonical artifact layout");
    }
    if manifest.tensor_contract.layers.is_empty() && !allow_missing_layers {
        anyhow::bail!("manifest has no layer shards");
    }

    let core_entries = [
        (&manifest.config_path, "config_path"),
        (&manifest.samples_path, "samples_path"),
        (&manifest.tokenization_path, "tokenization_path"),
        (&manifest.positions_path, "positions_path"),
        (&manifest.report_path, "report_path"),
        (&manifest.checksums_path, "checksums_path"),
    ];
    let mut core_paths = Vec::with_capacity(core_entries.len());
    for (value, field) in core_entries {
        core_paths.push(resolve_artifact_path(
            &canonical_run_dir,
            value,
            &format!("manifest {field}"),
        )?);
    }
    let [config_path, samples_path, tokenization_path, positions_path, report_path, checksums_path] =
        core_paths.try_into().expect("six core paths");
    let declared_core_paths = [
        manifest.config_path.as_str(),
        MANIFEST_FILENAME,
        manifest.samples_path.as_str(),
        manifest.tokenization_path.as_str(),
        manifest.positions_path.as_str(),
        manifest.report_path.as_str(),
        manifest.checksums_path.as_str(),
    ];
    if declared_core_paths
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != declared_core_paths.len()
    {
        anyhow::bail!("manifest core artifact paths must be distinct");
    }

    let config_bytes = fs::read(&config_path)
        .with_context(|| format!("failed to read config artifact: {}", config_path.display()))?;
    let computed_config_hash = stable_bytes_hash(&config_bytes);
    if computed_config_hash != manifest.config_hash {
        anyhow::bail!(
            "config_hash mismatch: manifest {}, computed {}",
            manifest.config_hash,
            computed_config_hash
        );
    }

    let mut expected_checksum_paths = HashSet::from([
        manifest.config_path.clone(),
        MANIFEST_FILENAME.to_string(),
        manifest.samples_path.clone(),
        manifest.tokenization_path.clone(),
        manifest.positions_path.clone(),
        manifest.report_path.clone(),
    ]);
    for layer in &manifest.tensor_contract.layers {
        if !expected_checksum_paths.insert(layer.path.clone()) {
            anyhow::bail!("layer path '{}' collides with another artifact", layer.path);
        }
    }
    if let Some(path) = &manifest.logits_path
        && !expected_checksum_paths.insert(path.clone())
    {
        anyhow::bail!("logits path '{path}' collides with another artifact");
    }

    let checksums_text = fs::read_to_string(&checksums_path)
        .with_context(|| format!("failed to read checksums: {}", checksums_path.display()))?;
    let checksums: BTreeMap<String, String> = serde_json::from_str(&checksums_text)
        .with_context(|| format!("failed to parse checksums: {}", checksums_path.display()))?;
    for expected_path in &expected_checksum_paths {
        if !checksums.contains_key(expected_path) {
            anyhow::bail!("checksums.json is missing required artifact: {expected_path}");
        }
    }
    for (relative_path, expected) in &checksums {
        if !expected_checksum_paths.contains(relative_path) {
            anyhow::bail!("checksums.json contains undeclared artifact: {relative_path}");
        }
        validate_sha256(expected, &format!("checksum for {relative_path}"))?;
        let path = resolve_artifact_path(
            &canonical_run_dir,
            relative_path,
            "checksums.json artifact path",
        )?;
        let actual = sha256_file_result(&path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            anyhow::bail!(
                "checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            );
        }
    }

    let samples: Vec<SampleArtifactRecord> = read_jsonl_records(&samples_path)?;
    let tokenization: Vec<TokenizationArtifactRecord> = read_jsonl_records(&tokenization_path)?;
    let positions: Vec<PositionArtifactRecord> = read_jsonl_records(&positions_path)?;

    if samples.len() != manifest.sample_count {
        anyhow::bail!(
            "samples.jsonl has {} rows but manifest sample_count is {}",
            samples.len(),
            manifest.sample_count
        );
    }
    if tokenization.len() != samples.len() || positions.len() != samples.len() {
        anyhow::bail!(
            "artifact row count mismatch: samples={}, tokenization={}, positions={}",
            samples.len(),
            tokenization.len(),
            positions.len()
        );
    }

    let mut order = Vec::with_capacity(samples.len());
    let mut sample_ids = HashSet::new();
    for (index, sample) in samples.iter().enumerate() {
        if sample.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "sample row {index} has schema_version {}",
                sample.schema_version
            );
        }
        if sample.sample_index != index {
            anyhow::bail!(
                "samples.jsonl row {index} has sample_index {}",
                sample.sample_index
            );
        }
        if sample.sample_id.trim().is_empty() {
            anyhow::bail!("samples.jsonl row {index} has an empty sample_id");
        }
        if !sample_ids.insert(sample.sample_id.clone()) {
            anyhow::bail!(
                "samples.jsonl contains duplicate sample_id '{}'",
                sample.sample_id
            );
        }
        validate_stable_hash(&sample.prompt_hash, &format!("sample {index} prompt_hash"))?;
        if manifest.extraction_config.prompt_hashes_only == sample.prompt.is_some() {
            anyhow::bail!(
                "sample prompt presence at sample_index {index} does not match prompt_hashes_only={}",
                manifest.extraction_config.prompt_hashes_only
            );
        }
        if let Some(prompt) = &sample.prompt {
            let computed = stable_prompt_hash(prompt);
            if computed != sample.prompt_hash {
                anyhow::bail!(
                    "prompt_hash mismatch at sample_index {index}: stored {}, computed {computed}",
                    sample.prompt_hash
                );
            }
        }
        let token_row = &tokenization[index];
        let position_row = &positions[index];
        if token_row.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "tokenization row {index} has schema_version {}",
                token_row.schema_version
            );
        }
        if position_row.schema_version != ARTIFACT_CONTRACT_VERSION {
            anyhow::bail!(
                "position row {index} has schema_version {}",
                position_row.schema_version
            );
        }
        if token_row.sample_index != index || position_row.sample_index != index {
            anyhow::bail!("sample_index mismatch at row {index}");
        }
        if token_row.sample_id != sample.sample_id || position_row.sample_id != sample.sample_id {
            anyhow::bail!("sample_id mismatch at sample_index {index}");
        }
        if token_row.prompt_hash != sample.prompt_hash {
            anyhow::bail!("prompt_hash mismatch at sample_index {index}");
        }
        if token_row.token_count != token_row.token_ids.len() {
            anyhow::bail!(
                "token_count mismatch at sample_index {index}: {} vs {} token_ids",
                token_row.token_count,
                token_row.token_ids.len()
            );
        }
        if token_row.token_ids.is_empty() {
            anyhow::bail!("empty token_ids at sample_index {index}");
        }
        if token_row.offset_unit != "unicode_character_index" {
            anyhow::bail!(
                "unsupported offset_unit '{}' at sample_index {index}",
                token_row.offset_unit
            );
        }
        validate_token_offsets(
            sample.prompt.as_deref(),
            &token_row.token_ids,
            &token_row.offsets,
            index,
        )?;
        if position_row.position_mode != manifest.extraction_config.token_position.as_str() {
            anyhow::bail!(
                "position_mode '{}' at sample_index {index} does not match extraction config '{}'",
                position_row.position_mode,
                manifest.extraction_config.token_position.as_str()
            );
        }
        let expected_pooling = pooling_for_mode(manifest.extraction_config.token_position);
        if position_row.pooling != expected_pooling {
            anyhow::bail!(
                "pooling '{}' at sample_index {index} does not match expected '{expected_pooling}'",
                position_row.pooling
            );
        }
        if position_row.selected_token_positions.is_empty() {
            anyhow::bail!("empty selected_token_positions at sample_index {index}");
        }
        match position_row.pooling.as_str() {
            "single" => {
                if position_row.selected_token_positions.len() != 1 {
                    anyhow::bail!(
                        "single pooling at sample_index {index} selected {} positions",
                        position_row.selected_token_positions.len()
                    );
                }
            }
            "mean" => {}
            other => anyhow::bail!("unsupported pooling '{other}' at sample_index {index}"),
        }
        for position in &position_row.selected_token_positions {
            if *position >= token_row.token_count {
                anyhow::bail!(
                    "selected token position {} out of bounds for token_count {} at sample_index {index}",
                    position,
                    token_row.token_count
                );
            }
        }
        if !position_row
            .selected_token_positions
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            anyhow::bail!(
                "selected_token_positions must be strictly increasing at sample_index {index}"
            );
        }
        let expected_source_field = source_field_for_position(&manifest.extraction_config);
        if position_row.source_field != expected_source_field {
            anyhow::bail!("source_field mismatch at sample_index {index}");
        }
        if matches!(
            manifest.extraction_config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) && position_row
            .source_value
            .as_deref()
            .is_none_or(str::is_empty)
        {
            anyhow::bail!("word-based position has no source_value at sample_index {index}");
        }
        if let (Some(prompt), Some(source_value)) = (
            sample.prompt.as_deref(),
            position_row.source_value.as_deref(),
        ) && let Some([start, end]) = position_row.source_byte_span
            && prompt.get(start..end) != Some(source_value)
        {
            anyhow::bail!(
                        "source_byte_span [{start}, {end}] does not slice to source_value at sample_index {index}"
                    );
        }
        if !matches!(
            manifest.extraction_config.token_position,
            TokenPositionMode::WordFinalSubtoken | TokenPositionMode::WordMean
        ) && (position_row.source_value.is_some() || position_row.source_byte_span.is_some())
        {
            anyhow::bail!(
                "prompt-wide position unexpectedly declares source value/span at sample_index {index}"
            );
        }
        if sample.prompt.is_some() {
            let recomputed = select_token_positions(
                &token_row.token_ids,
                &token_row.offsets,
                &manifest.extraction_config,
                position_row.source_byte_span,
            )?;
            if recomputed != position_row.selected_token_positions {
                anyhow::bail!(
                    "selected_token_positions do not reproduce at sample_index {index}: stored {:?}, computed {:?}",
                    position_row.selected_token_positions,
                    recomputed
                );
            }
        }
        order.push((sample.sample_id.clone(), sample.prompt_hash.clone()));
    }

    let computed_order_hash = sample_order_hash(&order);
    if computed_order_hash != manifest.sample_order_hash {
        anyhow::bail!(
            "sample_order_hash mismatch: manifest {}, computed {}",
            manifest.sample_order_hash,
            computed_order_hash
        );
    }

    let mut previous_layer = None;
    let mut layer_paths = HashSet::new();
    for layer in &manifest.tensor_contract.layers {
        if previous_layer.is_some_and(|previous| layer.layer_index <= previous) {
            anyhow::bail!("manifest layer indices must be strictly increasing");
        }
        previous_layer = Some(layer.layer_index);
        if layer.layer_index >= manifest.model.n_layers {
            anyhow::bail!(
                "layer index {} exceeds model n_layers {}",
                layer.layer_index,
                manifest.model.n_layers
            );
        }
        if layer.layer_name != layer_name(layer.layer_index)
            || layer.path != layer_relative_path(layer.layer_index)
        {
            anyhow::bail!(
                "layer {} name/path does not match canonical layer-sharded layout",
                layer.layer_index
            );
        }
        if !layer_paths.insert(layer.path.clone()) {
            anyhow::bail!("duplicate layer artifact path '{}'", layer.path);
        }
        let expected_shape = vec![manifest.sample_count, manifest.model.embed_dim];
        if layer.shape != expected_shape {
            anyhow::bail!(
                "layer {} shape {:?} does not match expected {:?}",
                layer.layer_name,
                layer.shape,
                expected_shape
            );
        }
        let path = resolve_artifact_path(&canonical_run_dir, &layer.path, "layer artifact path")?;
        let (actual_shape, values) = crate::npy::read_npy_2d(
            path.to_str()
                .with_context(|| format!("layer path is not UTF-8: {}", path.display()))?,
        )?;
        if actual_shape != layer.shape {
            anyhow::bail!(
                "layer {} npy shape {:?} does not match manifest {:?}",
                layer.layer_name,
                actual_shape,
                layer.shape
            );
        }
        validate_finite_tensor(&values, &format!("layer {}", layer.layer_name))?;
    }

    match (&manifest.logits_path, &manifest.tensor_contract.logits) {
        (None, None) => {}
        (Some(path), Some(logits)) if path == &logits.path => {
            if logits.shape.len() != 2
                || logits.shape[0] != manifest.sample_count
                || logits.shape[1] == 0
            {
                anyhow::bail!("invalid logits shape in tensor contract: {:?}", logits.shape);
            }
            let resolved = resolve_artifact_path(
                &canonical_run_dir,
                path,
                "manifest logits_path",
            )?;
            let (actual_shape, values) = crate::npy::read_npy_2d(
                resolved.to_str().with_context(|| {
                    format!("logits path is not UTF-8: {}", resolved.display())
                })?,
            )?;
            if actual_shape != logits.shape {
                anyhow::bail!(
                    "logits npy shape {:?} does not match manifest {:?}",
                    actual_shape,
                    logits.shape
                );
            }
            validate_finite_tensor(&values, "logits")?;
        }
        _ => anyhow::bail!(
            "manifest logits_path and tensor_contract.logits must either both be absent or declare the same path"
        ),
    }

    let report_text = fs::read_to_string(&report_path)
        .with_context(|| format!("failed to read report: {}", report_path.display()))?;
    let report: Value = serde_json::from_str(&report_text)
        .with_context(|| format!("failed to parse report: {}", report_path.display()))?;
    if report.get("status").and_then(Value::as_str) != Some("complete") {
        anyhow::bail!("report status is not complete");
    }
    if report.get("schema_version").and_then(Value::as_u64)
        != Some(u64::from(ARTIFACT_CONTRACT_VERSION))
        || report.get("layout").and_then(Value::as_str) != Some(ARTIFACT_LAYOUT)
        || report.get("sample_count").and_then(Value::as_u64)
            != u64::try_from(manifest.sample_count).ok()
        || report.get("layer_count").and_then(Value::as_u64)
            != u64::try_from(manifest.tensor_contract.layers.len()).ok()
        || report.get("logits_written").and_then(Value::as_bool)
            != Some(manifest.logits_path.is_some())
    {
        anyhow::bail!("report schema/layout/count/logits fields do not match the manifest");
    }

    Ok(ArtifactValidationSummary {
        run_dir: canonical_run_dir
            .to_str()
            .map(str::to_string)
            .unwrap_or_else(|| run_dir.display().to_string()),
        sample_count: manifest.sample_count,
        layer_count: manifest.tensor_contract.layers.len(),
        logits_present: manifest.logits_path.is_some(),
        sample_order_hash: manifest.sample_order_hash,
    })
}

pub(super) fn resolve_artifact_path(
    canonical_run_dir: &Path,
    relative_path: &str,
    field: &str,
) -> Result<std::path::PathBuf> {
    require_non_empty(relative_path, field)?;
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        anyhow::bail!("{field} must be a normalized relative artifact path: {relative_path}");
    }
    let joined = canonical_run_dir.join(relative);
    if !joined.is_file() {
        anyhow::bail!("{field} is missing: {}", joined.display());
    }
    let canonical = fs::canonicalize(&joined)
        .with_context(|| format!("failed to canonicalize artifact: {}", joined.display()))?;
    if !canonical.starts_with(canonical_run_dir) {
        anyhow::bail!(
            "{field} escapes the artifact run directory through a symlink: {}",
            joined.display()
        );
    }
    Ok(canonical)
}

pub(super) fn validate_stable_hash(value: &str, field: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("fnv1a64:") else {
        anyhow::bail!("{field} must use the fnv1a64:<16 hex digits> format");
    };
    if hex.len() != 16 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("{field} must use the fnv1a64:<16 hex digits> format");
    }
    Ok(())
}

/// When the file exists, verify its recorded size and SHA-256 against the
/// manifest (skipping absent optional fields).
pub(super) fn verify_file_identity(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&String>,
    label: &str,
) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    if let Some(expected_size) = expected_size {
        let actual_size = fs::metadata(path)
            .with_context(|| format!("failed to stat {label}: {}", path.display()))?
            .len();
        if actual_size != expected_size {
            anyhow::bail!(
                "{label} file size mismatch: manifest {expected_size}, actual {actual_size}"
            );
        }
    }
    if let Some(expected_sha256) = expected_sha256 {
        let actual_sha256 = sha256_file_result(path)?;
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            anyhow::bail!("{label} SHA-256 does not match the manifest");
        }
    }
    Ok(())
}

pub(super) fn validate_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("{field} must contain exactly 64 hexadecimal digits");
    }
    Ok(())
}

pub(crate) fn validate_token_offsets(
    prompt: Option<&str>,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
    sample_index: usize,
) -> Result<()> {
    if offsets.len() != token_ids.len() {
        anyhow::bail!(
            "offset count {} does not match token count {} at sample_index {sample_index}",
            offsets.len(),
            token_ids.len()
        );
    }
    let byte_length = prompt.map(|text| text.len());
    let mut previous = None;
    for (token_index, &(start, end)) in offsets.iter().enumerate() {
        if start > end || byte_length.is_some_and(|len| end > len) {
            anyhow::bail!(
                "invalid token offset ({start}, {end}) for token {token_index} at sample_index {sample_index}"
            );
        }
        if start == end {
            continue;
        }
        if previous.is_some_and(|(previous_start, previous_end)| {
            start < previous_start || end < previous_end
        }) {
            anyhow::bail!(
                "non-monotonic token offset ({start}, {end}) for token {token_index} at sample_index {sample_index}"
            );
        }
        previous = Some((start, end));
    }
    Ok(())
}

pub(super) fn validate_finite_tensor(values: &[f32], artifact: &str) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("{artifact} contains non-finite value {value} at flat index {index}");
    }
    Ok(())
}

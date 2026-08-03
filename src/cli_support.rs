use crate::Args;
use anyhow::Context;
use ember::extraction::{git_commit, sha256_bytes, sha256_file_result, unix_timestamp};
use ember::loader::{GgufLoader, GgufValue};
use ember::tokenizer::EmberTokenizer;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Embedded Llama tokenizer, materialized only when the conventional external
/// tokenizer file is absent.
static EMBEDDED_LLAMA_TOKENIZER: &str = include_str!("../tokenizer.json");

pub(crate) enum ResolvedTokenizer {
    File(String),
    EmbeddedLlama,
}

impl ResolvedTokenizer {
    pub(crate) fn identity(&self) -> &str {
        match self {
            Self::File(path) => path,
            Self::EmbeddedLlama => "embedded:tokenizer.json",
        }
    }

    pub(crate) fn sha256(&self) -> anyhow::Result<String> {
        match self {
            Self::File(path) => sha256_file_result(path)
                .with_context(|| format!("failed to hash tokenizer '{path}'")),
            Self::EmbeddedLlama => Ok(sha256_bytes(EMBEDDED_LLAMA_TOKENIZER.as_bytes())),
        }
    }

    pub(crate) fn load(&self) -> anyhow::Result<EmberTokenizer> {
        match self {
            Self::File(path) => EmberTokenizer::from_file(path),
            Self::EmbeddedLlama => EmberTokenizer::from_bytes(EMBEDDED_LLAMA_TOKENIZER),
        }
    }
}

pub(crate) fn default_tokenizer_for_arch(arch: &str) -> &'static str {
    match arch {
        "gpt2" => "tokenizer-gpt2.json",
        "llama" => "tokenizer.json",
        "qwen3" => "tokenizer-qwen3.json",
        "gemma4" => "tokenizer-gemma4.json",
        _ => "tokenizer.json",
    }
}

pub(crate) fn resolve_generation_architecture(
    requested: &str,
    loader: &GgufLoader,
) -> anyhow::Result<String> {
    let declared = match loader.metadata.get("general.architecture") {
        Some(GgufValue::Str(value)) => value.as_str(),
        Some(_) => anyhow::bail!("GGUF general.architecture must be a string"),
        None => anyhow::bail!("GGUF is missing required general.architecture metadata"),
    };
    let detected = match declared {
        "gpt2" => "gpt2",
        "llama" => "llama",
        "qwen2" | "qwen3" => "qwen3",
        "gemma3" | "gemma4" => "gemma4",
        other => anyhow::bail!(
            "GGUF architecture '{other}' is not supported by generation; expected gpt2, llama, qwen2/qwen3, or gemma3/gemma4"
        ),
    };
    if requested != "auto" && requested != detected {
        anyhow::bail!(
            "--arch {requested} conflicts with GGUF general.architecture='{declared}' (use --arch {detected} or omit --arch)"
        );
    }
    Ok(detected.to_string())
}

pub(crate) fn resolve_tokenizer(path: &str) -> ResolvedTokenizer {
    if Path::new(path).exists() {
        return ResolvedTokenizer::File(path.to_string());
    }
    if path == "tokenizer.json" {
        return ResolvedTokenizer::EmbeddedLlama;
    }
    ResolvedTokenizer::File(path.to_string())
}

pub(crate) fn parse_layers_list(value: Option<&str>) -> anyhow::Result<Vec<usize>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid layer index '{value}'"))
        })
        .collect()
}

pub(crate) fn parse_temperature(value: &str) -> Result<f32, String> {
    let temperature = value
        .parse::<f32>()
        .map_err(|_| format!("invalid temperature '{value}'"))?;
    if temperature.is_finite() && temperature >= 0.0 {
        Ok(temperature)
    } else {
        Err("temperature must be a finite number >= 0".to_string())
    }
}

pub(crate) fn parse_top_k(value: &str) -> Result<usize, String> {
    let top_k = value
        .parse::<usize>()
        .map_err(|_| format!("invalid top-k '{value}'"))?;
    if top_k > 0 {
        Ok(top_k)
    } else {
        Err("top-k must be greater than 0".to_string())
    }
}

pub(crate) fn parse_top_p(value: &str) -> Result<f32, String> {
    let top_p = value
        .parse::<f32>()
        .map_err(|_| format!("invalid top-p '{value}'"))?;
    if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
        Ok(top_p)
    } else {
        Err("top-p must be in the range (0, 1]".to_string())
    }
}

pub(crate) fn parse_max_seq_len(value: &str) -> Result<usize, String> {
    let max_seq_len = value
        .parse::<usize>()
        .map_err(|_| format!("invalid max sequence length '{value}'"))?;
    if max_seq_len > 0 {
        Ok(max_seq_len)
    } else {
        Err("max sequence length must be greater than 0".to_string())
    }
}

pub(crate) fn gguf_metadata_json(loader: &GgufLoader) -> serde_json::Value {
    let mut entries = serde_json::Map::new();
    for (key, value) in &loader.metadata {
        entries.insert(key.clone(), gguf_value_json(value));
    }
    // per-tensor inventory: the original GGUF records, captured before any
    // dtype conversion. This is the auditable per-tensor type list (Q4_K_M
    // files mix Q4_K and Q6_K; the model-level "quantization" label alone
    // is not a tensor-level claim).
    let mut inventory = Vec::new();
    let mut names: Vec<&String> = loader.tensor_meta.keys().collect();
    names.sort();
    for name in names {
        let meta = &loader.tensor_meta[name];
        let element_count = meta
            .dims
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim));
        let byte_len = element_count
            .and_then(|count| ember::loader::gguf_dtype_byte_len(meta.dtype, count).ok());
        inventory.push(serde_json::json!({
            "name": name,
            "dims": meta.dims,
            "dtype_code": meta.dtype,
            "dtype": ember::loader::ggml_dtype_name(meta.dtype).unwrap_or("unknown"),
            "offset": meta.offset,
            "byte_len": byte_len,
        }));
    }
    entries.insert(
        "tensor_inventory".to_string(),
        serde_json::Value::Array(inventory),
    );
    serde_json::Value::Object(entries)
}

fn gguf_value_json(value: &GgufValue) -> serde_json::Value {
    match value {
        GgufValue::U8(v) => serde_json::json!(v),
        GgufValue::I8(v) => serde_json::json!(v),
        GgufValue::U16(v) => serde_json::json!(v),
        GgufValue::I16(v) => serde_json::json!(v),
        GgufValue::U32(v) => serde_json::json!(v),
        GgufValue::U64(v) => serde_json::json!(v),
        GgufValue::I32(v) => serde_json::json!(v),
        GgufValue::I64(v) => serde_json::json!(v),
        GgufValue::F32(v) => serde_json::json!(v),
        GgufValue::F64(v) => serde_json::json!(v),
        GgufValue::Bool(v) => serde_json::json!(v),
        GgufValue::Str(v) => serde_json::json!(v),
        GgufValue::Array(values) => {
            serde_json::Value::Array(values.iter().map(gguf_value_json).collect())
        }
    }
}

pub(crate) fn write_json_file(path: &str, value: &serde_json::Value) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    ember::atomic_file::atomic_write(path, &bytes)
        .with_context(|| format!("failed to atomically write JSON to '{path}'"))?;
    Ok(())
}

/// Derive a JSON sidecar next to an output without treating every occurrence
/// of `.npy` in the path as a suffix.
pub(crate) fn sidecar_path(output_path: &str, suffix: &str) -> anyhow::Result<String> {
    let path = Path::new(output_path);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .with_context(|| format!("output path has no usable filename: {output_path}"))?;
    let filename = format!("{stem}{suffix}");
    let sidecar = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(
            || Path::new(&filename).to_path_buf(),
            |parent| parent.join(&filename),
        );
    sidecar
        .to_str()
        .map(str::to_owned)
        .with_context(|| format!("sidecar path is not valid UTF-8: {}", sidecar.display()))
}

pub(crate) fn validate_token_ids_for_model(
    token_ids: &[u32],
    model_vocab_size: usize,
    context: &str,
) -> anyhow::Result<()> {
    if let Some((index, token_id)) = token_ids
        .iter()
        .enumerate()
        .find(|(_, token_id)| **token_id as usize >= model_vocab_size)
    {
        anyhow::bail!(
            "{context} token ID {token_id} at position {index} is outside model vocabulary size {model_vocab_size}"
        );
    }
    Ok(())
}

pub(crate) fn build_run_manifest(
    args: &Args,
    tokenizer_path: &str,
    model_sha256: Option<&str>,
    tokenizer_sha256: Option<&str>,
    gguf_metadata: &serde_json::Value,
) -> serde_json::Value {
    let mut execution = serde_json::json!({
        "max_seq_len": args.max_seq_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "probe": args.probe,
        "probe_stimuli": args.probe_stimuli,
        "probe_template": args.probe_template,
        "probe_templates": args.probe_templates,
        "probe_position": args.probe_position,
        "probe_positions": args.probe_positions,
        "probe_generate_tokens": args.probe_generate_tokens,
        "probe_limit": args.probe_limit,
        "k_strategy": args.k_strategy,
        "k_allow_fallback": args.k_allow_fallback,
    });
    if let Some(spec) = args.zero_layer_output {
        execution
            .as_object_mut()
            .expect("execution manifest is an object")
            .insert(
                "experiment".to_string(),
                serde_json::json!({
                    "name": "zero-layer-output",
                    "layer": spec.layer(),
                    "stage": spec.stage().to_string(),
                    "modifies_execution": true,
                }),
            );
    } else if let Some(path) = &args.activation_stats {
        execution
            .as_object_mut()
            .expect("execution manifest is an object")
            .insert(
                "experiment".to_string(),
                serde_json::json!({
                    "name": "activation-stats",
                    "output": path,
                    "modifies_execution": false,
                }),
            );
    }

    serde_json::json!({
        "schema_version": 1,
        "created_at_unix": unix_timestamp(),
        "command_argv": env::args().collect::<Vec<_>>(),
        "source": {
            "git_commit": git_commit(),
        },
        "compiler": {
            "rustc_version_verbose": command_output("rustc", &["--version", "--verbose"]),
        },
        "runtime": {
            "rayon_num_threads_env": env::var("RAYON_NUM_THREADS").ok(),
            "rayon_current_num_threads": rayon::current_num_threads(),
            "cpu_features_detected": cpu_features_detected(),
        },
        "model": {
            "path": args.model,
            "sha256": model_sha256,
            "file_size_bytes": fs::metadata(&args.model).ok().map(|m| m.len()),
            "architecture": args.arch,
            "gguf_metadata": gguf_metadata,
        },
        "tokenizer": {
            "path": tokenizer_path,
            "sha256": tokenizer_sha256,
        },
        "execution": execution,
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|s| s.trim().to_string())
}

fn cpu_features_detected() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            features.push("ssse3");
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1");
        }
        if std::arch::is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
        if std::arch::is_x86_feature_detected!("avx512vl") {
            features.push("avx512vl");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            features.push("avx512vnni");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            features.push("fp16");
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            features.push("sve");
        }
    }

    features
}

pub(crate) fn token_audit_json(
    prompt: &str,
    tokenizer_path: &str,
    tokenizer_sha256: Option<&str>,
    bos_token_id: Option<u32>,
    token_ids: &[u32],
    offsets: &[(usize, usize)],
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "prompt": prompt,
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "bos_token_id": bos_token_id,
        "token_ids": token_ids,
        "token_count": token_ids.len(),
        "offsets": offsets,
        "offset_unit": "unicode_character_index",
        "encode_with_offsets_matches_encode": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn loader_with_arch(architecture: &str) -> GgufLoader {
        GgufLoader {
            metadata: HashMap::from([(
                "general.architecture".to_string(),
                GgufValue::Str(architecture.to_string()),
            )]),
            tensors: HashMap::new(),
            k_strategy: ember::quant_k::KStrategy::EagerF32,
            k_decisions: HashMap::new(),
            tensor_meta: HashMap::new(),
        }
    }

    #[test]
    fn generation_architecture_is_detected_and_aliases_match_engine_families() {
        assert_eq!(
            resolve_generation_architecture("auto", &loader_with_arch("qwen2")).unwrap(),
            "qwen3"
        );
        assert_eq!(
            resolve_generation_architecture("auto", &loader_with_arch("gemma3")).unwrap(),
            "gemma4"
        );
        assert!(resolve_generation_architecture("gpt2", &loader_with_arch("llama")).is_err());
    }

    #[test]
    fn sidecar_paths_replace_only_the_final_extension() {
        assert_eq!(
            sidecar_path("runs.npy/logits.npy", "_metadata.json").unwrap(),
            "runs.npy/logits_metadata.json"
        );
        assert_eq!(
            sidecar_path("logits", "_metadata.json").unwrap(),
            "logits_metadata.json"
        );
    }

    #[test]
    fn token_ids_are_checked_against_model_rows() {
        validate_token_ids_for_model(&[0, 2], 3, "prompt").unwrap();
        assert!(validate_token_ids_for_model(&[3], 3, "prompt").is_err());
    }

    #[test]
    fn embedded_tokenizer_is_loaded_and_hashed_without_a_temporary_file() {
        let resolved = ResolvedTokenizer::EmbeddedLlama;
        assert_eq!(resolved.identity(), "embedded:tokenizer.json");
        assert_eq!(resolved.sha256().unwrap().len(), 64);
        assert!(resolved.load().unwrap().vocab_size() > 0);
    }
}

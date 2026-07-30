use crate::Args;
use anyhow::Context;
use ember::extraction::{git_commit, unix_timestamp};
use ember::loader::{GgufLoader, GgufValue};
use std::env;
use std::fs;
use std::process::Command;

/// Embedded Llama tokenizer, materialized only when the conventional external
/// tokenizer file is absent.
static EMBEDDED_LLAMA_TOKENIZER: &str = include_str!("../tokenizer.json");

pub(super) fn default_tokenizer_for_arch(arch: &str) -> &'static str {
    match arch {
        "gpt2" => "tokenizer-gpt2.json",
        "llama" => "tokenizer.json",
        "qwen3" => "tokenizer-qwen3.json",
        "gemma4" => "tokenizer-gemma4.json",
        _ => "tokenizer.json",
    }
}

pub(super) fn resolve_tokenizer(path: &str) -> String {
    if std::path::Path::new(path).exists() {
        return path.to_string();
    }
    if path == "tokenizer.json" {
        let tmp = env::temp_dir().join("ember-tokenizer.json");
        if !tmp.exists() {
            if let Err(error) = fs::write(&tmp, EMBEDDED_LLAMA_TOKENIZER) {
                eprintln!("warning: could not write embedded tokenizer: {error}");
                return path.to_string();
            }
        }
        return tmp.to_string_lossy().into_owned();
    }
    path.to_string()
}

pub(super) fn parse_layers_list(value: Option<&str>) -> anyhow::Result<Vec<usize>> {
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

pub(super) fn parse_temperature(value: &str) -> Result<f32, String> {
    let temperature = value
        .parse::<f32>()
        .map_err(|_| format!("invalid temperature '{value}'"))?;
    if temperature.is_finite() && temperature >= 0.0 {
        Ok(temperature)
    } else {
        Err("temperature must be a finite number >= 0".to_string())
    }
}

pub(super) fn parse_top_k(value: &str) -> Result<usize, String> {
    let top_k = value
        .parse::<usize>()
        .map_err(|_| format!("invalid top-k '{value}'"))?;
    if top_k > 0 {
        Ok(top_k)
    } else {
        Err("top-k must be greater than 0".to_string())
    }
}

pub(super) fn parse_top_p(value: &str) -> Result<f32, String> {
    let top_p = value
        .parse::<f32>()
        .map_err(|_| format!("invalid top-p '{value}'"))?;
    if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
        Ok(top_p)
    } else {
        Err("top-p must be in the range (0, 1]".to_string())
    }
}

pub(super) fn parse_max_seq_len(value: &str) -> Result<usize, String> {
    let max_seq_len = value
        .parse::<usize>()
        .map_err(|_| format!("invalid max sequence length '{value}'"))?;
    if max_seq_len > 0 {
        Ok(max_seq_len)
    } else {
        Err("max sequence length must be greater than 0".to_string())
    }
}

pub(super) fn gguf_metadata_json(loader: &GgufLoader) -> serde_json::Value {
    let mut entries = serde_json::Map::new();
    for (key, value) in &loader.metadata {
        entries.insert(key.clone(), gguf_value_json(value));
    }
    serde_json::Value::Object(entries)
}

fn gguf_value_json(value: &GgufValue) -> serde_json::Value {
    match value {
        GgufValue::U8(v) => serde_json::json!(v),
        GgufValue::U32(v) => serde_json::json!(v),
        GgufValue::U64(v) => serde_json::json!(v),
        GgufValue::I32(v) => serde_json::json!(v),
        GgufValue::F32(v) => serde_json::json!(v),
        GgufValue::Bool(v) => serde_json::json!(v),
        GgufValue::Str(v) => serde_json::json!(v),
        GgufValue::Array(values) => {
            serde_json::Value::Array(values.iter().map(gguf_value_json).collect())
        }
    }
}

pub(super) fn write_json_file(path: &str, value: &serde_json::Value) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

pub(super) fn build_run_manifest(
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

pub(super) fn token_audit_json(
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
        "encode_with_offsets_matches_encode": true,
    })
}

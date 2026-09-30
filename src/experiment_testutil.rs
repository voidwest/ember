//! Synthetic end-to-end fixtures for experiment-driver tests: a tiny
//! Llama-family GGUF (F32 or Q8_0 weights) and a matching word-level
//! `tokenizer.json`, written to a per-test directory, plus a spec builder.
//!
//! Test builds only (`#[cfg(test)]` at the declaration in `main.rs`).

use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;
const T_UINT32: u32 = 4;
const T_FLOAT32: u32 = 6;
const T_STRING: u32 = 8;
const DTYPE_F32: u32 = 0;
const DTYPE_Q8_0: u32 = 8;

/// Vocabulary size of the synthetic model and tokenizer.
pub(crate) const VOCAB: usize = 64;

/// A synthetic model + tokenizer on disk.
pub(crate) struct TinyModel {
    pub dir: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub n_layers: usize,
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(crate) fn temp_dir(tag: &str) -> PathBuf {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ember-{tag}-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

struct Rng(u64);

impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn fill(&mut self, count: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..count).map(|_| offset + scale * self.f32()).collect()
    }
}

enum Payload {
    F32(Vec<f32>),
    Q8(Vec<f32>),
}

struct Tensor {
    name: String,
    /// GGUF order: `[in_features, out_features]` (ne0 first).
    dims: Vec<u64>,
    payload: Payload,
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

fn q8_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks(32) {
        let amax = block.iter().fold(0.0f32, |max, value| max.max(value.abs()));
        let scale = amax / 127.0;
        let scale_f16 = half::f16::from_f32(scale);
        out.extend(scale_f16.to_bits().to_le_bytes());
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for &value in block {
            out.push(((value * inverse).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

/// Write a tiny Llama GGUF + word-level tokenizer into a fresh directory.
/// `embed` must be a multiple of 64 for Q8_0 (two heads of `embed / 4`).
pub(crate) fn tiny_model(tag: &str, n_layers: usize, embed: usize, q8: bool) -> TinyModel {
    tiny_model_with_vocab(tag, n_layers, embed, q8, VOCAB)
}

/// `tiny_model` with a chosen vocabulary size; with the Llama-3 vocabulary
/// (128256) the repository's `tokenizer.json` fits it, as the GUI assumes.
pub(crate) fn tiny_model_with_vocab(
    tag: &str,
    n_layers: usize,
    embed: usize,
    q8: bool,
    vocab: usize,
) -> TinyModel {
    let dir = temp_dir(tag);
    let heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = embed / heads;
    let kv_dim = kv_heads * head_dim;
    let inter = 2 * embed;
    let mut rng = Rng(0x5EED_CAFE ^ (n_layers as u64) << 8 ^ embed as u64);
    let weight = |rng: &mut Rng, name: String, in_f: usize, out_f: usize, q8: bool| {
        let values = rng.fill(in_f * out_f, 0.35, 0.0);
        Tensor {
            name,
            dims: vec![in_f as u64, out_f as u64],
            payload: if q8 {
                Payload::Q8(values)
            } else {
                Payload::F32(values)
            },
        }
    };
    let mut tensors = vec![weight(
        &mut rng,
        "token_embd.weight".into(),
        embed,
        vocab,
        false,
    )];
    for layer in 0..n_layers {
        let block = format!("blk.{layer}.");
        for (name, in_f, out_f) in [
            ("attn_q.weight", embed, embed),
            ("attn_k.weight", embed, kv_dim),
            ("attn_v.weight", embed, kv_dim),
            ("attn_output.weight", embed, embed),
            ("ffn_gate.weight", embed, inter),
            ("ffn_up.weight", embed, inter),
            ("ffn_down.weight", inter, embed),
        ] {
            tensors.push(weight(&mut rng, format!("{block}{name}"), in_f, out_f, q8));
        }
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            tensors.push(Tensor {
                name: format!("{block}{name}"),
                dims: vec![embed as u64],
                payload: Payload::F32(rng.fill(embed, 0.1, 1.0)),
            });
        }
    }
    tensors.push(Tensor {
        name: "output_norm.weight".into(),
        dims: vec![embed as u64],
        payload: Payload::F32(rng.fill(embed, 0.1, 1.0)),
    });
    tensors.push(weight(&mut rng, "output.weight".into(), embed, vocab, q8));

    let mut kvs: Vec<(&str, u32, Vec<u8>)> = Vec::new();
    let mut text = Vec::new();
    push_string(&mut text, "llama");
    kvs.push(("general.architecture", T_STRING, text));
    for (key, value) in [
        ("llama.block_count", n_layers as u32),
        ("llama.attention.head_count", heads as u32),
        ("llama.attention.head_count_kv", kv_heads as u32),
        ("llama.embedding_length", embed as u32),
        ("llama.feed_forward_length", inter as u32),
        ("llama.context_length", 64),
        ("llama.vocab_size", vocab as u32),
    ] {
        kvs.push((key, T_UINT32, value.to_le_bytes().to_vec()));
    }
    kvs.push((
        "llama.rope.freq_base",
        T_FLOAT32,
        10_000.0f32.to_le_bytes().to_vec(),
    ));
    kvs.push((
        "llama.attention.layer_norm_rms_epsilon",
        T_FLOAT32,
        1e-5f32.to_le_bytes().to_vec(),
    ));

    let mut out = Vec::new();
    out.extend(GGUF_MAGIC.to_le_bytes());
    out.extend(GGUF_VERSION.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (key, ty, value) in &kvs {
        push_string(&mut out, key);
        out.extend(ty.to_le_bytes());
        out.extend(value);
    }
    let blobs: Vec<(u32, Vec<u8>)> = tensors
        .iter()
        .map(|tensor| match &tensor.payload {
            Payload::F32(values) => (
                DTYPE_F32,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            Payload::Q8(values) => (DTYPE_Q8_0, q8_bytes(values)),
        })
        .collect();
    let mut offset = 0usize;
    for (tensor, (dtype, blob)) in tensors.iter().zip(&blobs) {
        push_string(&mut out, &tensor.name);
        out.extend((tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            out.extend(dim.to_le_bytes());
        }
        out.extend(dtype.to_le_bytes());
        out.extend((offset as u64).to_le_bytes());
        offset += blob.len().div_ceil(ALIGNMENT) * ALIGNMENT;
    }
    out.resize(out.len().div_ceil(ALIGNMENT) * ALIGNMENT, 0);
    for (_, blob) in &blobs {
        out.extend(blob);
        out.resize(out.len().div_ceil(ALIGNMENT) * ALIGNMENT, 0);
    }
    let model = dir.join("tiny.gguf");
    std::fs::write(&model, out).expect("write tiny model");

    let tokenizer = dir.join("tokenizer.json");
    std::fs::write(&tokenizer, tokenizer_json()).expect("write tiny tokenizer");
    TinyModel {
        dir,
        model,
        tokenizer,
        n_layers,
    }
}

/// Word-level tokenizer over `w0 .. w62` plus `<unk>` (id 63).
fn tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for id in 0..VOCAB - 1 {
        vocab.insert(format!("w{id}"), serde_json::json!(id));
    }
    vocab.insert("<unk>".into(), serde_json::json!(VOCAB - 1));
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
    .to_string()
}

/// A complete spec TOML over `model`: `mode`, 2 threads, `max_new_tokens`,
/// the given `[[inputs]]`/`[[captures]]`/`[[interventions]]` body, and an
/// output directory under the model's temp dir.
pub(crate) fn spec_text(
    model: &TinyModel,
    mode: &str,
    max_new_tokens: usize,
    body: &str,
) -> String {
    format!(
        r#"schema = "ember.experiment.v1"

[experiment]
name = "synthetic"
seed = 7

[model]
path = {model:?}
tokenizer = {tokenizer:?}

[execution]
mode = "{mode}"
threads = 2
deterministic = true

[generation]
max_new_tokens = {max_new_tokens}
temperature = 0.0

{body}

[output]
directory = "unused"
"#,
        model = model.model.display().to_string(),
        tokenizer = model.tokenizer.display().to_string(),
    )
}

/// Resolve a spec text.
pub(crate) fn resolve(text: &str) -> ember::v05::spec::ExperimentSpecV1 {
    ember::v05::spec::RawExperimentSpec::from_toml_str(text)
        .expect("spec parses")
        .resolve()
        .expect("spec resolves")
}

/// `dir/name`, as a path.
pub(crate) fn out_dir(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

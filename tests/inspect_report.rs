//! Integration coverage for the library `inspect` report, the `diff`
//! orchestration, and the execution-plan report that `ember inspect`/`diff`
//! and the Python binding both consume.

use ember::diff_outcome::evaluate_diff;
use ember::inspect::{inspect_path, FileKind};
use std::path::PathBuf;
use std::time::Duration;

/// Minimal valid GGUF v3: one metadata key and one f32 tensor. Mirrors
/// `tests/integration.rs::build_single_tensor_gguf` (test helpers are
/// file-local there, so the small builder is duplicated deliberately).
fn build_minimal_gguf() -> Vec<u8> {
    let mut data = Vec::with_capacity(8 * 4);
    for _ in 0..8 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());

    let key = b"general.name";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&8u32.to_le_bytes());
    let val = b"test";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    let tname = b"test.weight";
    buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    buf.extend_from_slice(tname);
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let current_pos = buf.len() as u64;
    let alignment = 32u64;
    let data_start = (current_pos + alignment - 1) & !(alignment - 1);
    let padding = (data_start - current_pos) as usize;
    buf.resize(buf.len() + padding, 0);
    buf.extend_from_slice(&data);
    buf
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ember-inspect-it-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ))
}

#[test]
fn gguf_report_carries_structure() {
    let path = temp_path("minimal.gguf");
    std::fs::write(&path, build_minimal_gguf()).unwrap();
    let report = inspect_path(&path, false).unwrap();
    assert_eq!(report.kind, FileKind::Gguf);
    assert!(report.sha256.is_none());
    let gguf = report.gguf.as_ref().expect("gguf digest");
    assert_eq!(gguf.tensor_count, 1);
    assert_eq!(gguf.tensors.len(), 1);
    assert_eq!(gguf.tensors[0].name, "test.weight");
    assert_eq!(gguf.tensors[0].dtype, "f32");
    assert_eq!(gguf.tensors[0].elements, 8);
    assert_eq!(gguf.dtype_histogram.get("f32"), Some(&1));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn sha256_hashes_gguf_but_not_unknown_files() {
    let path = temp_path("hashed.gguf");
    std::fs::write(&path, build_minimal_gguf()).unwrap();
    let report = inspect_path(&path, true).unwrap();
    let sha = report.sha256.expect("sha256 requested");
    assert_eq!(sha.len(), 64);
    std::fs::remove_file(&path).unwrap();

    // An unrecognized file is not an error and is not hashed.
    let junk = temp_path("junk.bin");
    std::fs::write(&junk, b"not a model").unwrap();
    let report = inspect_path(&junk, true).unwrap();
    assert_eq!(report.kind, FileKind::Unknown);
    assert!(report.sha256.is_none());
    assert_eq!(report.notes.len(), 1);
    std::fs::remove_file(&junk).unwrap();
}

#[test]
fn truncated_gguf_fails_structural_validation() {
    let path = temp_path("broken.gguf");
    // Magic claims GGUF; the body is missing, so the loader must reject it.
    std::fs::write(&path, b"GGUF").unwrap();
    assert!(inspect_path(&path, false).is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn diff_report_shape_without_externals() {
    let path = temp_path("diff.gguf");
    std::fs::write(&path, b"definitely not gguf").unwrap();
    let report = evaluate_diff(&path, &[], Duration::from_secs(1));
    assert_eq!(report.schema, "ember.diff.v1");
    assert_eq!(report.ember.runtime, "ember");
    assert!(report.externals.is_empty());
    assert!(report.agreement.all_agree);
    std::fs::remove_file(&path).unwrap();
}

// ---------------------------------------------------------------------------
// tiny llama builder — a positive `inspect_plan` path needs a constructible
// model. Mirrors `tests/embedding_parity.rs` (helpers are file-local there).
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;
const T_UINT32: u32 = 4;
const T_FLOAT32: u32 = 6;
const T_STRING: u32 = 8;
const DTYPE_F32: u32 = 0;

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    data: Vec<f32>,
}

struct Kv {
    key: &'static str,
    ty: u32,
    value: Vec<u8>,
}

fn kv_string(key: &'static str, value: &str) -> Kv {
    let mut v = Vec::new();
    v.extend((value.len() as u64).to_le_bytes());
    v.extend(value.as_bytes());
    Kv {
        key,
        ty: T_STRING,
        value: v,
    }
}

fn kv_u32(key: &'static str, value: u32) -> Kv {
    Kv {
        key,
        ty: T_UINT32,
        value: value.to_le_bytes().to_vec(),
    }
}

fn kv_f32(key: &'static str, value: f32) -> Kv {
    Kv {
        key,
        ty: T_FLOAT32,
        value: value.to_le_bytes().to_vec(),
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn build_gguf(tensors: &[TensorSpec], kvs: &[Kv]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(GGUF_MAGIC.to_le_bytes());
    out.extend(GGUF_VERSION.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        write_string(&mut out, kv.key);
        out.extend(kv.ty.to_le_bytes());
        out.extend(&kv.value);
    }

    let mut offset = 0u64;
    let mut infos = Vec::new();
    for t in tensors {
        infos.push((t, offset));
        let size = (t.data.len() * 4) as u64;
        offset += size.div_ceil(ALIGNMENT) * ALIGNMENT;
    }
    for (t, tensor_offset) in &infos {
        write_string(&mut out, &t.name);
        out.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(DTYPE_F32.to_le_bytes());
        out.extend(tensor_offset.to_le_bytes());
    }

    let data_start = out.len() as u64;
    let pad = (ALIGNMENT - (data_start % ALIGNMENT)) % ALIGNMENT;
    out.extend(std::iter::repeat_n(0u8, pad as usize));
    for t in tensors {
        let mut bytes = Vec::with_capacity(t.data.len() * 4);
        for v in &t.data {
            bytes.extend(v.to_le_bytes());
        }
        bytes.resize(
            bytes.len().div_ceil(ALIGNMENT as usize) * ALIGNMENT as usize,
            0,
        );
        out.extend(bytes);
    }
    out
}

/// Deterministic LCG so the test model is reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn fill(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.f32()).collect()
}

/// Tiny llama GGUF: embed=16, heads=4, kv_heads=2, layers=2, vocab=64,
/// intermediate=48, max_seq=64.
fn tiny_llama_gguf() -> Vec<u8> {
    let embed = 16usize;
    let vocab = 64usize;
    let layers = 2usize;
    let interm = 48usize;
    let mut rng = Rng::new(0x5EED_CAFE);
    let mut tensors = Vec::new();

    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![embed as u64, vocab as u64],
        data: fill(&mut rng, vocab, embed),
    });
    let kv_dim = 2 * (embed / 4);
    for l in 0..layers {
        let b = format!("blk.{l}.");
        for (name, in_f, out_f) in [
            ("attn_q.weight", embed, embed),
            ("attn_k.weight", embed, kv_dim),
            ("attn_v.weight", embed, kv_dim),
            ("attn_output.weight", embed, embed),
            ("ffn_gate.weight", embed, interm),
            ("ffn_up.weight", embed, interm),
            ("ffn_down.weight", interm, embed),
        ] {
            tensors.push(TensorSpec {
                name: format!("{b}{name}"),
                dims: vec![in_f as u64, out_f as u64],
                data: fill(&mut rng, out_f, in_f),
            });
        }
        tensors.push(TensorSpec {
            name: format!("{b}attn_norm.weight"),
            dims: vec![embed as u64],
            data: fill(&mut rng, 1, embed),
        });
        tensors.push(TensorSpec {
            name: format!("{b}ffn_norm.weight"),
            dims: vec![embed as u64],
            data: fill(&mut rng, 1, embed),
        });
    }
    tensors.push(TensorSpec {
        name: "output_norm.weight".into(),
        dims: vec![embed as u64],
        data: fill(&mut rng, 1, embed),
    });
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        dims: vec![embed as u64, vocab as u64],
        data: fill(&mut rng, vocab, embed),
    });

    let kvs = vec![
        kv_string("general.architecture", "llama"),
        kv_u32("llama.block_count", layers as u32),
        kv_u32("llama.attention.head_count", 4),
        kv_u32("llama.attention.head_count_kv", 2),
        kv_u32("llama.embedding_length", embed as u32),
        kv_u32("llama.context_length", 64),
        kv_f32("llama.rope.freq_base", 10_000.0),
        kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        kv_u32("llama.vocab_size", vocab as u32),
    ];
    build_gguf(&tensors, &kvs)
}

#[test]
fn plan_report_builds_for_tiny_llama() {
    let path = temp_path("tiny-llama-plan.gguf");
    std::fs::write(&path, tiny_llama_gguf()).unwrap();
    let report = ember::inspect::inspect_plan(&path, "auto", "planned").unwrap();
    assert_eq!(report.architecture, "llama");
    assert_eq!(report.execution, "planned");
    assert!(!report.plan.tensor_table.is_empty());
    assert_eq!(
        report.plan.kernel_revision,
        ember::plan::PLAN_KERNEL_REVISION
    );
    // The serialized plan is the CLI `--output` contract the binding returns.
    let json = serde_json::to_value(&report.plan).unwrap();
    assert!(json.get("kernel_revision").is_some(), "plan JSON: {json}");
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn plan_rejects_conflicting_arch_and_bad_execution() {
    let path = temp_path("tiny-llama-plan-bad.gguf");
    std::fs::write(&path, tiny_llama_gguf()).unwrap();
    // Explicit arch conflicts with the GGUF's declared `llama`.
    assert!(ember::inspect::inspect_plan(&path, "gpt2", "planned").is_err());
    // Unknown execution mode fails before loading.
    assert!(ember::inspect::inspect_plan(&path, "auto", "turbo").is_err());
    std::fs::remove_file(&path).unwrap();
}

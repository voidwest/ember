//! Precomputed-embedding prefill parity tests.
//!
//! The foundation for multimodal input: token-based prefill and
//! precomputed-embedding prefill must share one internal path, with
//! byte-identical behavior. These tests build a tiny deterministic Llama
//! (and GPT-2) GGUF in memory, load it through the real loader, and prove:
//!
//! 1. the token prefill path still works,
//! 2. token prefill == equivalent embedding prefill (same logits, bit-exact),
//! 3. position handling is identical (multi-turn / start_pos > 0),
//! 4. KV-cache state after equivalent prefills is identical,
//! 5. decode after either prefill produces identical logits,
//! 6. the GPT-2 learned-position-embedding contract (embeddings include wpe).

use ember::backend::{Backend, CpuBackend};
use ember::kv_cache::KVCache;
use ember::llama::Llama;
use ember::loader::load_gguf;
use ember::model::{ForwardModel, Gpt2};
use ember::tensor::CpuTensor;

// ---------------------------------------------------------------------------
// minimal GGUF v3 writer (f32 tensors only)
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
    /// GGUF dims (llama.cpp convention: reversed torch dims; dims[0] is the
    /// fastest-varying axis of the *stored* data only for 1-D metadata —
    /// we follow the exact convention the llama.cpp converter produces).
    dims: Vec<u64>,
    /// row-major payload with the LAST dim fastest.
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

    // tensor info table with aligned data offsets
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

    // pad to the first tensor offset, then write payloads
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

/// deterministic LCG so the test model is reproducible
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

/// fill a row-major [rows, cols] tensor with deterministic values
fn fill(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.f32()).collect()
}

/// Build a tiny llama GGUF: embed=16, heads=4, kv_heads=2, layers=2,
/// vocab=64, intermediate=48, max_seq=64.
fn tiny_llama_gguf() -> Vec<u8> {
    let embed = 16usize;
    let vocab = 64usize;
    let layers = 2usize;
    let interm = 48usize;
    let mut rng = Rng::new(0x5EED_CAFE);
    let mut tensors = Vec::new();

    // embedding: GGUF dims [embed, vocab], data row-major over [vocab, embed]
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![embed as u64, vocab as u64],
        data: fill(&mut rng, vocab, embed),
    });
    let kv_dim = 2 * (embed / 4); // n_kv_heads(2) * head_dim(embed/4)
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
                // llama GGUF convention: dims [in, out]; payload row-major
                // over [out, in] (in fastest) — what gguf_to_row_major_f32
                // consumes.
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

fn load_tiny_llama(tag: &str) -> Llama<CpuBackend> {
    // unique file per test: parallel tests must not truncate a file another
    // test's loader has mmap'd (mmap + truncate = SIGBUS)
    let bytes = tiny_llama_gguf();
    let dir = std::env::temp_dir().join("ember_parity_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("tiny_llama_{tag}.gguf"));
    std::fs::write(&path, &bytes).unwrap();
    let loader = load_gguf(&path).expect("tiny llama GGUF must load");
    Llama::from_loader(loader).expect("tiny llama model must build")
}

/// Look up token embeddings exactly the way the model does internally
/// (row copy from the embedding table) so the parity test feeds the
/// embeddings path the same rows the token path uses.
fn embed_tokens_like_model(
    backend: &CpuBackend,
    model: &Llama<CpuBackend>,
    token_ids: &[u32],
) -> CpuTensor {
    let embed_dim = model.embed_dim();
    let mut out = backend.zeroes(&[token_ids.len(), embed_dim]).unwrap();
    let ember::llama::LlamaEmbedding::F32(table) = &model.embed_tokens else {
        panic!("test model must use F32 embeddings");
    };
    for (row, &token) in token_ids.iter().enumerate() {
        backend
            .assign_row_from_table(&mut out, row, table, token as usize)
            .unwrap();
    }
    out
}

static TEST_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn test_tag() -> String {
    format!(
        "t{}",
        TEST_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

fn assert_logits_eq(a: &CpuTensor, b: &CpuTensor, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    let da = a.data();
    let db = b.data();
    assert_eq!(da.len(), db.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in da.iter().zip(db.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{ctx}: logit[{i}] differs: {x} != {y} (bits {} vs {})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

fn assert_cache_eq(a: &KVCache, b: &KVCache, ctx: &str) {
    assert_eq!(a.cursor(), b.cursor(), "{ctx}: cursor mismatch");
    assert_eq!(a.n_layers(), b.n_layers(), "{ctx}: layer count mismatch");
    for layer in 0..a.n_layers() {
        let (ka, va) = a.get(layer);
        let (kb, vb) = b.get(layer);
        assert_eq!(ka, kb, "{ctx}: cached K differs at layer {layer}");
        assert_eq!(va, vb, "{ctx}: cached V differs at layer {layer}");
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn token_prefill_still_works() {
    let backend = CpuBackend;
    let model = load_tiny_llama(&test_tag());
    let mut cache = model.create_cache(&backend, 64);
    let tokens = [3u32, 7, 1, 42, 9];
    let logits = model
        .forward_with_cache(&backend, &tokens, &mut cache, 0)
        .expect("token prefill must run");
    assert_eq!(logits.shape(), &[5, 64]);
    assert_eq!(cache.cursor(), 5);
    assert!(logits.data().iter().all(|v| v.is_finite()));
    // deterministic
    let mut cache2 = model.create_cache(&backend, 64);
    let logits2 = model
        .forward_with_cache(&backend, &tokens, &mut cache2, 0)
        .unwrap();
    assert_logits_eq(&logits, &logits2, "determinism");
}

#[test]
fn token_and_embedding_prefill_are_identical() {
    let backend = CpuBackend;
    let model = load_tiny_llama(&test_tag());
    let tokens = [3u32, 7, 1, 42, 9, 55, 2];

    let mut cache_tok = model.create_cache(&backend, 64);
    let logits_tok = model
        .forward_with_cache(&backend, &tokens, &mut cache_tok, 0)
        .expect("token prefill must run");

    let mut cache_emb = model.create_cache(&backend, 64);
    let embeddings = embed_tokens_like_model(&backend, &model, &tokens);
    assert_eq!(embeddings.shape(), &[tokens.len(), 16]);
    let logits_emb = model
        .forward_embeddings_with_cache(&backend, &embeddings, &mut cache_emb, 0)
        .expect("embedding prefill must run");

    assert_logits_eq(&logits_tok, &logits_emb, "prefill logits");
    assert_cache_eq(&cache_tok, &cache_emb, "prefill KV cache");

    // trait-level entry points agree as well
    let mut cache_tok2 = model.create_cache(&backend, 64);
    let logits_tok2 = model
        .prefill_tokens_with_cache(&backend, &tokens, &mut cache_tok2)
        .unwrap();
    let mut cache_emb2 = model.create_cache(&backend, 64);
    let logits_emb2 = model
        .prefill_embeddings_with_cache(&backend, &embeddings, &mut cache_emb2)
        .unwrap();
    assert_logits_eq(&logits_tok2, &logits_emb2, "prefill_* entry points");
    assert_logits_eq(&logits_tok, &logits_tok2, "trait == inherent token path");
}

#[test]
fn multi_turn_position_handling_is_identical() {
    let backend = CpuBackend;
    let model = load_tiny_llama(&test_tag());
    let first = [3u32, 7, 1, 42];
    let second = [9u32, 55, 2];

    // token path, two turns
    let mut cache_tok = model.create_cache(&backend, 64);
    model
        .forward_with_cache(&backend, &first, &mut cache_tok, 0)
        .unwrap();
    let logits_tok = model
        .forward_with_cache(&backend, &second, &mut cache_tok, 4)
        .unwrap();

    // embeddings path, two turns (same start_pos sequence)
    let mut cache_emb = model.create_cache(&backend, 64);
    let emb1 = embed_tokens_like_model(&backend, &model, &first);
    model
        .forward_embeddings_with_cache(&backend, &emb1, &mut cache_emb, 0)
        .unwrap();
    let emb2 = embed_tokens_like_model(&backend, &model, &second);
    let logits_emb = model
        .forward_embeddings_with_cache(&backend, &emb2, &mut cache_emb, 4)
        .unwrap();

    assert_logits_eq(&logits_tok, &logits_emb, "multi-turn logits");
    assert_cache_eq(&cache_tok, &cache_emb, "multi-turn KV cache");
}

#[test]
fn decode_after_either_prefill_is_identical() {
    let backend = CpuBackend;
    let model = load_tiny_llama(&test_tag());
    let prompt = [3u32, 7, 1, 42, 9];

    let mut cache_tok = model.create_cache(&backend, 64);
    model
        .forward_with_cache(&backend, &prompt, &mut cache_tok, 0)
        .unwrap();

    let mut cache_emb = model.create_cache(&backend, 64);
    let embeddings = embed_tokens_like_model(&backend, &model, &prompt);
    model
        .forward_embeddings_with_cache(&backend, &embeddings, &mut cache_emb, 0)
        .unwrap();

    // decode steps (token-based, per the design). Two independent cache
    // pairs: one for logits comparison, one for greedy token comparison
    // (greedy advances the cursor too, so it needs its own cache).
    let mut cache_tok_g = model.create_cache(&backend, 64);
    let mut cache_emb_g = model.create_cache(&backend, 64);
    model
        .forward_with_cache(&backend, &prompt, &mut cache_tok_g, 0)
        .unwrap();
    model
        .forward_embeddings_with_cache(&backend, &embeddings, &mut cache_emb_g, 0)
        .unwrap();

    let mut generated_tok = Vec::new();
    let mut generated_emb = Vec::new();
    let mut tok = 11u32;
    for step in 0..6 {
        let logits_tok = model
            .forward_last_logits_with_cache(&backend, &[tok], &mut cache_tok, 5 + step)
            .unwrap();
        let logits_emb = model
            .forward_last_logits_with_cache(&backend, &[tok], &mut cache_emb, 5 + step)
            .unwrap();
        assert_logits_eq(&logits_tok, &logits_emb, &format!("decode step {step}"));
        let (gt, _) = model
            .greedy_next_token_with_cache(&backend, &[tok], &mut cache_tok_g, 5 + step)
            .unwrap();
        let (ge, _) = model
            .greedy_next_token_with_cache(&backend, &[tok], &mut cache_emb_g, 5 + step)
            .unwrap();
        assert_eq!(gt, ge, "greedy tokens must agree at step {step}");
        generated_tok.push(gt);
        generated_emb.push(ge);
        tok = gt;
    }
    assert_eq!(generated_tok, generated_emb);
    assert_cache_eq(&cache_tok, &cache_emb, "post-decode KV cache");
    assert_cache_eq(&cache_tok_g, &cache_emb_g, "post-greedy KV cache");
}

// ---------------------------------------------------------------------------
// GPT-2: learned position embeddings live in the embedding sequence
// ---------------------------------------------------------------------------

fn tiny_gpt2_gguf() -> Vec<u8> {
    let embed = 12usize;
    let vocab = 32usize;
    let layers = 1usize;
    let mut rng = Rng::new(0x6F2F);
    let mut tensors = Vec::new();
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![embed as u64, vocab as u64],
        data: fill(&mut rng, vocab, embed),
    });
    tensors.push(TensorSpec {
        name: "position_embd.weight".into(),
        dims: vec![embed as u64, 64],
        data: fill(&mut rng, 64, embed),
    });
    let b = "blk.0.";
    for (name, in_f, out_f) in [
        ("attn_qkv.weight", embed, 3 * embed),
        ("attn_output.weight", embed, embed),
        ("ffn_up.weight", embed, 4 * embed),
        ("ffn_down.weight", 4 * embed, embed),
    ] {
        tensors.push(TensorSpec {
            name: format!("{b}{name}"),
            dims: vec![out_f as u64, in_f as u64],
            data: fill(&mut rng, in_f, out_f),
        });
        // bias for each linear (gpt2 has them)
        tensors.push(TensorSpec {
            name: format!("{b}{name}").replace(".weight", ".bias"),
            dims: vec![out_f as u64],
            data: fill(&mut rng, 1, out_f),
        });
    }
    for name in [
        "attn_norm.weight",
        "attn_norm.bias",
        "ffn_norm.weight",
        "ffn_norm.bias",
    ] {
        tensors.push(TensorSpec {
            name: format!("{b}{name}"),
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
        name: "output_norm.bias".into(),
        dims: vec![embed as u64],
        data: fill(&mut rng, 1, embed),
    });
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        dims: vec![vocab as u64, embed as u64],
        data: fill(&mut rng, embed, vocab),
    });
    let kvs = vec![
        kv_string("general.architecture", "gpt2"),
        kv_u32("gpt2.block_count", layers as u32),
        kv_u32("gpt2.attention.head_count", 3),
    ];
    build_gguf(&tensors, &kvs)
}

#[test]
fn gpt2_embeddings_include_wpe() {
    let backend = CpuBackend;
    let bytes = tiny_gpt2_gguf();
    let dir = std::env::temp_dir().join("ember_parity_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny_gpt2.gguf");
    std::fs::write(&path, &bytes).unwrap();
    let loader = load_gguf(&path).expect("tiny gpt2 GGUF must load");
    let model = Gpt2::from_loader(loader).expect("tiny gpt2 model must build");

    let tokens = [5u32, 9, 2, 17];
    let mut cache_tok = model.create_cache(&backend, 64);
    let logits_tok = model
        .forward_with_cache(&backend, &tokens, &mut cache_tok, 0)
        .unwrap();

    // embeddings path: wte + wpe rows assembled by the caller (the
    // documented GPT-2 contract for precomputed embeddings)
    let mut x = backend.zeroes(&[tokens.len(), model.embed_dim()]).unwrap();
    for (row, &token) in tokens.iter().enumerate() {
        backend
            .assign_row_sum_from_tables(&mut x, row, &model.wte, token as usize, &model.wpe, row)
            .unwrap();
    }
    let mut cache_emb = model.create_cache(&backend, 64);
    let logits_emb = model
        .forward_embeddings_with_cache(&backend, &x, &mut cache_emb, 0)
        .unwrap();
    assert_logits_eq(&logits_tok, &logits_emb, "gpt2 prefill logits");
    assert_cache_eq(&cache_tok, &cache_emb, "gpt2 KV cache");
}

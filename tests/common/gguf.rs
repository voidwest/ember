//! Minimal GGUF v3 writer (f32 tensors only) and a tiny deterministic
//! Llama fixture, shared by integration tests that need a constructible
//! model in memory.
//!
//! Include with `#[path = "common/gguf.rs"] mod gguf;` from the tests that
//! use it. It is deliberately not a submodule of `common/mod.rs`, so test
//! crates that only need the env gate do not compile it.
#![allow(dead_code)]

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
pub const GGUF_VERSION: u32 = 3;
pub const ALIGNMENT: u64 = 32;

pub const T_UINT32: u32 = 4;
pub const T_FLOAT32: u32 = 6;
pub const T_STRING: u32 = 8;
pub const DTYPE_F32: u32 = 0;

pub struct TensorSpec {
    pub name: String,
    /// GGUF dims (llama.cpp convention: reversed torch dims; dims[0] is the
    /// fastest-varying axis of the *stored* data only for 1-D metadata —
    /// we follow the exact convention the llama.cpp converter produces).
    pub dims: Vec<u64>,
    /// row-major payload with the LAST dim fastest.
    pub data: Vec<f32>,
}

pub struct Kv {
    pub key: &'static str,
    pub ty: u32,
    pub value: Vec<u8>,
}

pub fn kv_string(key: &'static str, value: &str) -> Kv {
    let mut v = Vec::new();
    v.extend((value.len() as u64).to_le_bytes());
    v.extend(value.as_bytes());
    Kv {
        key,
        ty: T_STRING,
        value: v,
    }
}

pub fn kv_u32(key: &'static str, value: u32) -> Kv {
    Kv {
        key,
        ty: T_UINT32,
        value: value.to_le_bytes().to_vec(),
    }
}

pub fn kv_f32(key: &'static str, value: f32) -> Kv {
    Kv {
        key,
        ty: T_FLOAT32,
        value: value.to_le_bytes().to_vec(),
    }
}

pub fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

pub fn build_gguf(tensors: &[TensorSpec], kvs: &[Kv]) -> Vec<u8> {
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
pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// fill a row-major [rows, cols] tensor with deterministic values
pub fn fill(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.f32()).collect()
}

/// Build a tiny llama GGUF: embed=16, heads=4, kv_heads=2, layers=2,
/// vocab=64, intermediate=48, max_seq=64.
pub fn tiny_llama_gguf() -> Vec<u8> {
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

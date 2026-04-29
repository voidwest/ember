use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;

pub struct GgufLoader {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, CpuTensor>,
}

pub enum GgufValue {
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
}

pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLoader> {
    let mut f = File::open(&path).with_context(|| format!("failed to open {:?}", path.as_ref()))?;

    let magic = read_u32(&mut f)?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic: {:#x})", magic);
    }

    let version = read_u32(&mut f)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version: {}", version);
    }

    let tensor_count = read_u64(&mut f)?;
    let metadata_kv_count = read_u64(&mut f)?;

    // TODO(you): implement read_metadata_kv for all value types
    let metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        // stub - read and discard for now
        let _key = read_gguf_string(&mut f)?;
        let _val_type = read_u32(&mut f)?;
        // TODO: read value based on type
    }

    // TODO(you): read tensor name, n_dims, dims, type, offset
    let tensors = HashMap::new();
    for _ in 0..tensor_count {
        let _name = read_gguf_string(&mut f)?;
        // TODO: read shape, dtype, offset -> load into CpuTensor
    }

    Ok(GgufLoader { metadata, tensors })
}

fn read_u32(f: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).context("read_u32 failed")?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).context("read_u64 failed")?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string(f: &mut File) -> Result<String> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).context("read string failed")?;
    String::from_utf8(buf).context("invalid utf8 in string")
}

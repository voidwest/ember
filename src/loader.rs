use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Ok, Result};
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
    U8(u8),
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

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut f)?;
        let val_type = read_u32(&mut f)?;
        let value = read_gguf_value(&mut f, val_type)?;
        metadata.insert(key, value);
    }
    struct TensorInfo {
        name: String,
        dims: Vec<usize>,
        dtype: u32,
        offset: u64,
    }

    let tensors = HashMap::new();
    for _ in 0..tensor_count {
        let _name = read_gguf_string(&mut f)?;
    }

    Ok(GgufLoader { metadata, tensors })
}

fn read_u8(f: &mut File) -> Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(u8::from_le_bytes(buf))
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

fn read_i32(f: &mut File) -> Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(f: &mut File) -> Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_gguf_string(f: &mut File) -> Result<String> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).context("read string failed")?;
    String::from_utf8(buf).context("invalid utf8 in string")
}

fn read_gguf_value(f: &mut File, val_type: u32) -> Result<GgufValue> {
    match val_type {
        0 => Ok(GgufValue::U8(read_u8(f)?)),
        4 => Ok(GgufValue::I32(read_i32(f)?)),
        6 => Ok(GgufValue::U32(read_u32(f)?)),
        7 => Ok(GgufValue::F32(read_f32(f)?)),
        8 => Ok(GgufValue::Bool(read_u8(f)? != 0)),
        9 => Ok(GgufValue::Str(read_gguf_string(f)?)),
        11 => Ok(GgufValue::U64(read_u64(f)?)),
        _ => bail!("unsupported GGUF value type: {}", val_type),
    }
}

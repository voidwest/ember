use crate::quant::{dequantize_q8_0, Q8_0_TYPE_SIZE};
use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Ok, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

/// holds the parsed contents of a GGUF v3 file:
/// metadata key-value pairs and named tensors.
pub struct GgufLoader {
    /// metadata key-value pairs from the gguf header
    pub metadata: HashMap<String, GgufValue>,
    /// named tensors, dequantized to f32
    pub tensors: HashMap<String, CpuTensor>,
}

/// a typed value from GGUF metadata.
pub enum GgufValue {
    U8(u8),
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// nested array of gguf values (val_type 9)
    Array(Vec<GgufValue>),
}

/// load a GGUF file from disk using memory-mapped i/o.
/// avoids copying the file into userspace buffers and lets the OS
/// lazily page in data. the mmap is borrowed via cursor and
/// dropped when the function returns; tensors are copied out.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLoader> {
    let f = File::open(&path).with_context(|| format!("failed to open {:?}", path.as_ref()))?;
    // safety: the backing file is our own model file; no external writers.
    // if the file is modified while mapped, behavior is undefined.
    // for a local inference engine, this is acceptable.
    let mmap = unsafe { memmap2::Mmap::map(&f)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    load_gguf_from_reader(&mut cursor)
}

/// load a GGUF file from any readable + seekable source.
/// useful for testing with in-memory buffers (std::io::Cursor<Vec<u8>>).
pub fn load_gguf_from_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufLoader> {
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic: {:#x})", magic);
    }

    let version = read_u32(reader)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version: {}", version);
    }

    let tensor_count = read_u64(reader)?;
    let metadata_kv_count = read_u64(reader)?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(reader)?;
        let val_type = read_u32(reader)?;
        let value = read_gguf_value(reader, val_type)?;
        metadata.insert(key, value);
    }

    let mut tensor_info = read_tensor_info(reader, tensor_count as usize)?;

    let current_pos = reader.stream_position()?;
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(a)) => *a as u64,
        Some(GgufValue::U64(a)) => *a,
        _ => DEFAULT_ALIGNMENT,
    };
    let data_start = (current_pos + alignment - 1) & !(alignment - 1);

    let mut tensors = HashMap::new();
    for info in tensor_info.drain(..) {
        reader.seek(SeekFrom::Start(data_start + info.offset))?;
        let element_count: usize = info.dims.iter().product();

        let (data, dims) = match info.dtype {
            0 => {
                let mut data = vec![0.0f32; element_count];
                let mut buf = vec![0u8; element_count * 4];
                reader.read_exact(&mut buf)?;
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 4;
                    let bytes: [u8; 4] = buf[start..start + 4]
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("failed to read f32 at index {}", i))?;
                    *dst = f32::from_le_bytes(bytes);
                }
                (data, info.dims)
            }
            1 => {
                use half::f16;
                let mut buf = vec![0u8; element_count * 2];
                reader.read_exact(&mut buf)?;
                let mut data = vec![0.0f32; element_count];
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 2;
                    let bits = u16::from_le_bytes(
                        buf[start..start + 2]
                            .try_into()
                            .map_err(|_| anyhow::anyhow!("failed to read f16 at index {}", i))?,
                    );
                    *dst = f16::from_bits(bits).to_f32();
                }
                // F16 tensors may be stored column-major for quantized models.
                // Reverse the dimension order so the reshape matches the storage layout.
                let mut dims = info.dims;
                dims.reverse();
                (data, dims)
            }
            8 => {
                let n_blocks = element_count / 32;
                let mut buf = vec![0u8; n_blocks * Q8_0_TYPE_SIZE];
                reader.read_exact(&mut buf)?;

                let mut f32_data = vec![0.0f32; element_count];
                dequantize_q8_0(&buf, &mut f32_data)?;
                // Q8_0 blocks are stored along the innermost dimension, which
                // must be a multiple of 32. the gguf tensor info reports the
                // logical shape (e.g. [embed, vocab]), but the data is laid out
                // column-major so that the innermost dimension (vocab) becomes
                // the outer dimension for blocking. reverse the dims here.
                let mut dims = info.dims;
                dims.reverse();
                (f32_data, dims)
            }
            _ => {
                log::warn!(
                    "skipping tensor '{}' with unknown dtype {}",
                    info.name,
                    info.dtype
                );
                continue;
            }
        };

        let tensor = CpuTensor::from_data(dims, data);
        tensors.insert(info.name, tensor);
    }
    Ok(GgufLoader { metadata, tensors })
}

struct TensorInfo {
    name: String,
    dims: Vec<usize>,
    dtype: u32,
    offset: u64,
}

fn read_tensor_info<R: Read + Seek>(reader: &mut R, count: usize) -> Result<Vec<TensorInfo>> {
    let mut info = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_gguf_string(reader)?;
        let n_dims = read_u32(reader)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(reader)? as usize);
        }
        let dtype = read_u32(reader)?;
        let offset = read_u64(reader)?;
        info.push(TensorInfo {
            name,
            dims,
            dtype,
            offset,
        });
    }
    Ok(info)
}

fn read_u8<R: Read>(f: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(u8::from_le_bytes(buf))
}

fn read_u32<R: Read>(f: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).context("read_u32 failed")?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(f: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).context("read_u64 failed")?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(f: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(f: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(f: &mut R) -> Result<String> {
    let len = read_u64(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).context("read string failed")?;
    String::from_utf8(buf).context("invalid utf8 in string")
}

fn read_gguf_value<R: Read>(f: &mut R, val_type: u32) -> Result<GgufValue> {
    match val_type {
        0 => Ok(GgufValue::U8(read_u8(f)?)),
        5 => Ok(GgufValue::I32(read_i32(f)?)),
        4 => Ok(GgufValue::U32(read_u32(f)?)),
        6 => Ok(GgufValue::F32(read_f32(f)?)),
        7 => Ok(GgufValue::Bool(read_u8(f)? != 0)),
        8 => Ok(GgufValue::Str(read_gguf_string(f)?)),
        10 => Ok(GgufValue::U64(read_u64(f)?)),
        9 => {
            let element_type = read_u32(f)?;
            let count = read_u64(f)?;
            let mut elements = Vec::with_capacity(count as usize);
            for _ in 0..count {
                elements.push(read_gguf_value(f, element_type)?);
            }
            Ok(GgufValue::Array(elements))
        }
        _ => bail!("unsupported GGUF value type: {}", val_type),
    }
}

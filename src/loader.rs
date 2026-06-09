use crate::quant::{QuantizedWeight, Q8_0_TYPE_SIZE};
use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Ok, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

/// a tensor as loaded from a gguf file.
///
/// f32 and f16 tensors are stored as `CpuTensor`.  q8_0 tensors are kept
/// in raw block-compressed form (`QuantizedWeight`) - they are never
/// dequantized to f32, keeping the in-memory footprint at the quantized size.
#[derive(Clone)]
pub enum LoadedTensor {
    /// dequantized f32 tensor (for f32, f16, and small/direct-access tensors)
    F32(CpuTensor),
    /// raw q8_0 block-compressed weight (dequantized on the fly during matmul)
    Q8_0(QuantizedWeight),
}

/// holds the parsed contents of a GGUF v3 file:
/// metadata key-value pairs and named tensors.
pub struct GgufLoader {
    /// metadata key-value pairs from the gguf header
    pub metadata: HashMap<String, GgufValue>,
    /// named tensors.  linear weights are stored as `LoadedTensor::Q8_0`
    /// when the gguf dtype is q8_0; everything else is `LoadedTensor::F32`.
    pub tensors: HashMap<String, LoadedTensor>,
}

/// a typed value from GGUF metadata.
#[derive(Debug)]
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
        log::debug!(
            "loading tensor '{}' dtype={} dims={:?}",
            info.name,
            info.dtype,
            info.dims
        );
        let loaded = match info.dtype {
            0 => {
                // f32: read directly, no dim reversal
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
                LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
            }
            1 => {
                // f16: read and convert to f32. Keep the logical GGUF shape
                // unchanged; model builders handle any linear-weight transpose
                // the same way they do for native f32 tensors.
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
                LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
            }
            8 => {
                // q8_0: store raw, dequantize on the fly during matmul.
                // reverse dims to match the column-major storage convention
                // (same as the old path did for f16/q8_0 tensors).
                let n_blocks = element_count / 32;
                let mut raw = vec![0u8; n_blocks * Q8_0_TYPE_SIZE];
                reader.read_exact(&mut raw)?;
                let mut dims = info.dims;
                dims.reverse();
                LoadedTensor::Q8_0(QuantizedWeight::try_new(raw, dims)?)
            }
            30 => {
                // bf16: brain floating point — upper 16 bits of f32.
                let mut buf = vec![0u8; element_count * 2];
                reader.read_exact(&mut buf)?;
                let mut data = vec![0.0f32; element_count];
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 2;
                    let bits = u16::from_le_bytes(
                        buf[start..start + 2]
                            .try_into()
                            .map_err(|_| anyhow::anyhow!("failed to read bf16 at index {}", i))?,
                    );
                    *dst = f32::from_bits((bits as u32) << 16);
                }
                LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
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
        tensors.insert(info.name, loaded);
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

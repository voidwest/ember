use crate::quant::{QuantizedWeight, Q8_0_TYPE_SIZE};
use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Ok, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

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
/// GGUF stores 2D tensors with the first dim contiguous, i.e. the data is
/// row-major over `[out, in]` for a logical `[in, out]` tensor. The f32
/// matmul expects row-major `[in, out]`, so reinterpret and transpose once.
pub fn gguf_to_row_major_f32(tensor: crate::tensor::CpuTensor) -> crate::tensor::CpuTensor {
    let shape = tensor.shape();
    debug_assert_eq!(shape.len(), 2);
    let reordered =
        crate::tensor::CpuTensor::from_data(vec![shape[1], shape[0]], tensor.data().to_vec());
    reordered.transpose()
}

pub struct GgufLoader {
    /// metadata key-value pairs from the gguf header
    pub metadata: HashMap<String, GgufValue>,
    /// named tensors.  linear weights are stored as `LoadedTensor::Q8_0`
    /// when the gguf dtype is q8_0; everything else is `LoadedTensor::F32`.
    pub tensors: HashMap<String, LoadedTensor>,
}

impl GgufLoader {
    pub(crate) fn take_tensor(&mut self, name: &str) -> Result<LoadedTensor> {
        self.tensors
            .remove(name)
            .with_context(|| format!("Missing tensor: {name}"))
    }

    pub(crate) fn take_f32(&mut self, name: &str) -> Result<CpuTensor> {
        match self.take_tensor(name)? {
            LoadedTensor::F32(tensor) => Ok(tensor),
            LoadedTensor::Q8_0(weight) => Ok(weight.dequantize_all()),
        }
    }

    pub(crate) fn take_optional_f32(&mut self, names: &[String]) -> Option<CpuTensor> {
        let name = names
            .iter()
            .find(|name| self.tensors.contains_key(name.as_str()))?;
        match self.tensors.remove(name.as_str())? {
            LoadedTensor::F32(tensor) => Some(tensor),
            LoadedTensor::Q8_0(weight) => Some(weight.dequantize_all()),
        }
    }
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

/// Load a GGUF file from disk using memory-mapped I/O.
///
/// Q8_0 weights retain shared ranges into the read-only mapping, avoiding a
/// second anonymous-memory copy and allowing the OS to page weights lazily.
/// Dtypes that require conversion (F16/BF16) are materialized as F32.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLoader> {
    let f = File::open(&path).with_context(|| format!("failed to open {:?}", path.as_ref()))?;
    // Safety: the read-only mapping remains alive through every QuantizedWeight
    // that references it. As with all file mappings, callers must not truncate
    // or concurrently mutate the GGUF while it is loaded.
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f)? });
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    load_gguf_from_reader_impl(&mut cursor, Some(Arc::clone(&mmap)))
}

/// load a GGUF file from any readable + seekable source.
/// useful for testing with in-memory buffers (std::io::Cursor<Vec<u8>>).
pub fn load_gguf_from_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufLoader> {
    load_gguf_from_reader_impl(reader, None)
}

fn load_gguf_from_reader_impl<R: Read + Seek>(
    reader: &mut R,
    mmap: Option<Arc<memmap2::Mmap>>,
) -> Result<GgufLoader> {
    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic: {:#x})", magic);
    }

    let version = read_u32(reader)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version: {}", version);
    }

    let tensor_count = usize::try_from(read_u64(reader)?)
        .context("GGUF tensor count does not fit in memory address space")?;
    let metadata_kv_count = usize::try_from(read_u64(reader)?)
        .context("GGUF metadata count does not fit in memory address space")?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(reader)?;
        let val_type = read_u32(reader)?;
        let value = read_gguf_value(reader, val_type)?;
        metadata.insert(key, value);
    }

    let mut tensor_info = read_tensor_info(reader, tensor_count)?;

    let current_pos = reader.stream_position()?;
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(a)) => *a as u64,
        Some(GgufValue::U64(a)) => *a,
        _ => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment {alignment}: expected a power of two");
    }
    let data_start = current_pos
        .checked_add(alignment - 1)
        .context("GGUF aligned data offset overflow")?
        & !(alignment - 1);

    let mut tensors = HashMap::new();
    for info in tensor_info.drain(..) {
        let tensor_offset = data_start
            .checked_add(info.offset)
            .with_context(|| format!("tensor '{}' file offset overflow", info.name))?;
        reader.seek(SeekFrom::Start(tensor_offset))?;
        let element_count = info.dims.iter().try_fold(1usize, |count, &dim| {
            count.checked_mul(dim).with_context(|| {
                format!(
                    "tensor '{}' shape product overflow for dimensions {:?}",
                    info.name, info.dims
                )
            })
        })?;
        log::debug!(
            "loading tensor '{}' dtype={} dims={:?}",
            info.name,
            info.dtype,
            info.dims
        );
        let loaded =
            match info.dtype {
                0 => {
                    // f32: read directly, no dim reversal
                    let mut data = vec![0.0f32; element_count];
                    let byte_len = element_count.checked_mul(4).with_context(|| {
                        format!("tensor '{}' f32 byte size overflow", info.name)
                    })?;
                    let mut buf = vec![0u8; byte_len];
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
                    let byte_len = element_count.checked_mul(2).with_context(|| {
                        format!("tensor '{}' f16 byte size overflow", info.name)
                    })?;
                    let mut buf = vec![0u8; byte_len];
                    reader.read_exact(&mut buf)?;
                    let mut data = vec![0.0f32; element_count];
                    for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                        let start = i * 2;
                        let bits =
                            u16::from_le_bytes(buf[start..start + 2].try_into().map_err(|_| {
                                anyhow::anyhow!("failed to read f16 at index {}", i)
                            })?);
                        *dst = f16::from_bits(bits).to_f32();
                    }
                    LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
                }
                8 => {
                    // q8_0: store raw, dequantize on the fly during matmul.
                    // reverse dims to match the column-major storage convention
                    // (same as the old path did for f16/q8_0 tensors).
                    if !element_count.is_multiple_of(32) {
                        bail!(
                            "tensor '{}' Q8_0 element count {} is not block-aligned",
                            info.name,
                            element_count
                        );
                    }
                    let n_blocks = element_count / 32;
                    let byte_len = n_blocks.checked_mul(Q8_0_TYPE_SIZE).with_context(|| {
                        format!("tensor '{}' Q8_0 byte size overflow", info.name)
                    })?;
                    let mut dims = info.dims;
                    dims.reverse();
                    let weight = if let Some(mmap) = mmap.as_ref() {
                        let start = usize::try_from(tensor_offset).with_context(|| {
                            format!("tensor '{}' offset exceeds address space", info.name)
                        })?;
                        let end = start.checked_add(byte_len).with_context(|| {
                            format!("tensor '{}' mapped range overflow", info.name)
                        })?;
                        if end > mmap.len() {
                            bail!(
                                "tensor '{}' data range {}..{} exceeds file length {}",
                                info.name,
                                start,
                                end,
                                mmap.len()
                            );
                        }
                        QuantizedWeight::try_from_mmap(Arc::clone(mmap), start..end, dims)?
                    } else {
                        let mut raw = vec![0u8; byte_len];
                        reader.read_exact(&mut raw)?;
                        QuantizedWeight::try_new(raw, dims)?
                    };
                    LoadedTensor::Q8_0(weight)
                }
                10..=14 => {
                    // K-family super-blocks (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K):
                    // dequantize to f32 at load time. Keep the logical GGUF
                    // shape; linear-weight transpose handling is identical to
                    // the f32/f16 arms.
                    if !element_count.is_multiple_of(crate::quant_k::QK_K) {
                        bail!(
                            "tensor '{}' dtype {} element count {} is not 256-block-aligned",
                            info.name,
                            info.dtype,
                            element_count
                        );
                    }
                    let n_blocks = element_count / crate::quant_k::QK_K;
                    let block_bytes = crate::quant_k::k_block_bytes(info.dtype)
                        .with_context(|| format!("tensor '{}'", info.name))?;
                    let byte_len = n_blocks.checked_mul(block_bytes).with_context(|| {
                        format!("tensor '{}' K-quant byte size overflow", info.name)
                    })?;
                    let mut raw = vec![0u8; byte_len];
                    reader.read_exact(&mut raw)?;
                    let mut data = vec![0.0f32; element_count];
                    crate::quant_k::dequant_tensor(info.dtype, &raw, &mut data)
                        .map_err(|e| anyhow::anyhow!("tensor '{}': {e}", info.name))?;
                    LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
                }
                30 => {
                    // bf16: brain floating point — upper 16 bits of f32.
                    let byte_len = element_count.checked_mul(2).with_context(|| {
                        format!("tensor '{}' bf16 byte size overflow", info.name)
                    })?;
                    let mut buf = vec![0u8; byte_len];
                    reader.read_exact(&mut buf)?;
                    let mut data = vec![0.0f32; element_count];
                    for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                        let start = i * 2;
                        let bits =
                            u16::from_le_bytes(buf[start..start + 2].try_into().map_err(|_| {
                                anyhow::anyhow!("failed to read bf16 at index {}", i)
                            })?);
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
            dims.push(
                usize::try_from(read_u64(reader)?).with_context(|| {
                    format!("tensor '{}' dimension exceeds address space", name)
                })?,
            );
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
    let len = usize::try_from(read_u64(f)?).context("GGUF string length exceeds address space")?;
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
            let count =
                usize::try_from(read_u64(f)?).context("GGUF array length exceeds address space")?;
            let mut elements = Vec::with_capacity(count);
            for _ in 0..count {
                elements.push(read_gguf_value(f, element_type)?);
            }
            Ok(GgufValue::Array(elements))
        }
        _ => bail!("unsupported GGUF value type: {}", val_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_f32_moves_tensor_storage() {
        let tensor = CpuTensor::from_data(vec![2], vec![1.0, 2.0]);
        let allocation = tensor.data().as_ptr();
        let mut loader = GgufLoader {
            metadata: HashMap::new(),
            tensors: HashMap::from([("weight".to_string(), LoadedTensor::F32(tensor))]),
        };

        let taken = loader.take_f32("weight").unwrap();
        assert_eq!(taken.data().as_ptr(), allocation);
        assert!(!loader.tensors.contains_key("weight"));
    }
}

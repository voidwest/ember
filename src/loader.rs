use crate::quant::{QuantizedWeight, Q8_0_TYPE_SIZE};
use crate::tensor::CpuTensor;
use anyhow::{bail, Context, Ok, Result};
use std::collections::{HashMap, HashSet};
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
    /// raw q4_k/q6_k super-block-compressed weight, kept resident under a
    /// compressed K strategy (dequantized at block granularity during matmul)
    KQuant(crate::quant_k::KQuantWeight),
}

/// holds the parsed contents of a GGUF v3 file:
/// metadata key-value pairs and named tensors.
/// GGUF stores 2D tensors with the first dim contiguous, i.e. the data is
/// row-major over `[out, in]` for a logical `[in, out]` tensor. The f32
/// matmul expects row-major `[in, out]`, so reinterpret and transpose once.
pub fn gguf_to_row_major_f32(tensor: crate::tensor::CpuTensor) -> crate::tensor::CpuTensor {
    let shape = tensor.shape();
    assert_eq!(
        shape.len(),
        2,
        "GGUF row-major conversion requires a 2D tensor"
    );
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
    /// the K-family strategy requested at load time.
    pub k_strategy: crate::quant_k::KStrategy,
    /// per-tensor K-family execution decisions (original dtype, chosen
    /// path, fallback reason). Every decision is recorded; nothing is
    /// silent.
    pub k_decisions: HashMap<String, crate::quant_k::KTensorDecision>,
    /// original per-tensor GGUF records (native dims, dtype code, file
    /// offset), captured before any dtype conversion. `LoadedTensor` only
    /// reflects the post-conversion representation, so this map is the
    /// source of truth for tensor inventory and provenance.
    pub tensor_meta: HashMap<String, TensorMeta>,
}

/// The original GGUF record for one tensor, before any dtype conversion.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// GGUF-native dimensions (first dim contiguous).
    pub dims: Vec<usize>,
    /// GGUF dtype code (see [`ggml_dtype_name`]).
    pub dtype: u32,
    /// Data offset relative to the aligned tensor-data start.
    pub offset: u64,
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
            LoadedTensor::KQuant(weight) => Ok(weight.dequantize_all()),
        }
    }

    pub(crate) fn take_optional_f32(&mut self, names: &[String]) -> Option<CpuTensor> {
        let name = names
            .iter()
            .find(|name| self.tensors.contains_key(name.as_str()))?;
        match self.tensors.remove(name.as_str())? {
            LoadedTensor::F32(tensor) => Some(tensor),
            LoadedTensor::Q8_0(weight) => Some(weight.dequantize_all()),
            LoadedTensor::KQuant(weight) => Some(weight.dequantize_all()),
        }
    }
}

/// a typed value from GGUF metadata.
#[derive(Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
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
///
/// Uses the eager-f32 K strategy — the v0.1/v0.2 reference behavior. Use
/// [`load_gguf_with_k_strategy`] for the compressed-resident paths.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLoader> {
    load_gguf_with_k_strategy(path, crate::quant_k::KStrategy::EagerF32, true)
}

/// Load a GGUF file with an explicit K-family execution policy.
///
/// `allow_fallback` permits per-tensor downgrades (eager-f32 or scalar)
/// when the requested strategy has no native path for a tensor's dtype;
/// without it, such tensors are a hard error naming the tensor and its
/// GGUF dtype. Every decision is recorded in `GgufLoader.k_decisions`;
/// a downgrade is never silent.
pub fn load_gguf_with_k_strategy<P: AsRef<Path>>(
    path: P,
    strategy: crate::quant_k::KStrategy,
    allow_fallback: bool,
) -> Result<GgufLoader> {
    let f = File::open(&path).with_context(|| format!("failed to open {:?}", path.as_ref()))?;
    // Safety: the read-only mapping remains alive through every QuantizedWeight
    // that references it. As with all file mappings, callers must not truncate
    // or concurrently mutate the GGUF while it is loaded.
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f)? });
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    load_gguf_from_reader_impl(
        &mut cursor,
        Some(Arc::clone(&mmap)),
        strategy,
        allow_fallback,
    )
}

/// load a GGUF file from any readable + seekable source.
/// useful for testing with in-memory buffers (std::io::Cursor<Vec<u8>>).
/// Uses the eager-f32 K strategy (reference behavior).
pub fn load_gguf_from_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufLoader> {
    load_gguf_from_reader_impl(reader, None, crate::quant_k::KStrategy::EagerF32, true)
}

/// reader variant of [`load_gguf_with_k_strategy`]; see its docs.
pub fn load_gguf_from_reader_with_k_strategy<R: Read + Seek>(
    reader: &mut R,
    strategy: crate::quant_k::KStrategy,
    allow_fallback: bool,
) -> Result<GgufLoader> {
    load_gguf_from_reader_impl(reader, None, strategy, allow_fallback)
}

fn load_gguf_from_reader_impl<R: Read + Seek>(
    reader: &mut R,
    mmap: Option<Arc<memmap2::Mmap>>,
    k_strategy: crate::quant_k::KStrategy,
    allow_fallback: bool,
) -> Result<GgufLoader> {
    let initial_position = reader.stream_position()?;
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(initial_position))?;
    if file_len.saturating_sub(initial_position) < 24 {
        bail!("GGUF file is too short to contain a complete header");
    }

    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic: {:#x})", magic);
    }

    let version = read_u32(reader)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version: {}", version);
    }

    let tensor_count_raw = read_u64(reader)?;
    let metadata_kv_count_raw = read_u64(reader)?;
    if tensor_count_raw > file_len / 32 {
        bail!("GGUF tensor count {tensor_count_raw} is impossible for a {file_len}-byte file");
    }
    if metadata_kv_count_raw > file_len / 13 {
        bail!(
            "GGUF metadata count {metadata_kv_count_raw} is impossible for a {file_len}-byte file"
        );
    }
    let tensor_count = usize::try_from(tensor_count_raw)
        .context("GGUF tensor count does not fit in memory address space")?;
    let metadata_kv_count = usize::try_from(metadata_kv_count_raw)
        .context("GGUF metadata count does not fit in memory address space")?;

    let mut metadata = HashMap::new();
    metadata
        .try_reserve(metadata_kv_count)
        .context("failed to reserve GGUF metadata table")?;
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(reader)?;
        if key.is_empty() {
            bail!("GGUF metadata keys must not be empty");
        }
        let val_type = read_u32(reader)?;
        let value = read_gguf_value(reader, val_type)?;
        if metadata.insert(key.clone(), value).is_some() {
            bail!("duplicate GGUF metadata key '{key}'");
        }
    }

    let mut tensor_info = read_tensor_info(reader, tensor_count)?;
    let mut tensor_meta = HashMap::new();
    tensor_meta
        .try_reserve(tensor_info.len())
        .context("failed to reserve GGUF tensor-metadata table")?;
    for info in &tensor_info {
        tensor_meta.insert(
            info.name.clone(),
            TensorMeta {
                dims: info.dims.clone(),
                dtype: info.dtype,
                offset: info.offset,
            },
        );
    }
    let mut k_decisions = HashMap::new();
    k_decisions
        .try_reserve(tensor_info.len())
        .context("failed to reserve GGUF K-decision table")?;

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

    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(tensor_info.len())
        .context("failed to reserve GGUF tensor range table")?;
    for info in &tensor_info {
        let byte_len = tensor_byte_len(info)?;
        let start = data_start
            .checked_add(info.offset)
            .with_context(|| format!("tensor '{}' file offset overflow", info.name))?;
        let end = start
            .checked_add(u64::try_from(byte_len).context("tensor byte length exceeds u64")?)
            .with_context(|| format!("tensor '{}' file range overflow", info.name))?;
        if end > file_len {
            bail!(
                "tensor '{}' data range {start}..{end} exceeds file length {file_len}",
                info.name
            );
        }
        ranges.push((start, end, info.name.as_str()));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        let (_, previous_end, previous_name) = pair[0];
        let (next_start, _, next_name) = pair[1];
        if next_start < previous_end {
            bail!(
                "GGUF tensor ranges overlap: '{previous_name}' ends at {previous_end}, \
                 '{next_name}' starts at {next_start}"
            );
        }
    }

    let mut tensors = HashMap::new();
    tensors
        .try_reserve(tensor_info.len())
        .context("failed to reserve GGUF tensor table")?;
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
        let loaded = match info.dtype {
            0 => {
                // f32: read directly, no dim reversal
                let mut data = vec![0.0f32; element_count];
                let byte_len = element_count
                    .checked_mul(4)
                    .with_context(|| format!("tensor '{}' f32 byte size overflow", info.name))?;
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
                let byte_len = element_count
                    .checked_mul(2)
                    .with_context(|| format!("tensor '{}' f16 byte size overflow", info.name))?;
                let mut buf = vec![0u8; byte_len];
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
                if !element_count.is_multiple_of(32) {
                    bail!(
                        "tensor '{}' Q8_0 element count {} is not block-aligned",
                        info.name,
                        element_count
                    );
                }
                let n_blocks = element_count / 32;
                let byte_len = n_blocks
                    .checked_mul(Q8_0_TYPE_SIZE)
                    .with_context(|| format!("tensor '{}' Q8_0 byte size overflow", info.name))?;
                let mut dims = info.dims;
                dims.reverse();
                let weight = if let Some(mmap) = mmap.as_ref() {
                    let start = usize::try_from(tensor_offset).with_context(|| {
                        format!("tensor '{}' offset exceeds address space", info.name)
                    })?;
                    let end = start
                        .checked_add(byte_len)
                        .with_context(|| format!("tensor '{}' mapped range overflow", info.name))?;
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
                // K-family super-blocks (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K).
                // Under a compressed strategy the supported dtypes
                // (Q4_K/Q6_K) stay packed and resident; everything
                // else dequantizes to f32 at load (the eager-f32
                // reference), with every decision recorded.
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
                let native = crate::quant_k::KQuantDtype::from_gguf(info.dtype);
                let (execution, fallback_reason) =
                    resolve_k_execution(&info, k_strategy, native, allow_fallback)?;
                k_decisions.insert(
                    info.name.clone(),
                    crate::quant_k::KTensorDecision {
                        gguf_dtype: info.dtype,
                        execution,
                        fallback_reason,
                    },
                );
                match execution {
                    crate::quant_k::KExecution::EagerF32 => {
                        let mut raw = vec![0u8; byte_len];
                        reader.read_exact(&mut raw)?;
                        let mut data = vec![0.0f32; element_count];
                        crate::quant_k::dequant_tensor(info.dtype, &raw, &mut data)
                            .map_err(|e| anyhow::anyhow!("tensor '{}': {e}", info.name))?;
                        LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
                    }
                    crate::quant_k::KExecution::CompressedScalar
                    | crate::quant_k::KExecution::CompressedX86 => {
                        let native = native.expect("compressed execution requires a native dtype");
                        let mut dims = info.dims.clone();
                        dims.reverse();
                        if dims.len() != 2 {
                            bail!(
                                "tensor '{}' K-quant must be 2D for compressed residency, got {:?}",
                                info.name,
                                info.dims
                            );
                        }
                        let shape = [dims[0], dims[1]];
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
                            crate::quant_k::KQuantWeight::try_from_mmap(
                                Arc::clone(mmap),
                                start..end,
                                shape,
                                native,
                                execution,
                            )?
                        } else {
                            let mut raw = vec![0u8; byte_len];
                            reader.read_exact(&mut raw)?;
                            crate::quant_k::KQuantWeight::try_new(raw, shape, native)?
                        };
                        LoadedTensor::KQuant(weight)
                    }
                }
            }
            30 => {
                // bf16: brain floating point — upper 16 bits of f32.
                let byte_len = element_count
                    .checked_mul(2)
                    .with_context(|| format!("tensor '{}' bf16 byte size overflow", info.name))?;
                let mut buf = vec![0u8; byte_len];
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
                bail!(
                    "tensor '{}' uses unsupported GGML dtype {}",
                    info.name,
                    info.dtype
                );
            }
        };
        if tensors.insert(info.name.clone(), loaded).is_some() {
            bail!("duplicate GGUF tensor name '{}'", info.name);
        }
    }
    Ok(GgufLoader {
        metadata,
        tensors,
        k_strategy,
        k_decisions,
        tensor_meta,
    })
}

/// Decide the per-tensor K-family execution path under the requested
/// strategy, returning the decision and a fallback reason (if any).
/// Hard errors name the tensor, its GGUF dtype, and any missing CPU
/// feature; downgrades only happen when `allow_fallback` permits them
/// and are recorded by the caller in `k_decisions`.
fn resolve_k_execution(
    info: &TensorInfo,
    strategy: crate::quant_k::KStrategy,
    native: Option<crate::quant_k::KQuantDtype>,
    allow_fallback: bool,
) -> Result<(crate::quant_k::KExecution, Option<String>)> {
    use crate::quant_k::{KExecution, KStrategy};
    let x86_available = crate::k_matmul_x86::avx2_supported();
    let no_native_kernel = || {
        format!(
            "dtype {} has no native kernel in v0.3",
            ggml_dtype_name(info.dtype).unwrap_or("unknown")
        )
    };
    match strategy {
        KStrategy::EagerF32 => Ok((KExecution::EagerF32, None)),
        KStrategy::Auto => Ok(match native {
            Some(_) if x86_available => (KExecution::CompressedX86, None),
            Some(_) => (KExecution::CompressedScalar, None),
            None => (KExecution::EagerF32, Some(no_native_kernel())),
        }),
        KStrategy::Scalar => match native {
            Some(_) => Ok((KExecution::CompressedScalar, None)),
            None if allow_fallback => Ok((KExecution::EagerF32, Some(no_native_kernel()))),
            None => bail!(
                "tensor '{}' uses GGUF dtype {} which has no native kernel in v0.3; \
                 pass --k-allow-fallback to run it through the eager-f32 path",
                info.name,
                ggml_dtype_name(info.dtype).unwrap_or("unknown")
            ),
        },
        KStrategy::X86 => match (native, x86_available) {
            (Some(_), true) => Ok((KExecution::CompressedX86, None)),
            (Some(_), false) if allow_fallback => Ok((
                KExecution::CompressedScalar,
                Some("x86 feature set unavailable (avx2+fma)".to_string()),
            )),
            (Some(_), false) => bail!(
                "--k-strategy x86 requires the AVX2+FMA feature set (avx2, fma); \
                 pass --k-allow-fallback to run the scalar path"
            ),
            (None, _) if allow_fallback => Ok((KExecution::EagerF32, Some(no_native_kernel()))),
            (None, _) => bail!(
                "tensor '{}' uses GGUF dtype {} which has no native kernel in v0.3; \
                 pass --k-allow-fallback to run it through the eager-f32 path",
                info.name,
                ggml_dtype_name(info.dtype).unwrap_or("unknown")
            ),
        },
    }
}

struct TensorInfo {
    name: String,
    dims: Vec<usize>,
    dtype: u32,
    offset: u64,
}

fn read_tensor_info<R: Read + Seek>(reader: &mut R, count: usize) -> Result<Vec<TensorInfo>> {
    let mut info = Vec::new();
    info.try_reserve_exact(count)
        .context("failed to reserve GGUF tensor-info table")?;
    let mut names = HashSet::new();
    names
        .try_reserve(count)
        .context("failed to reserve GGUF tensor-name table")?;
    for _ in 0..count {
        let name = read_gguf_string(reader)?;
        if name.is_empty() {
            bail!("GGUF tensor names must not be empty");
        }
        if !names.insert(name.clone()) {
            bail!("duplicate GGUF tensor name '{name}'");
        }
        let n_dims = read_u32(reader)?;
        if !(1..=4).contains(&n_dims) {
            bail!("tensor '{name}' has invalid dimension count {n_dims}; expected 1..=4");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = usize::try_from(read_u64(reader)?)
                .with_context(|| format!("tensor '{}' dimension exceeds address space", name))?;
            if dim == 0 {
                bail!("tensor '{name}' has a zero-sized dimension");
            }
            dims.push(dim);
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

fn tensor_byte_len(info: &TensorInfo) -> Result<usize> {
    let element_count = info.dims.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).with_context(|| {
            format!(
                "tensor '{}' shape product overflow for dimensions {:?}",
                info.name, info.dims
            )
        })
    })?;
    gguf_dtype_byte_len(info.dtype, element_count)
        .with_context(|| format!("tensor '{}'", info.name))
}

/// Encoded byte length of `element_count` values of a GGUF dtype.
///
/// Shared by the loader's range validation and the tensor-inventory dump.
pub fn gguf_dtype_byte_len(dtype: u32, element_count: usize) -> Result<usize> {
    match dtype {
        0 => element_count
            .checked_mul(4)
            .with_context(|| "f32 byte size overflow".to_string()),
        1 | 30 => element_count
            .checked_mul(2)
            .with_context(|| "16-bit byte size overflow".to_string()),
        8 => {
            if !element_count.is_multiple_of(crate::quant::Q8_0_BLOCK_SIZE) {
                bail!("Q8_0 element count is not block-aligned");
            }
            (element_count / crate::quant::Q8_0_BLOCK_SIZE)
                .checked_mul(Q8_0_TYPE_SIZE)
                .with_context(|| "Q8_0 byte size overflow".to_string())
        }
        10..=14 => {
            if !element_count.is_multiple_of(crate::quant_k::QK_K) {
                bail!("dtype {dtype} element count is not K-block-aligned");
            }
            let block_bytes =
                crate::quant_k::k_block_bytes(dtype).with_context(|| format!("dtype {dtype}"))?;
            (element_count / crate::quant_k::QK_K)
                .checked_mul(block_bytes)
                .with_context(|| "K-quant byte size overflow".to_string())
        }
        dtype => bail!("unsupported GGML dtype {dtype}"),
    }
}

/// GGUF dtype code -> lowercase type name (current GGUF numbering).
pub fn ggml_dtype_name(dtype: u32) -> Option<&'static str> {
    match dtype {
        0 => Some("f32"),
        1 => Some("f16"),
        2 => Some("q4_0"),
        3 => Some("q4_1"),
        6 => Some("q5_0"),
        7 => Some("q5_1"),
        8 => Some("q8_0"),
        9 => Some("q8_1"),
        10 => Some("q2_k"),
        11 => Some("q3_k"),
        12 => Some("q4_k"),
        13 => Some("q5_k"),
        14 => Some("q6_k"),
        15 => Some("q8_k"),
        30 => Some("bf16"),
        _ => None,
    }
}

fn read_u8<R: Read>(f: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(u8::from_le_bytes(buf))
}

fn read_i8<R: Read>(f: &mut R) -> Result<i8> {
    Ok(read_u8(f)? as i8)
}

fn read_u16<R: Read>(f: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(f: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
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

fn read_i64<R: Read>(f: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(f: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(f: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn remaining_bytes<R: Seek>(reader: &mut R) -> Result<u64> {
    let position = reader.stream_position()?;
    let end = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(position))?;
    end.checked_sub(position)
        .context("reader position exceeds its end")
}

fn read_gguf_string<R: Read + Seek>(f: &mut R) -> Result<String> {
    let len = usize::try_from(read_u64(f)?).context("GGUF string length exceeds address space")?;
    let remaining = remaining_bytes(f)?;
    if u64::try_from(len).context("GGUF string length exceeds u64")? > remaining {
        bail!("GGUF string length {len} exceeds the {remaining} bytes remaining in the file");
    }
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .context("failed to reserve GGUF string buffer")?;
    buf.resize(len, 0);
    f.read_exact(&mut buf).context("read string failed")?;
    String::from_utf8(buf).context("invalid utf8 in string")
}

fn minimum_value_size(val_type: u32) -> Result<u64> {
    match val_type {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        8 => Ok(8),
        9 => Ok(12),
        10..=12 => Ok(8),
        _ => bail!("unsupported GGUF value type: {val_type}"),
    }
}

fn read_gguf_value<R: Read + Seek>(f: &mut R, val_type: u32) -> Result<GgufValue> {
    read_gguf_value_inner(f, val_type, 0)
}

fn read_gguf_value_inner<R: Read + Seek>(
    f: &mut R,
    val_type: u32,
    depth: usize,
) -> Result<GgufValue> {
    if depth > 16 {
        bail!("GGUF metadata arrays are nested more than 16 levels deep");
    }
    match val_type {
        0 => Ok(GgufValue::U8(read_u8(f)?)),
        1 => Ok(GgufValue::I8(read_i8(f)?)),
        2 => Ok(GgufValue::U16(read_u16(f)?)),
        3 => Ok(GgufValue::I16(read_i16(f)?)),
        5 => Ok(GgufValue::I32(read_i32(f)?)),
        4 => Ok(GgufValue::U32(read_u32(f)?)),
        6 => Ok(GgufValue::F32(read_f32(f)?)),
        7 => {
            let value = read_u8(f)?;
            if value > 1 {
                bail!("invalid GGUF boolean value {value}; expected 0 or 1");
            }
            Ok(GgufValue::Bool(value == 1))
        }
        8 => Ok(GgufValue::Str(read_gguf_string(f)?)),
        10 => Ok(GgufValue::U64(read_u64(f)?)),
        11 => Ok(GgufValue::I64(read_i64(f)?)),
        12 => Ok(GgufValue::F64(read_f64(f)?)),
        9 => {
            let element_type = read_u32(f)?;
            let count =
                usize::try_from(read_u64(f)?).context("GGUF array length exceeds address space")?;
            let minimum_bytes = minimum_value_size(element_type)?
                .checked_mul(u64::try_from(count).context("GGUF array length exceeds u64")?)
                .context("GGUF array minimum byte size overflow")?;
            let remaining = remaining_bytes(f)?;
            if minimum_bytes > remaining {
                bail!(
                    "GGUF array of {count} type-{element_type} values requires at least \
                     {minimum_bytes} bytes but only {remaining} remain"
                );
            }
            let mut elements = Vec::new();
            elements
                .try_reserve_exact(count)
                .context("failed to reserve GGUF metadata array")?;
            for _ in 0..count {
                elements.push(read_gguf_value_inner(f, element_type, depth + 1)?);
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
            k_strategy: crate::quant_k::KStrategy::EagerF32,
            k_decisions: HashMap::new(),
            tensor_meta: HashMap::new(),
        };

        let taken = loader.take_f32("weight").unwrap();
        assert_eq!(taken.data().as_ptr(), allocation);
        assert!(!loader.tensors.contains_key("weight"));
    }

    /// Build a minimal GGUF v3 file with one K-family tensor
    /// (`blk.0.attn_q.weight`, dims [256, blocks], zero payload) and no
    /// metadata. Zero blocks dequantize to zeros.
    fn write_minimal_gguf_with_k_tensor(dtype: u32, blocks: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        out.extend_from_slice(&0u64.to_le_bytes()); // no metadata kv
        let name = b"blk.0.attn_q.weight";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        out.extend_from_slice(&256u64.to_le_bytes());
        out.extend_from_slice(&(blocks as u64).to_le_bytes());
        out.extend_from_slice(&(dtype).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // offset
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.resize(
            out.len() + blocks * crate::quant_k::k_block_bytes(dtype).unwrap(),
            0,
        );
        out
    }

    #[test]
    fn tensor_meta_records_original_gguf_dtype() {
        let bytes = write_minimal_gguf_with_k_tensor(12, 128);
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader(&mut cursor).unwrap();

        let meta = loader
            .tensor_meta
            .get("blk.0.attn_q.weight")
            .expect("tensor metadata recorded before dtype conversion");
        assert_eq!(meta.dims, vec![256, 128]);
        assert_eq!(meta.dtype, 12);
        assert_eq!(meta.offset, 0);
        assert_eq!(ggml_dtype_name(meta.dtype), Some("q4_k"));
        assert_eq!(
            gguf_dtype_byte_len(meta.dtype, 256 * 128).unwrap(),
            128 * crate::quant_k::Q4_K_BLOCK_BYTES
        );

        // the eager loader still materializes the tensor as f32
        match loader.tensors.get("blk.0.attn_q.weight") {
            Some(LoadedTensor::F32(tensor)) => assert_eq!(tensor.shape(), &[256, 128]),
            _ => panic!("expected eager f32 tensor from the current loader"),
        }
    }

    #[test]
    fn scalar_strategy_keeps_q4_k_compressed_resident() {
        let bytes = write_minimal_gguf_with_k_tensor(12, 128);
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Scalar,
            false,
        )
        .unwrap();

        let decision = loader
            .k_decisions
            .get("blk.0.attn_q.weight")
            .expect("per-tensor decision recorded");
        assert_eq!(decision.gguf_dtype, 12);
        assert_eq!(
            decision.execution,
            crate::quant_k::KExecution::CompressedScalar
        );
        assert!(decision.fallback_reason.is_none());

        match loader.tensors.get("blk.0.attn_q.weight") {
            Some(LoadedTensor::KQuant(weight)) => {
                assert_eq!((weight.out_features(), weight.in_features()), (128, 256));
                assert_eq!(weight.dtype(), crate::quant_k::KQuantDtype::Q4K);
                assert_eq!(weight.byte_len(), 128 * crate::quant_k::Q4_K_BLOCK_BYTES);
                assert!(!weight.is_mapped(), "reader loads are owned");
            }
            other => panic!(
                "expected compressed KQuant tensor, got {}",
                match other {
                    Some(LoadedTensor::F32(_)) => "F32",
                    Some(LoadedTensor::Q8_0(_)) => "Q8_0",
                    Some(LoadedTensor::KQuant(_)) => "KQuant",
                    None => "none",
                }
            ),
        }
    }

    #[test]
    fn auto_strategy_falls_back_to_eager_for_q2_k_with_recorded_reason() {
        let bytes = write_minimal_gguf_with_k_tensor(10, 8); // q2_k, no native kernel
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Auto,
            false,
        )
        .unwrap();

        let decision = loader
            .k_decisions
            .get("blk.0.attn_q.weight")
            .expect("per-tensor decision recorded");
        assert_eq!(decision.execution, crate::quant_k::KExecution::EagerF32);
        let reason = decision
            .fallback_reason
            .as_deref()
            .expect("auto fallback reason recorded");
        assert!(reason.contains("q2_k"), "reason: {reason}");
        assert!(matches!(
            loader.tensors.get("blk.0.attn_q.weight"),
            Some(LoadedTensor::F32(_))
        ));
    }

    #[test]
    fn scalar_strategy_hard_fails_q2_k_without_allow_fallback() {
        let bytes = write_minimal_gguf_with_k_tensor(10, 8);
        let mut cursor = std::io::Cursor::new(bytes);
        let err = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Scalar,
            false,
        )
        .err()
        .expect("scalar strategy must hard-fail q2_k without allow-fallback");
        let message = err.to_string();
        assert!(
            message.contains("blk.0.attn_q.weight"),
            "message: {message}"
        );
        assert!(message.contains("q2_k"), "message: {message}");

        // with --k-allow-fallback the same file loads eager and records why
        let mut cursor = std::io::Cursor::new(write_minimal_gguf_with_k_tensor(10, 8));
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Scalar,
            true,
        )
        .unwrap();
        let decision = &loader.k_decisions["blk.0.attn_q.weight"];
        assert_eq!(decision.execution, crate::quant_k::KExecution::EagerF32);
        assert!(decision.fallback_reason.is_some());
    }

    #[test]
    fn execution_inventory_records_compressed_residency() {
        let bytes = write_minimal_gguf_with_k_tensor(12, 128);
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Scalar,
            false,
        )
        .unwrap();
        let inventory = crate::artifact::ExecutionInventory::from_loader(&loader);

        assert_eq!(inventory.requested_strategy, "compressed-scalar");
        assert_eq!(inventory.tensors.len(), 1);
        let tensor = &inventory.tensors[0];
        assert_eq!(tensor.name, "blk.0.attn_q.weight");
        assert_eq!(tensor.gguf_dtype, "q4_k");
        assert_eq!(tensor.gguf_dtype_code, 12);
        assert_eq!(tensor.resident, "compressed");
        assert_eq!(tensor.strategy, "compressed-scalar");
        assert_eq!(tensor.kernel, "scalar-q4k");
        assert_eq!(tensor.cpu_features, "none");
        assert!(tensor.fallback_reason.is_none());
        assert_eq!(tensor.workspace_bytes, crate::quant_k::QK_K * 4);

        let summary = &inventory.summary;
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.fallback_count, 0);
        assert_eq!(summary.compressed_bytes, (128 * 144) as u64);
        assert_eq!(summary.expanded_bytes, 0);
        assert_eq!(summary.per_dtype.len(), 1);
        assert_eq!(summary.per_dtype[0].dtype, "q4_k");
        assert_eq!(summary.per_dtype[0].tensor_count, 1);
        assert_eq!(summary.per_dtype[0].compressed_bytes, (128 * 144) as u64);
        assert_eq!(summary.per_dtype[0].expanded_bytes, 0);
    }

    #[test]
    fn execution_inventory_records_eager_and_fallback() {
        // auto on q2_k: eager-f32 with a recorded fallback reason
        let bytes = write_minimal_gguf_with_k_tensor(10, 8);
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Auto,
            false,
        )
        .unwrap();
        let inventory = crate::artifact::ExecutionInventory::from_loader(&loader);

        assert_eq!(inventory.requested_strategy, "auto");
        let tensor = &inventory.tensors[0];
        assert_eq!(tensor.resident, "f32");
        assert_eq!(tensor.strategy, "eager-f32");
        assert_eq!(tensor.kernel, "eager-f32-dequant");
        assert!(tensor.fallback_reason.is_some());
        assert_eq!(inventory.summary.fallback_count, 1);
        assert_eq!(inventory.summary.compressed_bytes, 0);
        assert_eq!(inventory.summary.expanded_bytes, (8 * 256 * 4) as u64);
    }

    #[test]
    fn x86_strategy_records_compressed_x86_when_avx2_available() {
        let bytes = write_minimal_gguf_with_k_tensor(14, 4); // q6_k
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::X86,
            false,
        )
        .unwrap();

        let decision = &loader.k_decisions["blk.0.attn_q.weight"];
        if crate::k_matmul_x86::avx2_supported() {
            assert_eq!(
                decision.execution,
                crate::quant_k::KExecution::CompressedX86
            );
            assert!(decision.fallback_reason.is_none());
            let inventory = crate::artifact::ExecutionInventory::from_loader(&loader);
            assert_eq!(inventory.tensors[0].kernel, "avx2-q6k");
            assert_eq!(inventory.tensors[0].strategy, "compressed-x86");
            assert_eq!(inventory.tensors[0].cpu_features, "avx2+fma");
        } else {
            // without the feature set the request hard-fails (no
            // allow-fallback); this branch only runs on non-AVX2 hosts
            assert!(decision.fallback_reason.is_some());
        }
    }

    #[test]
    fn auto_strategy_selects_x86_when_avx2_available() {
        let bytes = write_minimal_gguf_with_k_tensor(12, 8); // q4_k
        let mut cursor = std::io::Cursor::new(bytes);
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::Auto,
            false,
        )
        .unwrap();
        let decision = &loader.k_decisions["blk.0.attn_q.weight"];
        let expected = if crate::k_matmul_x86::avx2_supported() {
            crate::quant_k::KExecution::CompressedX86
        } else {
            crate::quant_k::KExecution::CompressedScalar
        };
        assert_eq!(decision.execution, expected);
        assert!(decision.fallback_reason.is_none());
    }

    #[test]
    fn x86_strategy_falls_back_to_scalar_only_with_allow_fallback() {
        // on hosts without AVX2 the request must hard-fail unless the
        // user explicitly allows the downgrade; on AVX2 hosts both paths
        // still validate the fallback recording machinery
        let bytes = write_minimal_gguf_with_k_tensor(14, 4);
        let mut cursor = std::io::Cursor::new(bytes);
        let result = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::X86,
            false,
        );
        if crate::k_matmul_x86::avx2_supported() {
            assert!(result.is_ok(), "avx2 hosts accept the x86 request");
        } else {
            let message = result.err().expect("hard-fail without AVX2").to_string();
            assert!(message.contains("avx2"), "message: {message}");
        }

        // with allow-fallback the load always succeeds and records why
        let mut cursor = std::io::Cursor::new(write_minimal_gguf_with_k_tensor(14, 4));
        let loader = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::X86,
            true,
        )
        .unwrap();
        let decision = &loader.k_decisions["blk.0.attn_q.weight"];
        if crate::k_matmul_x86::avx2_supported() {
            assert_eq!(
                decision.execution,
                crate::quant_k::KExecution::CompressedX86
            );
            assert!(decision.fallback_reason.is_none());
        } else {
            assert_eq!(
                decision.execution,
                crate::quant_k::KExecution::CompressedScalar
            );
            assert!(decision.fallback_reason.is_some());
        }
    }
}

use crate::quant::{QuantizedWeight, Q8_0_TYPE_SIZE};
use crate::tensor::CpuTensor;
/// Structured errors from the GGUF loading boundary. The loader is the
/// crate's main external-data seam: malformed or unsupported files are
/// expected inputs, so they get typed errors (Luminal lesson: `Result`
/// everywhere external data enters; rich errors, not panics).
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// Structural or semantic problem in the GGUF data itself.
    #[error("{0}")]
    Malformed(String),
    /// A GGUF count/offset does not fit the host address space.
    #[error("{0}")]
    Overflow(String),
    /// Memory reservation failed (counts are validated before reserving,
    /// so this is a defensive path).
    #[error("{0}")]
    Reservation(String),
    /// Underlying I/O failure.
    #[error("failed to read GGUF: {0}")]
    Io(#[from] std::io::Error),
}

impl LoaderError {
    fn malformed(message: impl Into<String>) -> Self {
        Self::Malformed(message.into())
    }
    fn overflow(message: impl Into<String>) -> Self {
        Self::Overflow(message.into())
    }
    fn reservation(message: impl Into<String>) -> Self {
        Self::Reservation(message.into())
    }
}

type Result<T> = std::result::Result<T, LoaderError>;
use anyhow::Context as _;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

/// Limits for values that cross from GGUF metadata into model construction.
///
/// These bounds are deliberately generous for supported models, but keep
/// hostile metadata from driving unbounded block, cache, or RoPE allocations.
pub mod limits {
    /// Maximum model context length.
    pub const MAX_CONTEXT_LEN: usize = 2_000_000;
    /// Maximum transformer layer count.
    pub const MAX_LAYERS: usize = 4_096;
    /// Maximum embedding width.
    pub const MAX_EMBED_DIM: usize = 1_000_000;
    /// Maximum vocabulary size.
    pub const MAX_VOCAB_SIZE: usize = 16_000_000;
    /// Maximum attention head count.
    pub const MAX_HEADS: usize = 8_192;
    /// Maximum attention head dimension.
    pub const MAX_HEAD_DIM: usize = 512;
    /// Maximum feed-forward intermediate width.
    pub const MAX_INTERMEDIATE_DIM: usize = 4_000_000;
    /// Maximum sliding-attention window.
    pub const MAX_SLIDING_WINDOW: usize = 16 * 1024 * 1024;
    /// Conservative maximum for the `context * head_dim` RoPE allocation
    /// product. `compute_rope_freqs` stores half as many values per table;
    /// this cap keeps the cosine+sine pair near 128 MiB at worst.
    pub const MAX_ROPE_TABLE_ELEMENTS: usize = 1 << 25;
    /// Maximum number of tensor records accepted from one GGUF header.
    /// Real model files are orders of magnitude smaller; this bound prevents
    /// sparse headers from forcing unbounded hash-map reservations.
    pub const MAX_TENSOR_COUNT: usize = 100_000;
    /// Maximum number of metadata key/value records accepted from one header.
    pub const MAX_METADATA_KV_COUNT: usize = 100_000;
    /// Maximum number of values in one metadata array.
    pub const MAX_METADATA_ARRAY_ELEMENTS: usize = 1_000_000;
    /// Aggregate cap on materialized metadata values. This includes array
    /// containers and scalar elements, not just top-level key/value records.
    pub const MAX_METADATA_VALUES: usize = 1_000_000;
    /// Maximum bytes in an individual GGUF string (keys, names, and values).
    pub const MAX_STRING_BYTES: usize = 1 << 20;
    /// Aggregate bytes available to metadata strings in one GGUF header.
    pub const MAX_METADATA_STRING_BYTES: usize = 64 << 20;
    /// Aggregate bytes available to tensor names in one GGUF header.
    pub const MAX_TENSOR_NAME_BYTES: usize = 64 << 20;
    /// Maximum encoded payload bytes declared by one tensor.
    pub const MAX_TENSOR_BYTES: u64 = 1 << 30;
    /// Maximum transient/final bytes the loader may materialize for one
    /// tensor. Mapped packed tensors count only anonymous allocations here.
    pub const MAX_TENSOR_ALLOCATION_BYTES: u64 = 1 << 30;
    /// Maximum aggregate anonymous bytes materialized while loading a model.
    /// This keeps the default loader bounded on the project's 16 GiB hosts;
    /// packed mmap-backed tensors do not consume this budget.
    pub const MAX_TOTAL_TENSOR_ALLOCATION_BYTES: u64 = 8 << 30;
    /// Maximum logical size of a GGUF file accepted by the mmap-backed loader.
    ///
    /// This intentionally leaves room below typical 64-bit address-space
    /// limits while preserving the project's current models (the largest
    /// supported Q8 files are about 8.5 GiB). In particular, checking the
    /// length before mapping prevents a sparse hostile file from reserving an
    /// unbounded virtual mapping.
    pub const MAX_GGUF_FILE_BYTES: u64 = 16 << 30;
    /// Maximum aggregate encoded Q8_0 bytes eligible for automatic packed
    /// decode. The per-layer builders repack every eligible Q8_0 projection
    /// into a same-size VNNI buffer (anonymous memory on top of the file
    /// mapping), so models whose Q8_0 payload exceeds this budget skip the
    /// automatic packing step instead of doubling the model's resident size
    /// on a 16 GiB host.
    pub const MAX_PACKED_DECODE_BYTES: u64 = 8 << 30;
}

/// a tensor as loaded from a gguf file.
///
/// f32 and f16 tensors are stored as `CpuTensor`.  q8_0 tensors are kept
/// in raw block-compressed form (`QuantizedWeight`) - they are never
/// dequantized to f32, keeping the in-memory footprint at the quantized size.
#[derive(Clone)]
pub enum LoadedTensor {
    /// dequantized f32 tensor (for f32, f16, and small/direct-access tensors)
    F32(CpuTensor),
    /// raw q8_0 block-compressed weight (consumed by packed integer matmul)
    Q8_0(QuantizedWeight),
    /// raw q4_k/q6_k super-block-compressed weight, kept resident under a
    /// compressed K strategy (multiplied by transient Q8_K activation rows)
    KQuant(crate::quant_k::KQuantWeight),
}

/// holds the parsed contents of a GGUF v3 file:
/// metadata key-value pairs and named tensors.
/// GGUF stores 2D tensors with the first dim contiguous, i.e. the data is
/// row-major over `[out, in]` for a logical `[in, out]` tensor. The f32
/// matmul expects row-major `[in, out]`, so reinterpret and transpose once.
///
/// This checked variant is used by model loaders at the untrusted GGUF
/// boundary. In particular, a malformed rank-1/3 tensor must be rejected
/// rather than reaching [`CpuTensor::transpose`]'s assertion.
pub fn try_gguf_to_row_major_f32(
    tensor: crate::tensor::CpuTensor,
) -> std::result::Result<crate::tensor::CpuTensor, LoaderError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(LoaderError::malformed(format!(
            "GGUF row-major conversion requires a 2D tensor, got shape {shape:?}"
        )));
    }
    let reordered =
        crate::tensor::CpuTensor::from_data(vec![shape[1], shape[0]], tensor.data().to_vec());
    Ok(reordered.transpose())
}

/// Convert a validated 2-D GGUF tensor to row-major layout.
///
/// This compatibility wrapper retains the historical infallible API for
/// callers that have already checked the tensor rank. New loaders should use
/// [`try_gguf_to_row_major_f32`] so malformed external data returns a
/// structured [`LoaderError`] instead of panicking.
pub fn gguf_to_row_major_f32(tensor: crate::tensor::CpuTensor) -> crate::tensor::CpuTensor {
    try_gguf_to_row_major_f32(tensor)
        .expect("GGUF row-major conversion requires a validated 2D tensor")
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

pub(crate) fn f32_dequantization_bytes(
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<u64> {
    let elements = out_features.checked_mul(in_features).ok_or_else(|| {
        LoaderError::overflow(format!(
            "tensor '{name}' f32 dequantization shape product overflow"
        ))
    })?;
    let bytes = u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| {
            LoaderError::overflow(format!(
                "tensor '{name}' f32 dequantization allocation size overflow"
            ))
        })?;
    if bytes > limits::MAX_TENSOR_ALLOCATION_BYTES {
        return Err(LoaderError::malformed(format!(
            "tensor '{name}' requires {} bytes for f32 dequantization, exceeding the {}-byte limit",
            bytes,
            limits::MAX_TENSOR_ALLOCATION_BYTES
        )));
    }
    Ok(bytes)
}

pub(crate) fn check_f32_dequantization_size(
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    f32_dequantization_bytes(name, out_features, in_features).map(|_| ())
}

/// Require a model builder's mandatory tensor inventory before allocating
/// metadata-sized derived state (for example, RoPE tables). This is a cheap
/// fail-closed gate for malformed or truncated model inputs.
pub(crate) fn require_tensors(loader: &GgufLoader, names: &[String]) -> Result<()> {
    let mut missing = Vec::new();
    for name in names {
        if !loader.tensors.contains_key(name) {
            missing.push(name.as_str());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(LoaderError::malformed(format!(
            "missing {} required tensor(s): {}",
            missing.len(),
            missing.join(", ")
        )))
    }
}

impl GgufLoader {
    pub(crate) fn take_tensor(&mut self, name: &str) -> Result<LoadedTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| LoaderError::malformed(format!("Missing tensor: {name}")))
    }

    pub(crate) fn take_f32(&mut self, name: &str) -> Result<CpuTensor> {
        match self.take_tensor(name)? {
            LoadedTensor::F32(tensor) => Ok(tensor),
            LoadedTensor::Q8_0(weight) => {
                check_f32_dequantization_size(name, weight.out_features(), weight.in_features())?;
                Ok(weight.dequantize_all())
            }
            LoadedTensor::KQuant(weight) => {
                check_f32_dequantization_size(name, weight.out_features(), weight.in_features())?;
                Ok(weight.dequantize_all())
            }
        }
    }

    pub(crate) fn take_optional_f32(&mut self, names: &[String]) -> Result<Option<CpuTensor>> {
        let Some(name) = names
            .iter()
            .find(|name| self.tensors.contains_key(name.as_str()))
        else {
            return Ok(None);
        };
        let tensor = match self.tensors.remove(name.as_str()) {
            Some(LoadedTensor::F32(tensor)) => tensor,
            Some(LoadedTensor::Q8_0(weight)) => {
                check_f32_dequantization_size(name, weight.out_features(), weight.in_features())?;
                weight.dequantize_all()
            }
            Some(LoadedTensor::KQuant(weight)) => {
                check_f32_dequantization_size(name, weight.out_features(), weight.in_features())?;
                weight.dequantize_all()
            }
            None => return Ok(None),
        };
        Ok(Some(tensor))
    }

    /// Check the aggregate f32 storage that model builders may materialize
    /// from mapped compressed tensors after the loader's mmap allocation
    /// accounting has completed. Linear weights stay compressed; this covers
    /// norms, biases, RoPE factors, and GPT-2 embedding tables, all of which
    /// are passed through `take_f32` by the builders.
    /// Check a boundary (such as a vision/audio mmproj) whose builder
    /// materializes every compressed tensor through `take_f32`.
    pub(crate) fn check_all_f32_dequantization_budget(&self) -> Result<()> {
        let mut total = 0u64;
        for (name, tensor) in &self.tensors {
            let (out_features, in_features) = match tensor {
                LoadedTensor::F32(_) => continue,
                LoadedTensor::Q8_0(weight) => (weight.out_features(), weight.in_features()),
                LoadedTensor::KQuant(weight) => (weight.out_features(), weight.in_features()),
            };
            let bytes = f32_dequantization_bytes(name, out_features, in_features)?;
            total = total.checked_add(bytes).ok_or_else(|| {
                LoaderError::overflow("aggregate f32 dequantization size overflow")
            })?;
            if total > limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES {
                return Err(LoaderError::malformed(format!(
                    "compressed tensors require {total} bytes for f32 materialization, exceeding the {}-byte limit",
                    limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES
                )));
            }
        }
        Ok(())
    }

    /// Total encoded bytes of every Q8_0 tensor in the loader. Builders
    /// repack eligible Q8_0 projections into same-size VNNI decode buffers,
    /// so this is the anonymous-memory cost of automatic packed decode.
    pub(crate) fn q8_0_encoded_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for (name, tensor) in &self.tensors {
            if let LoadedTensor::Q8_0(weight) = tensor {
                let bytes = u64::try_from(weight.data().len()).map_err(|_| {
                    LoaderError::overflow(format!("tensor '{name}' byte size overflow"))
                })?;
                total = total.checked_add(bytes).ok_or_else(|| {
                    LoaderError::overflow("aggregate Q8_0 packed-decode size overflow")
                })?;
            }
        }
        Ok(total)
    }

    pub(crate) fn check_model_dequantization_budget(&self) -> Result<()> {
        let is_gpt2 = matches!(
            self.metadata.get("general.architecture"),
            Some(GgufValue::Str(architecture)) if architecture == "gpt2"
        );
        let mut total = 0u64;
        for (name, tensor) in &self.tensors {
            let may_materialize = name.ends_with(".bias")
                || name.contains("norm")
                || name.ends_with("layer_output_scale.weight")
                || name == "rope_freqs.weight"
                || (is_gpt2
                    && matches!(name.as_str(), "token_embd.weight" | "position_embd.weight"));
            if !may_materialize {
                continue;
            }
            let (out_features, in_features) = match tensor {
                LoadedTensor::F32(_) => continue,
                LoadedTensor::Q8_0(weight) => (weight.out_features(), weight.in_features()),
                LoadedTensor::KQuant(weight) => (weight.out_features(), weight.in_features()),
            };
            let bytes = f32_dequantization_bytes(name, out_features, in_features)?;
            total = total.checked_add(bytes).ok_or_else(|| {
                LoaderError::overflow("aggregate model dequantization size overflow")
            })?;
            if total > limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES {
                return Err(LoaderError::malformed(format!(
                    "model f32 dequantization requires {total} bytes, exceeding the {}-byte limit",
                    limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES
                )));
            }
        }
        Ok(())
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
    let f = File::open(&path).map_err(|error| {
        LoaderError::malformed(format!(
            "failed to open {}: {error}",
            path.as_ref().display()
        ))
    })?;
    // Check the logical length before creating a file mapping. Mmap::map would
    // otherwise reserve the entire length of a sparse hostile file even when
    // parsing would reject its contents later.
    let file_len = f.metadata()?.len();
    if file_len > crate::loader::limits::MAX_GGUF_FILE_BYTES {
        return Err(LoaderError::malformed(format!(
            "GGUF file length {file_len} exceeds the {}-byte limit",
            crate::loader::limits::MAX_GGUF_FILE_BYTES
        )));
    }
    let map_len = usize::try_from(file_len).map_err(|error| {
        LoaderError::overflow(format!("GGUF file length exceeds address space: {error}"))
    })?;
    // Safety: the read-only mapping remains alive through every QuantizedWeight
    // that references it. As with all file mappings, callers must not truncate
    // or concurrently mutate the GGUF while it is loaded.
    //
    // Pass the checked length explicitly rather than letting memmap2 infer it;
    // this keeps the parser's byte slice boundary tied to the preflight check.
    let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().len(map_len).map(&f)? });
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    load_gguf_from_reader_impl(
        &mut cursor,
        Some(Arc::clone(&mmap)),
        strategy,
        allow_fallback,
    )
}

/// Stream-hash a GGUF file into a 64-bit content identity for feature-cache
/// keys, without materializing the whole file (which may be up to
/// [`limits::MAX_GGUF_FILE_BYTES`]). Binds the path to one descriptor,
/// rejects symlinks and non-regular files, caps the streamed bytes at the
/// GGUF limit, and verifies the path still resolves to the same file after
/// reading (metadata identity on all platforms, device/inode on unix).
pub(crate) fn gguf_content_identity(path: &Path) -> anyhow::Result<u64> {
    use sha2::{Digest, Sha256};

    let path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat GGUF {:?}", path))?;
    anyhow::ensure!(
        path_metadata.file_type().is_file(),
        "GGUF {:?} is not a regular file",
        path
    );
    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open GGUF {:?}", path))?;
    let initial = file
        .metadata()
        .with_context(|| format!("failed to stat GGUF {:?}", path))?;
    anyhow::ensure!(
        initial.file_type().is_file()
            && initial.len() == path_metadata.len()
            && initial.modified().ok() == path_metadata.modified().ok(),
        "GGUF file changed while opening {:?}",
        path
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            initial.dev() == path_metadata.dev() && initial.ino() == path_metadata.ino(),
            "GGUF file changed while opening {:?}",
            path
        );
    }
    let mut hasher = Sha256::new();
    let mut chunk = [0u8; 64 * 1024];
    let mut remaining = limits::MAX_GGUF_FILE_BYTES;
    loop {
        let read = file
            .read(&mut chunk)
            .with_context(|| format!("failed to read GGUF {:?}", path))?;
        if read == 0 {
            break;
        }
        let read = read as u64;
        anyhow::ensure!(
            read <= remaining,
            "GGUF file {:?} exceeds the {}-byte limit",
            path,
            limits::MAX_GGUF_FILE_BYTES
        );
        hasher.update(&chunk[..read as usize]);
        remaining -= read;
    }
    let final_metadata = file
        .metadata()
        .with_context(|| format!("failed to stat GGUF {:?} after reading", path))?;
    let final_path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat GGUF {:?} after reading", path))?;
    anyhow::ensure!(
        final_metadata.len() == initial.len()
            && final_metadata.modified().ok() == initial.modified().ok()
            && final_path_metadata.file_type().is_file()
            && final_path_metadata.len() == initial.len()
            && final_path_metadata.modified().ok() == initial.modified().ok(),
        "GGUF file changed while reading {:?}",
        path
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            final_path_metadata.dev() == initial.dev()
                && final_path_metadata.ino() == initial.ino(),
            "GGUF file changed while reading {:?}",
            path
        );
    }
    let digest = hasher.finalize();
    Ok(u64::from_le_bytes(
        digest[..8].try_into().expect("sha256 >= 8 bytes"),
    ))
}

/// load a GGUF file from any readable + seekable source.
/// useful for testing with in-memory buffers (`std::io::Cursor<Vec<u8>>`).
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
    let remaining_file_len = file_len.checked_sub(initial_position).ok_or_else(|| {
        LoaderError::malformed("GGUF reader position is beyond the end of the file")
    })?;
    if remaining_file_len < 24 {
        return Err(LoaderError::malformed(
            "GGUF file is too short to contain a complete header".to_string(),
        ));
    }
    if remaining_file_len > limits::MAX_GGUF_FILE_BYTES {
        return Err(LoaderError::malformed(format!(
            "GGUF file length {remaining_file_len} exceeds the {}-byte limit",
            limits::MAX_GGUF_FILE_BYTES
        )));
    }

    let magic = read_u32(reader)?;
    if magic != GGUF_MAGIC {
        return Err(LoaderError::malformed(format!(
            "not a GGUF file (bad magic: {:#x})",
            magic
        )));
    }

    let version = read_u32(reader)?;
    if version != GGUF_VERSION {
        return Err(LoaderError::malformed(format!(
            "unsupported GGUF version: {}",
            version
        )));
    }

    let tensor_count_raw = read_u64(reader)?;
    let metadata_kv_count_raw = read_u64(reader)?;
    if tensor_count_raw > limits::MAX_TENSOR_COUNT as u64 {
        return Err(LoaderError::malformed(format!(
            "GGUF tensor count {tensor_count_raw} exceeds the {}-record limit",
            limits::MAX_TENSOR_COUNT
        )));
    }
    if metadata_kv_count_raw > limits::MAX_METADATA_KV_COUNT as u64 {
        return Err(LoaderError::malformed(format!(
            "GGUF metadata count {metadata_kv_count_raw} exceeds the {}-record limit",
            limits::MAX_METADATA_KV_COUNT
        )));
    }
    // Check counts against the bytes still available in the source as well as
    // the hard caps above. The old check used the absolute file length, which
    // was wrong for a reader positioned at a non-zero offset and was also too
    // permissive for sparse files.
    const MIN_TENSOR_RECORD_BYTES: u64 = 33; // non-empty name + one dimension
    const MIN_METADATA_RECORD_BYTES: u64 = 14; // non-empty key + one-byte value
    if tensor_count_raw > remaining_file_len / MIN_TENSOR_RECORD_BYTES {
        return Err(LoaderError::malformed(format!(
            "GGUF tensor count {tensor_count_raw} is impossible for the remaining {remaining_file_len}-byte file"
        )));
    }
    if metadata_kv_count_raw > remaining_file_len / MIN_METADATA_RECORD_BYTES {
        return Err(LoaderError::malformed(format!(
            "GGUF metadata count {metadata_kv_count_raw} is impossible for the remaining {remaining_file_len}-byte file"
        )));
    }
    let tensor_count = usize::try_from(tensor_count_raw).map_err(|error| {
        LoaderError::overflow(format!(
            "GGUF tensor count does not fit in memory address space: {error}"
        ))
    })?;
    let metadata_kv_count = usize::try_from(metadata_kv_count_raw).map_err(|error| {
        LoaderError::overflow(format!(
            "GGUF metadata count does not fit in memory address space: {error}"
        ))
    })?;

    let mut metadata = HashMap::new();
    metadata.try_reserve(metadata_kv_count).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF metadata table: {error}"))
    })?;
    let mut metadata_budget = MetadataBudget::new();
    for _ in 0..metadata_kv_count {
        let key =
            read_gguf_string_with_budget(reader, &mut metadata_budget.remaining_string_bytes)?;
        if key.is_empty() {
            return Err(LoaderError::malformed(
                "GGUF metadata keys must not be empty".to_string(),
            ));
        }
        let val_type = read_u32(reader)?;
        let value = read_gguf_value(reader, val_type, &mut metadata_budget)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(LoaderError::malformed(format!(
                "duplicate GGUF metadata key '{key}'"
            )));
        }
    }

    let mut tensor_info = read_tensor_info(reader, tensor_count)?;
    let mut tensor_meta = HashMap::new();
    tensor_meta
        .try_reserve(tensor_info.len())
        .map_err(|error| {
            LoaderError::reservation(format!(
                "failed to reserve GGUF tensor-metadata table: {error}"
            ))
        })?;
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
        .map_err(|error| {
            LoaderError::reservation(format!("failed to reserve GGUF K-decision table: {error}"))
        })?;

    let current_pos = reader.stream_position()?;
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(a)) => *a as u64,
        Some(GgufValue::U64(a)) => *a,
        _ => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(LoaderError::malformed(format!(
            "invalid GGUF alignment {alignment}: expected a power of two"
        )));
    }
    let data_start = current_pos
        .checked_add(alignment - 1)
        .ok_or_else(|| LoaderError::overflow("GGUF aligned data offset overflow"))?
        & !(alignment - 1);

    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(tensor_info.len())
        .map_err(|error| {
            LoaderError::reservation(format!(
                "failed to reserve GGUF tensor range table: {error}"
            ))
        })?;
    // GPT-2's embedding builder dequantizes its Q8_0 embedding tensors into
    // owned f32 tables. Account for those downstream allocations here even
    // though ordinary mmap-backed Q8 linear weights remain zero-copy.
    let gpt2_materializes_q8_embedding = matches!(
        metadata.get("general.architecture"),
        Some(GgufValue::Str(architecture)) if architecture == "gpt2"
    );
    let mut total_tensor_allocation_bytes = 0u64;
    for info in &tensor_info {
        let element_count = tensor_element_count(info)?;
        let byte_len = tensor_byte_len(info)?;
        let encoded_bytes = u64::try_from(byte_len).map_err(|error| {
            LoaderError::overflow(format!("tensor byte length exceeds u64: {error}"))
        })?;
        if encoded_bytes > limits::MAX_TENSOR_BYTES {
            return Err(LoaderError::malformed(format!(
                "tensor '{}' declares {} encoded bytes, exceeding the {}-byte limit",
                info.name,
                encoded_bytes,
                limits::MAX_TENSOR_BYTES
            )));
        }
        let allocation_bytes = estimated_tensor_allocation_bytes(
            info,
            element_count,
            encoded_bytes,
            k_strategy,
            allow_fallback,
            mmap.is_some(),
            gpt2_materializes_q8_embedding
                && matches!(
                    info.name.as_str(),
                    "token_embd.weight" | "position_embd.weight"
                ),
        )?;
        if allocation_bytes > limits::MAX_TENSOR_ALLOCATION_BYTES {
            return Err(LoaderError::malformed(format!(
                "tensor '{}' requires an estimated {} bytes while loading, exceeding the {}-byte limit",
                info.name,
                allocation_bytes,
                limits::MAX_TENSOR_ALLOCATION_BYTES
            )));
        }
        total_tensor_allocation_bytes = total_tensor_allocation_bytes
            .checked_add(allocation_bytes)
            .ok_or_else(|| LoaderError::overflow("aggregate tensor allocation size overflow"))?;
        if total_tensor_allocation_bytes > limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES {
            return Err(LoaderError::malformed(format!(
                "GGUF tensors require an estimated {} bytes while loading, exceeding the {}-byte aggregate limit",
                total_tensor_allocation_bytes,
                limits::MAX_TOTAL_TENSOR_ALLOCATION_BYTES
            )));
        }
        let start = data_start.checked_add(info.offset).ok_or_else(|| {
            LoaderError::overflow(format!("tensor '{}' file offset overflow", info.name))
        })?;
        let end = start.checked_add(encoded_bytes).ok_or_else(|| {
            LoaderError::overflow(format!("tensor '{}' file range overflow", info.name))
        })?;
        if end > file_len {
            return Err(LoaderError::malformed(format!(
                "tensor '{}' data range {start}..{end} exceeds file length {file_len}",
                info.name
            )));
        }
        ranges.push((start, end, info.name.as_str()));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        let (_, previous_end, previous_name) = pair[0];
        let (next_start, _, next_name) = pair[1];
        if next_start < previous_end {
            return Err(LoaderError::malformed(format!(
                "GGUF tensor ranges overlap: '{previous_name}' ends at {previous_end}, \
                 '{next_name}' starts at {next_start}"
            )));
        }
    }

    let mut tensors = HashMap::new();
    tensors.try_reserve(tensor_info.len()).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF tensor table: {error}"))
    })?;
    for info in tensor_info.drain(..) {
        let tensor_offset = data_start.checked_add(info.offset).ok_or_else(|| {
            LoaderError::overflow(format!("tensor '{}' file offset overflow", info.name))
        })?;
        reader.seek(SeekFrom::Start(tensor_offset))?;
        let element_count = info.dims.iter().try_fold(1usize, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| {
                LoaderError::overflow(format!(
                    "tensor '{}' shape product overflow for dimensions {:?}",
                    info.name, info.dims
                ))
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
                let byte_len = element_count.checked_mul(4).ok_or_else(|| {
                    LoaderError::overflow(format!("tensor '{}' f32 byte size overflow", info.name))
                })?;
                let mut buf = vec![0u8; byte_len];
                reader.read_exact(&mut buf)?;
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 4;
                    let bytes: [u8; 4] = buf[start..start + 4].try_into().map_err(|_| {
                        LoaderError::malformed(format!("failed to read f32 at index {i}"))
                    })?;
                    *dst = f32::from_le_bytes(bytes);
                }
                LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
            }
            1 => {
                // f16: read and convert to f32. Keep the logical GGUF shape
                // unchanged; model builders handle any linear-weight transpose
                // the same way they do for native f32 tensors.
                use half::f16;
                let byte_len = element_count.checked_mul(2).ok_or_else(|| {
                    LoaderError::overflow(format!("tensor '{}' f16 byte size overflow", info.name))
                })?;
                let mut buf = vec![0u8; byte_len];
                reader.read_exact(&mut buf)?;
                let mut data = vec![0.0f32; element_count];
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 2;
                    let bits =
                        u16::from_le_bytes(buf[start..start + 2].try_into().map_err(|_| {
                            LoaderError::malformed(format!("failed to read f16 at index {i}"))
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
                    return Err(LoaderError::malformed(format!(
                        "tensor '{}' Q8_0 element count {} is not block-aligned",
                        info.name, element_count
                    )));
                }
                let n_blocks = element_count / 32;
                let byte_len = n_blocks.checked_mul(Q8_0_TYPE_SIZE).ok_or_else(|| {
                    LoaderError::overflow(format!("tensor '{}' Q8_0 byte size overflow", info.name))
                })?;
                let mut dims = info.dims;
                dims.reverse();
                let weight = if let Some(mmap) = mmap.as_ref() {
                    let start = usize::try_from(tensor_offset).map_err(|error| {
                        LoaderError::overflow(format!(
                            "tensor '{}' offset exceeds address space: {error}",
                            info.name
                        ))
                    })?;
                    let end = start.checked_add(byte_len).ok_or_else(|| {
                        LoaderError::overflow(format!(
                            "tensor '{}' mapped range overflow",
                            info.name
                        ))
                    })?;
                    if end > mmap.len() {
                        return Err(LoaderError::malformed(format!(
                            "tensor '{}' data range {}..{} exceeds file length {}",
                            info.name,
                            start,
                            end,
                            mmap.len()
                        )));
                    }
                    QuantizedWeight::try_from_mmap(Arc::clone(mmap), start..end, dims)
                        .map_err(|error| LoaderError::malformed(format!("{error:#}")))?
                } else {
                    let mut raw = vec![0u8; byte_len];
                    reader.read_exact(&mut raw)?;
                    QuantizedWeight::try_new(raw, dims)
                        .map_err(|error| LoaderError::malformed(format!("{error:#}")))?
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
                    return Err(LoaderError::malformed(format!(
                        "tensor '{}' dtype {} element count {} is not 256-block-aligned",
                        info.name, info.dtype, element_count
                    )));
                }
                let n_blocks = element_count / crate::quant_k::QK_K;
                let block_bytes = crate::quant_k::k_block_bytes(info.dtype)
                    .ok_or_else(|| LoaderError::malformed(format!("tensor '{}'", info.name)))?;
                let byte_len = n_blocks.checked_mul(block_bytes).ok_or_else(|| {
                    LoaderError::overflow(format!(
                        "tensor '{}' K-quant byte size overflow",
                        info.name
                    ))
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
                        crate::quant_k::dequant_tensor(info.dtype, &raw, &mut data).map_err(
                            |e| LoaderError::malformed(format!("tensor '{}': {e}", info.name)),
                        )?;
                        LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
                    }
                    crate::quant_k::KExecution::CompressedScalar
                    | crate::quant_k::KExecution::CompressedX86 => {
                        let native = native.ok_or_else(|| {
                            LoaderError::malformed(format!(
                                "tensor '{}' selected compressed execution without a native dtype",
                                info.name
                            ))
                        })?;
                        let mut dims = info.dims.clone();
                        dims.reverse();
                        if dims.len() != 2 {
                            return Err(LoaderError::malformed(format!(
                                "tensor '{}' K-quant must be 2D for compressed residency, got {:?}",
                                info.name, info.dims
                            )));
                        }
                        let shape = [dims[0], dims[1]];
                        let weight = if let Some(mmap) = mmap.as_ref() {
                            let start = usize::try_from(tensor_offset).map_err(|error| {
                                LoaderError::overflow(format!(
                                    "tensor '{}' offset exceeds address space: {error}",
                                    info.name
                                ))
                            })?;
                            let end = start.checked_add(byte_len).ok_or_else(|| {
                                LoaderError::overflow(format!(
                                    "tensor '{}' mapped range overflow",
                                    info.name
                                ))
                            })?;
                            if end > mmap.len() {
                                return Err(LoaderError::malformed(format!(
                                    "tensor '{}' data range {}..{} exceeds file length {}",
                                    info.name,
                                    start,
                                    end,
                                    mmap.len()
                                )));
                            }
                            crate::quant_k::KQuantWeight::try_from_mmap(
                                Arc::clone(mmap),
                                start..end,
                                shape,
                                native,
                                execution,
                            )
                            .map_err(|error| LoaderError::malformed(format!("{error:#}")))?
                        } else {
                            let mut raw = vec![0u8; byte_len];
                            reader.read_exact(&mut raw)?;
                            crate::quant_k::KQuantWeight::try_new_with_execution(
                                raw, shape, native, execution,
                            )
                            .map_err(|error| LoaderError::malformed(format!("{error:#}")))?
                        };
                        LoadedTensor::KQuant(weight)
                    }
                }
            }
            30 => {
                // bf16: brain floating point — upper 16 bits of f32.
                let byte_len = element_count.checked_mul(2).ok_or_else(|| {
                    LoaderError::overflow(format!("tensor '{}' bf16 byte size overflow", info.name))
                })?;
                let mut buf = vec![0u8; byte_len];
                reader.read_exact(&mut buf)?;
                let mut data = vec![0.0f32; element_count];
                for (i, dst) in data.iter_mut().enumerate().take(element_count) {
                    let start = i * 2;
                    let bits =
                        u16::from_le_bytes(buf[start..start + 2].try_into().map_err(|_| {
                            LoaderError::malformed(format!("failed to read bf16 at index {i}"))
                        })?);
                    *dst = f32::from_bits((bits as u32) << 16);
                }
                LoadedTensor::F32(CpuTensor::from_data(info.dims, data))
            }
            _ => {
                return Err(LoaderError::malformed(format!(
                    "tensor '{}' uses unsupported GGML dtype {}",
                    info.name, info.dtype
                )));
            }
        };
        if tensors.insert(info.name.clone(), loaded).is_some() {
            return Err(LoaderError::malformed(format!(
                "duplicate GGUF tensor name '{}'",
                info.name
            )));
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
    let x86_available = crate::k_quant_matmul::x86_k_supported();
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
            None => Err(LoaderError::malformed(format!(
                "tensor '{}' uses GGUF dtype {} which has no native kernel in v0.3; \
                 pass --k-allow-fallback to run it through the eager-f32 path",
                info.name,
                ggml_dtype_name(info.dtype).unwrap_or("unknown")
            ))),
        },
        KStrategy::X86 => match (native, x86_available) {
            (Some(_), true) => Ok((KExecution::CompressedX86, None)),
            (Some(_), false) if allow_fallback => Ok((
                KExecution::CompressedScalar,
                Some("x86 feature set unavailable (avx2+fma+f16c+ssse3)".to_string()),
            )),
            (Some(_), false) => Err(LoaderError::malformed("--k-strategy x86 requires the AVX2+FMA+F16C+SSSE3 feature set (avx2, fma, f16c, ssse3); \
                 pass --k-allow-fallback to run the scalar path".to_string())),
            (None, _) if allow_fallback => Ok((KExecution::EagerF32, Some(no_native_kernel()))),
            (None, _) => Err(LoaderError::malformed(format!(
                "tensor '{}' uses GGUF dtype {} which has no native kernel in v0.3; \
                 pass --k-allow-fallback to run it through the eager-f32 path",
                info.name,
                ggml_dtype_name(info.dtype).unwrap_or("unknown")
            ))),
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
    info.try_reserve_exact(count).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF tensor-info table: {error}"))
    })?;
    let mut names = HashSet::new();
    names.try_reserve(count).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF tensor-name table: {error}"))
    })?;
    let mut name_bytes_remaining = limits::MAX_TENSOR_NAME_BYTES;
    for _ in 0..count {
        let name = read_gguf_string_with_budget(reader, &mut name_bytes_remaining)?;
        if name.is_empty() {
            return Err(LoaderError::malformed(
                "GGUF tensor names must not be empty".to_string(),
            ));
        }
        if !names.insert(name.clone()) {
            return Err(LoaderError::malformed(format!(
                "duplicate GGUF tensor name '{name}'"
            )));
        }
        let n_dims = read_u32(reader)?;
        if !(1..=4).contains(&n_dims) {
            return Err(LoaderError::malformed(format!(
                "tensor '{name}' has invalid dimension count {n_dims}; expected 1..=4"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = usize::try_from(read_u64(reader)?).map_err(|error| {
                LoaderError::overflow(format!(
                    "tensor '{name}' dimension exceeds address space: {error}"
                ))
            })?;
            if dim == 0 {
                return Err(LoaderError::malformed(format!(
                    "tensor '{name}' has a zero-sized dimension"
                )));
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

fn tensor_element_count(info: &TensorInfo) -> Result<usize> {
    info.dims.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            LoaderError::overflow(format!(
                "tensor '{}' shape product overflow for dimensions {:?}",
                info.name, info.dims
            ))
        })
    })
}

fn tensor_byte_len(info: &TensorInfo) -> Result<usize> {
    let element_count = tensor_element_count(info)?;
    gguf_dtype_byte_len(info.dtype, element_count)
        .map_err(|error| LoaderError::malformed(format!("tensor '{}': {error}", info.name)))
}

/// Estimate anonymous storage needed while materializing one tensor.
///
/// Range validation runs this before the loader reads any payload. In
/// particular, a sparse file can advertise a huge logical range while having
/// almost no blocks on disk; checking only `file_len` therefore is not enough.
/// Mmap-backed packed tensors do not allocate their payload at load time, but
/// f32 conversion buffers and optional K-quant pre-split lanes still count.
fn estimated_tensor_allocation_bytes(
    info: &TensorInfo,
    element_count: usize,
    encoded_bytes: u64,
    strategy: crate::quant_k::KStrategy,
    allow_fallback: bool,
    mmap_present: bool,
    q8_dequantized_downstream: bool,
) -> Result<u64> {
    let elements = u64::try_from(element_count).map_err(|error| {
        LoaderError::overflow(format!(
            "tensor '{}' element count exceeds u64: {error}",
            info.name
        ))
    })?;
    let f32_bytes = elements.checked_mul(4).ok_or_else(|| {
        LoaderError::overflow(format!(
            "tensor '{}' f32 allocation size overflow",
            info.name
        ))
    })?;
    let add = |left: u64, right: u64| {
        left.checked_add(right).ok_or_else(|| {
            LoaderError::overflow(format!("tensor '{}' allocation size overflow", info.name))
        })
    };

    match info.dtype {
        // Conversion reads an encoded buffer and keeps a f32 destination until
        // CpuTensor takes ownership of it.
        0 | 1 | 30 => add(encoded_bytes, f32_bytes),
        // Q8_0 is mapped directly for file loads. A small number of model
        // builders (currently GPT-2 embeddings) explicitly dequantize a Q8
        // tensor into an owned f32 table, so include that downstream storage.
        8 => {
            if q8_dequantized_downstream {
                if mmap_present {
                    Ok(f32_bytes)
                } else {
                    add(encoded_bytes, f32_bytes)
                }
            } else if mmap_present {
                Ok(0)
            } else {
                Ok(encoded_bytes)
            }
        }
        10..=14 => {
            let (execution, _) = resolve_k_execution(
                info,
                strategy,
                crate::quant_k::KQuantDtype::from_gguf(info.dtype),
                allow_fallback,
            )?;
            match execution {
                crate::quant_k::KExecution::EagerF32 => add(encoded_bytes, f32_bytes),
                crate::quant_k::KExecution::CompressedScalar
                | crate::quant_k::KExecution::CompressedX86 => {
                    let mut owned = if mmap_present { 0 } else { encoded_bytes };
                    if matches!(execution, crate::quant_k::KExecution::CompressedX86)
                        && presplit_requested()
                    {
                        let blocks = u64::try_from(element_count / crate::quant_k::QK_K).map_err(
                            |error| {
                                LoaderError::overflow(format!(
                                    "tensor '{}' K-block count exceeds u64: {error}",
                                    info.name
                                ))
                            },
                        )?;
                        owned = add(
                            owned,
                            blocks.checked_mul(256).ok_or_else(|| {
                                LoaderError::overflow(format!(
                                    "tensor '{}' pre-split allocation size overflow",
                                    info.name
                                ))
                            })?,
                        )?;
                    }
                    Ok(owned)
                }
            }
        }
        dtype => Err(LoaderError::malformed(format!(
            "tensor '{}' uses unsupported GGML dtype {}",
            info.name, dtype
        ))),
    }
}

fn presplit_requested() -> bool {
    matches!(
        std::env::var("EMBER_PRESPLIT").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Encoded byte length of `element_count` values of a GGUF dtype.
///
/// Shared by the loader's range validation and the tensor-inventory dump.
pub fn gguf_dtype_byte_len(dtype: u32, element_count: usize) -> Result<usize> {
    match dtype {
        0 => element_count
            .checked_mul(4)
            .ok_or_else(|| LoaderError::overflow("f32 byte size overflow".to_string())),
        1 | 30 => element_count
            .checked_mul(2)
            .ok_or_else(|| LoaderError::overflow("16-bit byte size overflow".to_string())),
        8 => {
            if !element_count.is_multiple_of(crate::quant::Q8_0_BLOCK_SIZE) {
                return Err(LoaderError::malformed(
                    "Q8_0 element count is not block-aligned".to_string(),
                ));
            }
            (element_count / crate::quant::Q8_0_BLOCK_SIZE)
                .checked_mul(Q8_0_TYPE_SIZE)
                .ok_or_else(|| LoaderError::overflow("Q8_0 byte size overflow".to_string()))
        }
        10..=14 => {
            if !element_count.is_multiple_of(crate::quant_k::QK_K) {
                return Err(LoaderError::malformed(format!(
                    "dtype {dtype} element count is not K-block-aligned"
                )));
            }
            let block_bytes = crate::quant_k::k_block_bytes(dtype)
                .ok_or_else(|| LoaderError::malformed(format!("dtype {dtype}")))?;
            (element_count / crate::quant_k::QK_K)
                .checked_mul(block_bytes)
                .ok_or_else(|| LoaderError::overflow("K-quant byte size overflow".to_string()))
        }
        dtype => Err(LoaderError::malformed(format!(
            "unsupported GGML dtype {dtype}"
        ))),
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
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(f: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
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
        .ok_or_else(|| LoaderError::malformed("reader position exceeds its end"))
}

struct MetadataBudget {
    remaining_values: usize,
    remaining_string_bytes: usize,
}

impl MetadataBudget {
    fn new() -> Self {
        Self {
            remaining_values: limits::MAX_METADATA_VALUES,
            remaining_string_bytes: limits::MAX_METADATA_STRING_BYTES,
        }
    }

    fn consume_value(&mut self) -> Result<()> {
        if self.remaining_values == 0 {
            return Err(LoaderError::malformed(format!(
                "GGUF metadata contains more than the {}-value limit",
                limits::MAX_METADATA_VALUES
            )));
        }
        self.remaining_values -= 1;
        Ok(())
    }
}

fn read_gguf_string_with_budget<R: Read + Seek>(
    f: &mut R,
    remaining_string_bytes: &mut usize,
) -> Result<String> {
    let declared_len = read_u64(f)?;
    if declared_len > limits::MAX_STRING_BYTES as u64 {
        return Err(LoaderError::malformed(format!(
            "GGUF string length {declared_len} exceeds the {}-byte limit",
            limits::MAX_STRING_BYTES
        )));
    }
    let len = usize::try_from(declared_len).map_err(|error| {
        LoaderError::overflow(format!("GGUF string length exceeds address space: {error}"))
    })?;
    if len > *remaining_string_bytes {
        return Err(LoaderError::malformed(format!(
            "GGUF strings exceed the aggregate byte limit ({} bytes)",
            limits::MAX_METADATA_STRING_BYTES
        )));
    }
    let remaining = remaining_bytes(f)?;
    if declared_len > remaining {
        return Err(LoaderError::malformed(format!(
            "GGUF string length {len} exceeds the {remaining} bytes remaining in the file"
        )));
    }
    *remaining_string_bytes -= len;
    let mut buf = Vec::new();
    buf.try_reserve_exact(len).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF string buffer: {error}"))
    })?;
    buf.resize(len, 0);
    f.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|error| LoaderError::malformed(format!("invalid utf8 in string: {error}")))
}

fn minimum_value_size(val_type: u32) -> Result<u64> {
    match val_type {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        8 => Ok(8),
        9 => Ok(12),
        10..=12 => Ok(8),
        _ => Err(LoaderError::malformed(format!(
            "unsupported GGUF value type: {val_type}"
        ))),
    }
}

fn read_gguf_value<R: Read + Seek>(
    f: &mut R,
    val_type: u32,
    budget: &mut MetadataBudget,
) -> Result<GgufValue> {
    read_gguf_value_inner(f, val_type, 0, budget)
}

fn read_gguf_value_inner<R: Read + Seek>(
    f: &mut R,
    val_type: u32,
    depth: usize,
    budget: &mut MetadataBudget,
) -> Result<GgufValue> {
    if depth > 16 {
        return Err(LoaderError::malformed(
            "GGUF metadata arrays are nested more than 16 levels deep".to_string(),
        ));
    }
    budget.consume_value()?;
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
                return Err(LoaderError::malformed(format!(
                    "invalid GGUF boolean value {value}; expected 0 or 1"
                )));
            }
            Ok(GgufValue::Bool(value == 1))
        }
        8 => Ok(GgufValue::Str(read_gguf_string_with_budget(
            f,
            &mut budget.remaining_string_bytes,
        )?)),
        10 => Ok(GgufValue::U64(read_u64(f)?)),
        11 => Ok(GgufValue::I64(read_i64(f)?)),
        12 => Ok(GgufValue::F64(read_f64(f)?)),
        9 => {
            let element_type = read_u32(f)?;
            let count = usize::try_from(read_u64(f)?).map_err(|error| {
                LoaderError::overflow(format!("GGUF array length exceeds address space: {error}"))
            })?;
            if count > limits::MAX_METADATA_ARRAY_ELEMENTS {
                return Err(LoaderError::malformed(format!(
                    "GGUF metadata array has {count} elements, exceeding the {}-element limit",
                    limits::MAX_METADATA_ARRAY_ELEMENTS
                )));
            }
            if count > budget.remaining_values {
                return Err(LoaderError::malformed(format!(
                    "GGUF metadata arrays exceed the {}-value limit",
                    limits::MAX_METADATA_VALUES
                )));
            }
            let minimum_bytes = minimum_value_size(element_type)?
                .checked_mul(u64::try_from(count).map_err(|error| {
                    LoaderError::overflow(format!("GGUF array length exceeds u64: {error}"))
                })?)
                .ok_or_else(|| LoaderError::overflow("GGUF array minimum byte size overflow"))?;
            let remaining = remaining_bytes(f)?;
            if minimum_bytes > remaining {
                return Err(LoaderError::malformed(format!(
                    "GGUF array of {count} type-{element_type} values requires at least {minimum_bytes} bytes but only {remaining} remain"
                )));
            }
            let mut elements = Vec::new();
            elements.try_reserve_exact(count).map_err(|error| {
                LoaderError::reservation(format!("failed to reserve GGUF metadata array: {error}"))
            })?;
            for _ in 0..count {
                elements.push(read_gguf_value_inner(f, element_type, depth + 1, budget)?);
            }
            Ok(GgufValue::Array(elements))
        }
        _ => Err(LoaderError::malformed(format!(
            "unsupported GGUF value type: {}",
            val_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_row_major_conversion_rejects_non_2d_tensors() {
        let tensor = CpuTensor::from_data(vec![2, 3, 4], vec![0.0; 24]);
        let error = try_gguf_to_row_major_f32(tensor)
            .expect_err("rank-3 GGUF tensors must be rejected by the checked converter");
        assert!(
            matches!(error, LoaderError::Malformed(ref message) if message.contains("requires a 2D") && message.contains("[2, 3, 4]")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn checked_row_major_conversion_preserves_valid_layout() {
        let tensor = CpuTensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let converted = try_gguf_to_row_major_f32(tensor).expect("rank-2 conversion");
        assert_eq!(converted.shape(), &[2, 3]);
        assert_eq!(converted.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

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

    fn gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&tensor_count.to_le_bytes());
        out.extend_from_slice(&metadata_count.to_le_bytes());
        out
    }

    fn push_gguf_string(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }

    fn gguf_one_tensor(dims: &[u64], dtype: u32) -> Vec<u8> {
        let mut out = gguf_header(1, 0);
        push_gguf_string(&mut out, b"hostile");
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &dim in dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&dtype.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out
    }

    #[test]
    fn dequantization_size_guard_rejects_hostile_mapped_weights() {
        let error = check_f32_dequantization_size("hostile", 1 << 29, 2)
            .expect_err("mapped packed weights must not dequantize beyond the cap");
        assert!(error.to_string().contains("dequantization"), "{error}");
        assert!(error.to_string().contains("byte limit"), "{error}");

        check_f32_dequantization_size("small", 1024, 1024)
            .expect("ordinary norm/embedding-sized dequantization should pass");
    }

    /// A reader wrapper that reports an oversized logical length without
    /// materializing it, mirroring a sparse hostile file behind a reader.
    struct OversizedLen<R>(R);

    impl<R: std::io::Read + std::io::Seek> std::io::Read for OversizedLen<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl<R: std::io::Read + std::io::Seek> std::io::Seek for OversizedLen<R> {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            match pos {
                std::io::SeekFrom::End(_) => Ok(limits::MAX_GGUF_FILE_BYTES + 1),
                other => self.0.seek(other),
            }
        }
    }

    #[test]
    fn reader_loader_rejects_oversized_claimed_length() {
        let mut reader = OversizedLen(std::io::Cursor::new(gguf_header(0, 0)));
        let error = load_gguf_from_reader(&mut reader)
            .err()
            .expect("a reader claiming more than the GGUF cap must be rejected");
        assert!(error.to_string().contains("GGUF file length"), "{error}");
    }

    #[test]
    fn gguf_content_identity_hashes_regular_files_and_rejects_symlinks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ember-gguf-identity-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).expect("create identity fixture dir");
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"GGUF-identity-fixture").expect("write identity fixture");
        let first = gguf_content_identity(&path).expect("hash regular file");
        let second = gguf_content_identity(&path).expect("hash regular file again");
        assert_eq!(first, second);
        std::fs::write(&path, b"GGUF-identity-fixture-changed").expect("rewrite identity fixture");
        assert_ne!(
            first,
            gguf_content_identity(&path).expect("hash changed file")
        );
        #[cfg(unix)]
        {
            let link = dir.join("link.gguf");
            std::os::unix::fs::symlink(&path, &link).expect("create symlink fixture");
            let error = gguf_content_identity(&link).expect_err("symlinked GGUF must be rejected");
            assert!(error.to_string().contains("regular file"), "{error}");
        }
        std::fs::remove_dir_all(&dir).expect("remove identity fixture dir");
    }

    #[test]
    fn mmap_loader_rejects_oversized_sparse_files_before_mapping() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ember-mmap-oversized-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let file = File::create(&path).expect("create sparse GGUF fixture");
        file.set_len(limits::MAX_GGUF_FILE_BYTES + 1)
            .expect("extend sparse GGUF fixture");
        drop(file);

        let result = load_gguf(&path);
        std::fs::remove_file(&path).expect("remove sparse GGUF fixture");
        let error = result
            .err()
            .expect("an oversized sparse GGUF must be rejected before mmap");
        assert!(error.to_string().contains("GGUF file length"), "{error}");
        assert!(
            error
                .to_string()
                .contains(&limits::MAX_GGUF_FILE_BYTES.to_string()),
            "{error}"
        );
    }

    #[test]
    fn hostile_sparse_gguf_declarations_are_rejected_before_allocation() {
        let mut cursor =
            std::io::Cursor::new(gguf_header((limits::MAX_TENSOR_COUNT as u64) + 1, 0));
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("a sparse tensor count must not reserve an unbounded table");
        assert!(error.to_string().contains("tensor count"), "{error}");
        assert!(error.to_string().contains("record limit"), "{error}");

        let mut cursor = std::io::Cursor::new(gguf_one_tensor(&[1u64 << 30], 0));
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("a sparse f32 tensor declaration must not allocate");
        assert!(error.to_string().contains("encoded bytes"), "{error}");

        // This declaration is exactly at the encoded-byte cap but still needs
        // two f32-sized buffers during conversion, so the transient cap must
        // reject it before reading the payload.
        let mut cursor = std::io::Cursor::new(gguf_one_tensor(&[1u64 << 28], 0));
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("transient tensor allocation must be bounded");
        assert!(error.to_string().contains("while loading"), "{error}");
    }

    #[test]
    fn hostile_sparse_metadata_is_bounded() {
        let mut cursor =
            std::io::Cursor::new(gguf_header(0, (limits::MAX_METADATA_KV_COUNT as u64) + 1));
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("a sparse metadata count must not reserve an unbounded table");
        assert!(error.to_string().contains("metadata count"), "{error}");
        assert!(error.to_string().contains("record limit"), "{error}");

        let mut bytes = gguf_header(0, 1);
        bytes.extend_from_slice(&((limits::MAX_STRING_BYTES as u64) + 1).to_le_bytes());
        let mut cursor = std::io::Cursor::new(bytes);
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("a sparse metadata string must be bounded before reserve");
        assert!(error.to_string().contains("string length"), "{error}");
        assert!(error.to_string().contains("byte limit"), "{error}");

        let mut bytes = gguf_header(0, 1);
        push_gguf_string(&mut bytes, b"array");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array value
        bytes.extend_from_slice(&4u32.to_le_bytes()); // u32 elements
        bytes.extend_from_slice(&((limits::MAX_METADATA_ARRAY_ELEMENTS as u64) + 1).to_le_bytes());
        let mut cursor = std::io::Cursor::new(bytes);
        let error = load_gguf_from_reader(&mut cursor)
            .err()
            .expect("a sparse metadata array must be bounded before reserve");
        assert!(error.to_string().contains("metadata array"), "{error}");
        assert!(error.to_string().contains("element limit"), "{error}");
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
        assert_eq!(tensor.kernel, "q4-k-q8-k-scalar");
        assert_eq!(tensor.kernel_revision, crate::plan::PLAN_KERNEL_REVISION);
        assert_eq!(tensor.cpu_features, "none");
        assert_eq!(tensor.operations.len(), 1);
        assert_eq!(tensor.operations[0].operation, "linear-matmul");
        assert_eq!(tensor.operations[0].kernel, tensor.kernel);
        assert!(tensor.fallback_reason.is_none());
        assert_eq!(
            tensor.workspace_bytes,
            crate::k_quant_matmul::Q8_K_BLOCK_BYTES
        );

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

    // x86-only: asserts the AVX2+FMA+F16C+SSSE3 feature set.
    #[cfg(target_arch = "x86_64")]
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
        if crate::k_quant_matmul::x86_k_supported() {
            assert_eq!(
                decision.execution,
                crate::quant_k::KExecution::CompressedX86
            );
            assert!(decision.fallback_reason.is_none());
            let inventory = crate::artifact::ExecutionInventory::from_loader(&loader);
            assert_eq!(inventory.tensors[0].kernel, "q6-k-q8-k-avx2");
            assert_eq!(
                inventory.tensors[0].kernel_revision,
                crate::plan::PLAN_KERNEL_REVISION
            );
            assert_eq!(inventory.tensors[0].strategy, "compressed-x86");
            assert_eq!(inventory.tensors[0].cpu_features, "avx2+fma+f16c+ssse3");
        } else {
            // without the feature set the request hard-fails (no
            // allow-fallback); this branch only runs on unsupported x86 hosts
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
        let expected = if crate::k_quant_matmul::x86_k_supported() {
            crate::quant_k::KExecution::CompressedX86
        } else {
            crate::quant_k::KExecution::CompressedScalar
        };
        assert_eq!(decision.execution, expected);
        assert!(decision.fallback_reason.is_none());
        let LoadedTensor::KQuant(weight) = &loader.tensors["blk.0.attn_q.weight"] else {
            panic!("auto strategy must keep Q4_K compressed");
        };
        assert_eq!(
            weight.execution(),
            expected,
            "reader load dropped execution policy"
        );
    }

    #[test]
    fn x86_strategy_falls_back_to_scalar_only_with_allow_fallback() {
        // on hosts without the complete x86 feature set, the request must hard-fail unless the
        // user explicitly allows the downgrade; on supported x86 hosts both paths
        // still validate the fallback recording machinery
        let bytes = write_minimal_gguf_with_k_tensor(14, 4);
        let mut cursor = std::io::Cursor::new(bytes);
        let result = load_gguf_from_reader_with_k_strategy(
            &mut cursor,
            crate::quant_k::KStrategy::X86,
            false,
        );
        if crate::k_quant_matmul::x86_k_supported() {
            assert!(result.is_ok(), "supported x86 hosts accept the x86 request");
        } else {
            let message = result
                .err()
                .expect("hard-fail without the x86 feature set")
                .to_string();
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
        if crate::k_quant_matmul::x86_k_supported() {
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

use super::*;

#[test]
fn floating_tensor_reads_are_bounded_and_preserve_bits() {
    struct BoundedReader<'a> {
        cursor: std::io::Cursor<&'a [u8]>,
        largest_read: usize,
    }
    impl Read for BoundedReader<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(buffer.len() <= FLOAT_READ_BUFFER_BYTES);
            self.largest_read = self.largest_read.max(buffer.len());
            self.cursor.read(buffer)
        }
    }
    let expected: Vec<u32> = [
        0,
        0x8000_0000,
        1,
        0x3f80_0001,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
    ]
    .into_iter()
    .cycle()
    .take(5_003)
    .collect();
    let raw: Vec<_> = expected
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let mut reader = BoundedReader {
        cursor: std::io::Cursor::new(&raw),
        largest_read: 0,
    };
    let values = read_float_values(&mut reader, expected.len(), f32::from_le_bytes).unwrap();
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(reader.largest_read, FLOAT_READ_BUFFER_BYTES);
    assert_eq!(reader.cursor.position(), raw.len() as u64);
    for truncated in [raw.len() - 1, FLOAT_READ_BUFFER_BYTES - 1] {
        let error = read_float_values(
            &mut std::io::Cursor::new(&raw[..truncated]),
            expected.len(),
            f32::from_le_bytes,
        )
        .unwrap_err();
        assert!(
            matches!(error, LoaderError::Io(ref error) if error.kind() == std::io::ErrorKind::UnexpectedEof)
        );
    }
}

#[test]
fn floating_tensor_reads_preserve_all_half_encodings() {
    let raw: Vec<_> = (u16::MIN..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let half_values = read_float_values(&mut std::io::Cursor::new(&raw), 65_536, |bytes| {
        half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
    })
    .unwrap();
    let brain_values = read_float_values(&mut std::io::Cursor::new(&raw), 65_536, |bytes| {
        f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16)
    })
    .unwrap();
    for bits in u16::MIN..=u16::MAX {
        assert_eq!(
            half_values[usize::from(bits)].to_bits(),
            half::f16::from_bits(bits).to_f32().to_bits()
        );
        assert_eq!(
            brain_values[usize::from(bits)].to_bits(),
            u32::from(bits) << 16
        );
    }
}

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
    let mut loader = GgufLoader::for_test(
        HashMap::new(),
        HashMap::from([("weight".to_string(), LoadedTensor::F32(tensor))]),
    );

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
    let mut cursor = std::io::Cursor::new(gguf_header((limits::MAX_TENSOR_COUNT as u64) + 1, 0));
    let error = load_gguf_from_reader(&mut cursor)
        .err()
        .expect("a sparse tensor count must not reserve an unbounded table");
    assert!(error.to_string().contains("tensor count"), "{error}");
    assert!(error.to_string().contains("record limit"), "{error}");

    let mut cursor = std::io::Cursor::new(gguf_one_tensor(&[1u64 << 31], 0));
    let error = load_gguf_from_reader(&mut cursor)
        .err()
        .expect("a sparse f32 tensor declaration must not allocate");
    assert!(error.to_string().contains("encoded bytes"), "{error}");

    // A 1 GiB f16 declaration is well under the encoded-byte cap, but its
    // f32 destination is 2 GiB, so the allocation cap must reject it
    // before reading the payload.
    let mut cursor = std::io::Cursor::new(gguf_one_tensor(&[1u64 << 29], 1));
    let error = load_gguf_from_reader(&mut cursor)
        .err()
        .expect("transient tensor allocation must be bounded");
    assert!(error.to_string().contains("while loading"), "{error}");

    // f32 conversion streams through a small stack buffer, so a 1 GiB
    // f32 tensor needs exactly its 1 GiB destination: it passes the
    // allocation cap and is rejected only because the file is short.
    let mut cursor = std::io::Cursor::new(gguf_one_tensor(&[1u64 << 28], 0));
    let error = load_gguf_from_reader(&mut cursor)
        .err()
        .expect("a truncated tensor must be rejected");
    assert!(error.to_string().contains("exceeds file length"), "{error}");
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

#[test]
fn metadata_budget_admits_gemma4_scale_tokenizer_arrays() {
    // Gemma 4 tokenizers total ~1.30M metadata values (three 262,144-value
    // arrays plus a 514,906-entry merges array and the scalar keys). The
    // aggregate cap must admit that with headroom: this is the calibration
    // evidence for MAX_METADATA_VALUES.
    let mut budget = MetadataBudget::new();
    for _ in 0..1_301_500 {
        budget.consume_value().unwrap();
    }
    assert!(budget.consume_value().is_ok());
}

#[test]
fn metadata_value_budget_is_bounded_and_enforced() {
    let footprint = std::mem::size_of::<GgufValue>() * limits::MAX_METADATA_VALUES;
    assert!(
        footprint <= 256 << 20,
        "metadata envelope grows past 256 MiB: {footprint} bytes"
    );
    let mut budget = MetadataBudget::new();
    for _ in 0..limits::MAX_METADATA_VALUES {
        budget.consume_value().unwrap();
    }
    assert!(budget.consume_value().is_err());
}

#[test]
fn tensor_byte_cap_admits_gemma4_ple_and_stays_below_the_file_bound() {
    const {
        // per_layer_token_embd.weight (Gemma 4) declares 2,495,610,880
        // encoded bytes; the cap must admit it and stay under the file
        // bound.
        assert!(limits::MAX_TENSOR_BYTES >= 2_495_610_880);
        assert!(limits::MAX_TENSOR_BYTES < limits::MAX_GGUF_FILE_BYTES);
    }
}

#[test]
fn rope_table_cap_fits_the_documented_envelope() {
    const {
        // One f32 cosine+sine pair at the cap; Gemma 4 E2B needs
        // 131,072 x 512 for its global pair.
        let pair_bytes = limits::MAX_ROPE_TABLE_ELEMENTS * std::mem::size_of::<f32>();
        assert!(pair_bytes <= 512 << 20);
        assert!(limits::MAX_ROPE_TABLE_ELEMENTS >= 131_072 * 512);
    }
}

#[test]
fn gemma4_scale_header_parses_end_to_end() {
    // Two 700k-value arrays (the Gemma 4 pattern: several tokenizer arrays
    // whose total exceeds the old 1M aggregate cap) must parse.
    let mut bytes = gguf_header(0, 2);
    for key in [b"tokens".as_slice(), b"merges".as_slice()] {
        push_gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array value
        bytes.extend_from_slice(&0u32.to_le_bytes()); // u8 elements
        bytes.extend_from_slice(&700_000u64.to_le_bytes());
        bytes.resize(bytes.len() + 700_000, 0);
    }
    let mut cursor = std::io::Cursor::new(bytes);
    let loader = load_gguf_from_reader(&mut cursor).expect("gemma4-scale metadata must parse");
    assert_eq!(loader.metadata.len(), 2);
    match loader.metadata.get("tokens") {
        Some(GgufValue::Array(values)) => assert_eq!(values.len(), 700_000),
        other => panic!("expected a 700k-element array, got {other:?}"),
    }
}

#[test]
fn tokenizer_arrays_are_validated_but_not_materialized() {
    let mut bytes = gguf_header(0, 3);
    // String array under a skipped key.
    push_gguf_string(&mut bytes, b"tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes()); // string elements
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for token in [b"a".as_slice(), b"bb".as_slice(), b"ccc".as_slice()] {
        push_gguf_string(&mut bytes, token);
    }
    // Numeric array under a skipped key.
    push_gguf_string(&mut bytes, b"tokenizer.ggml.scores");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&6u32.to_le_bytes()); // f32 elements
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    // A key outside the skip list is still materialized.
    push_gguf_string(&mut bytes, b"custom.values");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&5u32.to_le_bytes()); // u32 elements
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.extend_from_slice(&9u32.to_le_bytes());

    let mut cursor = std::io::Cursor::new(bytes);
    let loader = load_gguf_from_reader(&mut cursor).expect("metadata parses");
    assert_eq!(loader.metadata.len(), 3, "keys are preserved, not dropped");
    match loader.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::SkippedArray {
            element_type,
            elements,
        }) => {
            assert_eq!(*element_type, 8);
            assert_eq!(*elements, 3);
        }
        other => panic!("expected a skipped tokenizer array, got {other:?}"),
    }
    match loader.metadata.get("tokenizer.ggml.scores") {
        Some(GgufValue::SkippedArray {
            element_type,
            elements,
        }) => {
            assert_eq!(*element_type, 6);
            assert_eq!(*elements, 2);
        }
        other => panic!("expected a skipped numeric array, got {other:?}"),
    }
    match loader.metadata.get("custom.values") {
        Some(GgufValue::Array(values)) => assert_eq!(values.len(), 2),
        other => panic!("expected a materialized array, got {other:?}"),
    }
}

#[test]
fn malformed_skipped_tokenizer_arrays_still_fail() {
    // A truncated string element inside a skipped array is a hard error.
    let mut bytes = gguf_header(0, 1);
    push_gguf_string(&mut bytes, b"tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    push_gguf_string(&mut bytes, b"ok");
    bytes.extend_from_slice(&64u64.to_le_bytes()); // declares 64 bytes; none follow
    let mut cursor = std::io::Cursor::new(bytes);
    let error = match load_gguf_from_reader(&mut cursor) {
        Ok(_) => panic!("truncated string must fail"),
        Err(error) => error,
    };
    assert!(
        format!("{error}").contains("remaining"),
        "unexpected error: {error}"
    );

    // Invalid UTF-8 inside a skipped array is a hard error too.
    let mut bytes = gguf_header(0, 1);
    push_gguf_string(&mut bytes, b"tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    push_gguf_string(&mut bytes, &[0xff]);
    let mut cursor = std::io::Cursor::new(bytes);
    let error = match load_gguf_from_reader(&mut cursor) {
        Ok(_) => panic!("invalid utf8 must fail"),
        Err(error) => error,
    };
    assert!(
        format!("{error}").contains("utf8"),
        "unexpected error: {error}"
    );
}

/// Build a minimal GGUF v3 file with one K-family tensor
/// (`blk.0.attn_q.weight`, dims [256, blocks], zero payload) and no
/// metadata. Zero blocks dequantize to zeros.
fn write_minimal_gguf_with_k_tensor(dtype: u32, blocks: usize) -> Vec<u8> {
    let mut out = gguf_header(1, 0);
    push_gguf_string(&mut out, b"blk.0.attn_q.weight");
    out.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    out.extend_from_slice(&256u64.to_le_bytes());
    out.extend_from_slice(&(blocks as u64).to_le_bytes());
    out.extend_from_slice(&(dtype).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // offset
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out.resize(
        out.len() + blocks * crate::quant_k::k_block_bytes(dtype).unwrap(),
        0,
    );
    out
}

/// Reader-load the [`write_minimal_gguf_with_k_tensor`] fixture under a
/// K strategy.
fn load_k(
    dtype: u32,
    blocks: usize,
    strategy: crate::quant_k::KStrategy,
    allow_fallback: bool,
) -> Result<GgufLoader> {
    let mut cursor = std::io::Cursor::new(write_minimal_gguf_with_k_tensor(dtype, blocks));
    load_gguf_from_reader_with_k_strategy(&mut cursor, strategy, allow_fallback)
}

/// GGUF v3 with two metadata keys and the given `(name, dims, dtype)`
/// tensors, each payload zero-filled and 32-byte aligned.
fn write_gguf_with_tensors(tensors: &[(&str, &[u64], u32)]) -> Vec<u8> {
    let mut out = gguf_header(tensors.len() as u64, 2);
    push_gguf_string(&mut out, b"general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    push_gguf_string(&mut out, b"llama");
    push_gguf_string(&mut out, b"llama.block_count");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    let mut offset = 0u64;
    let mut payload_lens = Vec::new();
    for &(name, dims, dtype) in tensors {
        push_gguf_string(&mut out, name.as_bytes());
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &dim in dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&dtype.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        let elements = dims.iter().product::<u64>() as usize;
        let len = gguf_dtype_byte_len(dtype, elements).unwrap_or(elements * 4);
        payload_lens.push(len);
        offset += (len as u64).div_ceil(32) * 32;
    }
    for len in payload_lens {
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        out.resize(out.len() + len, 0);
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out
}

fn write_temp_gguf(label: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "ember-header-{label}-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn sorted_debug<V: std::fmt::Debug>(map: &HashMap<String, V>) -> Vec<String> {
    let mut entries: Vec<String> = map.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
    entries.sort();
    entries
}

/// F16/BF16 tensors stay in their 16-bit encoding (shared with the file
/// mapping for file loads, copied for reader loads) and widen to exactly
/// the f32 tensor the eager loader produced, for every one of the 65,536
/// encodings (NaN payloads, infinities, subnormals, signed zeros).
#[test]
fn half_tensors_stay_encoded_and_widen_like_the_eager_loader() {
    use crate::half_weight::{widen_scalar, HalfDtype};
    let mut bytes = write_gguf_with_tensors(&[
        ("f16.weight", &[256, 256], 1),
        ("bf16.weight", &[256, 256], 30),
    ]);
    // Both payloads are 128 KiB (32-aligned), so they end the file.
    let payload = bytes.len() - 2 * 131_072;
    for (index, chunk) in bytes[payload..].chunks_exact_mut(2).enumerate() {
        chunk.copy_from_slice(&((index % 65_536) as u16).to_le_bytes());
    }
    let path = write_temp_gguf("half", &bytes);
    let mapped = load_gguf(&path).unwrap();
    let read = load_gguf_from_reader(&mut std::io::Cursor::new(bytes)).unwrap();
    std::fs::remove_file(&path).unwrap();
    for (loader, expect_mapped) in [(mapped, true), (read, false)] {
        for (name, dtype) in [
            ("f16.weight", HalfDtype::F16),
            ("bf16.weight", HalfDtype::Bf16),
        ] {
            let Some(LoadedTensor::Half(weight)) = loader.tensors.get(name) else {
                panic!("{name} was not kept as a Half tensor");
            };
            assert_eq!(weight.dtype(), dtype);
            assert_eq!(weight.is_mapped(), expect_mapped, "{name}");
            assert_eq!(weight.dims(), &[256, 256]);
            let tensor = weight.to_f32_tensor();
            assert_eq!(tensor.shape(), &[256, 256]);
            for (bits, value) in (0..=u16::MAX).zip(tensor.data()) {
                assert_eq!(
                    value.to_bits(),
                    widen_scalar(dtype, bits).to_bits(),
                    "{name} {bits:#06x}"
                );
            }
        }
        let mut loader = loader;
        loader.materialize_half_tensors();
        assert!(loader
            .tensors
            .values()
            .all(|tensor| matches!(tensor, LoadedTensor::F32(_))));
    }
}

/// The header-only load (used by `ember inspect`) reports exactly the
/// metadata, tensor inventory and K decisions of the full load, across
/// every loadable dtype (f32/f16/bf16/q8_0, native and fallback K), but
/// materializes no tensor.
#[test]
fn header_only_load_matches_full_load_inventory() {
    let bytes = write_gguf_with_tensors(&[
        ("norm.weight", &[64], 0),
        ("f16.weight", &[64, 2], 1),
        ("bf16.weight", &[64, 2], 30),
        ("q8.weight", &[64, 4], 8),
        ("q4k.weight", &[256, 2], 12),
        ("q6k.weight", &[256, 2], 14),
        ("q5k.weight", &[256, 2], 13),
    ]);
    let path = write_temp_gguf("inventory", &bytes);
    for strategy in [
        crate::quant_k::KStrategy::Auto,
        crate::quant_k::KStrategy::EagerF32,
        crate::quant_k::KStrategy::Scalar,
    ] {
        let full = load_gguf_with_k_strategy(&path, strategy, true).unwrap();
        let header = load_gguf_header(&path, strategy, true).unwrap();
        assert_eq!(full.tensors.len(), 7);
        assert!(
            header.tensors.is_empty(),
            "header-only load decoded tensors"
        );
        assert_eq!(sorted_debug(&header.metadata), sorted_debug(&full.metadata));
        assert_eq!(
            sorted_debug(&header.tensor_meta),
            sorted_debug(&full.tensor_meta)
        );
        assert_eq!(
            sorted_debug(&header.k_decisions),
            sorted_debug(&full.k_decisions)
        );
        assert_eq!(header.k_strategy, full.k_strategy);
    }
    std::fs::remove_file(&path).unwrap();
}

/// Structural rejections are identical: the header-only load runs the
/// same per-tensor checks (duplicate names, block alignment, shape and
/// strict-strategy decisions) without decoding payloads.
#[test]
fn header_only_load_rejects_what_the_full_load_rejects() {
    let cases: [(&str, Vec<u8>, bool); 4] = [
        (
            "duplicate",
            write_gguf_with_tensors(&[("a.weight", &[64], 0), ("a.weight", &[64], 1)]),
            true,
        ),
        (
            "q8-misaligned",
            write_gguf_with_tensors(&[("q8.weight", &[48, 1], 8)]),
            true,
        ),
        (
            "q8-not-2d",
            write_gguf_with_tensors(&[("q8.weight", &[64], 8)]),
            true,
        ),
        (
            "strict-no-fallback",
            write_gguf_with_tensors(&[("q5k.weight", &[256, 2], 13)]),
            false,
        ),
    ];
    for (label, bytes, allow_fallback) in cases {
        let path = write_temp_gguf(label, &bytes);
        let strategy = crate::quant_k::KStrategy::Scalar;
        let full = load_gguf_with_k_strategy(&path, strategy, allow_fallback)
            .err()
            .unwrap_or_else(|| panic!("{label}: full load accepted a malformed file"));
        let header = load_gguf_header(&path, strategy, allow_fallback)
            .err()
            .unwrap_or_else(|| panic!("{label}: header load accepted a malformed file"));
        assert_eq!(header.to_string(), full.to_string(), "{label}");
        std::fs::remove_file(&path).unwrap();
    }
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

/// Scalar strategy on Q4_K keeps the weight compressed-resident, and the
/// execution inventory reports that residency.
#[test]
fn scalar_strategy_keeps_q4_k_compressed_resident_and_inventory_records_it() {
    let loader = load_k(12, 128, crate::quant_k::KStrategy::Scalar, false).unwrap();

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

    let Some(LoadedTensor::KQuant(weight)) = loader.tensors.get("blk.0.attn_q.weight") else {
        panic!("expected compressed KQuant tensor");
    };
    assert_eq!((weight.out_features(), weight.in_features()), (128, 256));
    assert_eq!(weight.dtype(), crate::quant_k::KQuantDtype::Q4K);
    assert_eq!(weight.byte_len(), 128 * crate::quant_k::Q4_K_BLOCK_BYTES);
    assert!(!weight.is_mapped(), "reader loads are owned");

    let inventory = crate::execution_inventory::ExecutionInventory::from_loader(&loader);
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

/// Auto strategy on Q2_K (no native kernel) falls back to eager f32 with a
/// recorded reason, and the execution inventory reports the fallback.
#[test]
fn auto_strategy_falls_back_to_eager_for_q2_k_and_inventory_records_it() {
    let loader = load_k(10, 8, crate::quant_k::KStrategy::Auto, false).unwrap();

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

    let inventory = crate::execution_inventory::ExecutionInventory::from_loader(&loader);
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
fn scalar_strategy_hard_fails_q2_k_without_allow_fallback() {
    let err = load_k(10, 8, crate::quant_k::KStrategy::Scalar, false)
        .err()
        .expect("scalar strategy must hard-fail q2_k without allow-fallback");
    let message = err.to_string();
    assert!(
        message.contains("blk.0.attn_q.weight"),
        "message: {message}"
    );
    assert!(message.contains("q2_k"), "message: {message}");

    // with --k-allow-fallback the same file loads eager and records why
    let loader = load_k(10, 8, crate::quant_k::KStrategy::Scalar, true).unwrap();
    let decision = &loader.k_decisions["blk.0.attn_q.weight"];
    assert_eq!(decision.execution, crate::quant_k::KExecution::EagerF32);
    assert!(decision.fallback_reason.is_some());
}

// x86-only: asserts the AVX2+FMA+F16C+SSSE3 feature set. Without it the
// no-fallback X86 request hard-fails; that path is covered by
// `x86_strategy_falls_back_to_scalar_only_with_allow_fallback`.
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_strategy_records_compressed_x86_when_avx2_available() {
    if !crate::k_quant_matmul::x86_k_supported() {
        return;
    }
    let loader = load_k(14, 4, crate::quant_k::KStrategy::X86, false).unwrap(); // q6_k

    let decision = &loader.k_decisions["blk.0.attn_q.weight"];
    assert_eq!(
        decision.execution,
        crate::quant_k::KExecution::CompressedX86
    );
    assert!(decision.fallback_reason.is_none());
    let inventory = crate::execution_inventory::ExecutionInventory::from_loader(&loader);
    assert_eq!(inventory.tensors[0].kernel, "q6-k-q8-k-avx2");
    assert_eq!(
        inventory.tensors[0].kernel_revision,
        crate::plan::PLAN_KERNEL_REVISION
    );
    assert_eq!(inventory.tensors[0].strategy, "compressed-x86");
    assert_eq!(inventory.tensors[0].cpu_features, "avx2+fma+f16c+ssse3");
}

#[test]
fn eager_k_mapped_payload_matches_owned_without_duplicate_budget() {
    for dtype in 10..=14 {
        let bytes = write_minimal_gguf_with_k_tensor(dtype, 4);
        let mut mapping = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        mapping.copy_from_slice(&bytes);
        let mapped = Arc::new(mapping.make_read_only().unwrap());
        let (mapped_loader, _) = load_gguf_from_reader_impl(
            &mut std::io::Cursor::new(&bytes),
            Some(mapped),
            crate::quant_k::KStrategy::EagerF32,
            false,
            LoadMode::Full,
        )
        .unwrap();
        let owned_loader = load_gguf_from_reader_with_k_strategy(
            &mut std::io::Cursor::new(&bytes),
            crate::quant_k::KStrategy::EagerF32,
            false,
        )
        .unwrap();
        let name = "blk.0.attn_q.weight";
        let (LoadedTensor::F32(a), LoadedTensor::F32(b)) =
            (&mapped_loader.tensors[name], &owned_loader.tensors[name])
        else {
            panic!("expected eager weights")
        };
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.data(), b.data());
        let info = TensorInfo {
            name: name.into(),
            dims: vec![256, 4],
            dtype,
            offset: 0,
        };
        let encoded = 4 * crate::quant_k::k_block_bytes(dtype).unwrap() as u64;
        assert_eq!(
            estimated_tensor_allocation_bytes(
                &info,
                1024,
                encoded,
                crate::quant_k::KStrategy::EagerF32,
                false,
                true,
                false
            )
            .unwrap(),
            4096
        );
        assert_eq!(
            estimated_tensor_allocation_bytes(
                &info,
                1024,
                encoded,
                crate::quant_k::KStrategy::EagerF32,
                false,
                false,
                false
            )
            .unwrap(),
            4096 + encoded
        );
    }
}

#[test]
fn float_tensor_estimate_counts_only_the_f32_destination() {
    // read_float_values streams through a fixed stack buffer; the encoded
    // bytes are never held in an allocation of their own.
    for (dtype, width) in [(0u32, 4u64), (1, 2), (30, 2)] {
        let info = TensorInfo {
            name: "float".into(),
            dims: vec![1024, 16],
            dtype,
            offset: 0,
        };
        for mmap_present in [false, true] {
            assert_eq!(
                estimated_tensor_allocation_bytes(
                    &info,
                    16 * 1024,
                    16 * 1024 * width,
                    crate::quant_k::KStrategy::EagerF32,
                    false,
                    mmap_present,
                    false,
                )
                .unwrap(),
                16 * 1024 * 4,
                "dtype {dtype}"
            );
        }
    }
}

#[test]
fn arm_strategy_records_native_tier_or_rejects_unavailable_cpu() {
    for (dtype, kernel) in [
        (12, "q4-k-q8-k-neon-dotprod"),
        (14, "q6-k-q8-k-neon-dotprod"),
    ] {
        let result = load_k(dtype, 4, crate::quant_k::KStrategy::Arm, false);
        if !crate::k_quant_matmul::arm_k_supported() {
            assert!(result.is_err());
            continue;
        }
        let loader = result.unwrap();
        let decision = &loader.k_decisions["blk.0.attn_q.weight"];
        assert_eq!(
            decision.execution,
            crate::quant_k::KExecution::CompressedArm
        );
        assert!(decision.fallback_reason.is_none());
        let inventory = crate::execution_inventory::ExecutionInventory::from_loader(&loader);
        assert_eq!(inventory.tensors[0].kernel, kernel);
        assert_eq!(inventory.tensors[0].strategy, "compressed-arm");
        assert_eq!(inventory.tensors[0].cpu_features, "neon+dotprod");
    }
}

#[test]
fn auto_strategy_selects_x86_when_avx2_available() {
    let loader = load_k(12, 8, crate::quant_k::KStrategy::Auto, false).unwrap(); // q4_k
    let decision = &loader.k_decisions["blk.0.attn_q.weight"];
    let expected = if crate::k_quant_matmul::x86_k_supported() {
        crate::quant_k::KExecution::CompressedX86
    } else if crate::k_quant_matmul::arm_k_supported() {
        crate::quant_k::KExecution::CompressedArm
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
    let result = load_k(14, 4, crate::quant_k::KStrategy::X86, false);
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
    let loader = load_k(14, 4, crate::quant_k::KStrategy::X86, true).unwrap();
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

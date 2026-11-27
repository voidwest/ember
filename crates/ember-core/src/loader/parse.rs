use super::*;

pub(super) fn read_u8<R: Read>(f: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(u8::from_le_bytes(buf))
}

pub(super) fn read_i8<R: Read>(f: &mut R) -> Result<i8> {
    Ok(read_u8(f)? as i8)
}

pub(super) fn read_u16<R: Read>(f: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub(super) fn read_i16<R: Read>(f: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

pub(super) fn read_u32<R: Read>(f: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn read_u64<R: Read>(f: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub(super) fn read_i32<R: Read>(f: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub(super) fn read_i64<R: Read>(f: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub(super) fn read_f32<R: Read>(f: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub(super) fn read_f64<R: Read>(f: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub(super) fn remaining_bytes<R: Seek>(reader: &mut R) -> Result<u64> {
    let position = reader.stream_position()?;
    let end = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(position))?;
    end.checked_sub(position)
        .ok_or_else(|| LoaderError::malformed("reader position exceeds its end"))
}

pub(super) struct MetadataBudget {
    pub(super) remaining_values: usize,
    pub(super) remaining_string_bytes: usize,
}

impl MetadataBudget {
    pub(super) fn new() -> Self {
        Self {
            remaining_values: limits::MAX_METADATA_VALUES,
            remaining_string_bytes: limits::MAX_METADATA_STRING_BYTES,
        }
    }

    pub(super) fn consume_value(&mut self) -> Result<()> {
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

/// Read one GGUF string into `scratch`, enforcing every string limit.
///
/// Shared by the materializing reader and the skipping reader so both paths
/// apply the identical bounds (declared length, aggregate string budget,
/// bytes remaining in the file) and UTF-8 validation.
pub(super) fn read_gguf_string_bytes<R: Read + Seek>(
    f: &mut R,
    remaining_string_bytes: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<()> {
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
    scratch.clear();
    scratch.try_reserve_exact(len).map_err(|error| {
        LoaderError::reservation(format!("failed to reserve GGUF string buffer: {error}"))
    })?;
    scratch.resize(len, 0);
    f.read_exact(scratch)?;
    std::str::from_utf8(scratch)
        .map_err(|error| LoaderError::malformed(format!("invalid utf8 in string: {error}")))?;
    Ok(())
}

/// Walk `count` GGUF strings, enforcing every limit enforced by
/// [`read_gguf_string_bytes`] (declared length, aggregate string budget, bytes
/// remaining in the file, UTF-8) without materializing them.
///
/// The bytes remaining in the file are tracked by subtraction so the
/// `remaining_bytes` seek dance is paid once per array instead of once per
/// string; tokenizer arrays hold ~10^5-10^6 strings.
pub(super) fn skip_gguf_strings<R: Read + Seek>(
    f: &mut R,
    count: usize,
    remaining_string_bytes: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    let mut file_remaining = remaining_bytes(f)?;
    for _ in 0..count {
        let declared_len = read_u64(f)?;
        file_remaining = file_remaining.saturating_sub(8);
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
        if declared_len > file_remaining {
            return Err(LoaderError::malformed(format!(
                "GGUF string length {len} exceeds the {file_remaining} bytes remaining in the file"
            )));
        }
        *remaining_string_bytes -= len;
        file_remaining -= declared_len;
        scratch.clear();
        scratch.try_reserve_exact(len).map_err(|error| {
            LoaderError::reservation(format!("failed to reserve GGUF string buffer: {error}"))
        })?;
        scratch.resize(len, 0);
        f.read_exact(scratch)?;
        std::str::from_utf8(scratch)
            .map_err(|error| LoaderError::malformed(format!("invalid utf8 in string: {error}")))?;
    }
    Ok(())
}

pub(super) fn read_gguf_string_with_budget<R: Read + Seek>(
    f: &mut R,
    remaining_string_bytes: &mut usize,
) -> Result<String> {
    let mut bytes = Vec::new();
    read_gguf_string_bytes(f, remaining_string_bytes, &mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| LoaderError::malformed(format!("invalid utf8 in string: {error}")))
}

pub(super) fn minimum_value_size(val_type: u32) -> Result<u64> {
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

pub(super) fn read_gguf_value<R: Read + Seek>(
    f: &mut R,
    val_type: u32,
    budget: &mut MetadataBudget,
    skip_array: bool,
) -> Result<GgufValue> {
    read_gguf_value_inner(f, val_type, 0, budget, skip_array)
}

pub(super) fn read_gguf_value_inner<R: Read + Seek>(
    f: &mut R,
    val_type: u32,
    depth: usize,
    budget: &mut MetadataBudget,
    skip_array: bool,
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
            let declared = read_u64(f)?;
            let count = usize::try_from(declared).map_err(|error| {
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
            if skip_array && element_type != 9 {
                // Declared unused by the caller: keep the same validation and
                // budget accounting, but no values. The per-element budget
                // check is equivalent to consuming `count` values here.
                budget.remaining_values -= count;
                if element_type == 8 {
                    let mut scratch = Vec::new();
                    skip_gguf_strings(f, count, &mut budget.remaining_string_bytes, &mut scratch)?;
                } else {
                    let stride = minimum_value_size(element_type)?;
                    let bytes = stride
                        .checked_mul(declared)
                        .ok_or_else(|| LoaderError::overflow("GGUF array byte size overflow"))?;
                    f.seek(SeekFrom::Current(i64::try_from(bytes).map_err(
                        |error| {
                            LoaderError::overflow(format!(
                                "GGUF array length exceeds seek range: {error}"
                            ))
                        },
                    )?))?;
                }
                return Ok(GgufValue::SkippedArray {
                    element_type,
                    elements: declared,
                });
            }
            let mut elements = Vec::new();
            elements.try_reserve_exact(count).map_err(|error| {
                LoaderError::reservation(format!("failed to reserve GGUF metadata array: {error}"))
            })?;
            for _ in 0..count {
                elements.push(read_gguf_value_inner(
                    f,
                    element_type,
                    depth + 1,
                    budget,
                    false,
                )?);
            }
            Ok(GgufValue::Array(elements))
        }
        _ => Err(LoaderError::malformed(format!(
            "unsupported GGUF value type: {}",
            val_type
        ))),
    }
}

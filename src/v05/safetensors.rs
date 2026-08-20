//! Minimal, strict safetensors codec for v0.5 bundle payloads.
//!
//! Implements the published safetensors file format exactly (8-byte
//! little-endian header length, JSON header padded to an 8-byte boundary
//! *inside* the length field, data immediately after) so payloads are
//! byte-compatible with the Python `safetensors` library.
//! Serialization is deterministic (BTreeMap key order); deserialization
//! validates every offset, size, and alignment before any slice is taken.

use serde::Serialize;
use std::collections::BTreeMap;

/// Tensor dtypes supported by the v0.5 bundle format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
}

impl TensorDType {
    pub const fn byte_size(self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::F16 => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            TensorDType::F32 => "F32",
            TensorDType::F16 => "F16",
        }
    }

    pub fn parse(name: &str) -> Result<TensorDType, String> {
        match name {
            "F32" => Ok(TensorDType::F32),
            "F16" => Ok(TensorDType::F16),
            other => Err(format!("unsupported safetensors dtype '{other}'")),
        }
    }
}

/// One tensor to serialize.
pub struct TensorData<'a> {
    pub name: &'a str,
    pub dtype: TensorDType,
    pub shape: &'a [usize],
    /// Little-endian bytes of the tensor (dtype-consistent length).
    pub bytes: &'a [u8],
}

/// Serialize tensors into the safetensors byte format.
///
/// Deterministic: tensors are emitted in name order (BTreeMap), the data
/// start is 8-byte aligned, and the total file length is a multiple of 8.
pub fn serialize(tensors: &[TensorData<'_>]) -> Result<Vec<u8>, String> {
    // Validate sizes and compute offsets in name order.
    let mut by_name: BTreeMap<&str, &TensorData<'_>> = BTreeMap::new();
    for tensor in tensors {
        let element_count: usize = tensor
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| format!("tensor '{}' shape product overflow", tensor.name))?;
        let expected = element_count
            .checked_mul(tensor.dtype.byte_size())
            .ok_or_else(|| format!("tensor '{}' byte size overflow", tensor.name))?;
        if tensor.bytes.len() != expected {
            return Err(format!(
                "tensor '{}': {} bytes for shape {:?} {} (expected {expected})",
                tensor.name,
                tensor.bytes.len(),
                tensor.shape,
                tensor.dtype.name()
            ));
        }
        if by_name.insert(tensor.name, tensor).is_some() {
            return Err(format!("duplicate tensor name '{}'", tensor.name));
        }
    }

    // Header JSON with data_offsets, in name order.
    let mut header = serde_json::Map::new();
    let mut data_offset = 0usize;
    for (name, tensor) in &by_name {
        let start = data_offset;
        let end = start + tensor.bytes.len();
        let mut entry = serde_json::Map::new();
        entry.insert(
            "dtype".into(),
            serde_json::Value::String(tensor.dtype.name().to_string()),
        );
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(tensor.shape.iter().map(|&d| d.into()).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![start.into(), end.into()]),
        );
        header.insert(name.to_string(), serde_json::Value::Object(entry));
        data_offset = end;
    }
    let mut header_json = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|error| format!("safetensors header serialization failed: {error}"))?;
    if header_json.len() > u64::MAX as usize {
        return Err("safetensors header exceeds u64 length".into());
    }
    // The published format pads the header *string itself* (ASCII spaces)
    // so that 8 + header_len is 8-byte aligned and header_len includes the
    // padding; readers then derive data_start = 8 + header_len directly.
    // (Padding the data start instead breaks spec-compliant readers such as
    // the Python `safetensors` library whenever the unpadded header length
    // is not 8-aligned.)
    let header_padding = (8 + header_json.len())
        .checked_rem(8)
        .map(|r| (8 - r) % 8)
        .unwrap_or(0);
    header_json.extend(std::iter::repeat_n(b' ', header_padding));
    let header_len = header_json.len() as u64;
    let data_start = 8 + header_json.len();
    debug_assert_eq!(data_start % 8, 0, "safetensors data must be 8-byte aligned");
    let mut out = Vec::with_capacity(data_start + data_offset);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_json);
    for tensor in by_name.values() {
        out.extend_from_slice(tensor.bytes);
    }
    // The safetensors format requires the total file length to be a
    // multiple of 8; pad the tail deterministically.
    let tail_padding = (8 - out.len() % 8) % 8;
    out.extend(std::iter::repeat_n(0u8, tail_padding));
    Ok(out)
}

/// A deserialized tensor view.
#[derive(Debug, Clone)]
pub struct TensorView {
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Strictly parse a safetensors byte buffer.
///
/// Validates the header length, JSON structure, offsets, alignment, and
/// bounds; returns the tensors in file order plus the data slice.
pub fn deserialize(bytes: &[u8]) -> Result<Vec<(String, TensorView)>, String> {
    if bytes.len() < 8 {
        return Err("safetensors buffer shorter than the 8-byte header length".into());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| "safetensors header length overflow".to_string())?;
    if data_start > bytes.len() {
        return Err(format!(
            "safetensors header claims {header_len} bytes but the buffer has only {}",
            bytes.len()
        ));
    }
    // Tolerant of both the published form (header_len already includes
    // padding, so `padded == 8 + header_len`) and the legacy pre-0.6.4
    // form (padding emitted after the header instead of inside it).
    let padded = 8usize
        .checked_add(header_len)
        .and_then(|v| v.checked_add(7))
        .map(|v| v & !7)
        .ok_or_else(|| "safetensors offset overflow".to_string())?;
    if padded > bytes.len() {
        return Err("safetensors data region exceeds buffer".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|error| format!("safetensors header is not valid JSON: {error}"))?;
    let object = header
        .as_object()
        .ok_or_else(|| "safetensors header must be a JSON object".to_string())?;
    let mut tensors = Vec::with_capacity(object.len());
    for (name, value) in object {
        let entry = value
            .as_object()
            .ok_or_else(|| format!("safetensors entry '{name}' must be an object"))?;
        let dtype_name = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("safetensors entry '{name}' lacks a dtype string"))?;
        let dtype = TensorDType::parse(dtype_name)?;
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("safetensors entry '{name}' lacks a shape array"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| format!("safetensors entry '{name}' has a non-integer shape"))
            })
            .collect::<Result<_, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("safetensors entry '{name}' lacks data_offsets"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "safetensors entry '{name}' data_offsets must have 2 entries"
            ));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("safetensors entry '{name}' has an invalid start offset"))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("safetensors entry '{name}' has an invalid end offset"))?
            as usize;
        if start >= end {
            return Err(format!(
                "safetensors entry '{name}' has a non-increasing offset range"
            ));
        }
        let element_count: usize = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| format!("safetensors entry '{name}' shape product overflow"))?;
        let expected = element_count
            .checked_mul(dtype.byte_size())
            .ok_or_else(|| format!("safetensors entry '{name}' byte size overflow"))?;
        if end - start != expected {
            return Err(format!(
                "safetensors entry '{name}': offset range {}..{} does not match shape {:?} {} \
                 (expected {expected} bytes)",
                start,
                end,
                shape,
                dtype.name()
            ));
        }
        let absolute_start = padded
            .checked_add(start)
            .ok_or_else(|| "safetensors offset overflow".to_string())?;
        let absolute_end = padded
            .checked_add(end)
            .ok_or_else(|| "safetensors offset overflow".to_string())?;
        if absolute_end > bytes.len() {
            return Err(format!(
                "safetensors entry '{name}' range {}..{} exceeds the buffer",
                absolute_start, absolute_end
            ));
        }
        tensors.push((
            name.clone(),
            TensorView {
                dtype,
                shape,
                data_offsets: (absolute_start, absolute_end),
            },
        ));
    }
    Ok(tensors)
}

/// Load one tensor's f32 data (with f16 conversion when stored as f16).
pub fn tensor_f32(bytes: &[u8], view: &TensorView) -> Result<Vec<f32>, String> {
    let raw = &bytes[view.data_offsets.0..view.data_offsets.1];
    match view.dtype {
        TensorDType::F32 => {
            if !raw.len().is_multiple_of(4) {
                return Err("f32 tensor byte length is not 4-aligned".into());
            }
            Ok(raw
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
                .collect())
        }
        TensorDType::F16 => {
            if !raw.len().is_multiple_of(2) {
                return Err("f16 tensor byte length is not 2-aligned".into());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes(chunk.try_into().expect("2 bytes")).to_f32())
                .collect())
        }
    }
}

/// Convert f32 values to f16 bytes (little-endian).
pub fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
        .collect()
}

/// A serializable tensor entry for the capture index.
#[derive(Debug, Clone, Serialize)]
pub struct PayloadEntry<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub dtype: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensors() -> Vec<TensorData<'static>> {
        let a_bytes: &'static [u8] = Box::leak(
            [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        let b_bytes: &'static [u8] = Box::leak(
            [1.0f32, 2.0]
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        vec![
            TensorData {
                name: "a",
                dtype: TensorDType::F32,
                shape: &[2, 2],
                bytes: a_bytes,
            },
            TensorData {
                name: "b",
                dtype: TensorDType::F16,
                shape: &[2],
                bytes: b_bytes,
            },
        ]
    }

    #[test]
    fn round_trip() {
        let bytes = serialize(&sample_tensors()).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        let tensors = deserialize(&bytes).unwrap();
        assert_eq!(tensors.len(), 2);
        let (name, view) = &tensors[0];
        assert_eq!(name, "a");
        assert_eq!(view.dtype, TensorDType::F32);
        assert_eq!(view.shape, vec![2, 2]);
        let data = tensor_f32(&bytes, view).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        let (_, view_b) = &tensors[1];
        let data_b = tensor_f32(&bytes, view_b).unwrap();
        assert_eq!(data_b, vec![1.0, 2.0]);
    }

    #[test]
    fn deterministic_serialization() {
        let mut a = sample_tensors();
        a.reverse();
        let first = serialize(&sample_tensors()).unwrap();
        let second = serialize(&a).unwrap();
        assert_eq!(first, second, "serialization must be name-ordered");
    }

    #[test]
    fn corrupted_buffer_fails() {
        let bytes = serialize(&sample_tensors()).unwrap();
        // Truncated data region: the structural parser must reject it.
        let truncated = &bytes[..bytes.len() - 5];
        assert!(deserialize(truncated).is_err());
        // Header length claiming more bytes than exist.
        let mut bad = Vec::new();
        bad.extend_from_slice(&(u64::MAX).to_le_bytes());
        bad.extend_from_slice(&bytes);
        assert!(deserialize(&bad).is_err());
    }

    #[test]
    fn size_mismatch_fails() {
        let tensors = vec![TensorData {
            name: "x",
            dtype: TensorDType::F32,
            shape: &[3],
            bytes: &[0u8; 4],
        }];
        assert!(serialize(&tensors).is_err());
    }

    #[test]
    fn header_is_spec_padded_for_any_header_length() {
        // Published format: 8 + header_len must be 8-aligned with
        // header_len *including* the padding, so external readers compute
        // data_start = 8 + header_len directly. Vary the header length
        // (via tensor-name length) to force both aligned and unaligned
        // unpadded headers.
        for name_len in 1..40usize {
            let name = format!("t{:0width$}", name_len, width = 3);
            let payload = vec![0u8; 8];
            let tensors = vec![TensorData {
                name: &name,
                dtype: TensorDType::F32,
                shape: &[2],
                bytes: &payload,
            }];
            let bytes = serialize(&tensors).unwrap();
            let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
            assert_eq!(
                (8 + header_len) % 8,
                0,
                "name_len {name_len}: 8 + header_len must be 8-aligned"
            );
            // The recorded length is the full header: JSON + padding.
            let header: serde_json::Value =
                serde_json::from_slice(&bytes[8..8 + header_len]).expect("header parses");
            assert!(header.is_object());
            // Data begins immediately after the header.
            let entry = header.as_object().unwrap().get(&name).unwrap();
            assert_eq!(entry["data_offsets"][0].as_u64(), Some(0));
            assert_eq!(entry["data_offsets"][1].as_u64(), Some(8));
            // Ember's own reader still round-trips.
            let views = deserialize(&bytes).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(tensor_f32(&bytes, &views[0].1).unwrap(), vec![0.0; 2]);
        }
    }

    #[test]
    fn legacy_unaligned_header_still_reads() {
        // Pre-fix bundles (header padding emitted *after* the header and
        // excluded from header_len) must keep loading. Build one by hand:
        // an unpadded header whose 8 + len is deliberately not 8-aligned.
        let mut header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while (8 + header.len()).is_multiple_of(8) {
            header.push(b' ');
        }
        let header_len = header.len() as u64; // excludes the following padding
        let data_start = (8 + header.len() + 7) & !7; // legacy: padding after header
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.resize(data_start, 0); // legacy zero padding
        bytes.extend_from_slice(&1.5f32.to_le_bytes()); // one f32 (shape [1])
        while bytes.len() % 8 != 0 {
            bytes.push(0);
        }
        let views = deserialize(&bytes).unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].0, "x");
        assert_eq!(tensor_f32(&bytes, &views[0].1).unwrap(), vec![1.5]);
    }

    #[test]
    fn f16_conversion() {
        let values = [0.5f32, -2.0, 3.25];
        let bytes = f32_to_f16_bytes(&values);
        let view = TensorView {
            dtype: TensorDType::F16,
            shape: vec![3],
            data_offsets: (0, bytes.len()),
        };
        let back = tensor_f32(&bytes, &view).unwrap();
        for (a, b) in values.iter().zip(back) {
            assert!((a - b).abs() < 1e-3);
        }
    }
}

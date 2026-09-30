//! Steering and direction interventions (`steer`, `ablate-projection`).
//!
//! A direction operation adds to (or removes from) the selected rows along
//! one vector per layer. The vector comes from the intervention's source:
//!
//! * `inline-vector`: the values in the spec;
//! * `vector-file`: a `.npy` or `.safetensors` file whose SHA-256 the spec
//!   pins, so the file is part of the experiment's semantic identity;
//! * `contrastive`: `mean(capture | positive prompts) - mean(capture |
//!   negative prompts)` at the intervention's site and layer, computed by the
//!   driver in the run before the inputs execute.
//!
//! Every resolved (file or contrastive) direction is written into the bundle
//! as an artifact (`artifacts/directions/<intervention>.safetensors` plus a
//! provenance record), so the payload and semantic hashes cover the exact
//! vectors that were applied, and `experiment verify` checks the record
//! against the tensors and the spec.
//!
//! Arithmetic is deterministic: norms and coefficients accumulate in `f64`
//! in index order, and a steering term that rounds to zero leaves the value
//! untouched, so `alpha = 0` is bit-identical to no intervention.

use crate::v05::hook::SemanticHookSite;
use crate::v05::intervention::{InterventionSource, InterventionSpec, SteerNormalization};
use crate::v05::manifest::{sha256_hex, SemanticManifest};
use crate::v05::safetensors::{self, TensorDType, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Direction artifact record schema.
pub const DIRECTION_SCHEMA_V1: &str = "ember.direction.v1";

/// Bundle directory that holds direction artifacts.
pub const DIRECTION_DIR: &str = "artifacts/directions";

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt()
}

/// Apply `steer` to one row in place.
pub fn steer_row(
    row: &mut [f32],
    direction: &[f32],
    alpha: f32,
    normalize: SteerNormalization,
) -> Result<(), String> {
    if row.len() != direction.len() {
        return Err(format!(
            "direction has {} values; the row has {}",
            direction.len(),
            row.len()
        ));
    }
    let scale = match normalize {
        SteerNormalization::None => 1.0,
        SteerNormalization::Unit | SteerNormalization::MatchResidualNorm => {
            let norm = l2_norm(direction);
            if norm == 0.0 || !norm.is_finite() {
                return Err(format!(
                    "cannot normalize a direction of norm {norm} ({normalize:?})"
                ));
            }
            let target = if normalize == SteerNormalization::Unit {
                1.0
            } else {
                l2_norm(row)
            };
            target / norm
        }
    };
    let coefficient = f64::from(alpha) * scale;
    for (value, &d) in row.iter_mut().zip(direction) {
        let delta = (coefficient * f64::from(d)) as f32;
        if delta != 0.0 {
            *value += delta;
        }
    }
    Ok(())
}

/// Apply `ablate-projection` to one row in place: remove the component
/// along `direction`.
pub fn ablate_projection_row(row: &mut [f32], direction: &[f32]) -> Result<(), String> {
    if row.len() != direction.len() {
        return Err(format!(
            "direction has {} values; the row has {}",
            direction.len(),
            row.len()
        ));
    }
    let norm_sq: f64 = direction.iter().map(|&d| f64::from(d) * f64::from(d)).sum();
    if norm_sq == 0.0 || !norm_sq.is_finite() {
        return Err("cannot project onto a zero or non-finite direction".into());
    }
    let dot: f64 = row
        .iter()
        .zip(direction)
        .map(|(&x, &d)| f64::from(x) * f64::from(d))
        .sum();
    let coefficient = dot / norm_sq;
    for (value, &d) in row.iter_mut().zip(direction) {
        let delta = (coefficient * f64::from(d)) as f32;
        if delta != 0.0 {
            *value -= delta;
        }
    }
    Ok(())
}

/// The mean of `rows` (`count x columns`, row-major) in `f64`.
pub fn mean_rows(rows: &[f32], columns: usize) -> Result<Vec<f64>, String> {
    if columns == 0 || rows.is_empty() || !rows.len().is_multiple_of(columns) {
        return Err(format!(
            "cannot average {} values as rows of {columns}",
            rows.len()
        ));
    }
    let count = rows.len() / columns;
    let mut sum = vec![0.0f64; columns];
    for row in rows.chunks_exact(columns) {
        for (acc, &value) in sum.iter_mut().zip(row) {
            *acc += f64::from(value);
        }
    }
    for value in &mut sum {
        *value /= count as f64;
    }
    Ok(sum)
}

/// `mean(positive) - mean(negative)` over per-prompt mean rows, in `f64`,
/// rounded once to `f32`.
pub fn mean_difference(positive: &[Vec<f64>], negative: &[Vec<f64>]) -> Result<Vec<f32>, String> {
    let columns = positive
        .first()
        .map(Vec::len)
        .ok_or("no positive rows to average")?;
    if negative.is_empty() {
        return Err("no negative rows to average".into());
    }
    if positive
        .iter()
        .chain(negative)
        .any(|row| row.len() != columns)
    {
        return Err("contrastive rows have different widths".into());
    }
    let mean = |rows: &[Vec<f64>]| {
        let mut sum = vec![0.0f64; columns];
        for row in rows {
            for (acc, value) in sum.iter_mut().zip(row) {
                *acc += value;
            }
        }
        sum.into_iter()
            .map(|value| value / rows.len() as f64)
            .collect::<Vec<f64>>()
    };
    let (a, b) = (mean(positive), mean(negative));
    Ok(a.iter().zip(&b).map(|(x, y)| (x - y) as f32).collect())
}

/// A parsed direction file: a `[rows, columns]` f32 matrix (a 1-D vector
/// is one row).
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionMatrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

impl DirectionMatrix {
    fn from_shape(shape: &[usize], values: Vec<f32>) -> Result<DirectionMatrix, String> {
        let (rows, columns) = match shape {
            [columns] => (1, *columns),
            [rows, columns] => (*rows, *columns),
            other => {
                return Err(format!(
                    "a direction must be 1-D [d] or 2-D [rows, d]; found shape {other:?}"
                ))
            }
        };
        if rows == 0 || columns == 0 || rows * columns != values.len() {
            return Err(format!(
                "direction shape {shape:?} does not match its {} values",
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("direction values must be finite".into());
        }
        Ok(DirectionMatrix {
            rows,
            columns,
            values,
        })
    }

    fn row(&self, index: usize) -> Vec<f32> {
        self.values[index * self.columns..(index + 1) * self.columns].to_vec()
    }
}

/// Parse a `.npy` array (format versions 1-3, little-endian `f4`/`f8`,
/// C order, 1-D or 2-D).
pub fn parse_npy(bytes: &[u8]) -> Result<DirectionMatrix, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not a .npy file (bad magic)".into());
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(".npy header is truncated".into());
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        other => return Err(format!("unsupported .npy format version {other}")),
    };
    let data_start = header_start
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .ok_or(".npy header is truncated")?;
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|_| ".npy header is not text")?;
    let field = |key: &str| -> Result<&str, String> {
        let at = header
            .find(&format!("'{key}'"))
            .ok_or_else(|| format!(".npy header has no '{key}'"))?;
        let rest = &header[at + key.len() + 2..];
        let colon = rest.find(':').ok_or(".npy header is malformed")?;
        Ok(rest[colon + 1..].trim_start())
    };
    let descr = field("descr")?;
    let quote = descr.chars().next().ok_or(".npy descr is empty")?;
    let descr = descr[1..]
        .split(quote)
        .next()
        .ok_or(".npy descr is malformed")?;
    let fortran = field("fortran_order")?;
    if !fortran.starts_with("False") {
        return Err(".npy arrays in Fortran order are not supported".into());
    }
    let shape_text = field("shape")?;
    let shape_text = shape_text
        .strip_prefix('(')
        .and_then(|text| text.split(')').next())
        .ok_or(".npy shape is malformed")?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().map_err(|_| ".npy shape is malformed"))
        .collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product();
    let data = &bytes[data_start..];
    let values: Vec<f32> = match descr {
        "<f4" => {
            if data.len() != count * 4 {
                return Err(format!(
                    ".npy data holds {} bytes; shape {shape:?} of f4 needs {}",
                    data.len(),
                    count * 4
                ));
            }
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
                .collect()
        }
        "<f8" => {
            if data.len() != count * 8 {
                return Err(format!(
                    ".npy data holds {} bytes; shape {shape:?} of f8 needs {}",
                    data.len(),
                    count * 8
                ));
            }
            data.chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")) as f32)
                .collect()
        }
        other => {
            return Err(format!(
                ".npy dtype '{other}' is not supported (use little-endian float32 '<f4' or \
                 float64 '<f8')"
            ))
        }
    };
    DirectionMatrix::from_shape(&shape, values)
}

/// Serialize a `.npy` (version 1.0, `<f4`) array; used by tests and the
/// probe tooling that writes directions.
pub fn write_npy(shape: &[usize], values: &[f32]) -> Vec<u8> {
    let shape_text = match shape {
        [one] => format!("({one},)"),
        many => format!(
            "({})",
            many.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    // Pad so the data starts on a 64-byte boundary, newline-terminated.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    for value in values {
        out.extend(value.to_le_bytes());
    }
    out
}

/// Parse a direction from a safetensors file: `tensor` names it, or the
/// file must hold exactly one tensor.
pub fn parse_safetensors_direction(
    bytes: &[u8],
    tensor: Option<&str>,
) -> Result<DirectionMatrix, String> {
    let tensors = safetensors::deserialize(bytes)?;
    let (name, view) = match tensor {
        Some(name) => tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .ok_or_else(|| format!("safetensors file has no tensor '{name}'"))?,
        None => match tensors.as_slice() {
            [single] => single,
            [] => return Err("safetensors file holds no tensor".into()),
            _ => {
                return Err(format!(
                    "safetensors file holds {} tensors; name one with `tensor`",
                    tensors.len()
                ))
            }
        },
    };
    let values = safetensors::tensor_f32(bytes, view)
        .map_err(|error| format!("tensor '{name}': {error}"))?;
    DirectionMatrix::from_shape(&view.shape, values)
}

/// Read, hash-check and parse a direction file.
pub fn read_direction_file(
    path: &Path,
    expected_sha256: &str,
    tensor: Option<&str>,
) -> Result<DirectionMatrix, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read direction file '{}': {error}", path.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != expected_sha256 {
        return Err(format!(
            "direction file '{}' hashes to {actual} but the spec pins {expected_sha256}",
            path.display()
        ));
    }
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "npy" => {
            if tensor.is_some() {
                return Err("`tensor` applies to safetensors files, not .npy".into());
            }
            parse_npy(&bytes)
        }
        "safetensors" => parse_safetensors_direction(&bytes, tensor),
        other => Err(format!(
            "direction file '{}' has extension '{other}'; use .npy or .safetensors",
            path.display()
        )),
    }
    .map_err(|error| format!("direction file '{}': {error}", path.display()))
}

/// Map a direction matrix onto the intervention's layers: one row applies
/// to every layer; `n_layers` rows give row `L` to layer `L` (per-layer
/// sites only). The width must equal `columns` (the site's width).
pub fn directions_for_layers(
    matrix: &DirectionMatrix,
    layers: &[usize],
    n_layers: usize,
    columns: usize,
    per_layer_site: bool,
) -> Result<BTreeMap<usize, Vec<f32>>, String> {
    if matrix.columns != columns {
        return Err(format!(
            "direction dimension mismatch: the direction has {} columns but the site has \
             {columns} (the model's width there)",
            matrix.columns
        ));
    }
    let mut out = BTreeMap::new();
    for &layer in layers {
        let row = if matrix.rows == 1 {
            matrix.row(0)
        } else if per_layer_site && matrix.rows == n_layers {
            matrix.row(layer)
        } else {
            return Err(format!(
                "direction has {} rows; expected 1 (every layer) or {n_layers} (one per layer)",
                matrix.rows
            ));
        };
        out.insert(layer, row);
    }
    Ok(out)
}

/// A direction resolved by the driver, written into the bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedDirection {
    pub intervention_id: String,
    pub site: SemanticHookSite,
    /// `vector-file` or `contrastive`.
    pub source_kind: String,
    /// Per-layer vectors (layer 0 for head sites).
    pub layers: BTreeMap<usize, Vec<f32>>,
}

/// Provenance record of one direction artifact
/// (`artifacts/directions/<id>.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionRecord {
    pub schema: String,
    pub intervention_id: String,
    pub site: SemanticHookSite,
    pub source_kind: String,
    /// The pinned file hash (vector-file sources).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_sha256: Option<String>,
    /// SHA-256 of each positive / negative prompt (contrastive sources).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub positive_prompt_sha256: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub negative_prompt_sha256: Vec<String>,
    pub tensor_file: String,
    pub layers: Vec<DirectionLayerRecord>,
}

/// One layer of a direction record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionLayerRecord {
    pub layer: usize,
    pub tensor: String,
    pub columns: usize,
    /// SHA-256 of the tensor's little-endian f32 bytes.
    pub checksum: String,
    /// Informational (not verified bit-for-bit).
    pub l2_norm: f64,
}

fn layer_tensor_name(layer: usize) -> String {
    format!("layer-{layer}")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn direction_paths(intervention_id: &str) -> (String, String) {
    (
        format!("{DIRECTION_DIR}/{intervention_id}.json"),
        format!("{DIRECTION_DIR}/{intervention_id}.safetensors"),
    )
}

/// The bundle files for resolved directions: a safetensors payload and a
/// provenance record per intervention.
pub fn direction_artifact_files(
    directions: &[ResolvedDirection],
    interventions: &[InterventionSpec],
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut files = BTreeMap::new();
    for direction in directions {
        let spec = interventions
            .iter()
            .find(|spec| spec.id == direction.intervention_id)
            .ok_or_else(|| {
                format!(
                    "direction for unknown intervention '{}'",
                    direction.intervention_id
                )
            })?;
        let (record_path, tensor_path) = direction_paths(&direction.intervention_id);
        let owned: Vec<(String, Vec<u8>, [usize; 1])> = direction
            .layers
            .iter()
            .map(|(layer, values)| (layer_tensor_name(*layer), f32_bytes(values), [values.len()]))
            .collect();
        let tensors: Vec<TensorData<'_>> = owned
            .iter()
            .map(|(name, bytes, shape)| TensorData {
                name,
                dtype: TensorDType::F32,
                shape,
                bytes,
            })
            .collect();
        let payload = safetensors::serialize(&tensors)?;
        let (file_sha256, positive, negative) = match &spec.source {
            Some(InterventionSource::VectorFile { sha256, .. }) => {
                (Some(sha256.clone()), Vec::new(), Vec::new())
            }
            Some(InterventionSource::Contrastive {
                positive, negative, ..
            }) => (
                None,
                positive.iter().map(|p| sha256_hex(p.as_bytes())).collect(),
                negative.iter().map(|p| sha256_hex(p.as_bytes())).collect(),
            ),
            _ => (None, Vec::new(), Vec::new()),
        };
        let record = DirectionRecord {
            schema: DIRECTION_SCHEMA_V1.to_string(),
            intervention_id: direction.intervention_id.clone(),
            site: direction.site,
            source_kind: direction.source_kind.clone(),
            file_sha256,
            positive_prompt_sha256: positive,
            negative_prompt_sha256: negative,
            tensor_file: tensor_path
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string(),
            layers: direction
                .layers
                .iter()
                .map(|(layer, values)| DirectionLayerRecord {
                    layer: *layer,
                    tensor: layer_tensor_name(*layer),
                    columns: values.len(),
                    checksum: sha256_hex(&f32_bytes(values)),
                    l2_norm: stable(l2_norm(values)),
                })
                .collect(),
        };
        let mut json = serde_json::to_value(&record).map_err(|error| error.to_string())?;
        crate::plan::sort_value_keys(&mut json);
        let mut bytes = serde_json::to_vec_pretty(&json).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        files.insert(record_path, bytes);
        files.insert(tensor_path, payload);
    }
    Ok(files)
}

/// Round to 9 significant digits so the value survives JSON unchanged.
fn stable(value: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    format!("{value:.8e}").parse().unwrap_or(value)
}

/// Read a direction artifact back: per-layer vectors by layer.
pub fn read_direction_tensors(bytes: &[u8]) -> Result<BTreeMap<usize, Vec<f32>>, String> {
    let mut out = BTreeMap::new();
    for (name, view) in safetensors::deserialize(bytes)? {
        let layer = name
            .strip_prefix("layer-")
            .and_then(|layer| layer.parse::<usize>().ok())
            .ok_or_else(|| format!("unexpected direction tensor '{name}'"))?;
        out.insert(layer, safetensors::tensor_f32(bytes, &view)?);
    }
    Ok(out)
}

/// Verify direction artifacts against the semantic manifest: every
/// intervention with a resolved-direction source has a record and a tensor
/// file that agree with each other and with the spec (pinned file hash,
/// prompt hashes, site, layers), and no other direction artifact exists.
/// `file` returns a listed bundle file's verified bytes.
pub fn verify_direction_artifacts(
    semantic: &SemanticManifest,
    listed: &[String],
    file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut expected_paths = std::collections::BTreeSet::new();
    let n_layers = semantic.model.layer_count;
    for spec in &semantic.interventions {
        let Some(source) = spec.source.as_ref().filter(|s| s.is_resolved_direction()) else {
            continue;
        };
        let (record_path, tensor_path) = direction_paths(&spec.id);
        expected_paths.insert(record_path.clone());
        expected_paths.insert(tensor_path.clone());
        let (Some(record_bytes), Some(tensor_bytes)) = (file(&record_path), file(&tensor_path))
        else {
            errors.push(format!("'{}': direction artifact missing", spec.id));
            continue;
        };
        let record: DirectionRecord = match serde_json::from_slice(&record_bytes) {
            Ok(record) => record,
            Err(error) => {
                errors.push(format!(
                    "'{}': malformed direction record: {error}",
                    spec.id
                ));
                continue;
            }
        };
        let tensors = match read_direction_tensors(&tensor_bytes) {
            Ok(tensors) => tensors,
            Err(error) => {
                errors.push(format!("'{}': {error}", spec.id));
                continue;
            }
        };
        if record.schema != DIRECTION_SCHEMA_V1
            || record.intervention_id != spec.id
            || record.site != spec.site
            || record.source_kind != source.kind_name()
        {
            errors.push(format!(
                "'{}': direction record does not describe this intervention",
                spec.id
            ));
        }
        match source {
            InterventionSource::VectorFile { sha256, .. } => {
                if record.file_sha256.as_deref() != Some(sha256.as_str()) {
                    errors.push(format!(
                        "'{}': recorded file hash differs from the spec's pin",
                        spec.id
                    ));
                }
            }
            InterventionSource::Contrastive {
                positive, negative, ..
            } => {
                let hashes = |prompts: &[String]| -> Vec<String> {
                    prompts.iter().map(|p| sha256_hex(p.as_bytes())).collect()
                };
                if record.positive_prompt_sha256 != hashes(positive)
                    || record.negative_prompt_sha256 != hashes(negative)
                {
                    errors.push(format!(
                        "'{}': recorded prompt hashes differ from the spec's prompts",
                        spec.id
                    ));
                }
            }
            _ => {}
        }
        let expected_layers: Result<Vec<usize>, String> = if spec.site.is_per_layer() {
            spec.layers.resolve(n_layers)
        } else {
            Ok(vec![0])
        };
        let record_layers: Vec<usize> = record.layers.iter().map(|layer| layer.layer).collect();
        let tensor_layers: Vec<usize> = tensors.keys().copied().collect();
        match expected_layers {
            Ok(layers) if layers == record_layers && layers == tensor_layers => {}
            Ok(layers) => errors.push(format!(
                "'{}': direction layers {record_layers:?} (tensors {tensor_layers:?}) differ \
                 from the intervention's layers {layers:?}",
                spec.id
            )),
            Err(error) => errors.push(format!("'{}': {error}", spec.id)),
        }
        for layer in &record.layers {
            match tensors.get(&layer.layer) {
                Some(values)
                    if values.len() == layer.columns
                        && sha256_hex(&f32_bytes(values)) == layer.checksum => {}
                _ => errors.push(format!(
                    "'{}': layer {} tensor does not match its recorded checksum",
                    spec.id, layer.layer
                )),
            }
        }
    }
    for path in listed {
        if path.starts_with(&format!("{DIRECTION_DIR}/")) && !expected_paths.contains(path) {
            errors.push(format!("unexpected direction artifact '{path}'"));
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steer_matches_the_closed_form_and_alpha_zero_is_identity() {
        let original = vec![1.0f32, -2.0, 0.5, -0.0];
        let direction = vec![3.0f32, 0.0, 4.0, 0.0];
        let mut row = original.clone();
        steer_row(&mut row, &direction, 0.0, SteerNormalization::None).unwrap();
        assert_eq!(
            row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            original.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut row = original.clone();
        steer_row(&mut row, &direction, 2.0, SteerNormalization::None).unwrap();
        assert_eq!(row, vec![7.0, -2.0, 8.5, -0.0]);
        let mut row = original.clone();
        steer_row(&mut row, &direction, 2.0, SteerNormalization::Unit).unwrap();
        assert_eq!(row, vec![1.0 + 1.2, -2.0, 0.5 + 1.6, -0.0]);
        let mut row = vec![3.0f32, 4.0, 0.0, 0.0];
        steer_row(
            &mut row,
            &direction,
            0.5,
            SteerNormalization::MatchResidualNorm,
        )
        .unwrap();
        // |row| = 5, unit d = (0.6, 0, 0.8, 0): += 0.5 * 5 * d.
        assert_eq!(row, vec![3.0 + 1.5, 4.0, 2.0, 0.0]);
        let mut row = original.clone();
        assert!(steer_row(&mut row, &[0.0; 4], 1.0, SteerNormalization::Unit).is_err());
        assert!(steer_row(&mut row, &[1.0; 3], 1.0, SteerNormalization::None).is_err());
    }

    #[test]
    fn ablation_removes_exactly_the_projection() {
        let mut row = vec![3.0f32, 4.0, 5.0];
        ablate_projection_row(&mut row, &[0.0, 2.0, 0.0]).unwrap();
        assert_eq!(row, vec![3.0, 0.0, 5.0]);
        let mut row = vec![1.0f32, 1.0];
        ablate_projection_row(&mut row, &[1.0, -1.0]).unwrap();
        assert_eq!(row, vec![1.0, 1.0]);
        assert!(ablate_projection_row(&mut row, &[0.0, 0.0]).is_err());
    }

    #[test]
    fn npy_round_trips_and_rejects_bad_input() {
        let bytes = write_npy(&[4], &[1.0, 2.0, 3.0, 4.5]);
        let parsed = parse_npy(&bytes).unwrap();
        assert_eq!((parsed.rows, parsed.columns), (1, 4));
        assert_eq!(parsed.values, vec![1.0, 2.0, 3.0, 4.5]);
        let bytes = write_npy(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parsed = parse_npy(&bytes).unwrap();
        assert_eq!((parsed.rows, parsed.columns), (2, 3));
        assert!(parse_npy(b"not numpy").is_err());
        let mut truncated = write_npy(&[4], &[1.0; 4]);
        truncated.pop();
        assert!(parse_npy(&truncated).is_err());
        let text = String::from_utf8_lossy(&write_npy(&[4], &[1.0; 4])).replace("<f4", "<i4");
        assert!(parse_npy(text.as_bytes()).is_err());
    }

    #[test]
    fn layer_mapping_checks_dimensions() {
        let one = DirectionMatrix::from_shape(&[3], vec![1.0, 2.0, 3.0]).unwrap();
        let mapped = directions_for_layers(&one, &[1, 2], 4, 3, true).unwrap();
        assert_eq!(mapped[&1], mapped[&2]);
        let error = directions_for_layers(&one, &[1], 4, 5, true).unwrap_err();
        assert!(error.contains("dimension mismatch"), "{error}");
        let per_layer = DirectionMatrix::from_shape(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mapped = directions_for_layers(&per_layer, &[1], 2, 2, true).unwrap();
        assert_eq!(mapped[&1], vec![3.0, 4.0]);
        assert!(directions_for_layers(&per_layer, &[1], 3, 2, true).is_err());
    }

    #[test]
    fn contrastive_mean_difference() {
        let positive = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let negative = vec![vec![0.0, 1.0]];
        assert_eq!(
            mean_difference(&positive, &negative).unwrap(),
            vec![2.0, 2.0]
        );
        assert_eq!(mean_rows(&[1.0, 2.0, 3.0, 6.0], 2).unwrap(), vec![2.0, 4.0]);
    }
}

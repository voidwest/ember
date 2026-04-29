use crate::tensor::CpuTensor;
use anyhow::{Context, Ok, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, CpuTensor>> {
    let data =
        std::fs::read(&path).with_context(|| format!("failed to read {:?}", path.as_ref()))?;

    let tensors = SafeTensors::deserialize(&data).context("invalid safetensors file")?;

    let mut map = HashMap::new();

    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();

        match view.dtype() {
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                map.insert(name.to_string(), CpuTensor::from_data(shape, floats));
            }
            other => {
                anyhow::bail!("unsupported dtype {:?} for tensor {}", other, name);
            }
        }
    }

    Ok(map)
}

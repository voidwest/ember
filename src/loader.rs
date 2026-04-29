use crate::tensor::CpuTensor;
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, CpuTensor>> {}

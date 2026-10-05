//! Per-tensor execution inventory: what the loader decided for every
//! K-family tensor (resident form, strategy, kernel, CPU features, fallback
//! reason) and the run-level summary. Recorded in capture manifests and
//! bundles; `ember::artifact` re-exports it.

use serde::{Deserialize, Serialize};

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

/// One operation-specific kernel use of a resident tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOperationExecution {
    /// Semantic use such as `embedding-lookup`, `linear-matmul`, or
    /// `lm-head-matmul`.
    pub operation: String,
    pub kernel: String,
    pub cpu_features: String,
    /// Transient workspace bytes per activation row for this operation.
    pub workspace_bytes: usize,
}

/// One per-tensor K-family execution record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorExecution {
    pub name: String,
    /// GGUF dtype name (`ggml_dtype_name`), e.g. "q4_k".
    pub gguf_dtype: String,
    pub gguf_dtype_code: u32,
    /// Resident representation: "compressed" or "f32".
    pub resident: String,
    /// Execution strategy: "eager-f32", "compressed-scalar", or
    /// "compressed-x86".
    pub strategy: String,
    /// Selected kernel: "eager-f32-dequant", "q4-k-q8-k-scalar", "q6-k-q8-k-scalar",
    /// "q4-k-q8-k-avx2", "q6-k-q8-k-avx2".
    pub kernel: String,
    /// Numerical/runtime kernel ABI revision. Additive for older artifact
    /// readers; zero means the pre-revision historical inventory.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub kernel_revision: u32,
    /// CPU feature requirement for this kernel ("none" for scalar/eager).
    pub cpu_features: String,
    /// Operation-specific routing. This disambiguates embedding row lookup
    /// from tied LM-head matmul when both use the same resident tensor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub operations: Vec<TensorOperationExecution>,
    /// Why the requested strategy was not honored, if it was not.
    pub fallback_reason: Option<String>,
    /// Aggregate thread-local workspace bytes per activation row. Multi-row prefill
    /// scales this value by its runtime row count, and the reusable vector may
    /// retain the peak capacity for the life of the worker thread.
    pub workspace_bytes: usize,
}

/// Per-dtype residency totals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtypeExecutionSummary {
    pub dtype: String,
    pub tensor_count: usize,
    /// Resident compressed bytes (packed path; zero for eager tensors).
    pub compressed_bytes: u64,
    /// Resident f32 bytes (eager path; zero for compressed tensors).
    pub expanded_bytes: u64,
}

/// Model-level execution/residency summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    pub tensor_count: usize,
    pub fallback_count: usize,
    /// Total resident compressed bytes across compressed-path tensors.
    pub compressed_bytes: u64,
    /// Total resident f32 bytes across eager-path tensors.
    pub expanded_bytes: u64,
    pub per_dtype: Vec<DtypeExecutionSummary>,
}

/// v0.3 execution provenance: the per-tensor K-family decisions made at
/// load time plus model-level residency totals. Additive field on the
/// manifest; older artifacts simply lack it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInventory {
    /// Requested `--k-strategy` name.
    pub requested_strategy: String,
    pub tensors: Vec<TensorExecution>,
    pub summary: ExecutionSummary,
}

impl ExecutionInventory {
    /// Build the inventory from the loader's recorded per-tensor K
    /// decisions and original GGUF metadata.
    pub fn from_loader(loader: &crate::loader::GgufLoader) -> Self {
        use crate::quant_k::{KExecution, KQuantDtype};
        use std::collections::BTreeMap;

        let mut tensors = Vec::new();
        let mut per_dtype = BTreeMap::<String, DtypeExecutionSummary>::new();
        let mut fallback_count = 0usize;
        let mut compressed_bytes = 0u64;
        let mut expanded_bytes = 0u64;

        let mut names: Vec<&String> = loader.k_decisions.keys().collect();
        names.sort();
        for name in names {
            let decision = &loader.k_decisions[name];
            let dtype_name = crate::loader::ggml_dtype_name(decision.gguf_dtype)
                .unwrap_or("unknown")
                .to_string();
            let element_count = loader.tensor_meta.get(name).and_then(|meta| {
                meta.dims
                    .iter()
                    .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            });
            let byte_len = element_count.and_then(|count| {
                crate::loader::gguf_dtype_byte_len(decision.gguf_dtype, count).ok()
            });

            // Per-row transient Q8_K workspace. GGUF linear dims are
            // [in_features, out_features], with the first dimension contiguous.
            let q8_k_workspace_bytes = loader
                .tensor_meta
                .get(name)
                .and_then(|meta| meta.dims.first().copied())
                .map_or(0, |input_features| {
                    (input_features / crate::quant_k::QK_K)
                        * crate::k_quant_matmul::Q8_K_BLOCK_BYTES
                });

            let (resident, strategy, kernel, cpu_features, workspace_bytes) = match decision
                .execution
            {
                KExecution::EagerF32 => ("f32", "eager-f32", "eager-f32-dequant", "none", 0usize),
                KExecution::CompressedScalar => match KQuantDtype::from_gguf(decision.gguf_dtype) {
                    Some(KQuantDtype::Q4K) => (
                        "compressed",
                        "compressed-scalar",
                        "q4-k-q8-k-scalar",
                        "none",
                        q8_k_workspace_bytes,
                    ),
                    Some(KQuantDtype::Q6K) => (
                        "compressed",
                        "compressed-scalar",
                        "q6-k-q8-k-scalar",
                        "none",
                        q8_k_workspace_bytes,
                    ),
                    None => ("f32", "eager-f32", "eager-f32-dequant", "none", 0),
                },
                KExecution::CompressedX86 => match KQuantDtype::from_gguf(decision.gguf_dtype) {
                    Some(KQuantDtype::Q4K) => (
                        "compressed",
                        "compressed-x86",
                        "q4-k-q8-k-avx2",
                        "avx2+fma+f16c+ssse3",
                        q8_k_workspace_bytes,
                    ),
                    Some(KQuantDtype::Q6K) => (
                        "compressed",
                        "compressed-x86",
                        "q6-k-q8-k-avx2",
                        "avx2+fma+f16c+ssse3",
                        q8_k_workspace_bytes,
                    ),
                    None => ("f32", "eager-f32", "eager-f32-dequant", "none", 0),
                },
                KExecution::CompressedArm => match KQuantDtype::from_gguf(decision.gguf_dtype) {
                    Some(KQuantDtype::Q4K) => (
                        "compressed",
                        "compressed-arm",
                        "q4-k-q8-k-neon-dotprod",
                        "neon+dotprod",
                        q8_k_workspace_bytes,
                    ),
                    Some(KQuantDtype::Q6K) => (
                        "compressed",
                        "compressed-arm",
                        "q6-k-q8-k-neon-dotprod",
                        "neon+dotprod",
                        q8_k_workspace_bytes,
                    ),
                    None => ("f32", "eager-f32", "eager-f32-dequant", "none", 0),
                },
            };

            let matmul = TensorOperationExecution {
                operation: if name.as_str() == "output.weight" {
                    "lm-head-matmul"
                } else {
                    "linear-matmul"
                }
                .to_string(),
                kernel: kernel.to_string(),
                cpu_features: cpu_features.to_string(),
                workspace_bytes,
            };
            let (kernel, cpu_features, workspace_bytes, operations) =
                if name.as_str() == "token_embd.weight" {
                    let row_kernel = match decision.execution {
                        KExecution::EagerF32 => "embedding-f32-row",
                        KExecution::CompressedScalar
                        | KExecution::CompressedX86
                        | KExecution::CompressedArm => {
                            match KQuantDtype::from_gguf(decision.gguf_dtype) {
                                Some(KQuantDtype::Q4K) => "embedding-q4-k-row-dequant",
                                Some(KQuantDtype::Q6K) => "embedding-q6-k-row-dequant",
                                None => "embedding-f32-row",
                            }
                        }
                    };
                    let embedding = TensorOperationExecution {
                        operation: "embedding-lookup".to_string(),
                        kernel: row_kernel.to_string(),
                        cpu_features: "none".to_string(),
                        workspace_bytes: 0,
                    };
                    if loader.tensors.contains_key("output.weight") {
                        (
                            row_kernel.to_string(),
                            "none".to_string(),
                            0,
                            vec![embedding],
                        )
                    } else {
                        let mut tied_matmul = matmul;
                        tied_matmul.operation = "lm-head-matmul".to_string();
                        (
                            "multiple-see-operations".to_string(),
                            tied_matmul.cpu_features.clone(),
                            tied_matmul.workspace_bytes,
                            vec![embedding, tied_matmul],
                        )
                    }
                } else {
                    (
                        matmul.kernel.clone(),
                        matmul.cpu_features.clone(),
                        matmul.workspace_bytes,
                        vec![matmul],
                    )
                };

            if decision.fallback_reason.is_some() {
                fallback_count += 1;
            }
            let compressed = byte_len.unwrap_or(0) as u64;
            let expanded = element_count.unwrap_or(0) as u64 * 4;
            match decision.execution {
                KExecution::EagerF32 => {
                    expanded_bytes += expanded;
                    let entry = per_dtype.entry(dtype_name.clone()).or_insert_with(|| {
                        DtypeExecutionSummary {
                            dtype: dtype_name.clone(),
                            tensor_count: 0,
                            compressed_bytes: 0,
                            expanded_bytes: 0,
                        }
                    });
                    entry.tensor_count += 1;
                    entry.expanded_bytes += expanded;
                }
                KExecution::CompressedScalar
                | KExecution::CompressedX86
                | KExecution::CompressedArm => {
                    compressed_bytes += compressed;
                    let entry = per_dtype.entry(dtype_name.clone()).or_insert_with(|| {
                        DtypeExecutionSummary {
                            dtype: dtype_name.clone(),
                            tensor_count: 0,
                            compressed_bytes: 0,
                            expanded_bytes: 0,
                        }
                    });
                    entry.tensor_count += 1;
                    entry.compressed_bytes += compressed;
                }
            }

            tensors.push(TensorExecution {
                name: name.clone(),
                gguf_dtype: dtype_name,
                gguf_dtype_code: decision.gguf_dtype,
                resident: resident.to_string(),
                strategy: strategy.to_string(),
                kernel,
                kernel_revision: crate::plan::PLAN_KERNEL_REVISION,
                cpu_features,
                operations,
                fallback_reason: decision.fallback_reason.clone(),
                workspace_bytes,
            });
        }

        let tensor_count = tensors.len();
        Self {
            requested_strategy: loader.k_strategy.name().to_string(),
            tensors,
            summary: ExecutionSummary {
                tensor_count,
                fallback_count,
                compressed_bytes,
                expanded_bytes,
                per_dtype: per_dtype.into_values().collect(),
            },
        }
    }
}

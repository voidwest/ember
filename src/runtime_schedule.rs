//! Runtime-only decode schedule derived from an [`ExecutionPlan`].
//!
//! The schedule answers "which kernel and how many workers would each matvec
//! use on this host, right now" without entering the serialized plan:
//! thread counts, detected ISA tiers and cache/topology state are host
//! properties, and the plan hash is the v0.5 semantic identity, so
//! host-dependent fields must not change it. This module is a derived view
//! for `ember inspect plan` and diagnostics; the kernels stay authoritative
//! for their own dispatch (the decision predicates are shared with them).

use crate::plan::{ExecutionPlan, KernelId, TensorRecord};
use serde::Serialize;

/// Host state that influences decode dispatch but must not be serialized
/// into the plan.
#[derive(Debug, Clone, Serialize)]
pub struct HostSnapshot {
    pub arch: &'static str,
    /// Features recorded when the plan was built (`plan.cpu.features`).
    pub detected_features: Vec<String>,
    /// Features the plan's selected kernels require.
    pub required_features: Vec<String>,
    pub rayon_threads: usize,
    pub available_parallelism: usize,
    /// Compressed-resident K-quant AVX2/FMA/F16C/SSSE3 kernels available.
    pub k_quant_x86: bool,
    /// Opt-in AVX-512 K-quant dot tier (`EMBER_K_AVX512=1`).
    pub k_avx512_opt_in: bool,
    /// 16-output packed Q8_0 decode layout available.
    pub packed_q8_vnni: bool,
    /// Interleaved Q8_0 lm-head layout available.
    pub interleaved_q8: bool,
}

impl HostSnapshot {
    pub fn detect(plan: &ExecutionPlan) -> Self {
        Self {
            arch: std::env::consts::ARCH,
            detected_features: plan.cpu.features.clone(),
            required_features: plan.cpu.required.clone(),
            rayon_threads: rayon::current_num_threads(),
            available_parallelism: std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1),
            k_quant_x86: crate::k_quant_matmul::x86_k_supported(),
            k_avx512_opt_in: crate::k_quant_matmul::k_avx512_opt_in(),
            packed_q8_vnni: crate::simd::packed_q8_0_vnni_supported(),
            interleaved_q8: crate::simd::interleaved_q8_0_supported(),
        }
    }
}

/// One matrix weight's kernel/thread decision.
#[derive(Debug, Clone, Serialize)]
pub struct ScheduledMatvec {
    pub tensor: String,
    /// Layer index parsed from `blk.<n>.*` names, if any.
    pub layer: Option<usize>,
    /// `q` | `k` | `v` | `o` | `gate` | `up` | `down` | `lm_head` | `other`.
    pub operator: &'static str,
    pub dtype: String,
    pub kernel: String,
    pub out_features: usize,
    pub in_features: usize,
    pub macs: u64,
    pub weight_bytes: usize,
    /// `mmap` when the weight reads directly from the file mapping.
    pub storage: &'static str,
    pub parallel_requested: bool,
    /// `serial` | `row-parallel-rayon` | `column-parallel-rayon`.
    pub scheduled: &'static str,
}

/// Derived per-tensor decode schedule for the current host.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSchedule {
    pub host: HostSnapshot,
    /// The plan's single thread-strategy string (`serial` |
    /// `column-parallel-rayon`).
    pub thread_strategy: String,
    pub matvecs: Vec<ScheduledMatvec>,
    pub parallel_matvecs: usize,
    pub serial_matvecs: usize,
    pub total_macs: u64,
    pub total_weight_bytes: u64,
    /// Decode scratch arena size from the plan.
    pub scratch_bytes: usize,
    /// KV-cache bytes appended per decoded token across all layers.
    pub kv_bytes_per_token: u64,
}

impl RuntimeSchedule {
    pub fn from_plan(plan: &ExecutionPlan) -> Self {
        let host = HostSnapshot::detect(plan);
        let requested = plan.dispatch.thread_strategy == "column-parallel-rayon";
        let mut matvecs: Vec<ScheduledMatvec> = plan
            .tensor_table
            .iter()
            .filter(|record| is_matvec_record(record))
            .map(|record| schedule_matvec(record, requested, host.rayon_threads))
            .collect();
        matvecs.sort_by(|a, b| {
            (a.layer.unwrap_or(usize::MAX), a.tensor.as_str())
                .cmp(&(b.layer.unwrap_or(usize::MAX), b.tensor.as_str()))
        });
        let parallel_matvecs = matvecs
            .iter()
            .filter(|entry| entry.scheduled != "serial")
            .count();
        let total_macs = matvecs.iter().map(|entry| entry.macs).sum();
        let total_weight_bytes = matvecs.iter().map(|entry| entry.weight_bytes as u64).sum();
        let kv_bytes_per_token = kv_bytes_per_token(plan);
        Self {
            host,
            thread_strategy: plan.dispatch.thread_strategy.clone(),
            serial_matvecs: matvecs.len() - parallel_matvecs,
            parallel_matvecs,
            total_macs,
            total_weight_bytes,
            scratch_bytes: plan.scratch.total_bytes,
            kv_bytes_per_token,
            matvecs,
        }
    }

    pub fn to_summary_text(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let host = &self.host;
        let _ = write!(
            out,
            "runtime schedule (derived; host state is not part of the plan hash)\n\
             \x20 arch {} | rayon threads {} | available parallelism {}\n\
             \x20 features: {} (required: {})\n\
             \x20 tiers: k-quant x86 {} | avx-512 k-dot opt-in {} | packed q8 {} | interleaved q8 {}\n\
             \x20 thread strategy: {} | matvecs: {} (parallel {}, serial {})\n\
             \x20 weight bytes streamed per token: {} | macs per token: {}\n\
             \x20 scratch arena: {} bytes | kv bytes per token: {}\n",
            host.arch,
            host.rayon_threads,
            host.available_parallelism,
            host.detected_features.join(", "),
            host.required_features.join("+"),
            yesno(host.k_quant_x86),
            yesno(host.k_avx512_opt_in),
            yesno(host.packed_q8_vnni),
            yesno(host.interleaved_q8),
            self.thread_strategy,
            self.matvecs.len(),
            self.parallel_matvecs,
            self.serial_matvecs,
            self.total_weight_bytes,
            self.total_macs,
            self.scratch_bytes,
            self.kv_bytes_per_token,
        );
        for entry in &self.matvecs {
            let _ = writeln!(
                out,
                "  {:<40} {:<8} {}x{} {:<22} {:>10} B  {:<6} {}",
                entry.tensor,
                entry.operator,
                entry.out_features,
                entry.in_features,
                entry.kernel,
                entry.weight_bytes,
                entry.storage,
                entry.scheduled,
            );
        }
        out
    }
}

fn yesno(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn is_matvec_record(record: &TensorRecord) -> bool {
    record.shape.len() == 2
        && matches!(
            record.kernel,
            KernelId::EagerF32
                | KernelId::Q8Packed
                | KernelId::KQuantScalarQ4K
                | KernelId::KQuantScalarQ6K
                | KernelId::KQuantAvx2Q4K
                | KernelId::KQuantAvx2Q6K
        )
}

fn schedule_matvec(record: &TensorRecord, requested: bool, threads: usize) -> ScheduledMatvec {
    let out_features = record.shape.first().copied().unwrap_or(0);
    let in_features = record.shape.get(1).copied().unwrap_or(0);
    let scheduled = match record.kernel {
        KernelId::Q8Packed => {
            if crate::simd::q8_decode_uses_row_parallel(out_features, in_features) {
                "row-parallel-rayon"
            } else {
                "serial"
            }
        }
        KernelId::KQuantScalarQ4K
        | KernelId::KQuantScalarQ6K
        | KernelId::KQuantAvx2Q4K
        | KernelId::KQuantAvx2Q6K => {
            if crate::k_quant_matmul::parallel_for_shape(
                1,
                in_features,
                out_features,
                threads,
                requested,
            ) {
                "column-parallel-rayon"
            } else {
                "serial"
            }
        }
        _ => "serial",
    };
    ScheduledMatvec {
        tensor: record.name.clone(),
        layer: layer_of(&record.name),
        operator: operator_of(&record.name),
        dtype: record.gguf_dtype.clone(),
        kernel: record.kernel.name().to_string(),
        out_features,
        in_features,
        macs: (out_features as u64).saturating_mul(in_features as u64),
        weight_bytes: record.resident_bytes,
        storage: if record.mmap { "mmap" } else { "resident" },
        parallel_requested: requested,
        scheduled,
    }
}

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|index| index.parse().ok())
}

fn operator_of(name: &str) -> &'static str {
    if name.contains("attn_q") {
        "q"
    } else if name.contains("attn_k") {
        "k"
    } else if name.contains("attn_v") {
        "v"
    } else if name.contains("attn_output") {
        "o"
    } else if name.contains("ffn_gate") {
        "gate"
    } else if name.contains("ffn_up") {
        "up"
    } else if name.contains("ffn_down") {
        "down"
    } else if name.starts_with("output.weight") {
        "lm_head"
    } else {
        "other"
    }
}

fn kv_bytes_per_token(plan: &ExecutionPlan) -> u64 {
    let element_bytes = match plan.kv.precision.as_str() {
        "f16" | "bf16" => 2,
        _ => 4,
    };
    let layers = plan.layers.len() as u64;
    (layers * plan.kv.n_kv_heads as u64)
        .saturating_mul(plan.kv.head_dim as u64)
        .saturating_mul(element_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str, shape: [usize; 2], kernel: KernelId) -> TensorRecord {
        TensorRecord {
            id: 0,
            name: name.to_string(),
            shape: shape.to_vec(),
            gguf_dtype: "q4_k".to_string(),
            execution: "compressed_x86".to_string(),
            kernel,
            resident_bytes: shape[0] * shape[1] / 2,
            mmap: true,
        }
    }

    #[test]
    fn operator_and_layer_classification() {
        assert_eq!(operator_of("blk.3.attn_q.weight"), "q");
        assert_eq!(operator_of("blk.3.attn_output.weight"), "o");
        assert_eq!(operator_of("blk.3.ffn_gate.weight"), "gate");
        assert_eq!(operator_of("blk.3.ffn_down.weight"), "down");
        assert_eq!(
            operator_of("output.weight (tied to token_embd.weight)"),
            "lm_head"
        );
        assert_eq!(layer_of("blk.11.attn_k.weight"), Some(11));
        assert_eq!(layer_of("output.weight"), None);
    }

    #[test]
    fn k_quant_decisions_match_the_shared_predicate() {
        // Large projections parallelize; tiny ones and single-thread runs do not.
        let big = schedule_matvec(
            &record(
                "blk.0.ffn_down.weight",
                [2048, 8192],
                KernelId::KQuantAvx2Q4K,
            ),
            true,
            4,
        );
        assert_eq!(big.scheduled, "column-parallel-rayon");
        let small = schedule_matvec(
            &record("blk.0.tiny.weight", [256, 256], KernelId::KQuantAvx2Q4K),
            true,
            4,
        );
        assert_eq!(small.scheduled, "serial");
        let single = schedule_matvec(
            &record(
                "blk.0.ffn_down.weight",
                [2048, 8192],
                KernelId::KQuantAvx2Q4K,
            ),
            true,
            1,
        );
        assert_eq!(single.scheduled, "serial");
        let unrequested = schedule_matvec(
            &record(
                "blk.0.ffn_down.weight",
                [2048, 8192],
                KernelId::KQuantAvx2Q4K,
            ),
            false,
            4,
        );
        assert_eq!(unrequested.scheduled, "serial");
        assert_eq!(big.macs, 2048 * 8192);
    }

    #[test]
    fn f32_records_are_serial() {
        let entry = schedule_matvec(
            &record("blk.0.dense.weight", [2048, 2048], KernelId::EagerF32),
            true,
            8,
        );
        assert_eq!(entry.scheduled, "serial");
    }
}

//! Cross-machine reproducibility report.
//!
//! A bundle's semantic identity deliberately excludes the host: the same
//! experiment on two machines has the same spec, model, plan and inputs.
//! Numbers can still differ between hosts, because SIMD tiers accumulate
//! dot products and norms in different orders. This module records, in the
//! bundle's `runtime.json` (never in the semantic identity), what decided
//! those orders: the execution tier per op family, the CPU model and
//! features, the worker count and rayon configuration, the dispatch knobs in
//! the environment, and the build. [`explain_host_differences`] turns two
//! such records into a list of differences with a likely explanation, which
//! `experiment compare` and `experiment reproduce` print when results differ.

use crate::plan::ExecutionPlan;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Schema tag of the `host_profile` object in `runtime.json`.
pub const HOST_PROFILE_SCHEMA: &str = "ember.host-profile.v1";

/// Key of the host profile inside `runtime.json`.
pub const RUNTIME_KEY: &str = "host_profile";

/// Environment variables that change kernel dispatch or scheduling.
const DISPATCH_ENV: [&str; 8] = [
    "EMBER_K_AVX512",
    "EMBER_LLAMA_PACKED_Q8",
    "EMBER_PACKED_CACHE",
    "EMBER_PRESPLIT",
    "EMBER_FUSED_GREEDY",
    "EMBER_PARALLEL_REPACK",
    "EMBER_VISION_FAST_EXP",
    "RAYON_NUM_THREADS",
];

/// What decided floating-point reduction order on the host that ran a
/// bundle. Recorded in `runtime.json` under [`RUNTIME_KEY`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostProfile {
    pub schema: String,
    pub cpu: CpuProfile,
    /// Execution tier the kernels dispatch to on this host, per op family
    /// (`q8_0_matvec`, `k_quant_matvec`, `elementwise`, `f32_matmul`).
    pub op_tiers: BTreeMap<String, String>,
    /// Kernels the execution plan selected, per matvec operator
    /// (`q`, `k`, ..., `lm_head`).
    pub plan_kernels: BTreeMap<String, Vec<String>>,
    /// Plan kernel fallbacks, by reason, with their tensor counts.
    pub kernel_fallbacks: BTreeMap<String, usize>,
    pub threads: ThreadProfile,
    /// Dispatch-affecting environment variables that were set.
    pub env: BTreeMap<String, String>,
    pub build: BuildProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuProfile {
    pub arch: String,
    pub os: String,
    pub model: String,
    /// Runtime-detected ISA features relevant to Ember's kernels.
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreadProfile {
    /// Worker threads the run used.
    pub requested: usize,
    /// Threads in the rayon pool the run executed in.
    pub rayon_threads: usize,
    pub available_parallelism: usize,
    /// The plan's thread strategy (`serial` | `column-parallel-rayon`).
    pub strategy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildProfile {
    pub ember_version: String,
    pub git_commit: String,
    /// `debug` or `release` (from `debug_assertions`).
    pub profile: String,
    pub opt_level: String,
    pub rustc: String,
    pub target: String,
    /// SIMD features enabled at compile time (`-C target-feature` /
    /// `target-cpu`), as opposed to detected at runtime.
    pub compiled_target_features: Vec<String>,
}

impl HostProfile {
    /// Detect the profile of this process for a run of `plan` with
    /// `threads` workers. Call it inside the run's rayon pool.
    pub fn detect(plan: &ExecutionPlan, threads: usize) -> Self {
        let schedule = crate::runtime_schedule::RuntimeSchedule::from_plan(plan);
        let mut plan_kernels: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for matvec in &schedule.matvecs {
            let kernels = plan_kernels.entry(matvec.operator.to_string()).or_default();
            if !kernels.contains(&matvec.kernel) {
                kernels.push(matvec.kernel.clone());
            }
        }
        for kernels in plan_kernels.values_mut() {
            kernels.sort();
        }
        let mut kernel_fallbacks = BTreeMap::new();
        for entry in &plan.dispatch.kernel_per_tensor {
            if let Some(reason) = &entry.fallback {
                *kernel_fallbacks.entry(reason.clone()).or_insert(0) += 1;
            }
        }
        let env = DISPATCH_ENV
            .iter()
            .filter_map(|name| {
                std::env::var(name)
                    .ok()
                    .map(|value| (name.to_string(), value))
            })
            .collect();
        Self {
            schema: HOST_PROFILE_SCHEMA.to_string(),
            cpu: CpuProfile {
                arch: std::env::consts::ARCH.to_string(),
                os: std::env::consts::OS.to_string(),
                model: cpu_model(),
                features: detected_features(),
            },
            op_tiers: op_tiers(),
            plan_kernels,
            kernel_fallbacks,
            threads: ThreadProfile {
                requested: threads,
                rayon_threads: rayon::current_num_threads(),
                available_parallelism: std::thread::available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(1),
                strategy: plan.dispatch.thread_strategy.clone(),
            },
            env,
            build: BuildProfile {
                ember_version: env!("CARGO_PKG_VERSION").to_string(),
                git_commit: crate::build_info::GIT_COMMIT
                    .unwrap_or("unknown")
                    .to_string(),
                profile: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                }
                .to_string(),
                opt_level: crate::build_info::OPT_LEVEL
                    .unwrap_or("unknown")
                    .to_string(),
                rustc: crate::build_info::RUSTC_VERSION
                    .unwrap_or("unknown")
                    .to_string(),
                target: crate::build_info::TARGET.unwrap_or("unknown").to_string(),
                compiled_target_features: compiled_target_features(),
            },
        }
    }

    /// The profile as a `runtime.json` value.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// The CPU's model name. It cannot change while the process runs, so it is
/// looked up once (on macOS that spawns `sysctl`) and reused by every
/// bundle a sweep or GUI session writes.
fn cpu_model() -> String {
    static MODEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODEL.get_or_init(detect_cpu_model).clone()
}

fn detect_cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            && output.status.success()
        {
            let model = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !model.is_empty() {
                return model;
            }
        }
    }
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        // x86 names the model; ARM Linux usually only has implementer/part.
        for key in ["model name", "Hardware", "CPU part"] {
            if let Some(value) = cpuinfo.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_string())
            }) {
                return value;
            }
        }
    }
    "unknown".to_string()
}

fn detected_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut features: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if is_x86_feature_detected!($name) { features.push($name); })*
            };
        }
        probe!(
            "sse2",
            "ssse3",
            "sse4.1",
            "avx",
            "avx2",
            "fma",
            "f16c",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512vnni"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_aarch64_feature_detected!($name) { features.push($name); })*
            };
        }
        probe!("neon", "dotprod", "fp16", "i8mm", "sve");
    }
    features.into_iter().map(str::to_string).collect()
}

fn compiled_target_features() -> Vec<String> {
    let mut features = Vec::new();
    macro_rules! compiled {
        ($($name:tt),*) => {
            $(if cfg!(target_feature = $name) { features.push($name.to_string()); })*
        };
    }
    compiled!(
        "sse2",
        "avx",
        "avx2",
        "fma",
        "f16c",
        "avx512f",
        "avx512vnni",
        "neon",
        "dotprod",
        "fp16",
        "i8mm"
    );
    features
}

/// The tier each op family dispatches to on this host. The predicates are
/// the kernels' own dispatch checks.
fn op_tiers() -> BTreeMap<String, String> {
    let mut tiers = BTreeMap::new();
    let q8 = if cfg!(target_arch = "x86_64") {
        if crate::simd::interleaved_q8_0_supported() {
            "x86-avx512-vnni"
        } else if x86_has_avx2_f16c() {
            "x86-avx2"
        } else {
            "scalar"
        }
    } else if crate::simd::interleaved_q8_0_supported() {
        "arm-neon-dotprod"
    } else {
        "scalar"
    };
    tiers.insert("q8_0_matvec".to_string(), q8.to_string());
    let k_quant = if crate::k_quant_matmul::k_avx512_opt_in() {
        "x86-avx512 (EMBER_K_AVX512 opt-in)"
    } else if crate::k_quant_matmul::x86_k_supported() {
        "x86-avx2"
    } else if crate::k_quant_matmul::arm_k_supported() {
        "arm-neon-dotprod"
    } else {
        "scalar"
    };
    tiers.insert("k_quant_matvec".to_string(), k_quant.to_string());
    let elementwise = if x86_has_avx2_fma() {
        "x86-avx2-fma"
    } else if arm_has_neon() {
        "arm-neon"
    } else {
        "scalar"
    };
    tiers.insert("elementwise".to_string(), elementwise.to_string());
    // matrixmultiply picks its own microkernel at runtime.
    let f32_matmul = if cfg!(target_arch = "x86_64") {
        if x86_has_avx2_fma() {
            "matrixmultiply-fma"
        } else {
            "matrixmultiply-sse2"
        }
    } else if arm_has_neon() {
        "matrixmultiply-neon"
    } else {
        "matrixmultiply-generic"
    };
    tiers.insert("f32_matmul".to_string(), f32_matmul.to_string());
    tiers
}

fn x86_has_avx2_f16c() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn x86_has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn arm_has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// explaining differences
// ---------------------------------------------------------------------------

/// How likely a host difference is to change numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NumericImpact {
    Unlikely,
    Possible,
    Likely,
}

/// One field that differs between two host profiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostDifference {
    /// Dotted path inside the host profile (or a legacy `runtime.json` key).
    pub field: String,
    pub a: Value,
    pub b: Value,
    pub impact: NumericImpact,
    pub explanation: String,
}

/// Every host difference between two bundles, and the likeliest cause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostDifferenceReport {
    /// Differences, most numerically relevant first.
    pub differences: Vec<HostDifference>,
    /// One-sentence likely explanation for a numeric difference.
    pub likely_explanation: String,
    /// Caveats, e.g. a bundle that predates host profiles.
    pub notes: Vec<String>,
}

/// Top-level `runtime.json` keys compared when a bundle has no host profile.
const LEGACY_KEYS: [&str; 4] = ["os", "cpu_features", "threads", "compiler_version"];

/// Compare the `runtime.json` values of two bundles (either may be missing
/// or unparseable) and explain which host differences could account for a
/// numeric difference between their results.
pub fn explain_host_differences(
    runtime_a: Option<&Value>,
    runtime_b: Option<&Value>,
) -> HostDifferenceReport {
    let mut notes = Vec::new();
    let profile_a = runtime_a.and_then(|value| value.get(RUNTIME_KEY));
    let profile_b = runtime_b.and_then(|value| value.get(RUNTIME_KEY));
    let mut leaves_a = BTreeMap::new();
    let mut leaves_b = BTreeMap::new();
    match (profile_a, profile_b) {
        (Some(a), Some(b)) => {
            flatten("", a, &mut leaves_a);
            flatten("", b, &mut leaves_b);
        }
        _ => {
            for (side, present) in [("a", profile_a.is_some()), ("b", profile_b.is_some())] {
                if !present {
                    notes.push(format!(
                        "bundle {side} has no host profile (written before Ember recorded one); \
                         only the legacy runtime.json fields {} are compared",
                        LEGACY_KEYS.join(", ")
                    ));
                }
            }
            for key in LEGACY_KEYS {
                for (runtime, leaves) in [(runtime_a, &mut leaves_a), (runtime_b, &mut leaves_b)] {
                    if let Some(value) = runtime.and_then(|runtime| runtime.get(key)) {
                        leaves.insert(format!("legacy.{key}"), value.clone());
                    }
                }
            }
        }
    }
    let mut fields: Vec<&String> = leaves_a.keys().chain(leaves_b.keys()).collect();
    fields.sort();
    fields.dedup();
    let mut differences = Vec::new();
    for field in fields {
        let a = leaves_a.get(field).cloned().unwrap_or(Value::Null);
        let b = leaves_b.get(field).cloned().unwrap_or(Value::Null);
        if a == b || field == "schema" {
            continue;
        }
        let (impact, explanation) = classify(field, &a, &b);
        differences.push(HostDifference {
            field: field.clone(),
            a,
            b,
            impact,
            explanation,
        });
    }
    // Most relevant first; stable within an impact level (field order).
    differences.sort_by_key(|difference| std::cmp::Reverse(difference.impact));
    let likely_explanation = match differences.first() {
        None => "no recorded host difference: both runs used the same tiers, threads, \
                 environment and build, so a numeric difference is not explained by the host \
                 (check the spec, inputs and model, or suspect nondeterminism)"
            .to_string(),
        Some(top) if top.impact == NumericImpact::Unlikely => format!(
            "only differences that should not change numbers were recorded ({}); a numeric \
             difference is not explained by the host",
            differences
                .iter()
                .map(|difference| difference.field.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some(top) => top.explanation.clone(),
    };
    HostDifferenceReport {
        differences,
        likely_explanation,
        notes,
    }
}

/// Flatten objects to dotted leaves; arrays and scalars are leaves.
fn flatten(prefix: &str, value: &Value, out: &mut BTreeMap<String, Value>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&path, child, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn show(value: &Value) -> String {
    match value {
        Value::Null => "unset".to_string(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn classify(field: &str, a: &Value, b: &Value) -> (NumericImpact, String) {
    use NumericImpact::*;
    let (sa, sb) = (show(a), show(b));
    let tier = |family: &str| {
        (
            Likely,
            format!(
                "different {family} execution tier ({sa} vs {sb}): the tiers accumulate \
                 products in a different reduction order, so values differ in the last bits \
                 and greedy decoding can flip at near-ties; this is expected, not a bug"
            ),
        )
    };
    if let Some(family) = field.strip_prefix("op_tiers.") {
        return match family {
            "q8_0_matvec" => tier("Q8_0 matvec"),
            "k_quant_matvec" => tier("K-quant (Q4_K/Q6_K) matvec"),
            "elementwise" => tier("RMSNorm/SiLU/softmax vector"),
            "f32_matmul" => tier("f32 matmul"),
            other => tier(other),
        };
    }
    if let Some(operator) = field.strip_prefix("plan_kernels.") {
        return (
            Likely,
            format!(
                "the plan selected different kernels for the {operator} projection \
                 ({sa} vs {sb}); different kernels use different reduction orders"
            ),
        );
    }
    if field.starts_with("kernel_fallbacks.") {
        return (
            Likely,
            format!(
                "kernel fallback '{}' applied to {sa} vs {sb} tensors: the fallback kernel \
                 reduces in a different order",
                field.trim_start_matches("kernel_fallbacks.")
            ),
        );
    }
    if let Some(name) = field.strip_prefix("env.") {
        let impact = if name == "RAYON_NUM_THREADS" {
            Possible
        } else {
            Likely
        };
        return (
            impact,
            format!("dispatch knob {name} differs ({sa} vs {sb}); it changes which kernel runs"),
        );
    }
    match field {
        "cpu.arch" | "legacy.cpu_arch" => (
            Likely,
            format!(
                "different CPU architecture ({sa} vs {sb}): every SIMD kernel differs, so \
                 reduction orders differ throughout"
            ),
        ),
        "threads.requested" | "threads.rayon_threads" | "legacy.threads" => (
            Possible,
            format!(
                "different worker count ({sa} vs {sb}): column-parallel matvecs split their \
                 reductions by worker and change summation order; row-parallel kernels are \
                 unaffected"
            ),
        ),
        "threads.strategy" => (
            Possible,
            format!(
                "different thread strategy ({sa} vs {sb}): column-parallel splits change \
                 summation order"
            ),
        ),
        "cpu.features" | "legacy.cpu_features" => (
            Possible,
            format!(
                "different detected CPU features ({sa} vs {sb}); they matter through the op \
                 tiers they select"
            ),
        ),
        "cpu.os" | "legacy.os" => (
            Possible,
            format!(
                "different OS ({sa} vs {sb}): exp/ln/powf come from the platform libm, which \
                 can differ in the last ulp"
            ),
        ),
        "build.rustc" | "legacy.compiler_version" | "build.target" => (
            Possible,
            format!(
                "different compiler or target ({sa} vs {sb}): code generation and std math \
                 can differ in the last ulp"
            ),
        ),
        "build.ember_version" | "build.git_commit" => (
            Likely,
            format!(
                "different Ember build ({sa} vs {sb}): kernels may have changed between the \
                 two versions"
            ),
        ),
        "build.profile" | "build.opt_level" => (
            Unlikely,
            format!(
                "different build profile ({sa} vs {sb}): Rust never reassociates \
                 floating-point arithmetic at any optimization level"
            ),
        ),
        "build.compiled_target_features" => (
            Unlikely,
            format!(
                "different compile-time target features ({sa} vs {sb}): runtime dispatch, \
                 not these, selects the SIMD kernels, and Rust does not contract a*b+c into \
                 FMA on its own"
            ),
        ),
        "cpu.model" => (
            Unlikely,
            format!(
                "different CPU model ({sa} vs {sb}): with identical tiers and threads, \
                 results are normally bit-identical"
            ),
        ),
        "threads.available_parallelism" => (
            Unlikely,
            format!("different core count ({sa} vs {sb}); only the worker count used matters"),
        ),
        _ => (Possible, format!("{field} differs ({sa} vs {sb})")),
    }
}

/// Render a report as indented text lines for the CLI.
pub fn report_lines(report: &HostDifferenceReport) -> Vec<String> {
    let mut lines = Vec::new();
    for note in &report.notes {
        lines.push(format!("  note: {note}"));
    }
    if report.differences.is_empty() {
        lines.push("  no host differences recorded".to_string());
    }
    for difference in &report.differences {
        lines.push(format!(
            "  [{}] {}: {} vs {}",
            match difference.impact {
                NumericImpact::Likely => "likely",
                NumericImpact::Possible => "possible",
                NumericImpact::Unlikely => "unlikely",
            },
            difference.field,
            show(&difference.a),
            show(&difference.b)
        ));
    }
    lines.push(format!(
        "  likely explanation: {}",
        report.likely_explanation
    ));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn runtime(profile: Value) -> Value {
        json!({ "threads": 8, "os": "macos", RUNTIME_KEY: profile })
    }

    fn profile() -> Value {
        json!({
            "schema": HOST_PROFILE_SCHEMA,
            "cpu": {
                "arch": "x86_64",
                "os": "linux",
                "model": "AMD EPYC 9654",
                "features": ["avx2", "fma", "f16c", "avx512f", "avx512vnni"]
            },
            "op_tiers": {
                "q8_0_matvec": "x86-avx512-vnni",
                "k_quant_matvec": "x86-avx2",
                "elementwise": "x86-avx2-fma",
                "f32_matmul": "matrixmultiply-fma"
            },
            "plan_kernels": { "q": ["q8-packed"], "lm_head": ["q8-packed"] },
            "kernel_fallbacks": {},
            "threads": {
                "requested": 8,
                "rayon_threads": 8,
                "available_parallelism": 96,
                "strategy": "column-parallel-rayon"
            },
            "env": {},
            "build": {
                "ember_version": "1.0.0",
                "git_commit": "abc",
                "profile": "release",
                "opt_level": "3",
                "rustc": "rustc 1.92.0",
                "target": "x86_64-unknown-linux-gnu",
                "compiled_target_features": ["sse2"]
            }
        })
    }

    #[test]
    fn identical_profiles_report_no_host_explanation() {
        let report = explain_host_differences(Some(&runtime(profile())), Some(&runtime(profile())));
        assert!(report.differences.is_empty());
        assert!(report.notes.is_empty());
        assert!(report
            .likely_explanation
            .contains("not explained by the host"));
    }

    #[test]
    fn a_different_q8_tier_is_the_likely_explanation() {
        let mut other = profile();
        other["op_tiers"]["q8_0_matvec"] = json!("x86-avx2");
        other["cpu"]["model"] = json!("Intel Core i7-8700");
        other["cpu"]["features"] = json!(["avx2", "fma", "f16c"]);
        other["threads"]["requested"] = json!(6);
        other["threads"]["rayon_threads"] = json!(6);
        other["threads"]["available_parallelism"] = json!(12);
        let report = explain_host_differences(Some(&runtime(profile())), Some(&runtime(other)));
        let fields: Vec<&str> = report
            .differences
            .iter()
            .map(|difference| difference.field.as_str())
            .collect();
        assert_eq!(fields[0], "op_tiers.q8_0_matvec", "{fields:?}");
        for expected in [
            "cpu.model",
            "cpu.features",
            "threads.requested",
            "threads.rayon_threads",
            "threads.available_parallelism",
        ] {
            assert!(fields.contains(&expected), "{expected} missing: {fields:?}");
        }
        assert!(report.likely_explanation.contains("Q8_0 matvec"));
        assert!(report.likely_explanation.contains("reduction order"));
        assert_eq!(report.differences[0].impact, NumericImpact::Likely);
        let model = report
            .differences
            .iter()
            .find(|difference| difference.field == "cpu.model")
            .unwrap();
        assert_eq!(model.impact, NumericImpact::Unlikely);
        let text = report_lines(&report).join("\n");
        assert!(text.contains("[likely] op_tiers.q8_0_matvec: x86-avx512-vnni vs x86-avx2"));
    }

    #[test]
    fn only_benign_differences_do_not_explain_a_numeric_difference() {
        let mut other = profile();
        other["build"]["profile"] = json!("debug");
        other["cpu"]["model"] = json!("AMD EPYC 7763");
        let report = explain_host_differences(Some(&runtime(profile())), Some(&runtime(other)));
        assert_eq!(report.differences.len(), 2);
        assert!(report
            .differences
            .iter()
            .all(|difference| difference.impact == NumericImpact::Unlikely));
        assert!(report
            .likely_explanation
            .contains("not explained by the host"));
    }

    #[test]
    fn env_knobs_and_fallbacks_are_reported() {
        let mut other = profile();
        other["env"] = json!({ "EMBER_K_AVX512": "1" });
        other["kernel_fallbacks"] = json!({ "missing avx2": 3 });
        let report = explain_host_differences(Some(&runtime(profile())), Some(&runtime(other)));
        let env = report
            .differences
            .iter()
            .find(|difference| difference.field == "env.EMBER_K_AVX512")
            .unwrap();
        assert_eq!(env.a, Value::Null);
        assert_eq!(env.impact, NumericImpact::Likely);
        assert!(report
            .differences
            .iter()
            .any(|difference| difference.field == "kernel_fallbacks.missing avx2"));
    }

    #[test]
    fn legacy_bundles_fall_back_to_top_level_runtime_fields() {
        let legacy = json!({ "threads": 4, "os": "linux", "cpu_features": ["avx2"] });
        let report = explain_host_differences(Some(&legacy), Some(&runtime(profile())));
        assert_eq!(report.notes.len(), 1, "{:?}", report.notes);
        assert!(report.notes[0].contains("bundle a"));
        let fields: Vec<&str> = report
            .differences
            .iter()
            .map(|difference| difference.field.as_str())
            .collect();
        assert!(fields.contains(&"legacy.threads"), "{fields:?}");
        assert!(fields.contains(&"legacy.os"), "{fields:?}");
        let missing = explain_host_differences(None, None);
        assert_eq!(missing.notes.len(), 2);
        assert!(missing.differences.is_empty());
    }

    #[test]
    fn detected_profile_round_trips_and_names_every_op_family() {
        let tiers = op_tiers();
        for family in ["q8_0_matvec", "k_quant_matvec", "elementwise", "f32_matmul"] {
            assert!(tiers.contains_key(family), "{family}");
        }
        let features = detected_features();
        #[cfg(target_arch = "aarch64")]
        assert!(features.iter().any(|feature| feature == "neon"));
        let _ = features;
    }
}

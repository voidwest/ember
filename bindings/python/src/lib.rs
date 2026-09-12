//! Ember's investigate surface for Python: `inspect`, `plan`, `diff`, and
//! `diff_corpus` reports.
//!
//! This is a thin binding over the same library core as the `ember` CLI:
//! report dicts are produced by serializing the same structs the CLI
//! serializes for `--json`, so binding output and CLI output agree by
//! construction. The extension is optional and never required by the Rust
//! CLI/headless install path.

use ::ember::diff_corpus::{run_diff_corpus, CorpusMode, CorpusRequest};
use ::ember::diff_outcome::{evaluate_diff, ExternalRuntime};
use ::ember::inspect::{inspect_path, inspect_plan};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;
use std::time::Duration;

fn runtime_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}

/// Parse `against` names, de-duplicating in order; unknown names are refused
/// with the CLI's wording.
fn parse_runtimes(names: Vec<String>) -> PyResult<Vec<ExternalRuntime>> {
    let mut runtimes = Vec::new();
    for name in &names {
        match ExternalRuntime::parse(name) {
            Some(runtime) if !runtimes.contains(&runtime) => runtimes.push(runtime),
            Some(_) => {}
            None => {
                return Err(PyValueError::new_err(format!(
                    "unknown runtime '{name}'; supported: llama.cpp, candle"
                )));
            }
        }
    }
    Ok(runtimes)
}

/// Convert a serialized report into plain Python objects (dict/list/scalars).
fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(flag) => flag.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                int.into_pyobject(py)?.into_any().unbind()
            } else if let Some(uint) = number.as_u64() {
                uint.into_pyobject(py)?.into_any().unbind()
            } else {
                number
                    .as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        Value::String(text) => text.into_pyobject(py)?.into_any().unbind(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

/// Inspect a GGUF model, `tokenizer.json`, or KV snapshot directory.
///
/// Returns the same report dict as `ember inspect <path> --json`.
/// `sha256=True` also hashes GGUF/tokenizer files (snapshots hash on load).
#[pyfunction]
#[pyo3(signature = (path, sha256=false))]
fn inspect(py: Python<'_>, path: PathBuf, sha256: bool) -> PyResult<Py<PyAny>> {
    let report = py
        .detach(|| inspect_path(&path, sha256))
        .map_err(runtime_error)?;
    let value = serde_json::to_value(&report)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &value)
}

/// Evaluate a file with Ember and external runtimes (schema `ember.diff.v1`).
///
/// `against` accepts `llama.cpp` / `candle` (repeatable; duplicates are
/// ignored). Returns the same report dict as `ember diff <file> --json`.
#[pyfunction]
#[pyo3(signature = (file, against, timeout_secs=30.0))]
fn diff(
    py: Python<'_>,
    file: PathBuf,
    against: Vec<String>,
    timeout_secs: f64,
) -> PyResult<Py<PyAny>> {
    if !(timeout_secs.is_finite() && timeout_secs > 0.0) {
        return Err(PyValueError::new_err("timeout_secs must be positive"));
    }
    let runtimes = parse_runtimes(against)?;
    let report =
        py.detach(|| evaluate_diff(&file, &runtimes, Duration::from_secs_f64(timeout_secs)));
    let value = serde_json::to_value(&report)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &value)
}

/// Build the v0.4 execution plan for a llama-family GGUF
/// (`ember inspect <file> plan`).
///
/// Returns `{"architecture", "execution", "plan"}` where `plan` is the same
/// JSON the CLI writes with `--output`.
#[pyfunction]
#[pyo3(signature = (path, arch="auto", execution="planned"))]
fn plan(py: Python<'_>, path: PathBuf, arch: &str, execution: &str) -> PyResult<Py<PyAny>> {
    let report = py
        .detach(|| inspect_plan(&path, arch, execution))
        .map_err(runtime_error)?;
    let value = serde_json::to_value(&report)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &value)
}

/// Run a scaled differential corpus campaign (`ember diff-corpus`).
///
/// Returns the summary the CLI writes to `summary_{mode}-{n}-{seed}.json`
/// plus artifact paths and per-target counts. `against` accepts
/// `llama.cpp` / `candle`; `seeds` are explicit seed files (omit for the
/// mode defaults, which read `research/embersec/comparative/corpus.json`).
// Flat Python keyword surface; the request struct is assembled below.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (n, out_dir, seed=1, mode="raw", against=None, timeout_secs=8.0, jobs=4, seeds=None))]
fn diff_corpus(
    py: Python<'_>,
    n: usize,
    out_dir: PathBuf,
    seed: u64,
    mode: &str,
    against: Option<Vec<String>>,
    timeout_secs: f64,
    jobs: usize,
    seeds: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }
    if jobs == 0 {
        return Err(PyValueError::new_err("jobs must be positive"));
    }
    if !(timeout_secs.is_finite() && timeout_secs > 0.0) {
        return Err(PyValueError::new_err("timeout_secs must be positive"));
    }
    let Some(mode) = CorpusMode::parse(mode) else {
        return Err(PyValueError::new_err(format!(
            "unknown mode '{mode}'; supported: raw, construction"
        )));
    };
    let request = CorpusRequest {
        n,
        seed,
        mode,
        against: parse_runtimes(against.unwrap_or_default())?,
        timeout_secs,
        jobs,
        out_dir,
        seeds: seeds.unwrap_or_default(),
    };
    let outcome = py
        .detach(|| run_diff_corpus(&request, false))
        .map_err(runtime_error)?;
    let value = serde_json::to_value(&outcome)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    json_to_py(py, &value)
}

/// Ember's investigate surface: `inspect`, `plan`, `diff`, and `diff_corpus`
/// reports over the same library core as the `ember` CLI (dicts match the CLI
/// `--json` / `--output` contracts).
#[pymodule]
fn ember(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(diff_corpus, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

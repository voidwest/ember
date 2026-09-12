# Ember Python bindings

Optional PyO3 bindings over Ember's investigate surface. This is a thin layer
over the same library core as the `ember` CLI: report dictionaries are produced
by serializing the same structs the CLI serializes for `--json`, so
`ember.inspect(...)` equals `json.loads(ember inspect ... --json)` by
construction.

The core CLI/headless install path stays single-toolchain Rust; this extension
is never required by it. Root `cargo` commands do not build this crate
(`default-members = ["."]`); build it explicitly with maturin.

## Build

```bash
# from the repository root, with a virtualenv active
python -m pip install maturin
maturin develop -m bindings/python/Cargo.toml   # debug build into the venv
```

or build/install a wheel:

```bash
python -m pip install ./bindings/python
```

Requires Python >= 3.11 (abi3 wheel) and the repository's pinned Rust toolchain.

## Usage

```python
import ember

# Structural digest: GGUF model, tokenizer.json, or KV snapshot directory.
report = ember.inspect("model.gguf", sha256=True)
report["kind"]              # "gguf" | "tokenizer" | "kv-snapshot" | "unknown"
report["gguf"]["tensor_count"]

# v0.4 execution plan for a llama-family GGUF (the CLI --output JSON).
plan = ember.plan("model.gguf", execution="planned")
plan["plan"]["tensor_table"]

# Cross-runtime comparison (same taxonomy as `ember diff`).
result = ember.diff("model.gguf", against=["llama.cpp", "candle"], timeout_secs=30)
result["schema"]            # "ember.diff.v1"
result["agreement"]         # {"all_agree": bool, "distinct_outcomes": [...], ...}

# Scaled differential corpus campaign (writes blobs/log/summary under out_dir).
campaign = ember.diff_corpus(n=100, out_dir=".cache/corpus-run", against=["candle"])
campaign["summary"]["per_target"]
campaign["log_path"]
```

## API

| Function | Mirrors | Returns |
|---|---|---|
| `inspect(path, sha256=False)` | `ember inspect <path> --json` | report dict |
| `plan(path, arch="auto", execution="planned")` | `ember inspect <path> plan` | `{architecture, execution, plan}` |
| `diff(file, against, timeout_secs=30.0)` | `ember diff <file> --against … --json` | `ember.diff.v1` dict |
| `diff_corpus(n, out_dir, seed=1, mode="raw", against=None, timeout_secs=8.0, jobs=4, seeds=None)` | `ember diff-corpus …` | summary + artifact paths + per-target stats |

Errors: invalid arguments (unknown runtime/mode, non-positive n/jobs/timeout)
raise `ValueError`; report/campaign failures raise `RuntimeError` with the
CLI's message. The GIL is released while work runs.

## Scope and limitations

- The binding covers the settled investigate surface: `inspect` (digest and
  plan), `diff`, and `diff-corpus`, all over the same library core as the CLI.
- `diff`/`diff_corpus` spawn external runtime binaries exactly like the CLI; a
  missing binary is reported as `HARNESS_ERROR`, not raised. `diff_corpus`
  refuses an `out_dir` inside `research/embersec/comparative/` (frozen tree).
- Build coverage is Linux x86_64 CI plus local dev; the arm64 CI tier does not
  build Python wheels.

## Tests

```bash
python -m pytest tests/test_python_bindings.py -q
```

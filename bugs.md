# Bugs & Issues

---

## 2026-05-08 — `cargo fmt` CI failure

**Symptom**: `cargo fmt -- --check` failed on GitHub CI.

**Cause**: Two formatting violations slipped through:

1. `src/backend.rs:44` — `load_from_cpu` trait method signature was split across 4 lines; `rustfmt` wanted it on a single line.
2. `src/kv_cache.rs:119` — missing trailing newline at end of file.

**Fix** (commit `6bbad15`):

- Collapsed `load_from_cpu(&self, data: Vec<f32>, shape: &[usize])` onto one line.
- Added trailing `\n` to `src/kv_cache.rs`.

**Prevention**: Run `cargo fmt` before pushing. A pre-push hook or `cargo fmt -- --check` in CI already catches this.


# Ember Security Audit — 2026-08-16

Method: threat-modeled review of every trust boundary (untrusted GGUF,
tokenizer JSON, v0.5 experiment specs, bundles, NPY, KV snapshots, local
web console), a memory-safety audit of every unsafe kernel and its callers,
a dependency advisory pass (RustSec DB, 1216 advisories), and a dangerous-
API inventory. Two parallel read-only audit passes (memory-safety scope and
IO/web scope) were cross-verified against the code. Findings were either
fixed with regression tests or documented as accepted risk.

## Result summary

- RustSec: **0 exploitable vulnerabilities** in the lockfile (761 crates);
  11 "unmaintained" warnings (transitive: async-std/instant/number_prefix/
  paste/proc-macro-error2/rustls-pemfile — via gpui and tokenizers).
- Memory safety: **0 Critical / 0 High / 0 Medium** — every unsafe kernel's
  slice/alignment/block-count invariants are established by exact-length
  constructors (QuantizedWeight/KQuantWeight try_new*) or asserting
  wrappers before dispatch; the GGUF loader is checked-arithmetic
  throughout; arena accesses are bounds-checked slices.
- IO/web: **1 High + 1 Medium fixed**, 6 Low (3 fixed), 6 Info documented.

## Fixed (with regression tests)

| # | Sev | Area | Finding | Fix |
|---|-----|------|---------|-----|
| F1 | High | web GUI (src/gui.rs, gui_page.html) | Unauthenticated localhost JSON API: blind CSRF from any website could force arbitrary local GGUF loads, endless runs, disk fill; DNS rebinding could READ API responses (research data, model paths, generated text). | Per-session 128-bit bearer token (embedded in the served page, required on every API call, printed to terminal); strict Host-header allowlist (localhost/127.0.0.1/[::1]) on loopback binds — defeats rebinding; Origin hostname check on POSTs; 0.0.0.0 binds get a loud warning and the token still applies. Tests: host_allowlist_accepts_loopback_forms_only, cross_origin_posts_are_identified. |
| F2 | Medium | v05/verify.rs | checksums.sha256 + semantic-manifest payload keys from the untrusted bundle were `root.join(relative)`-ed without validation: absolute paths replace the root and `..`/symlinks escape → arbitrary-file hash oracle + full-buffer memory DoS during `ember experiment verify`. | Reuse `bundle::validate_relative_path` (reject empty/absolute/`..`/CurDir/non-Normal) for every key before joining; stream-hash via sha256_file_result instead of fs::read. Regression test: traversal_checksum_fails_even_with_the_correct_hash (correct hash still fails — proves rejection, not mismatch). |
| F3 | Low | v05/verify.rs | read_gguf_string did `vec![0u8; len]` on a header-declared u64 length → capacity-overflow panic (abort) from a crafted deep-check GGUF. | Bound len to 1 MiB before allocating. |
| F8a | Low | alloc_counter.rs | realloc double-counted TOTAL_ALLOCATED_BYTES (+2*new−old instead of new−old) — residency telemetry drift. | Event counter increments once; live bytes change by the exact delta. |
| F8b | Low | alloc_counter.rs | usize overflow inside #[global_allocator] would panic (catastrophic); tracking flag leaked on a panicking measurement. | saturating_add for thread-local cells; Drop guard clears the flag. |
| F9 | Low | plan_build.rs / plan.rs | Unchecked `elements*4`, `cursor += size`, `total_bytes + alignment` in arena sizing — adversarial-metadata wrap (DoS-only, but cheap to close). | checked_mul/checked_add with invariant messages. |
| F10 | Low | llama.rs / gemma4.rs | No sanity caps on GGUF metadata: context_length=0xFFFFFFFF drives ~1 TB rope-table/KV allocations (abort). | llama.cpp-style caps: context_length ≤ 2M, n_heads ≤ 8192, head_dim ≤ 512, vocab ≤ 16M (llama + gemma4). |
| F11 | Low | web GUI | Unbounded request-body read (read_to_end) → local memory DoS. | 1 MiB body cap with explicit reject. |
| F12 | Info | web GUI | All responses were HTTP 200; no nosniff. | Real 403/404 statuses; X-Content-Type-Options: nosniff on every response. |

## Holds (audited, no issue) — the notable ones

- **Unsafe kernels (simd.rs, k_quant_matmul.rs):** every raw pointer/deref/
  SIMD load-store invariant is established by asserting wrappers (length,
  block-count, alignment) or exact-length constructors; the four-row K path
  is entered only with exactly 4*blocks_per_row packed rows; unaligned
  loads/stores only; AVX-512 prefetch may over-read ≤64 B (prefetches never
  fault/write).
- **tensor.rs sgemm FFI:** dims bounded by checked element counts, no
  aliasing, zero-dim early return (matrixmultiply 0.3.10 contract).
- **GGUF loader:** counts/offsets/alignment checked before any reserve/
  seek/mmap; range re-validated against mmap.len(); overlapping tensor
  ranges rejected; metadata array nesting capped at 16; strings bounded.
- **mmap (SIGBUS on concurrent truncate/replace):** documented contract,
  llama.cpp-equivalent posture — accepted risk (Info).
- **safetensors codec:** zero unsafe; every offset/size/alignment validated
  before slicing.
- **npy reader:** magic/version, header bounded by file, exact payload
  length required before any allocation.
- **v0.5 spec TOML:** deny_unknown_fields, safe-id validation, no recursion
  exposure beyond toml's own limits.
- **atomic_file / staging:** O_EXCL sibling temps, rename-only publish,
  hard-link no-replace for new files, cleanup on failure.
- **kv_snapshot:** hash-verified before import, payload caps, symlink
  rejection on outputs, path validation on load.
- **Command::new sites:** argv-only, no shell; llama-cpp-external validates
  the executable path and passes structured args.
- **GUI XSS:** esc() on all user-controlled innerHTML inputs; generated
  text via textContent.
- **CountingAllocator:** no pointer arithmetic; delegates to System with
  identical layout; relaxed atomics; no underflow path.

## Accepted risks (documented, not exploitable without local access)

- mmap SIGBUS if a GGUF is truncated/replaced while loaded (self-inflicted
  local action; matches llama.cpp).
- Slowloris against the console holds 8 handler slots (requires a local
  process after the token fix — self-DoS).
- 0.0.0.0 binds disable the Host/Origin checks (loud warning printed;
  token still required).
- Temp files use process umask (0600 hardening deferred to a platform
  policy pass; O_EXCL prevents planting).
- verify.rs metadata files (manifest/checksums/index) have no size caps
  beyond F2's path validation — a hostile bundle can still be a large-file
  CPU/memory load; the path validation removes the *escape*, and the
  stream-hash removes the full-buffer amplification.

## Verification

415 Rust tests + 38 Python tests pass; clippy -D warnings under both
feature configs; fmt clean; cargo audit clean of exploitable advisories;
CI (x86_64 + aarch64 tiers) green on all fixes.

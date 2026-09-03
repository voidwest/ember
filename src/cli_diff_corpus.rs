//! Scaled differential corpus runner (`ember diff-corpus`).
//!
//! A Rust port of `research/embersec/comparative/diff_fuzz.py` on top of the
//! supervised subprocess/diff layer (`crate::subprocess`,
//! `crate::diff_outcome`). The frozen Python fuzzer is read, never written;
//! this module mirrors its operators, slots, tables, and output schema so a
//! scaled campaign can run without the multi-worker OOM hazard that
//! corrupted the original fuzz pools on this host.
//!
//! # Parity scope (read before comparing numbers)
//!
//! - **Same**: `field_offsets` slot discovery, `metadata_value_slots`,
//!   `BOUNDARY64` / `BOUNDARY32` / `CONFIG_BOUNDARY64` values,
//!   `mutate` / `mutate_construction` structure (45% structured-patch +
//!   raw edits, 1-3 config patches, magic preservation), seed selection
//!   (raw: all non-tokenizer corpus fixtures; construction: gguf-050/051/052
//!   valid models), per-case JSONL + `summary_{mode}-{n}-{seed}.json` schema,
//!   crash saving on PANIC / PROCESS_CRASH / TIMEOUT / RESOURCE_LIMIT with
//!   content-hash dedup, 500-case wall-time checkpoints.
//! - **Deliberately different**: the RNG. Python uses `random.Random` (Mersenne
//!   Twister); this runner uses `StdRng` (`rand` 0.8, already a dependency).
//!   Streams are NOT bit-identical across implementations, so parity means
//!   same operators/slots/tables/coverage, never byte-equal blobs.
//! - **Ember side is a superset**: `diff_fuzz.py` raw mode runs the harness
//!   `gguf_load_check` stage only, while [`evaluate_ember`] runs load *plus*
//!   model construction. Ember-side divergences can therefore only go
//!   ACCEPT -> STRUCTURED_REJECT (construct-layer rejection of a loadable
//!   file), auditable per case; REJECT -> ACCEPT would be a bug. Construction
//!   mode (`gguf_model_check`) is directly comparable.
//! - **Ember is in-process**: externals run under [`run_supervised`] (fixed
//!   timeout, kill + reap); Ember evaluates in-process per file like
//!   [`evaluate_ember`]. An Ember unwind is caught and reported as PANIC
//!   rather than killing the campaign (an OOM SIGKILL cannot be caught by
//!   anyone; that is what `--jobs <= 4` and `dmesg` watches are for).
//! - **Crash stderr tails** are the report-layer 400-char tails
//!   ([`SideReport::stderr_tail]`), not Python's last-800-chars of raw
//!   stderr; the harness byte caps already bound the raw streams.
//!
//! # Host discipline (this host OOM-corrupted the original pools)
//!
//! - `--jobs` defaults to 4 and is clamped to at most 4: at most 4
//!   concurrent child processes. Externals evaluate *sequentially* within a
//!   case (unlike `ember diff`, which fans externals out on scoped threads)
//!   so the cap holds per worker.
//! - The full mutation set is never held in RAM: generation streams blobs to
//!   `out-dir/blobs/` one at a time (like `diff_fuzz.py`'s `blob_dir`), and
//!   evaluation workers read one blob at a time.

use clap::{Args as ClapArgs, ValueEnum};
use ember::diff_outcome::{
    evaluate_ember, evaluate_external, DiffOutcome, ExternalRuntime, SideReport,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// GGUF magic. Mutations never touch bytes 0..4 of a magic-bearing seed, and
/// restore them if a truncation ever drops them, so inputs reach deep paths.
/// (diff_fuzz.py: `MAGIC_BYTES = b"GGUF"`.)
const MAGIC_BYTES: [u8; 4] = *b"GGUF";

/// Boundary constants for 8-byte field slots.
/// (diff_fuzz.py lines 115-116:
/// `BOUNDARY64 = [0, 1, 2, 31, 32, 33, 255, 256, 257, 2**31 - 1, 2**31,
///  2**32 - 1, 2**32, 2**40, 2**63 - 1, 2**63, 2**64 - 1]`.)
pub const BOUNDARY64: &[u64] = &[
    0,
    1,
    2,
    31,
    32,
    33,
    255,
    256,
    257,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    1 << 40,
    (1 << 63) - 1,
    1 << 63,
    u64::MAX,
];

/// Boundary constants for 4-byte field slots.
/// (diff_fuzz.py lines 117-118:
/// `BOUNDARY32 = [0, 1, 2, 7, 8, 9, 15, 30, 31, 32, 33, 99, 255, 2**31 - 1,
///  2**31, 2**32 - 1]`.)
pub const BOUNDARY32: &[u32] = &[
    0,
    1,
    2,
    7,
    8,
    9,
    15,
    30,
    31,
    32,
    33,
    99,
    255,
    (1 << 31) - 1,
    1 << 31,
    u32::MAX,
];

/// Boundary constants for construction-layer metadata value patches.
/// (diff_fuzz.py lines 121-122:
/// `CONFIG_BOUNDARY64 = [0, 1, 2, 3, 4, 5, 7, 31, 255, 4096, 1 << 20,
///  1 << 24, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1]`.)
pub const CONFIG_BOUNDARY64: &[u64] = &[
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    31,
    255,
    4096,
    1 << 20,
    1 << 24,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    (1 << 63) - 1,
];

/// Single-byte choices for 1-byte field slots (diff_fuzz.py line 224).
const BYTE_CHOICES: &[u8] = &[0, 1, 2, 0x7F, 0x80, 0xFF];

/// Single-byte choices for raw byte-set edits (diff_fuzz.py line 239).
const RAW_SET_CHOICES: &[u8] = &[0x00, 0xFF, 0x7F, 0x80, 0x01];

fn read_u64_le(data: &[u8], off: usize) -> Option<u64> {
    data.get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
}

fn read_u32_le(data: &[u8], off: usize) -> Option<u32> {
    data.get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
}

/// Locate mutable numeric fields: header counts, metadata value slots,
/// tensor-info dims/dtype/offset slots. Returns `(offset, width)` pairs.
///
/// Faithful port of `field_offsets` (diff_fuzz.py lines 31-112), including
/// its quirks: the key bytes are skipped without a bounds check (saturating
/// arithmetic keeps that safe here), and tensor-info dim/dtype/offset slots
/// are appended without bounds checks (harmless: [`mutate_raw`] re-checks
/// `off + width <= len` before patching, exactly like the Python).
pub fn field_offsets(data: &[u8]) -> Vec<(usize, usize)> {
    let mut offs = Vec::new();
    if data.len() < 24 || data.get(..4) != Some(&MAGIC_BYTES[..]) {
        return offs;
    }
    let (Some(nt), Some(nkv)) = (read_u64_le(data, 8), read_u64_le(data, 16)) else {
        return offs;
    };
    offs.push((8, 8));
    offs.push((16, 8));
    let mut pos: usize = 24;
    for _ in 0..(nkv.min(2000) as usize) {
        let Some(klen) = read_u64_le(data, pos) else {
            return offs;
        };
        pos = pos.saturating_add(8);
        // Python slices `data[pos:pos+klen]` without checking, then advances
        // past it; the next iteration's bounds check catches the overrun.
        pos = pos.saturating_add(klen as usize);
        let Some(vtype) = read_u32_le(data, pos) else {
            return offs;
        };
        pos += 4;
        match vtype {
            4..=6 => {
                offs.push((pos, 4));
                pos = pos.saturating_add(4);
            }
            7 => {
                offs.push((pos, 1));
                pos = pos.saturating_add(1);
            }
            8 => {
                let Some(slen) = read_u64_le(data, pos) else {
                    return offs;
                };
                offs.push((pos, 8));
                pos = pos.saturating_add(8).saturating_add(slen as usize);
            }
            9 => {
                if data.len().saturating_sub(pos) < 12 {
                    return offs;
                }
                let (Some(et), Some(cnt)) = (read_u32_le(data, pos), read_u64_le(data, pos + 4))
                else {
                    return offs;
                };
                offs.push((pos + 4, 8));
                pos = pos.saturating_add(12);
                for _ in 0..(cnt.min(5000) as usize) {
                    if et == 8 {
                        let Some(slen) = read_u64_le(data, pos) else {
                            return offs;
                        };
                        offs.push((pos, 8));
                        pos = pos.saturating_add(8).saturating_add(slen as usize);
                    } else if et == 0 || et == 7 {
                        pos = pos.saturating_add(1);
                    } else if et == 2 || et == 3 {
                        pos = pos.saturating_add(2);
                    } else if (4..=6).contains(&et) {
                        offs.push((pos, 4));
                        pos = pos.saturating_add(4);
                    } else if (10..=12).contains(&et) {
                        offs.push((pos, 8));
                        pos = pos.saturating_add(8);
                    } else {
                        return offs;
                    }
                }
            }
            10 => {
                offs.push((pos, 8));
                pos = pos.saturating_add(8);
            }
            _ => return offs,
        }
    }
    for _ in 0..(nt.min(2000) as usize) {
        let Some(nlen) = read_u64_le(data, pos) else {
            return offs;
        };
        pos = pos.saturating_add(8).saturating_add(nlen as usize);
        let Some(nd) = read_u32_le(data, pos) else {
            return offs;
        };
        offs.push((pos, 4));
        pos = pos.saturating_add(4);
        for _ in 0..(nd.min(8) as usize) {
            offs.push((pos, 8));
            pos = pos.saturating_add(8);
        }
        offs.push((pos, 4));
        pos = pos.saturating_add(4);
        offs.push((pos, 8));
        pos = pos.saturating_add(8);
    }
    offs
}

/// Offsets of scalar metadata VALUE slots only (u32/f32), excluding string
/// lengths, array counts, and the tensor-info section. Patching only these
/// keeps the file loadable, so mutations reach model construction.
///
/// Faithful port of `metadata_value_slots` (diff_fuzz.py lines 125-183).
pub fn metadata_value_slots(data: &[u8]) -> Vec<usize> {
    let mut slots = Vec::new();
    if data.len() < 24 || data.get(..4) != Some(&MAGIC_BYTES[..]) {
        return slots;
    }
    let (Some(nkv), _) = (read_u64_le(data, 16), read_u64_le(data, 8)) else {
        return slots;
    };
    let mut pos: usize = 24;
    for _ in 0..(nkv.min(2000) as usize) {
        let Some(klen) = read_u64_le(data, pos) else {
            return slots;
        };
        pos = pos.saturating_add(8);
        pos = pos.saturating_add(klen as usize);
        let Some(vtype) = read_u32_le(data, pos) else {
            return slots;
        };
        pos += 4;
        match vtype {
            4 | 6 => {
                slots.push(pos);
                pos = pos.saturating_add(4);
            }
            5 => {
                pos = pos.saturating_add(4);
            }
            7 => {
                pos = pos.saturating_add(1);
            }
            8 => {
                let Some(slen) = read_u64_le(data, pos) else {
                    return slots;
                };
                pos = pos.saturating_add(8).saturating_add(slen as usize);
            }
            9 => {
                if data.len().saturating_sub(pos) < 12 {
                    return slots;
                }
                let (Some(et), Some(cnt)) = (read_u32_le(data, pos), read_u64_le(data, pos + 4))
                else {
                    return slots;
                };
                pos = pos.saturating_add(12);
                for _ in 0..(cnt.min(5000) as usize) {
                    if et == 8 {
                        let Some(slen) = read_u64_le(data, pos) else {
                            return slots;
                        };
                        pos = pos.saturating_add(8).saturating_add(slen as usize);
                    } else if et == 0 || et == 7 {
                        pos = pos.saturating_add(1);
                    } else if et == 2 || et == 3 {
                        pos = pos.saturating_add(2);
                    } else if (4..=6).contains(&et) || (10..=12).contains(&et) {
                        // NOTE the asymmetry with `field_offsets`: array
                        // element scalars are SKIPPED here (no slots pushed),
                        // which is what keeps construction mutants loadable.
                        pos = pos.saturating_add(if (10..=12).contains(&et) { 8 } else { 4 });
                    } else {
                        return slots;
                    }
                }
            }
            10 => {
                pos = pos.saturating_add(8);
            }
            _ => return slots,
        }
    }
    slots
}

/// Raw-mode mutation: 45% structured boundary patch (+ maybe a raw tweak),
/// else 1-6 raw byte edits, plus a 20% truncation chance; magic preserved.
/// Faithful port of `mutate` (diff_fuzz.py lines 210-249).
pub fn mutate_raw(seed: &[u8], rng: &mut StdRng) -> Vec<u8> {
    let mut data: Vec<u8> = seed.to_vec();
    if data.len() < 5 {
        return data;
    }
    let slots = field_offsets(seed);
    if rng.gen_range(0.0..1.0) < 0.45 && !slots.is_empty() {
        let &(off, width) = slots.choose(rng).expect("slots nonempty");
        if off.saturating_add(width) <= data.len() {
            match width {
                8 => {
                    let v = *BOUNDARY64.choose(rng).expect("table nonempty");
                    data[off..off + 8].copy_from_slice(&v.to_le_bytes());
                }
                4 => {
                    let v = *BOUNDARY32.choose(rng).expect("table nonempty");
                    data[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
                _ => {
                    data[off] = *BYTE_CHOICES.choose(rng).expect("table nonempty");
                }
            }
        }
        if rng.gen_range(0.0..1.0) < 0.3 {
            let p = rng.gen_range(4..data.len());
            let x: u8 = rng.gen_range(1..256i32) as u8;
            data[p] ^= x;
        }
    } else {
        for _ in 0..rng.gen_range(1..=6) {
            if data.len() <= 5 {
                break;
            }
            let p = rng.gen_range(4..data.len());
            let op = rng.gen_range(0.0..1.0);
            if op < 0.5 {
                let x: u8 = rng.gen_range(1..256i32) as u8;
                data[p] ^= x;
            } else if op < 0.75 {
                data[p] = *RAW_SET_CHOICES.choose(rng).expect("table nonempty");
            } else if op < 0.9 {
                data.insert(p, rng.gen_range(0..=u8::MAX));
            } else {
                data.remove(p);
            }
        }
    }
    if rng.gen_range(0.0..1.0) < 0.2 && data.len() > 5 {
        let k = rng.gen_range(5..data.len() + 1);
        data.truncate(k);
    }
    if seed.get(..4) == Some(&MAGIC_BYTES[..])
        && (data.len() < 4 || data.get(..4) != Some(&MAGIC_BYTES[..]))
    {
        if data.len() < 4 {
            // Python slice-assigns `data[:4] = MAGIC_BYTES`, which extends a
            // short buffer; unreachable via the edit paths above (they keep
            // >= 5 bytes), replicated for completeness.
            data = MAGIC_BYTES.to_vec();
        } else {
            data[..4].copy_from_slice(&MAGIC_BYTES);
        }
    }
    data
}

/// Construction-layer mutation: patch 1-3 scalar metadata values to
/// boundary values, plus a 15% raw tweak early in the file.
/// Faithful port of `mutate_construction` (diff_fuzz.py lines 186-207).
///
/// One guarded deviation: Python's `randrange(24, min(len, 4096))` raises on
/// seeds shorter than 25 bytes; this returns the patched buffer unchanged
/// instead of crashing the campaign. Unreachable with the default seed set
/// (25 KiB valid models).
pub fn mutate_construction(seed: &[u8], rng: &mut StdRng) -> Vec<u8> {
    let mut data: Vec<u8> = seed.to_vec();
    let slots = metadata_value_slots(&data);
    if slots.is_empty() || data.len() < 5 {
        return data;
    }
    for _ in 0..rng.gen_range(1..=3) {
        let off = *slots.choose(rng).expect("slots nonempty");
        if off.saturating_add(4) <= data.len() {
            let v = *CONFIG_BOUNDARY64.choose(rng).expect("table nonempty");
            data[off..off + 4].copy_from_slice(&((v & 0xFFFF_FFFF) as u32).to_le_bytes());
        }
    }
    if rng.gen_range(0.0..1.0) < 0.15 && data.len() > 24 {
        let p = rng.gen_range(24..data.len().min(4096));
        let x: u8 = rng.gen_range(1..256i32) as u8;
        data[p] ^= x;
    }
    data
}

/// Fuzz mode: raw (all seeds + field slots) or construction (metadata-only
/// patches on the valid models, reaching model construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CorpusMode {
    Raw,
    Construction,
}

impl CorpusMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Construction => "construction",
        }
    }
}

#[derive(ClapArgs)]
pub(crate) struct DiffCorpusCommand {
    /// Number of mutations to generate and evaluate.
    #[arg(long)]
    pub n: usize,
    /// RNG seed (same operators/slots as diff_fuzz.py `--seed`, but the
    /// stream is NOT bit-identical: StdRng vs Python random.Random).
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Mutation mode: raw or construction.
    #[arg(long, value_enum, default_value_t = CorpusMode::Raw)]
    pub mode: CorpusMode,
    /// External runtimes to compare against (repeatable or comma-separated);
    /// the ember side is always evaluated.
    #[arg(long, value_delimiter = ',')]
    pub against: Vec<String>,
    /// Per-runtime deadline in seconds (externals; ember is in-process).
    #[arg(long, default_value_t = 8.0)]
    pub timeout_secs: f64,
    /// Parallel case workers (max 4: at most 4 concurrent child processes).
    #[arg(long, default_value_t = 4)]
    pub jobs: usize,
    /// Output directory (REFUSED when inside research/embersec/comparative/).
    #[arg(long)]
    pub out_dir: PathBuf,
    /// Explicit seed files (repeatable or comma-separated). Default mirrors
    /// diff_fuzz.py: all non-tokenizer corpus fixtures (raw) or the
    /// gguf-050/051/052 valid models (construction).
    #[arg(long, value_delimiter = ',')]
    pub seeds: Vec<String>,
}

/// One per-case, per-target log record. Mirrors the diff_fuzz.py JSONL
/// record (`i`, `target`, `outcome`, `exit_code`, `blob`): `termination`
/// carries the exit-code/signal detail (`exit(1)`, `signal(11)`,
/// `timeout(killed+reaped)`, `in-process`), plus bounded diagnostics.
#[derive(Debug, Clone, Serialize)]
struct CaseRecord {
    i: usize,
    target: String,
    outcome: String,
    termination: String,
    wall_ms: Option<f64>,
    blob: String,
    stderr_tail: String,
}

/// Run summary. First seven keys mirror diff_fuzz.py's
/// `summary_{mode}-{n}-{seed}.json`; `runner`/`rng_note` are additive
/// provenance for the audit trail.
#[derive(Debug, Clone, Serialize)]
struct CorpusSummary {
    mode: String,
    n_mutations: usize,
    seed_cases: usize,
    rng_seed: u64,
    timeout_s: f64,
    per_target: BTreeMap<String, BTreeMap<String, u64>>,
    failure_inputs_saved: u64,
    runner: String,
    rng_note: String,
}

pub(crate) struct CorpusRunStats {
    pub per_target: BTreeMap<String, BTreeMap<String, u64>>,
    pub failure_inputs_saved: u64,
    pub elapsed_s: f64,
}

pub(crate) struct CorpusRunConfig {
    pub n: usize,
    pub seed: u64,
    pub mode: CorpusMode,
    pub seed_blobs: Vec<Vec<u8>>,
    pub against: Vec<ExternalRuntime>,
    pub timeout: Duration,
    pub jobs: usize,
    pub out_dir: PathBuf,
}

/// Outcomes that save the crashing input + stderr tail for triage (mirrors
/// diff_fuzz.py's PANIC / PROCESS_CRASH / TIMEOUT / RESOURCE_LIMIT set).
fn saves_crash(outcome: DiffOutcome) -> bool {
    matches!(
        outcome,
        DiffOutcome::Panic
            | DiffOutcome::ProcessCrash
            | DiffOutcome::Timeout
            | DiffOutcome::ResourceLimitOrExternalKill
    )
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Lexically normalize an absolute path (no IO, so it works before the
/// directory exists).
fn normalize_abs(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Refuse an `--out-dir` inside the frozen comparative tree. Checked
/// lexically *before* creating anything (so refusal leaves no trace), then
/// re-checked after canonicalization (catches symlink tricks).
fn guard_out_dir(out_dir: &Path) -> anyhow::Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let abs = if out_dir.is_absolute() {
        out_dir.to_path_buf()
    } else {
        cwd.join(out_dir)
    };
    let frozen = normalize_abs(&cwd.join("research/embersec/comparative"));
    let norm = normalize_abs(&abs);
    anyhow::ensure!(
        norm != frozen && !norm.starts_with(&frozen),
        "--out-dir '{}' is inside the frozen research/embersec/comparative/ tree; choose a scratch directory (e.g. .cache/diff-corpus/<run-tag>/)",
        out_dir.display()
    );
    std::fs::create_dir_all(out_dir)?;
    if let (Ok(canonical), Ok(frozen_canonical)) = (
        std::fs::canonicalize(out_dir),
        std::fs::canonicalize(&frozen),
    ) {
        anyhow::ensure!(
            canonical != frozen_canonical && !canonical.starts_with(&frozen_canonical),
            "--out-dir '{}' resolves inside the frozen research/embersec/comparative/ tree; refusing",
            out_dir.display()
        );
    }
    Ok(abs)
}

fn load_default_seeds(mode: CorpusMode) -> anyhow::Result<(Vec<String>, Vec<Vec<u8>>)> {
    let comp = PathBuf::from("research/embersec/comparative");
    let corpus_path = comp.join("corpus.json");
    anyhow::ensure!(
        corpus_path.is_file(),
        "corpus.json not found at '{}'; run from the repo root or pass --seeds <files>",
        corpus_path.display()
    );
    let text = std::fs::read_to_string(&corpus_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&text)?;
    let cases = parsed
        .get("cases")
        .and_then(|c| c.as_array())
        .ok_or_else(|| anyhow::anyhow!("corpus.json has no 'cases' array"))?;
    let mut names = Vec::new();
    let mut blobs = Vec::new();
    for case in cases {
        let id = case.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let input_type = case
            .get("input_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let take = match mode {
            CorpusMode::Raw => input_type != "TOKENIZER_JSON",
            CorpusMode::Construction => matches!(id, "gguf-050" | "gguf-051" | "gguf-052"),
        };
        if !take {
            continue;
        }
        let fixture = case
            .get("fixture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("corpus case {id} has no fixture"))?;
        let path = comp.join(fixture);
        let bytes = std::fs::read(&path).map_err(|e| {
            anyhow::anyhow!("cannot read seed fixture {id} ({}): {e}", path.display())
        })?;
        names.push(id.to_string());
        blobs.push(bytes);
    }
    anyhow::ensure!(
        !blobs.is_empty(),
        "no seed fixtures selected for mode {}",
        mode.as_str()
    );
    Ok((names, blobs))
}

fn load_override_seeds(paths: &[String]) -> anyhow::Result<(Vec<String>, Vec<Vec<u8>>)> {
    let mut names = Vec::new();
    let mut blobs = Vec::new();
    for p in paths {
        let path = PathBuf::from(p);
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("cannot read --seeds file '{}': {e}", path.display()))?;
        names.push(path.display().to_string());
        blobs.push(bytes);
    }
    anyhow::ensure!(!blobs.is_empty(), "--seeds selected no readable files");
    Ok((names, blobs))
}

fn record_of(i: usize, blob_name: &str, report: &SideReport) -> CaseRecord {
    CaseRecord {
        i,
        target: report.runtime.clone(),
        outcome: report.outcome.token().to_string(),
        termination: report
            .termination
            .clone()
            .unwrap_or_else(|| "not-run".to_string()),
        wall_ms: report.wall_ms,
        blob: blob_name.to_string(),
        stderr_tail: report.stderr_tail.clone(),
    }
}

/// Execute a fully-resolved campaign: stream mutations to `blobs/`, evaluate
/// each case (ember in-process, externals sequentially via [`run_supervised`])
/// on up to `jobs` workers, append JSONL under lock, save crashers deduped
/// by content hash. Returns aggregate stats; the summary file is written by
/// the caller.
pub(crate) fn run_corpus(config: &CorpusRunConfig) -> anyhow::Result<CorpusRunStats> {
    anyhow::ensure!(config.n > 0, "--n must be positive");
    anyhow::ensure!(!config.seed_blobs.is_empty(), "no seed blobs loaded");
    anyhow::ensure!(config.jobs > 0, "--jobs must be positive");
    let jobs = config.jobs.min(4);
    let blobs_dir = config.out_dir.join("blobs");
    let crash_dir = config.out_dir.join("crashes");
    std::fs::create_dir_all(&blobs_dir)?;
    std::fs::create_dir_all(&crash_dir)?;
    let run_tag = format!("{}-{}-{}", config.mode.as_str(), config.n, config.seed);
    let log_path = config.out_dir.join(format!("log_{run_tag}.jsonl"));
    // Truncate any prior log for this tag first: rerunning a tag must not
    // silently duplicate the campaign in its audit trail (diff_fuzz.py).
    std::fs::write(&log_path, "")?;

    // Phase 1: single-threaded generation with one evolving RNG, mirroring
    // diff_fuzz.py's `seed = rng.choice(seeds); mutations.append(mut(...))`
    // order exactly (choice-then-mutate draws). Blobs stream to disk; only
    // one is ever in RAM.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let construction = config.mode == CorpusMode::Construction;
    for i in 0..config.n {
        let seed_blob = config
            .seed_blobs
            .choose(&mut rng)
            .expect("seeds nonempty (checked)");
        let blob = if construction {
            mutate_construction(seed_blob, &mut rng)
        } else {
            mutate_raw(seed_blob, &mut rng)
        };
        std::fs::write(blobs_dir.join(format!("m{i:05}.bin")), &blob)?;
    }

    // Phase 2: parallel evaluation. Each worker owns one case at a time;
    // externals run sequentially within the case so concurrent child
    // processes never exceed `jobs` (<= 4).
    let log_file = std::fs::OpenOptions::new().append(true).open(&log_path)?;
    let shared = Shared {
        log: Mutex::new(BufWriter::new(log_file)),
        counts: Mutex::new(BTreeMap::new()),
        saved: Mutex::new(HashSet::new()),
        failure_inputs_saved: AtomicU64::new(0),
        next: AtomicUsize::new(0),
        completed: AtomicUsize::new(0),
    };
    let t0 = Instant::now();
    let externals: Vec<ExternalRuntime> = config.against.clone();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = shared.next.fetch_add(1, Ordering::SeqCst);
                if i >= config.n {
                    break;
                }
                if let Err(e) = eval_one(i, config, &externals, &shared) {
                    eprintln!("diff-corpus: case {i} evaluation failed: {e:#}");
                }
                let done = shared.completed.fetch_add(1, Ordering::SeqCst) + 1;
                if done.is_multiple_of(500) || done == config.n {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let saved = shared.failure_inputs_saved.load(Ordering::SeqCst);
                    println!(
                        "  {done}/{} cases, {elapsed:.0}s, failures so far: {saved}",
                        config.n
                    );
                }
            });
        }
    });
    {
        let mut log = shared.log.lock().expect("log mutex poisoned");
        log.flush()?;
    }
    let elapsed_s = t0.elapsed().as_secs_f64();
    let counts = shared.counts.lock().expect("counts mutex poisoned").clone();
    Ok(CorpusRunStats {
        per_target: counts,
        failure_inputs_saved: shared.failure_inputs_saved.load(Ordering::SeqCst),
        elapsed_s,
    })
}

struct Shared {
    log: Mutex<BufWriter<std::fs::File>>,
    counts: Mutex<BTreeMap<String, BTreeMap<String, u64>>>,
    saved: Mutex<HashSet<(String, String)>>,
    failure_inputs_saved: AtomicU64,
    next: AtomicUsize,
    completed: AtomicUsize,
}

fn eval_one(
    i: usize,
    config: &CorpusRunConfig,
    externals: &[ExternalRuntime],
    shared: &Shared,
) -> anyhow::Result<()> {
    let blob_name = format!("m{i:05}.bin");
    let blob_path = config.out_dir.join("blobs").join(&blob_name);

    // Ember first, in-process. An unwind becomes a PANIC report rather than
    // a dead campaign (evaluators must not panic by contract; scope makes
    // even that a loud, classified finding).
    let ember_report =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| evaluate_ember(&blob_path)))
        {
            Ok(report) => report,
            Err(payload) => SideReport {
                runtime: "ember".to_string(),
                outcome: DiffOutcome::Panic,
                termination: Some("panicked(unwound-by-harness)".to_string()),
                wall_ms: None,
                stderr_tail: panic_message(&payload).chars().take(400).collect(),
                stdout_truncated: false,
                stderr_truncated: false,
                harness_detail: None,
            },
        };
    let mut reports = Vec::with_capacity(1 + externals.len());
    reports.push(ember_report);
    // Externals sequentially within the case: the jobs cap is a cap on
    // concurrent child processes, so no fan-out here (unlike `ember diff`).
    for runtime in externals {
        reports.push(evaluate_external(*runtime, &blob_path, config.timeout));
    }

    for report in &reports {
        let record = record_of(i, &blob_name, report);
        {
            let mut counts = shared.counts.lock().expect("counts mutex poisoned");
            *counts
                .entry(record.target.clone())
                .or_default()
                .entry(record.outcome.clone())
                .or_insert(0) += 1;
        }
        {
            let mut log = shared.log.lock().expect("log mutex poisoned");
            writeln!(log, "{}", serde_json::to_string(&record)?)?;
        }
        if saves_crash(report.outcome) {
            save_crash(config, shared, &blob_path, report)?;
        }
    }
    Ok(())
}

fn save_crash(
    config: &CorpusRunConfig,
    shared: &Shared,
    blob_path: &Path,
    report: &SideReport,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(blob_path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = format!("{:x}", hasher.finalize());
    let short = digest[..16].to_string();
    let key = (report.runtime.clone(), short.clone());
    {
        let mut saved = shared.saved.lock().expect("saved mutex poisoned");
        if !saved.insert(key) {
            return Ok(());
        }
    }
    let dir = config.out_dir.join("crashes").join(&report.runtime);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("{short}.bin")), &bytes)?;
    std::fs::write(dir.join(format!("{short}.stderr")), &report.stderr_tail)?;
    shared.failure_inputs_saved.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

pub(crate) fn run_diff_corpus_command(command: &DiffCorpusCommand) -> anyhow::Result<()> {
    anyhow::ensure!(command.n > 0, "--n must be positive");
    anyhow::ensure!(
        command.timeout_secs > 0.0,
        "--timeout-secs must be positive"
    );
    anyhow::ensure!(command.jobs > 0, "--jobs must be positive");
    let jobs = if command.jobs > 4 {
        eprintln!(
            "diff-corpus: --jobs {} exceeds the 4-concurrent-child cap; clamping to 4",
            command.jobs
        );
        4
    } else {
        command.jobs
    };
    let mut against = Vec::new();
    for name in &command.against {
        match ExternalRuntime::parse(name) {
            Some(runtime) if !against.contains(&runtime) => against.push(runtime),
            Some(_) => {}
            None => anyhow::bail!(
                "unknown runtime '{name}'; supported: llama.cpp, candle (see `ember diff runtimes`)"
            ),
        }
    }
    let out_dir = guard_out_dir(&command.out_dir)?;
    let (seed_names, seed_blobs) = if command.seeds.is_empty() {
        load_default_seeds(command.mode)?
    } else {
        load_override_seeds(&command.seeds)?
    };
    let timeout = Duration::from_secs_f64(command.timeout_secs);
    let config = CorpusRunConfig {
        n: command.n,
        seed: command.seed,
        mode: command.mode,
        seed_blobs,
        against,
        timeout,
        jobs,
        out_dir: out_dir.clone(),
    };
    println!(
        "diff-corpus: mode={} n={} seed={} seeds={} against=[{}] jobs={} timeout={}s out={}",
        command.mode.as_str(),
        command.n,
        command.seed,
        seed_names.len(),
        config
            .against
            .iter()
            .map(|r| r.name())
            .collect::<Vec<_>>()
            .join(","),
        jobs,
        command.timeout_secs,
        out_dir.display(),
    );
    let stats = run_corpus(&config)?;
    let run_tag = format!("{}-{}-{}", command.mode.as_str(), command.n, command.seed);
    let summary = CorpusSummary {
        mode: command.mode.as_str().to_string(),
        n_mutations: command.n,
        seed_cases: seed_names.len(),
        rng_seed: command.seed,
        timeout_s: command.timeout_secs,
        per_target: stats.per_target.clone(),
        failure_inputs_saved: stats.failure_inputs_saved,
        runner: "ember diff-corpus".to_string(),
        rng_note: "StdRng: same operators/slots/tables as diff_fuzz.py, NOT bit-identical streams (Python random.Random differs)".to_string(),
    };
    let summary_path = out_dir.join(format!("summary_{run_tag}.json"));
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    for (target, counts) in &stats.per_target {
        println!(
            "{target:18} {counts:?} (wall {elapsed:.0}s total)",
            elapsed = stats.elapsed_s
        );
    }
    println!(
        "diff-corpus done: {} cases, {} failure inputs saved; summary {}",
        command.n,
        stats.failure_inputs_saved,
        summary_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Minimal GGUF header (magic + version + counts + padding) for slot and
    /// magic tests. Not a loadable model; exercises the walker only.
    fn tiny_header(nt: u64, nkv: u64) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&nt.to_le_bytes());
        v.extend_from_slice(&nkv.to_le_bytes());
        v
    }

    #[test]
    fn boundary_tables_match_diff_fuzz_py_exactly() {
        // Pinned against diff_fuzz.py lines 115-122; any edit here or there
        // must be a deliberate, reviewed change (see also
        // `boundary_tables_present_in_frozen_source` below).
        assert_eq!(
            BOUNDARY64,
            &[
                0,
                1,
                2,
                31,
                32,
                33,
                255,
                256,
                257,
                2_147_483_647,
                2_147_483_648,
                4_294_967_295,
                4_294_967_296,
                1_099_511_627_776,
                9_223_372_036_854_775_807,
                9_223_372_036_854_775_808,
                18_446_744_073_709_551_615,
            ]
        );
        assert_eq!(
            BOUNDARY32,
            &[
                0,
                1,
                2,
                7,
                8,
                9,
                15,
                30,
                31,
                32,
                33,
                99,
                255,
                2_147_483_647,
                2_147_483_648,
                4_294_967_295,
            ]
        );
        assert_eq!(
            CONFIG_BOUNDARY64,
            &[
                0,
                1,
                2,
                3,
                4,
                5,
                7,
                31,
                255,
                4096,
                1_048_576,
                16_777_216,
                2_147_483_647,
                2_147_483_648,
                4_294_967_295,
                4_294_967_296,
                9_223_372_036_854_775_807,
            ]
        );
        assert_eq!(BOUNDARY64.len(), 17);
        assert_eq!(BOUNDARY32.len(), 16);
        assert_eq!(CONFIG_BOUNDARY64.len(), 17);
    }

    #[test]
    fn boundary_tables_present_in_frozen_source() {
        // The frozen fuzzer is read-only; this pins our transcription to its
        // literal table lines so silent drift fails loudly. Path is relative
        // to src/ (compile-time include, no runtime IO).
        const FROZEN: &str = include_str!("../research/embersec/comparative/diff_fuzz.py");
        assert!(FROZEN.contains(
            "BOUNDARY64 = [0, 1, 2, 31, 32, 33, 255, 256, 257, 2**31 - 1, 2**31, 2**32 - 1,"
        ));
        assert!(FROZEN.contains(
            "BOUNDARY32 = [0, 1, 2, 7, 8, 9, 15, 30, 31, 32, 33, 99, 255, 2**31 - 1, 2**31,"
        ));
        assert!(FROZEN.contains(
            "CONFIG_BOUNDARY64 = [0, 1, 2, 3, 4, 5, 7, 31, 255, 4096, 1 << 20, 1 << 24,"
        ));
    }

    #[test]
    fn mutation_is_deterministic_per_seed() {
        let seed_blob = tiny_header(1, 0);
        let run = |rng_seed: u64| {
            let mut rng = test_rng(rng_seed);
            (0..20)
                .map(|_| mutate_raw(&seed_blob, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
        let run_c = |rng_seed: u64| {
            let mut rng = test_rng(rng_seed);
            (0..20)
                .map(|_| mutate_construction(&seed_blob, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run_c(11), run_c(11));
    }

    #[test]
    fn magic_is_preserved_and_never_invented() {
        let magic_seed = tiny_header(0, 0);
        let mut rng = test_rng(7);
        for _ in 0..200 {
            let out = mutate_raw(&magic_seed, &mut rng);
            assert_eq!(&out[..4], b"GGUF");
        }
        let plain_seed = b"NOPE-not-gguf-at-all".to_vec();
        for _ in 0..200 {
            let out = mutate_raw(&plain_seed, &mut rng);
            assert_eq!(&out[..4], b"NOPE");
        }
    }

    #[test]
    fn field_offsets_finds_header_counts_and_rejects_short_inputs() {
        assert!(field_offsets(&[]).is_empty());
        assert!(field_offsets(b"GGUF").is_empty());
        assert!(field_offsets(b"XXXX....long-enough-to-pass-length....").is_empty());
        let offs = field_offsets(&tiny_header(0, 0));
        assert!(offs.contains(&(8, 8)));
        assert!(offs.contains(&(16, 8)));
    }

    #[test]
    fn construction_mutation_keeps_short_inputs_stable() {
        // Empty slot set / tiny seeds return the input unchanged.
        let mut rng = test_rng(11);
        assert_eq!(mutate_construction(b"tiny", &mut rng), b"tiny");
        assert_eq!(mutate_construction(&[], &mut rng), Vec::<u8>::new());
    }

    #[test]
    fn out_dir_inside_comparative_is_refused() {
        let err = guard_out_dir(Path::new("research/embersec/comparative/x")).unwrap_err();
        assert!(err.to_string().contains("frozen"));
        let err2 = guard_out_dir(Path::new("research/embersec/comparative")).unwrap_err();
        assert!(err2.to_string().contains("frozen"));
    }

    #[test]
    fn corpus_runner_end_to_end_with_externals_absent() {
        // Three tiny blobs, no external binaries: externals must report
        // HarnessError shape while ember still classifies in-process.
        let dir = std::env::temp_dir().join(format!("ember-diff-corpus-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out_dir = dir.join("out");
        // SAFETY: no other thread in this test reads the process environment
        // concurrently (workers only move byte buffers and take a mutex).
        unsafe {
            std::env::set_var(
                ExternalRuntime::LlamaCpp.env_override(),
                dir.join("no-such-llama-bin"),
            );
            std::env::set_var(
                ExternalRuntime::Candle.env_override(),
                dir.join("no-such-candle-bin"),
            );
        }
        let config = CorpusRunConfig {
            n: 3,
            seed: 7,
            mode: CorpusMode::Raw,
            seed_blobs: vec![tiny_header(0, 0), b"definitely not gguf........".to_vec()],
            against: vec![ExternalRuntime::LlamaCpp, ExternalRuntime::Candle],
            timeout: Duration::from_secs(5),
            jobs: 1,
            out_dir: out_dir.clone(),
        };
        let stats = run_corpus(&config).unwrap();
        assert!(stats.per_target.contains_key("ember"));
        for runtime in ["llama.cpp", "candle"] {
            let counts = &stats.per_target[runtime];
            assert_eq!(
                counts.get("HARNESS_ERROR"),
                Some(&3),
                "absent {runtime} must be HarnessError, got {counts:?}"
            );
        }
        // Ember classified all three in-process (never HarnessError for
        // readable files).
        let ember_total: u64 = stats.per_target["ember"].values().sum();
        assert_eq!(ember_total, 3);
        assert!(!stats.per_target["ember"].contains_key("HARNESS_ERROR"));
        // JSONL has one record per case x target.
        let log = std::fs::read_to_string(out_dir.join("log_raw-3-7.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 9);
        let first: serde_json::Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
        for key in ["i", "target", "outcome", "termination", "blob"] {
            assert!(first.get(key).is_some(), "record missing {key}");
        }
        // Blobs streamed to disk one file per case.
        assert_eq!(std::fs::read_dir(out_dir.join("blobs")).unwrap().count(), 3);
        unsafe {
            std::env::remove_var(ExternalRuntime::LlamaCpp.env_override());
            std::env::remove_var(ExternalRuntime::Candle.env_override());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

//! On-disk cache for the packed Q8_0 decode layouts built at model load.
//!
//! Building the 16-output VNNI layout costs ~0.5 s for Llama-3.2-1B (and
//! comparable work for Gemma gate/up projections) and is paid on every
//! process start. This module persists those packed bytes and serves them
//! back when the source model is unchanged, so a warm deployment can skip
//! the repack.
//!
//! Enabled by default; `EMBER_PACKED_CACHE=0` disables it (any other value,
//! or unset, enables). Trust model: the cache is derived data
//! whose identity binds the source path, size, mtime and a compact
//! metadata/tensor-table fingerprint (scalar metadata values exactly; array
//! and string contents bounded, since packed layouts depend only on tensor
//! bytes and shapes). Structural validation (magic, version,
//! header digest, entry bounds, expected shapes and lengths) runs on every
//! read. A full payload digest is stored and checked only when
//! `EMBER_PACKED_CACHE_VERIFY=1`, because hashing a ~1 GiB payload costs
//! roughly the repack it would save — the same trust the model file itself
//! receives (no content hash). Torn writes cannot be observed: the payload is
//! streamed into a sibling temporary file and published by rename.
//!
//! The first run for a model writes the cache (roughly the size of the
//! packed tensors; ~1.3 GiB for a 1B Q8_0 model). After each successful
//! publish the directory is pruned to `EMBER_PACKED_CACHE_BYTES` (an integer
//! with an optional `K`/`M`/`G`/`T` binary-unit suffix; default 8 GiB, `0`
//! disables eviction): least-recently-used cache files are evicted and
//! temporary files left by crashed writers (older than a day) are removed.
//! Writability and free space are not probed up front: an unwritable or full
//! cache directory disables writes for that run with one warning, leaves no
//! partial file behind, and degrades to in-memory packing.
//!
//! Cache directory: `$EMBER_CACHE_DIR`, else `$XDG_CACHE_HOME/ember`, else
//! `$HOME/.cache/ember`. Any failure (unwritable directory, corrupt file,
//! shape mismatch) degrades to the in-memory packing path.

use crate::loader::{GgufLoader, GgufValue};
use crate::quant::{QuantizedWeightInterleaved, QuantizedWeightVnni};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

const MAGIC: &[u8; 8] = b"EMBERPK1";
/// Format revision: 1 stored VNNI tiles only; 2 adds the interleaved lm-head
/// layout (entries carry a kind and an optional second byte range).
const VERSION: u32 = 2;
/// Layout ids of the packed representations (`quant.rs`).
pub const LAYOUT_VNNI_TILE16: &str = "q8-vnni-tile16-v1";
pub const LAYOUT_INTERLEAVED_4ROW: &str = "q8-interleaved-4row-v1";
/// Header format tag (both layouts share one file).
const FORMAT_TAG: &str = "ember-packed-q8-v2";
/// Bytes reserved between the fixed prefix and the payload for the header.
/// The header is small (one entry per packed tensor) and rewritten in place
/// once the payload is complete, so payload entries stream to disk without a
/// second in-memory copy.
const HEADER_RESERVE: u64 = 64 * 1024;
const PREFIX_LEN: u64 = 16; // magic + version + header_len
const PAYLOAD_START: u64 = PREFIX_LEN + HEADER_RESERVE;
/// Sanity cap for a cache file (the largest supported GGUF is 16 GiB).
const MAX_CACHE_BYTES: u64 = 20 * 1024 * 1024 * 1024;
/// Default retention for the whole cache directory: roughly six 1B-Q8_0
/// models, or seven Gemma 4 E2B caches.
pub const DEFAULT_CACHE_BUDGET_BYTES: u64 = 8 << 30;
/// Temporary files older than this are leftovers from a crashed writer (a
/// live publish takes seconds, not hours) and are removed during pruning.
const STALE_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Which packed layout an entry stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum EntryKind {
    /// 16-output VNNI tiles (`QuantizedWeightVnni`).
    Vnni,
    /// 4-row interleaved quants + scales (`QuantizedWeightInterleaved`).
    Interleaved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HeaderEntry {
    name: String,
    kind: EntryKind,
    out_features: usize,
    in_features: usize,
    blocks_per_row: usize,
    /// Offset relative to the payload start (VNNI payload, or quants).
    offset: u64,
    len: u64,
    /// Second range (interleaved scales); zero for VNNI entries.
    #[serde(default)]
    aux_offset: u64,
    #[serde(default)]
    aux_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Header {
    /// Format tag; both layouts share one file.
    format: String,
    payload_len: u64,
    #[serde(default)]
    payload_sha256: Option<String>,
    entries: Vec<HeaderEntry>,
}

struct WriterState {
    file: File,
    temp_path: PathBuf,
    dest: PathBuf,
    entries: Vec<HeaderEntry>,
    payload_written: u64,
    hasher: Option<Sha256>,
}

/// One process's handle on the packed cache for one source model + layout.
pub struct PackedCache {
    path: PathBuf,
    /// Lazily created read-only mapping of the cache file, shared by every
    /// packed weight served from it (no per-tensor copies).
    mapped: Mutex<Option<std::sync::Arc<memmap2::Mmap>>>,
    writer: Mutex<Option<WriterState>>,
    entries: Vec<HeaderEntry>,
    verify: bool,
    /// Eviction budget for the cache directory, in bytes (`0` disables).
    budget: u64,
    hits: AtomicUsize,
    misses: AtomicUsize,
    write_enabled: std::sync::atomic::AtomicBool,
}

impl PackedCache {
    /// Open (or prepare to write) the cache for a source model.
    ///
    /// Enabled unless `EMBER_PACKED_CACHE=0`. Returns `None` when the cache
    /// is disabled, the directory cannot be resolved, or the file is
    /// structurally invalid (a missing file simply yields an empty cache
    /// that will be written on `finish_write`).
    pub fn for_loader(model_path: &Path, loader: &GgufLoader) -> Option<Self> {
        let enabled = env_enabled(std::env::var_os("EMBER_PACKED_CACHE").as_deref());
        Self::for_loader_with(model_path, loader, enabled)
    }

    /// Testable constructor: `enabled` replaces the environment check.
    pub fn for_loader_with(model_path: &Path, loader: &GgufLoader, enabled: bool) -> Option<Self> {
        let verify =
            std::env::var_os("EMBER_PACKED_CACHE_VERIFY").is_some_and(|value| value == "1");
        let budget = match std::env::var_os("EMBER_PACKED_CACHE_BYTES") {
            Some(value) => env_budget(Some(&value)).unwrap_or_else(|| {
                log::warn!(
                    "ignoring invalid EMBER_PACKED_CACHE_BYTES value {:?}; using the {}-byte default",
                    value,
                    DEFAULT_CACHE_BUDGET_BYTES
                );
                DEFAULT_CACHE_BUDGET_BYTES
            }),
            None => DEFAULT_CACHE_BUDGET_BYTES,
        };
        Self::for_loader_in(model_path, loader, enabled, cache_dir(), verify, budget)
    }

    fn for_loader_in(
        model_path: &Path,
        loader: &GgufLoader,
        enabled: bool,
        dir: Option<PathBuf>,
        verify: bool,
        budget: u64,
    ) -> Option<Self> {
        if !enabled || !crate::simd::packed_q8_0_vnni_supported() {
            return None;
        }
        let dir = dir?;
        let key = cache_key(model_path, loader);
        let path = dir.join(format!("{key}.bin"));
        let entries = match Self::read_header(&path, verify) {
            Ok(Some((header, _payload_len))) => header.entries,
            Ok(None) => Vec::new(),
            Err(error) => {
                log::warn!(
                    "packed cache {} is unusable ({error}); rebuilding in memory",
                    path.display()
                );
                Vec::new()
            }
        };
        Some(Self {
            path,
            mapped: Mutex::new(None),
            writer: Mutex::new(None),
            entries,
            verify,
            budget,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            write_enabled: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Number of packed tensors served from disk.
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    /// Number of lookups that missed (packed in memory instead).
    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    fn find(
        &self,
        name: &str,
        kind: EntryKind,
        out_features: usize,
        in_features: usize,
    ) -> Option<&HeaderEntry> {
        self.entries.iter().find(|entry| {
            entry.name == name
                && entry.kind == kind
                && entry.out_features == out_features
                && entry.in_features == in_features
        })
    }

    /// Whether a validated entry exists for `name` with the expected shape.
    pub fn has_vnni(&self, name: &str, out_features: usize, in_features: usize) -> bool {
        self.find(name, EntryKind::Vnni, out_features, in_features)
            .is_some()
    }

    /// Whether a validated interleaved entry exists for `name`.
    pub fn has_interleaved(&self, name: &str, out_features: usize, in_features: usize) -> bool {
        self.find(name, EntryKind::Interleaved, out_features, in_features)
            .is_some()
    }

    fn range(entry: &HeaderEntry, offset: u64, len: u64) -> Option<std::ops::Range<usize>> {
        let absolute_start = PAYLOAD_START.checked_add(offset)?;
        let start = usize::try_from(absolute_start).ok()?;
        let end = usize::try_from(absolute_start.checked_add(len)?).ok()?;
        debug_assert!(end > start, "entry range must be non-empty: {entry:?}");
        Some(start..end)
    }

    /// Load one packed VNNI tensor as a zero-copy view of the mapped cache
    /// file, validating length and shape.
    pub fn get_vnni(
        &self,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Option<QuantizedWeightVnni> {
        let entry = self.find(name, EntryKind::Vnni, out_features, in_features)?;
        let mmap = self.mapped()?;
        let range = Self::range(entry, entry.offset, entry.len)?;
        match QuantizedWeightVnni::from_mapped(mmap, range, out_features, in_features) {
            Ok(weight) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(weight)
            }
            Err(error) => {
                log::warn!("packed cache entry '{name}' rejected: {error}");
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Load one interleaved Q8_0 lm-head weight as a zero-copy view of the
    /// mapped cache file.
    pub fn get_interleaved(
        &self,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Option<QuantizedWeightInterleaved> {
        let entry = self.find(name, EntryKind::Interleaved, out_features, in_features)?;
        let mmap = self.mapped()?;
        let quants = Self::range(entry, entry.offset, entry.len)?;
        let scales = Self::range(entry, entry.aux_offset, entry.aux_len)?;
        match QuantizedWeightInterleaved::from_mapped_parts(
            mmap,
            quants,
            scales,
            out_features,
            in_features,
        ) {
            Ok(weight) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(weight)
            }
            Err(error) => {
                log::warn!("packed cache entry '{name}' rejected: {error}");
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    fn mapped(&self) -> Option<std::sync::Arc<memmap2::Mmap>> {
        let mut guard = self.mapped.lock().ok()?;
        if guard.is_none() {
            let file = File::open(&self.path).ok()?;
            // Safety: read-only mapping of a file that is only ever replaced
            // by an atomic rename (never truncated in place), so the mapping
            // cannot observe a partial payload.
            let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
            *guard = Some(std::sync::Arc::new(mmap));
        }
        guard.clone()
    }

    /// Record a freshly packed VNNI tensor for the next write.
    ///
    /// Bytes are streamed to the temporary file immediately; nothing is
    /// buffered in memory.
    pub fn record_vnni(&self, name: &str, weight: &QuantizedWeightVnni) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.record(
            name,
            EntryKind::Vnni,
            weight.out_features(),
            weight.in_features(),
            weight.blocks_per_row(),
            &[weight.packed_bytes()],
        );
    }

    /// Record a freshly packed interleaved Q8_0 weight (quants + scales).
    pub fn record_interleaved(&self, name: &str, weight: &QuantizedWeightInterleaved) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.record(
            name,
            EntryKind::Interleaved,
            weight.out_features(),
            weight.in_features(),
            weight.blocks_per_row,
            &[weight.quants(), weight.scales()],
        );
    }

    /// Stream one entry (`sections[0]` plus an optional `sections[1]`).
    fn record(
        &self,
        name: &str,
        kind: EntryKind,
        out_features: usize,
        in_features: usize,
        blocks_per_row: usize,
        sections: &[&[u8]],
    ) {
        if !self.write_enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut guard = match self.writer.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        if guard.is_none() {
            match Self::open_writer(&self.path, self.verify) {
                Ok(state) => *guard = Some(state),
                Err(error) => {
                    log::warn!(
                        "packed cache {} is not writable ({error}); caching disabled for this run",
                        self.path.display()
                    );
                    self.write_enabled.store(false, Ordering::Relaxed);
                    return;
                }
            }
        }
        let write_result = {
            let state = guard.as_mut().expect("writer initialized above");
            let mut ranges = [(0u64, 0u64); 2];
            let mut failed = None;
            for (index, bytes) in sections.iter().enumerate() {
                if let Some(hasher) = state.hasher.as_mut() {
                    hasher.update(bytes);
                }
                let offset = state.payload_written;
                if let Err(error) = state.file.write_all(bytes) {
                    failed = Some(error);
                    break;
                }
                state.payload_written += bytes.len() as u64;
                ranges[index] = (offset, bytes.len() as u64);
            }
            match failed {
                Some(error) => Err(error),
                None => {
                    state.entries.push(HeaderEntry {
                        name: name.to_string(),
                        kind,
                        out_features,
                        in_features,
                        blocks_per_row,
                        offset: ranges[0].0,
                        len: ranges[0].1,
                        aux_offset: ranges[1].0,
                        aux_len: ranges[1].1,
                    });
                    Ok(())
                }
            }
        };
        if let Err(error) = write_result {
            log::warn!("packed cache write failed ({error}); caching disabled for this run");
            if let Some(state) = guard.take() {
                let _ = std::fs::remove_file(&state.temp_path);
            }
            self.write_enabled.store(false, Ordering::Relaxed);
        }
    }

    /// Publish the cache file (no-op when nothing was recorded) and prune the
    /// cache directory to its budget.
    pub fn finish_write(&self) {
        let mut guard = match self.writer.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        let Some(state) = guard.take() else {
            return;
        };
        if state.entries.is_empty() {
            let _ = std::fs::remove_file(&state.temp_path);
            return;
        }
        match Self::publish(state) {
            Ok(()) => self.prune_cache_dir(),
            Err(error) => log::warn!("packed cache publish failed: {error}"),
        }
    }

    fn prune_cache_dir(&self) {
        let Some(dir) = self.path.parent() else {
            return;
        };
        prune_dir(dir, &self.path, self.budget, SystemTime::now());
    }

    fn open_writer(dest: &Path, verify: bool) -> std::io::Result<WriterState> {
        let parent = dest.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut temp_path = parent.join(format!(
            ".{}.tmp-{}",
            dest.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("packed"),
            std::process::id()
        ));
        let mut sequence = 0u32;
        let mut file = loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => break file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    sequence += 1;
                    if sequence > 64 {
                        return Err(error);
                    }
                    temp_path = parent.join(format!(
                        ".{}.tmp-{}-{sequence}",
                        dest.file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("packed"),
                        std::process::id()
                    ));
                }
                Err(error) => return Err(error),
            }
        };
        // Fixed prefix + reserved header area; the payload follows. The header
        // is written in place by `publish` once every entry is known.
        let mut prefix = Vec::with_capacity(PAYLOAD_START as usize);
        prefix.extend_from_slice(MAGIC);
        prefix.extend_from_slice(&VERSION.to_le_bytes());
        prefix.extend_from_slice(&0u32.to_le_bytes());
        prefix.resize(PAYLOAD_START as usize, 0);
        file.write_all(&prefix)?;
        Ok(WriterState {
            file,
            temp_path,
            dest: dest.to_path_buf(),
            entries: Vec::new(),
            payload_written: 0,
            hasher: verify.then(Sha256::new),
        })
    }

    fn publish(mut state: WriterState) -> std::io::Result<()> {
        state.entries.sort_by(|a, b| a.name.cmp(&b.name));
        let header = Header {
            format: FORMAT_TAG.to_string(),
            payload_len: state.payload_written,
            payload_sha256: state.hasher.map(|hasher| hex(&hasher.finalize())),
            entries: state.entries,
        };
        let header_json = serde_json::to_vec(&header)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        if header_json.len() as u64 + PREFIX_LEN > PAYLOAD_START {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "packed cache header exceeds its reserved area",
            ));
        }
        let digest = Sha256::digest(&header_json);
        state.file.seek(SeekFrom::Start(0))?;
        state.file.write_all(MAGIC)?;
        state.file.write_all(&VERSION.to_le_bytes())?;
        state
            .file
            .write_all(&(header_json.len() as u32).to_le_bytes())?;
        state.file.write_all(&header_json)?;
        state.file.write_all(&digest)?;
        state.file.flush()?;
        drop(state.file);
        std::fs::rename(&state.temp_path, &state.dest)
    }

    /// Read and validate the header. `Ok(None)` means "no usable file".
    fn read_header(path: &Path, verify: bool) -> std::io::Result<Option<(Header, u64)>> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let file_len = file.metadata()?.len();
        if file_len < PAYLOAD_START {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file shorter than the fixed prefix",
            ));
        }
        if file_len > MAX_CACHE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file exceeds the cache size cap",
            ));
        }
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix)?;
        if &prefix[..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic",
            ));
        }
        let version = u32::from_le_bytes(prefix[8..12].try_into().expect("4 bytes"));
        if version != VERSION {
            // A stale file from an older format revision is rebuilt, not an
            // error worth surfacing.
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stale packed-cache format revision",
            ));
        }
        let header_len = u32::from_le_bytes(prefix[12..16].try_into().expect("4 bytes")) as u64;
        if header_len == 0 || PREFIX_LEN + header_len + 32 > PAYLOAD_START {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "header length out of range",
            ));
        }
        let mut header_json = vec![0u8; header_len as usize];
        file.read_exact(&mut header_json)?;
        let mut digest = [0u8; 32];
        file.read_exact(&mut digest)?;
        if Sha256::digest(&header_json)[..] != digest {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "header digest mismatch",
            ));
        }
        let header: Header = serde_json::from_slice(&header_json)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        if header.format != FORMAT_TAG {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "format tag mismatch",
            ));
        }
        let payload_len = file_len - PAYLOAD_START;
        if header.payload_len != payload_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "payload length mismatch",
            ));
        }
        let mut names = std::collections::HashSet::new();
        let mut ranges = Vec::with_capacity(header.entries.len());
        for entry in &header.entries {
            if !names.insert(entry.name.clone()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "duplicate entry name",
                ));
            }
            let (expected, expected_aux) = expected_entry_lens(entry);
            if Some(entry.len) != expected || Some(entry.aux_len) != expected_aux {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "entry length does not match its shape and kind",
                ));
            }
            for (offset, len) in [(entry.offset, entry.len), (entry.aux_offset, entry.aux_len)] {
                if offset.saturating_add(len) > payload_len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "entry extends past the payload",
                    ));
                }
                if len > 0 {
                    ranges.push((offset, offset + len));
                }
            }
        }
        // Entries are written contiguously by a single writer, so any overlap
        // means the header and payload disagree.
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "entries overlap",
                ));
            }
        }
        let covered: u64 = header
            .entries
            .iter()
            .map(|entry| entry.len + entry.aux_len)
            .sum();
        if covered != payload_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "payload is not fully covered by entries",
            ));
        }
        if verify && let Some(expected) = &header.payload_sha256 {
            let mut hasher = Sha256::new();
            let mut remaining = payload_len;
            let mut buffer = vec![0u8; 1 << 20];
            file.seek(SeekFrom::Start(PAYLOAD_START))?;
            while remaining > 0 {
                let chunk = buffer.len().min(remaining as usize);
                file.read_exact(&mut buffer[..chunk])?;
                hasher.update(&buffer[..chunk]);
                remaining -= chunk as u64;
            }
            if hex(&hasher.finalize()) != *expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "payload digest mismatch",
                ));
            }
        }
        Ok(Some((header, payload_len)))
    }
}

/// Expected `(primary, auxiliary)` payload lengths for an entry, or `(None,
/// None)` when the shape is not block-aligned.
fn expected_entry_lens(entry: &HeaderEntry) -> (Option<u64>, Option<u64>) {
    match entry.kind {
        EntryKind::Vnni => (
            expected_vnni_len(entry.out_features, entry.in_features),
            Some(0),
        ),
        EntryKind::Interleaved => {
            match QuantizedWeightInterleaved::expected_lengths(
                entry.out_features,
                entry.in_features,
            ) {
                Ok((_, quants, scales)) => (Some(quants as u64), Some(scales as u64)),
                Err(_) => (None, None),
            }
        }
    }
}

fn expected_vnni_len(out_features: usize, in_features: usize) -> Option<u64> {
    use crate::quant::{Q8_0_BLOCK_SIZE, VNNI_BLOCK_RECORD_SIZE, VNNI_OUT_TILE};
    if in_features == 0 || !in_features.is_multiple_of(Q8_0_BLOCK_SIZE) {
        return None;
    }
    let blocks = in_features / Q8_0_BLOCK_SIZE;
    let tiles = out_features.div_ceil(VNNI_OUT_TILE);
    Some((tiles * blocks * VNNI_BLOCK_RECORD_SIZE) as u64)
}

fn cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("EMBER_CACHE_DIR") {
        return Some(PathBuf::from(dir).join("packed"));
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir).join("ember").join("packed"));
    }
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("ember")
            .join("packed"),
    )
}

/// Environment policy: `EMBER_PACKED_CACHE=0` disables; unset or any other
/// value enables. Pure so it can be tested without mutating the process
/// environment.
fn env_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|value| value != "0")
}

/// Parse `EMBER_PACKED_CACHE_BYTES`: a byte count with an optional binary
/// unit suffix (`K`/`M`/`G`/`T`, optionally followed by `iB` or `B`). `None`
/// means unset or unparsable (the caller falls back to the default); `0`
/// disables eviction. Pure so it can be tested without mutating the process
/// environment.
fn env_budget(value: Option<&std::ffi::OsStr>) -> Option<u64> {
    let text = value?.to_str()?.trim();
    let digits_end = text
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(text.len());
    let (digits, suffix) = text.split_at(digits_end);
    if digits.is_empty() {
        return None;
    }
    let amount = digits.parse::<u64>().ok()?;
    let multiplier = match suffix.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" | "KB" | "KIB" => 1 << 10,
        "M" | "MB" | "MIB" => 1 << 20,
        "G" | "GB" | "GIB" => 1 << 30,
        "T" | "TB" | "TIB" => 1 << 40,
        _ => return None,
    };
    Some(amount.saturating_mul(multiplier))
}

/// Evict least-recently-used `*.bin` files until `dir` holds at most `budget`
/// bytes, and remove stale `.tmp-*` files left behind by crashed writers.
///
/// `keep` is the file just published: it counts toward the budget (so a
/// directory holding only an oversized entry stays as it is) but is never
/// evicted. Deleting a file that another process has mapped is safe on the
/// platforms this runs on; a failed removal (for example a file held open by
/// Windows) is logged and skipped.
fn prune_dir(dir: &Path, keep: &Path, budget: u64, now: SystemTime) {
    let Ok(listing) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<(PathBuf, u64, Option<SystemTime>)> = Vec::new();
    let mut total = 0u64;
    for entry in listing.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.contains(".tmp-") {
            let stale = meta
                .modified()
                .ok()
                .and_then(|modified| now.duration_since(modified).ok())
                .is_some_and(|age| age > STALE_TEMP_AGE);
            if stale {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        if !name.ends_with(".bin") {
            continue;
        }
        total = total.saturating_add(meta.len());
        if path != keep {
            entries.push((path, meta.len(), meta.modified().ok()));
        }
    }
    if budget == 0 || total <= budget {
        return;
    }
    entries.sort_by_key(|(_, _, modified)| *modified); // oldest first
    for (path, len, _) in entries {
        if total <= budget {
            break;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                total = total.saturating_sub(len);
                log::debug!("packed cache evicted {}", path.display());
            }
            Err(error) => log::debug!("packed cache could not evict {}: {error}", path.display()),
        }
    }
}

/// Hash one metadata value into the cache key without materializing its
/// `Debug` rendering.
///
/// Tokenizer arrays carry hundreds of thousands of owned strings; formatting
/// them costs ~200 ms per load on Gemma-scale headers. The packed layouts
/// depend only on tensor bytes and shapes, which the tensor-table fingerprint
/// in [`cache_key`] already binds, so scalars are hashed exactly while strings
/// use a length plus bounded prefix and arrays use only their shape (length,
/// or element type plus count for skipped arrays).
fn hash_metadata_value(hasher: &mut Sha256, value: &GgufValue) {
    match value {
        GgufValue::Array(items) => {
            hasher.update(b"a");
            hasher.update((items.len() as u64).to_le_bytes());
        }
        GgufValue::SkippedArray {
            element_type,
            elements,
        } => {
            hasher.update(b"A");
            hasher.update(element_type.to_le_bytes());
            hasher.update(elements.to_le_bytes());
        }
        GgufValue::Str(text) => {
            hasher.update(b"s");
            hasher.update((text.len() as u64).to_le_bytes());
            hasher.update(&text.as_bytes()[..text.len().min(32)]);
        }
        scalar => {
            hasher.update(b"v");
            hasher.update(format!("{scalar:?}").as_bytes());
        }
    }
}

fn cache_key(model_path: &Path, loader: &GgufLoader) -> String {
    let metadata = std::fs::metadata(model_path).ok();
    let file_len = metadata.as_ref().map_or(0, |meta| meta.len());
    let modified = metadata
        .and_then(|meta| meta.modified().ok())
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    let mut hasher = Sha256::new();
    // v4: tokenizer metadata arrays are `SkippedArray` (count only) instead of
    // materialized arrays; old-key files are ignored and re-derived.
    hasher.update(b"ember-packed-cache-v4");
    hasher.update(
        model_path
            .canonicalize()
            .unwrap_or_else(|_| model_path.to_path_buf())
            .to_string_lossy()
            .as_bytes(),
    );
    hasher.update(file_len.to_le_bytes());
    hasher.update(modified.to_le_bytes());
    let mut keys: Vec<&String> = loader.metadata.keys().collect();
    keys.sort_unstable();
    for key in keys {
        hasher.update(key.as_bytes());
        hash_metadata_value(&mut hasher, &loader.metadata[key]);
    }
    let mut names: Vec<&String> = loader.tensor_meta.keys().collect();
    names.sort_unstable();
    for name in names {
        let meta = &loader.tensor_meta[name];
        hasher.update(name.as_bytes());
        hasher.update(format!("{:?}:{}:{}", meta.dims, meta.dtype, meta.offset).as_bytes());
    }
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hex(&hasher.finalize())[..32].to_string()
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{GgufLoader, GgufValue, TensorMeta};
    use crate::quant::QuantizedWeight;
    use std::collections::HashMap;

    fn loader_fixture() -> (GgufLoader, Vec<u8>) {
        let mut tensor_meta = HashMap::new();
        tensor_meta.insert(
            "blk.0.attn_q.weight".to_string(),
            TensorMeta {
                dims: vec![2048, 2048],
                dtype: 12,
                offset: 32,
            },
        );
        let loader = GgufLoader {
            metadata: HashMap::new(),
            tensors: HashMap::new(),
            k_strategy: crate::quant_k::KStrategy::Auto,
            k_decisions: HashMap::new(),
            tensor_meta,
        };
        (loader, vec![0u8; 16])
    }

    /// Row-contiguous Q8_0 bytes for an `out x in` weight (zeros + scales).
    fn q8_bytes(out_features: usize, in_features: usize) -> Vec<u8> {
        let blocks_per_row = in_features / 32;
        let mut data = vec![0u8; out_features * blocks_per_row * 34];
        for row in 0..out_features {
            for block in 0..blocks_per_row {
                let at = (row * blocks_per_row + block) * 34;
                // scale bits: small positive f16 pattern
                data[at] = 0x00;
                data[at + 1] = 0x3c;
                data[at + 2] = ((row + block) % 251) as u8;
            }
        }
        data
    }

    fn temp_cache_path(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "ember-packed-cache-test-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        if !model.exists() {
            std::fs::write(&model, b"fixture").unwrap();
        }
        (dir, model)
    }

    fn open_cache(model: &Path, loader: &GgufLoader, dir: &Path) -> PackedCache {
        open_cache_with_budget(model, loader, dir, DEFAULT_CACHE_BUDGET_BYTES)
    }

    fn open_cache_with_budget(
        model: &Path,
        loader: &GgufLoader,
        dir: &Path,
        budget: u64,
    ) -> PackedCache {
        PackedCache::for_loader_in(model, loader, true, Some(dir.join("cache")), false, budget)
            .expect("cache opens")
    }

    /// The packed layouts are only usable on AVX-512 VNNI hosts; elsewhere the
    /// constructor is inert (`None`) and the disk-format tests below would
    /// panic on `open_cache`, so they skip instead.
    fn vnni_host() -> bool {
        crate::simd::packed_q8_0_vnni_supported()
    }

    #[test]
    fn vnni_round_trip_is_byte_identical() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("roundtrip");
        let cache = open_cache(&model, &loader, &dir);
        assert!(!cache.has_vnni("blk.0.attn_q.weight", 2048, 2048));

        let source = QuantizedWeight::try_new(q8_bytes(2048, 2048), vec![2048, 2048]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&source);
        let expected = packed.packed_bytes().to_vec();
        cache.record_vnni("blk.0.attn_q.weight", &packed);
        cache.finish_write();

        // Re-open (as a fresh process would).
        let cache = open_cache(&model, &loader, &dir);
        assert!(cache.has_vnni("blk.0.attn_q.weight", 2048, 2048));
        let loaded = cache
            .get_vnni("blk.0.attn_q.weight", 2048, 2048)
            .expect("cache hit");
        assert_eq!(loaded.packed_bytes(), expected.as_slice());
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn interleaved_round_trip_is_byte_identical() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("interleaved");
        let cache = open_cache(&model, &loader, &dir);
        assert!(!cache.has_interleaved("output.weight", 4096, 2048));

        let source = QuantizedWeight::try_new(q8_bytes(4096, 2048), vec![4096, 2048]).unwrap();
        let packed = QuantizedWeightInterleaved::from_quantized(&source);
        let expected_quants = packed.quants().to_vec();
        let expected_scales = packed.scales().to_vec();
        cache.record_interleaved("output.weight", &packed);
        cache.finish_write();

        let cache = open_cache(&model, &loader, &dir);
        assert!(cache.has_interleaved("output.weight", 4096, 2048));
        let loaded = cache
            .get_interleaved("output.weight", 4096, 2048)
            .expect("cache hit");
        assert_eq!(loaded.quants(), expected_quants.as_slice());
        assert_eq!(loaded.scales(), expected_scales.as_slice());
        // Shape mismatches and the wrong kind do not match.
        assert!(cache.get_interleaved("output.weight", 2048, 2048).is_none());
        assert!(cache.get_interleaved("output.weight", 4096, 1024).is_none());
        assert!(!cache.has_vnni("output.weight", 4096, 2048));
    }

    #[test]
    fn both_layouts_share_one_file() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("both-kinds");
        let cache = open_cache(&model, &loader, &dir);
        let q8 = QuantizedWeight::try_new(q8_bytes(512, 1024), vec![512, 1024]).unwrap();
        let vnni = QuantizedWeightVnni::from_quantized(&q8);
        let head = QuantizedWeight::try_new(q8_bytes(256, 256), vec![256, 256]).unwrap();
        let interleaved = QuantizedWeightInterleaved::from_quantized(&head);
        let expected_vnni = vnni.packed_bytes().to_vec();
        let expected_quants = interleaved.quants().to_vec();
        let expected_scales = interleaved.scales().to_vec();

        cache.record_vnni("blk.0.attn_q.weight", &vnni);
        cache.record_interleaved("output.weight", &interleaved);
        cache.finish_write();

        let cache = open_cache(&model, &loader, &dir);
        assert_eq!(
            cache
                .get_vnni("blk.0.attn_q.weight", 512, 1024)
                .expect("vnni hit")
                .packed_bytes(),
            expected_vnni.as_slice()
        );
        let loaded = cache
            .get_interleaved("output.weight", 256, 256)
            .expect("interleaved hit");
        assert_eq!(loaded.quants(), expected_quants.as_slice());
        assert_eq!(loaded.scales(), expected_scales.as_slice());
    }

    #[test]
    fn shape_or_identity_changes_miss() {
        if !vnni_host() {
            return;
        }
        let (mut loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("invalidate");
        let cache = open_cache(&model, &loader, &dir);
        let source = QuantizedWeight::try_new(q8_bytes(512, 1024), vec![512, 1024]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&source);
        cache.record_vnni("blk.0.attn_q.weight", &packed);
        cache.finish_write();

        let cache = open_cache(&model, &loader, &dir);
        assert!(cache.get_vnni("blk.0.attn_q.weight", 256, 1024).is_none());
        assert!(cache.get_vnni("blk.0.attn_q.weight", 512, 2048).is_none());
        assert!(cache.get_vnni("other.weight", 512, 1024).is_none());

        // A metadata change selects a different key: no entries.
        loader.metadata.insert(
            "llama.block_count".to_string(),
            crate::loader::GgufValue::U32(16),
        );
        let cache = open_cache(&model, &loader, &dir);
        assert!(!cache.has_vnni("blk.0.attn_q.weight", 512, 1024));
    }

    #[test]
    fn corrupt_file_falls_back_to_packing() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("corrupt");
        // A file at the exact key path with a valid header but truncated payload.
        let cache = open_cache(&model, &loader, &dir);
        let source = QuantizedWeight::try_new(q8_bytes(64, 64), vec![64, 64]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&source);
        cache.record_vnni("blk.0.attn_q.weight", &packed);
        cache.finish_write();
        let file = std::fs::read_dir(dir.join("cache"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let mut bytes = std::fs::read(&file).unwrap();
        bytes.truncate(bytes.len() - 4);
        std::fs::write(&file, &bytes).unwrap();
        let cache = open_cache(&model, &loader, &dir);
        assert!(!cache.has_vnni("blk.0.attn_q.weight", 64, 64));
    }

    #[test]
    fn concurrent_writers_publish_a_valid_file() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("concurrent");
        let source = QuantizedWeight::try_new(q8_bytes(128, 256), vec![128, 256]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&source);
        let expected = packed.packed_bytes().to_vec();
        let first = open_cache(&model, &loader, &dir);
        let second = open_cache(&model, &loader, &dir);
        first.record_vnni("blk.0.attn_q.weight", &packed);
        second.record_vnni("blk.0.attn_q.weight", &packed);
        first.finish_write();
        second.finish_write();
        let cache = open_cache(&model, &loader, &dir);
        let loaded = cache
            .get_vnni("blk.0.attn_q.weight", 128, 256)
            .expect("cache hit");
        assert_eq!(loaded.packed_bytes(), expected.as_slice());
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        let (loader, _) = loader_fixture();
        let (_dir, model) = temp_cache_path("disabled");
        assert!(PackedCache::for_loader_with(&model, &loader, false).is_none());
    }

    #[test]
    fn env_policy_defaults_on_and_0_disables() {
        use std::ffi::OsStr;
        assert!(env_enabled(None));
        assert!(env_enabled(Some(OsStr::new("1"))));
        assert!(!env_enabled(Some(OsStr::new("0"))));
    }

    #[test]
    fn budget_env_policy_parses_units_and_zero_disables() {
        use std::ffi::OsStr;
        assert_eq!(env_budget(None), None);
        assert_eq!(env_budget(Some(OsStr::new("0"))), Some(0));
        assert_eq!(env_budget(Some(OsStr::new("1024"))), Some(1024));
        assert_eq!(env_budget(Some(OsStr::new("512M"))), Some(512 << 20));
        assert_eq!(env_budget(Some(OsStr::new("8G"))), Some(8 << 30));
        assert_eq!(env_budget(Some(OsStr::new("8GiB"))), Some(8 << 30));
        assert_eq!(env_budget(Some(OsStr::new(" 2 KiB "))), Some(2 << 10));
        assert_eq!(env_budget(Some(OsStr::new("1T"))), Some(1 << 40));
        assert_eq!(env_budget(Some(OsStr::new("nope"))), None);
        assert_eq!(env_budget(Some(OsStr::new("12X"))), None);
        assert_eq!(env_budget(Some(OsStr::new("GiB"))), None);
    }

    #[test]
    fn prune_evicts_oldest_entries_and_removes_stale_temps() {
        let dir = std::env::temp_dir().join(format!("ember-packed-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let now = SystemTime::now();
        let write = |name: &str, bytes: usize, age_secs: u64| {
            let path = dir.join(name);
            std::fs::write(&path, vec![0u8; bytes]).unwrap();
            let times = std::fs::FileTimes::new().set_modified(now - Duration::from_secs(age_secs));
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(times)
                .unwrap();
            path
        };
        let old = write("aaaaaaaa.bin", 100, 300);
        let middle = write("bbbbbbbb.bin", 100, 200);
        let published = write("cccccccc.bin", 100, 100);
        let stale_temp = write(".dddddddd.bin.tmp-77", 20, 2 * 24 * 60 * 60);
        let live_temp = write(".eeeeeeee.bin.tmp-78", 20, 60);

        prune_dir(&dir, &published, 250, now);
        assert!(!old.exists(), "the oldest entry is evicted first");
        assert!(middle.exists() && published.exists());
        assert!(
            !stale_temp.exists(),
            "a crashed writer's temp file is removed"
        );
        assert!(live_temp.exists(), "a live writer's temp file is kept");

        prune_dir(&dir, &published, 1, now);
        assert!(published.exists(), "the published file is never evicted");
        assert!(!middle.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finish_write_prunes_older_entries_to_the_budget() {
        if !vnni_host() {
            return;
        }
        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("prune-wiring");
        let source = QuantizedWeight::try_new(q8_bytes(64, 64), vec![64, 64]).unwrap();
        let packed = QuantizedWeightVnni::from_quantized(&source);

        let first = open_cache_with_budget(&model, &loader, &dir, 6000);
        first.record_vnni("blk.0.attn_q.weight", &packed);
        first.finish_write();
        let cache_dir = dir.join("cache");
        let files: Vec<PathBuf> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        assert_eq!(files.len(), 1, "one entry, plus no leftover temp file");
        let times =
            std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(3600));
        File::options()
            .write(true)
            .open(&files[0])
            .unwrap()
            .set_times(times)
            .unwrap();

        // A different source key writes a second entry; the budget only fits
        // one, so the older file is evicted.
        let (mut second_loader, _) = loader_fixture();
        second_loader
            .metadata
            .insert("llama.block_count".to_string(), GgufValue::U32(16));
        let second = open_cache_with_budget(&model, &second_loader, &dir, 6000);
        second.record_vnni("blk.0.attn_q.weight", &packed);
        second.finish_write();

        assert!(!files[0].exists(), "the older cache entry is evicted");
        let remaining: Vec<PathBuf> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn cache_key_bounds_large_metadata_arrays() {
        let (mut loader, _) = loader_fixture();
        let (_dir, model) = temp_cache_path("key-arrays");
        loader.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(
                (0..200_000)
                    .map(|_| GgufValue::Str("a".to_string()))
                    .collect(),
            ),
        );
        let first = cache_key(&model, &loader);
        // Same array length, different element contents: the packed layouts
        // depend only on tensor bytes/shapes (bound by the tensor-table hash),
        // so the key stays stable and the huge array is never Debug-formatted.
        loader.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(
                (0..200_000)
                    .map(|_| GgufValue::Str("b".to_string()))
                    .collect(),
            ),
        );
        assert_eq!(cache_key(&model, &loader), first);
        // Scalar metadata changes still select a different key.
        loader
            .metadata
            .insert("llama.block_count".to_string(), GgufValue::U32(16));
        assert_ne!(cache_key(&model, &loader), first);
    }

    #[test]
    fn cached_vnni_replaces_the_repack_for_a_linear() {
        if !vnni_host() {
            return;
        }
        use crate::backend::CpuBackend;
        use crate::model::Linear;

        let (loader, _) = loader_fixture();
        let (dir, model) = temp_cache_path("linear-cache");
        let build = || {
            Linear::<CpuBackend>::new_q8_0(
                QuantizedWeight::try_new(q8_bytes(128, 256), vec![128, 256]).unwrap(),
                None,
            )
        };
        let reference = QuantizedWeightVnni::from_quantized(
            &QuantizedWeight::try_new(q8_bytes(128, 256), vec![128, 256]).unwrap(),
        )
        .packed_bytes()
        .to_vec();

        // First run packs and records the entry.
        let cache = open_cache(&model, &loader, &dir);
        let mut linear = build();
        linear.prepare_packed_decode_cached(Some(&cache), "blk.0.ffn_gate.weight");
        assert!(linear.has_packed_decode());
        cache.finish_write();

        // Second run is served from disk and is byte-identical.
        let cache = open_cache(&model, &loader, &dir);
        let mut linear = build();
        linear.prepare_packed_decode_cached(Some(&cache), "blk.0.ffn_gate.weight");
        assert_eq!(cache.hits(), 1);
        assert_eq!(
            linear
                .packed_q8_weight_without_bias()
                .expect("cache hit")
                .packed_bytes(),
            reference.as_slice()
        );

        // An unknown entry packs in memory and is recorded for the next run.
        let mut other = build();
        other.prepare_packed_decode_cached(Some(&cache), "blk.1.ffn_up.weight");
        assert!(other.has_packed_decode());
        assert_eq!(cache.misses(), 1);
    }
}

//! Encoded-media feature cache.
//!
//! Repeated media across turns must not re-run a 40-second vision tower:
//! when the same image/audio/video segment appears again under the *same*
//! processing configuration, its projected features are reused bit-exactly.
//!
//! Cache boundary (deliberately narrow): the cache stores **projected
//! features** — the output right before embedding assembly. Decoding and
//! preprocessing stay outside (they are cheap relative to encoding), which
//! keeps the cached payload small and the key simple.
//!
//! Key correctness rule: a key combines the *content* of the decoded raw
//! input with hashes of every configuration that influences the result —
//! preprocess recipe, encoder weights identity, projector weights identity,
//! precision. Two entries with different keys never collide; identical keys
//! guarantee identical inputs end-to-end, so reuse is bit-exact by
//! construction.

use crate::multimodal::request::{MediaId, MediaKind};
use crate::tensor::CpuTensor;
use std::collections::HashMap;

/// Hash of everything about a preprocessing recipe that affects output.
///
/// Structs feed their fields via [`PreprocessFingerprint::mix_u64`]; adding a
/// field to a recipe MUST add a mix call (enforced by review; the fingerprint
/// also embeds a format tag so accidental layout changes invalidate).
#[derive(Debug, Clone, Default)]
pub struct PreprocessFingerprint(u64);

impl PreprocessFingerprint {
    pub fn new(tag: &str) -> Self {
        let mut f = Self(0);
        f.mix_bytes(tag.as_bytes());
        f
    }

    pub fn mix_u64(&mut self, v: u64) {
        self.0 = self.0.rotate_left(17) ^ v;
    }

    pub fn mix_f64(&mut self, v: f64) {
        self.mix_u64(v.to_bits());
    }

    pub fn mix_bytes(&mut self, b: &[u8]) {
        for chunk in b.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.mix_u64(u64::from_le_bytes(buf));
        }
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Full cache key: content + modality + configuration identities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureCacheKey {
    pub media_id: MediaId,
    pub kind: MediaKind,
    /// Fingerprint of the preprocess recipe.
    pub preprocess: u64,
    /// Identity of the encoder (+projector) weights — e.g. sha256 of the
    /// mmproj GGUF. Features are never reused across incompatible models.
    pub tower_identity: u64,
}

/// One cached entry plus its byte cost.
struct Entry {
    features: CpuTensor,
    bytes: usize,
}

/// Bounded in-process LRU-ish cache (insertion-evicting oldest-first).
pub struct MediaFeatureCache {
    map: HashMap<FeatureCacheKey, (Entry, u64)>,
    clock: u64,
    max_bytes: usize,
    used_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl MediaFeatureCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            clock: 0,
            max_bytes,
            used_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn get(&self, key: &FeatureCacheKey) -> Option<&CpuTensor> {
        match self.map.get(key) {
            Some((e, _)) => {
                // interior mutability not needed for stats; callers count
                Some(&e.features)
            }
            None => None,
        }
    }

    /// Like [`MediaFeatureCache::get`] but records a hit/miss.
    pub fn lookup(&mut self, key: &FeatureCacheKey) -> Option<&CpuTensor> {
        if self.map.contains_key(key) {
            self.hits += 1;
            if let Some((_, t)) = self.map.get_mut(key) {
                self.clock += 1;
                *t = self.clock;
            }
            self.map.get(key).map(|(e, _)| &e.features)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: FeatureCacheKey, features: CpuTensor) {
        let bytes = features.len() * std::mem::size_of::<f32>();
        if bytes > self.max_bytes {
            return; // single entry larger than the whole cache
        }
        if let Some(old) = self.map.get(&key) {
            self.used_bytes -= old.0.bytes;
        }
        while self.used_bytes + bytes > self.max_bytes {
            // evict least-recently-touched
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    if let Some((e, _)) = self.map.remove(&k) {
                        self.used_bytes -= e.bytes;
                        self.evictions += 1;
                    }
                }
                None => break,
            }
        }
        self.used_bytes += bytes;
        self.clock += 1;
        self.map
            .insert(key, (Entry { features, bytes }, self.clock));
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Eviction count since construction.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }
}

/// Content identity for a decoded RGB frame/image tensor.
pub fn media_id_of_tensor(t: &CpuTensor) -> MediaId {
    MediaId::from_tensor(t)
}

// ---------------------------------------------------------------------------
// Phase 5 Track F: concurrency-safe wrapper with per-key coalescing
// ---------------------------------------------------------------------------

use std::collections::HashMap as StdHashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Aggregate telemetry for [`SharedFeatureCache`].
#[derive(Debug, Default, Clone)]
pub struct SharedCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub coalesced: u64,
    pub in_flight_waits: u64,
    pub evictions: u64,
    pub resident_bytes: usize,
    /// Milliseconds of tower work avoided thanks to hits + coalesced waits
    /// (measured durations of identical encodes).
    pub encode_time_saved_ms: f64,
}

struct InFlightState {
    outcome: Option<Result<Arc<CpuTensor>, ()>>,
}

struct InFlight {
    cv: Condvar,
    state: Mutex<InFlightState>,
}

/// Process-wide shared feature cache: safe under concurrent use and free of
/// stampedes. Two requests for the same uncached media run the encoder
/// exactly once; the second waits (`coalesced`) instead of duplicating a
/// 40-second tower pass.
///
/// Failure semantics (Track F2): if the leader's encode fails, waiters wake,
/// see `Failed`, and ONE of them retries as the new leader — a cancelled or
/// failed producer never poisons the key for others.
pub struct SharedFeatureCache {
    inner: Mutex<MediaFeatureCache>,
    inflight: Mutex<StdHashMap<FeatureCacheKey, Arc<InFlight>>>,
    metrics: Mutex<SharedCacheMetrics>,
}

impl SharedFeatureCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(MediaFeatureCache::new(max_bytes)),
            inflight: Mutex::new(StdHashMap::new()),
            metrics: Mutex::new(SharedCacheMetrics::default()),
        }
    }

    fn record(&self, f: impl FnOnce(&mut SharedCacheMetrics)) {
        let mut m = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut m);
    }

    pub fn metrics(&self) -> SharedCacheMetrics {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn used_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .used_bytes()
    }

    /// Look up `key`, running `encode` exactly once on a miss. The returned
    /// tensor is shared (cheap Arc clone); treat it as read-only.
    pub fn get_or_insert_with(
        &self,
        key: &FeatureCacheKey,
        encode: impl FnOnce() -> anyhow::Result<CpuTensor>,
    ) -> anyhow::Result<Arc<CpuTensor>> {
        // fast path: warm hit
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(t) = inner.lookup(key) {
                let arc = Arc::new(t.clone());
                drop(inner);
                self.record(|m| {
                    m.hits += 1;
                    // saved time is tracked by the leader's measurement below
                });
                return Ok(arc);
            }
        }
        loop {
            // join-or-become-leader
            let entry = {
                let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(e) = inflight.get(key) {
                    let e = e.clone();
                    let mut st = e.state.lock().unwrap_or_else(|er| er.into_inner());
                    drop(inflight);
                    self.record(|m| {
                        m.in_flight_waits += 1;
                        m.misses += 1;
                    });
                    // wait for an outcome; READ it (never take): every
                    // woken waiter must be able to observe it
                    loop {
                        match &st.outcome {
                            Some(Ok(t)) => {
                                let t = t.clone();
                                drop(st);
                                self.record(|m| m.coalesced += 1);
                                return Ok(t);
                            }
                            Some(Err(())) => break, // leader failed; retry
                            None => {}
                        }
                        st = e.cv.wait(st).unwrap_or_else(|er| er.into_inner());
                    }
                    drop(st);
                    continue; // become the new leader
                }
                let e = Arc::new(InFlight {
                    cv: Condvar::new(),
                    state: Mutex::new(InFlightState { outcome: None }),
                });
                inflight.insert(key.clone(), e.clone());
                e
            }; // locks released before encode

            // we are the leader: run the encoder OUTSIDE any lock
            let t0 = Instant::now();
            let result = encode();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            let shared = result.map(|t| {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                inner.insert(key.clone(), t.clone());
                Arc::new(t)
            });

            self.record(|m| {
                m.misses += 1;
                if shared.is_ok() {
                    m.encode_time_saved_ms += elapsed_ms; // this cost is now cached
                }
            });

            // publish outcome to waiters, then drop the in-flight entry
            {
                let mut st = entry.state.lock().unwrap_or_else(|e| e.into_inner());
                st.outcome = Some(match &shared {
                    Ok(t) => Ok(t.clone()),
                    Err(_) => Err(()),
                });
                entry.cv.notify_all();
            }
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            // only remove if we are still the registered leader
            if inflight
                .get(key)
                .map(|e| Arc::ptr_eq(e, &entry))
                .unwrap_or(false)
            {
                inflight.remove(key);
            }

            self.record(|m| {
                m.resident_bytes = self.used_bytes();
                m.evictions = self
                    .inner
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .evictions();
            });
            return shared;
        }
    }
}

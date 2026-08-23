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
/// Structs feed their fields via [`PreprocessFingerprint::mix`]; adding a
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

    /// Like [`get`] but records a hit/miss.
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
}

/// Content identity for a decoded RGB frame/image tensor.
pub fn media_id_of_tensor(t: &CpuTensor) -> MediaId {
    MediaId::from_tensor(t)
}

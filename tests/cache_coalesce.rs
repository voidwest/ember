//! Phase 5 Track F: cache coalescing under concurrency.
//!
//! Contract: two requests for the same uncached media execute the encoder
//! exactly once (stampede prevention); a failed/cancelled leader never
//! poisons the key for others; metrics count hits, misses, coalesced waits,
//! evictions and resident bytes honestly.

use ember::multimodal::cache::{FeatureCacheKey, PreprocessFingerprint, SharedFeatureCache};
use ember::multimodal::MediaKind;
use ember::tensor::CpuTensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn key(tag: u64) -> FeatureCacheKey {
    let mut fp = PreprocessFingerprint::new("test");
    fp.mix_u64(42);
    FeatureCacheKey {
        media_id: ember::multimodal::request::MediaId(0xdead_beef ^ tag),
        kind: MediaKind::Image,
        preprocess: fp.value(),
        tower_identity: 7,
    }
}

fn tensor(n: usize, fill: f32) -> CpuTensor {
    CpuTensor::from_data(vec![n], vec![fill; n])
}

#[test]
fn concurrent_cold_requests_encode_exactly_once() {
    let cache = Arc::new(SharedFeatureCache::new(64 * 1024 * 1024));
    let executions = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for i in 0..8 {
        let cache = cache.clone();
        let exec = executions.clone();
        handles.push(std::thread::spawn(move || {
            // stagger starts so some threads genuinely arrive mid-encode
            std::thread::sleep(Duration::from_millis(i * 15));
            cache
                .get_or_insert_with(&key(1), || {
                    exec.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(150)); // tower cost
                    Ok(tensor(1024, 1.0))
                })
                .expect("encode ok")
        }));
    }
    let results: Vec<Arc<CpuTensor>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "stampede must coalesce"
    );
    for r in &results {
        assert_eq!(r.shape(), &[1024]);
        assert_eq!(r.data()[0], 1.0);
    }
    let m = cache.metrics();
    assert_eq!(m.misses, 8);
    assert_eq!(m.coalesced, 7);
    assert!(m.encode_time_saved_ms >= 0.0);
    assert_eq!(m.resident_bytes, 1024 * 4);
}

#[test]
fn warm_hit_skips_encode() {
    let cache = SharedFeatureCache::new(64 * 1024 * 1024);
    let exec = AtomicUsize::new(0);
    cache
        .get_or_insert_with(&key(2), || {
            exec.fetch_add(1, Ordering::SeqCst);
            Ok(tensor(16, 2.0))
        })
        .unwrap();
    for _ in 0..5 {
        cache
            .get_or_insert_with(&key(2), || panic!("must not re-run"))
            .unwrap();
    }
    assert_eq!(exec.load(Ordering::SeqCst), 1);
    let m = cache.metrics();
    assert_eq!(m.hits, 5);
    assert_eq!(m.misses, 1);
}

#[test]
fn failed_leader_does_not_poison_key() {
    let cache = Arc::new(SharedFeatureCache::new(64 * 1024 * 1024));
    let attempts = Arc::new(AtomicUsize::new(0));

    // first attempt fails (simulated cancelled/failed encode)
    let a = attempts.clone();
    let r = cache.get_or_insert_with(&key(3), move || {
        a.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("encoder blew up"))
    });
    assert!(r.is_err());

    // the key must remain usable: a retry succeeds
    let a2 = attempts.clone();
    let t = cache
        .get_or_insert_with(&key(3), move || {
            a2.fetch_add(1, Ordering::SeqCst);
            Ok(tensor(8, 3.0))
        })
        .unwrap();
    assert_eq!(t.data()[0], 3.0);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[test]
fn waiting_threads_survive_leader_failure_and_retry() {
    let cache = Arc::new(SharedFeatureCache::new(64 * 1024 * 1024));
    let fail_first = Arc::new(AtomicUsize::new(0));
    let successes = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let cache = cache.clone();
        let ff = fail_first.clone();
        let okc = successes.clone();
        handles.push(std::thread::spawn(move || {
            cache
                .get_or_insert_with(&key(4), || {
                    if ff.fetch_add(1, Ordering::SeqCst) == 0 {
                        std::thread::sleep(Duration::from_millis(50));
                        Err(anyhow::anyhow!("first attempt dies"))
                    } else {
                        okc.fetch_add(1, Ordering::SeqCst);
                        Ok(tensor(4, 9.0))
                    }
                })
                .is_ok()
        }));
    }
    let oks: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // At most the failed LEADER may observe its own error; every other
    // participant must have been coalesced or retried into success, and
    // the key must never stay poisoned.
    let failures = oks.iter().filter(|&&ok| !ok).count();
    assert!(
        failures <= 1,
        "at most one thread (the failed leader) may see an error, got {failures}"
    );
    assert!(oks.iter().filter(|&&ok| ok).count() >= 3);
    assert!(successes.load(Ordering::SeqCst) >= 1);
    // and the cache remains usable afterwards
    let t = cache
        .get_or_insert_with(&key(4), || Ok(tensor(4, 9.0)))
        .unwrap();
    assert_eq!(t.data()[0], 9.0);
}

#[test]
fn different_keys_never_coalesce() {
    let cache = SharedFeatureCache::new(64 * 1024 * 1024);
    let exec = AtomicUsize::new(0);
    for tag in 10..14u64 {
        cache
            .get_or_insert_with(&key(tag), || {
                exec.fetch_add(1, Ordering::SeqCst);
                Ok(tensor(4, tag as f32))
            })
            .unwrap();
    }
    assert_eq!(
        exec.load(Ordering::SeqCst),
        4,
        "distinct content = distinct encodes"
    );
}

#[test]
fn eviction_keeps_budget_and_counts() {
    let cache = SharedFeatureCache::new(64); // tiny budget: ~4 floats per entry
    for tag in 20..30u64 {
        cache
            .get_or_insert_with(&key(tag), || Ok(tensor(4, tag as f32)))
            .unwrap();
        let m = cache.metrics();
        assert!(
            m.resident_bytes <= 64,
            "resident bytes {} exceeded budget",
            m.resident_bytes
        );
    }
    assert!(cache.metrics().evictions > 0, "evictions must be counted");
}

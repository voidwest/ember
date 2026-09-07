//! Phase 5 Track F: cache coalescing under concurrency.
//!
//! Contract: two requests for the same uncached media execute the encoder
//! exactly once (stampede prevention); a failed/cancelled leader never
//! poisons the key for others; metrics count hits, misses, coalesced waits,
//! evictions and resident bytes honestly.

use ember::multimodal::cache::{
    FeatureCacheKey, MediaFeatureCache, PreprocessFingerprint, SharedFeatureCache,
};
use ember::multimodal::MediaKind;
use ember::tensor::CpuTensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

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

fn wait_until(predicate: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !predicate() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::yield_now();
    }
    true
}

#[test]
fn concurrent_cold_requests_encode_exactly_once() {
    let cache = Arc::new(SharedFeatureCache::new(64 * 1024 * 1024));
    let executions = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(9));
    let finish = Arc::new((Mutex::new(false), Condvar::new()));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let cache = cache.clone();
        let exec = executions.clone();
        let start = start.clone();
        let finish = finish.clone();
        handles.push(std::thread::spawn(move || {
            start.wait();
            cache
                .get_or_insert_with(&key(1), || {
                    exec.fetch_add(1, Ordering::SeqCst);
                    let (lock, cv) = &*finish;
                    let (released, _) = cv
                        .wait_timeout_while(
                            lock.lock().unwrap(),
                            Duration::from_secs(10),
                            |ready| !*ready,
                        )
                        .unwrap();
                    assert!(*released, "test did not release the encoder");
                    Ok(tensor(1024, 1.0))
                })
                .expect("encode ok")
        }));
    }
    start.wait();
    let all_waiting = wait_until(|| cache.metrics().in_flight_waits == 7);
    *finish.0.lock().unwrap() = true;
    finish.1.notify_all();
    let results: Vec<Arc<CpuTensor>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        all_waiting,
        "all followers must join the leader before release"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "stampede must coalesce"
    );
    for r in &results {
        assert!(Arc::ptr_eq(r, &results[0]));
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
    let original = cache
        .get_or_insert_with(&key(2), || {
            exec.fetch_add(1, Ordering::SeqCst);
            Ok(tensor(16, 2.0))
        })
        .unwrap();
    for _ in 0..5 {
        let hit = cache
            .get_or_insert_with(&key(2), || panic!("must not re-run"))
            .unwrap();
        assert!(Arc::ptr_eq(&original, &hit), "warm hits must share storage");
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
    assert_eq!(successes.load(Ordering::SeqCst), 1);
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

#[test]
fn replacing_oldest_entry_accounts_for_its_bytes_once() {
    let mut cache = MediaFeatureCache::new(64);
    cache.insert(key(1), tensor(8, 1.0));
    cache.insert(key(2), tensor(8, 2.0));
    cache.insert(key(1), tensor(12, 3.0));

    assert_eq!(cache.used_bytes(), 48);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.evictions(), 1);
    assert!(cache.get(&key(2)).is_none());
    assert_eq!(cache.get(&key(1)).unwrap().data(), &[3.0; 12]);

    // Replacing with a smaller entry frees exactly the size difference.
    cache.insert(key(1), tensor(4, 4.0));
    cache.insert(key(2), tensor(12, 5.0));
    assert_eq!(cache.used_bytes(), 64);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.evictions(), 1);
}

#[test]
fn replacing_large_oldest_entry_cannot_underflow_accounting() {
    let mut cache = MediaFeatureCache::new(64);
    cache.insert(key(1), tensor(12, 1.0));
    cache.insert(key(2), tensor(4, 2.0));
    cache.insert(key(1), tensor(16, 3.0));
    assert_eq!(cache.used_bytes(), 64);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.evictions(), 1);
    assert_eq!(cache.lookup(&key(1)).unwrap().len(), 16);

    // Oversized replacements leave the previous cached value intact.
    cache.insert(key(1), tensor(17, 4.0));
    assert_eq!(cache.used_bytes(), 64);
    assert_eq!(cache.get(&key(1)).unwrap().data(), &[3.0; 16]);
}

#[test]
fn concurrent_cheap_encodes_do_not_register_a_leader_after_cache_fill() {
    let cache = Arc::new(SharedFeatureCache::new(64 * 1024));
    let executions = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cache = cache.clone();
            let executions = executions.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                (0..64)
                    .map(|tag| {
                        start.wait();
                        cache
                            .get_or_insert_with(&key(tag), || {
                                executions.fetch_add(1, Ordering::SeqCst);
                                Ok(tensor(4, tag as f32))
                            })
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(executions.load(Ordering::SeqCst), 64);
    for row in &results {
        for (value, first) in row.iter().zip(&results[0]) {
            assert!(Arc::ptr_eq(value, first));
        }
    }
    assert_eq!(cache.used_bytes(), 64 * 4 * 4);
}

#[test]
fn panicking_leader_wakes_waiter_and_retries() {
    let cache = Arc::new(SharedFeatureCache::new(1024));
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (panic_tx, panic_rx) = std::sync::mpsc::channel();
    let leader_cache = cache.clone();
    let leader = std::thread::spawn(move || {
        leader_cache.get_or_insert_with(&key(99), || {
            started_tx.send(()).unwrap();
            panic_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            panic!("test encoder panic");
        })
    });
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let waiter_cache = cache.clone();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let result = waiter_cache.get_or_insert_with(&key(99), || Ok(tensor(4, 9.0)));
        done_tx.send(result).unwrap();
    });
    let waiting = wait_until(|| cache.metrics().in_flight_waits == 1);
    panic_tx.send(()).unwrap();
    assert!(leader.join().is_err());
    let result = done_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    waiter.join().unwrap();
    assert!(
        waiting,
        "the follower must be waiting when the leader panics"
    );
    assert_eq!(result.data(), &[9.0; 4]);
    let hit = cache
        .get_or_insert_with(&key(99), || panic!("must hit"))
        .unwrap();
    assert!(Arc::ptr_eq(&result, &hit));
}

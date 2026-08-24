//! Cross-request vision batching benchmark: N independent single-image
//! requests encoded sequentially vs one grouped batched encode.
//!
//! Interleaved baseline/candidate runs with medians (thermal hygiene);
//! skips silently when the local mmproj fixture is absent.

use std::time::Instant;

fn main() {
    let mmproj = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            // `cargo test --all-targets` executes harness=false benches with
            // no arguments; skip silently like the absent-fixture path below
            // instead of failing the CI test run (this exact exit(2) turned
            // the v0.6.5 release CI red on both tiers).
            eprintln!("skip: usage: multimodal_batch_bench <mmproj.gguf> [n_images]");
            return;
        }
    };
    let n_images: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    use ember::backend::CpuBackend;
    use ember::loader::load_gguf;
    use ember::tensor::CpuTensor;
    let backend = CpuBackend;
    let mut loader = match load_gguf(std::path::Path::new(&mmproj)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("skip: cannot load {mmproj}: {e}");
            return;
        }
    };
    let vision = ember::multimodal::vision::VisionModel::from_mmproj_loader(&mut loader).unwrap();

    // synthetic normalized tiles at tower input size
    let size = vision.transformer.config.image_size;
    // deterministic "images": distinct fill values so no cache tricks apply
    let images: Vec<CpuTensor> = (0..n_images)
        .map(|i| {
            let mut t = vec![0.25f32; 3 * size * size];
            for v in t.iter_mut().step_by(97) {
                *v = 0.1 + 0.01 * i as f32;
            }
            CpuTensor::from_data(vec![1, 3, size, size], t)
        })
        .collect();

    let seq_once = || {
        for img in &images {
            let _ = vision.encode(&backend, img).unwrap();
        }
    };
    let batch_once = || {
        let batch = {
            // concatenate all tiles into [n,3,size,size]
            let per = 3 * size * size;
            let mut data = Vec::with_capacity(n_images * per);
            for img in &images {
                data.extend_from_slice(img.data());
            }
            CpuTensor::from_data(vec![n_images, 3, size, size], data)
        };
        let _ = vision.encode(&backend, &batch).unwrap();
    };

    // warm-up both paths once
    seq_once();
    batch_once();

    const ITERS: usize = 3;
    let mut seq_ms = Vec::with_capacity(ITERS);
    let mut bat_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        batch_once();
        bat_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        // cooldown gap reduces (but does not eliminate) thermal coupling on
        // this host; treat results as lower bounds either way
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let t0 = Instant::now();
        seq_once();
        seq_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        std::thread::sleep(std::time::Duration::from_millis(1500));
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (seq_med, bat_med) = (median(seq_ms), median(bat_ms));
    println!(
        "n_images={n_images} tile={size}px sequential_median={:.1}ms batched_median={:.1}ms speedup={:.2}x",
        seq_med,
        bat_med,
        seq_med / bat_med
    );
}

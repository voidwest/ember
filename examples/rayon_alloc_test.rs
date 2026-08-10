fn main() {
    let mut v = vec![0.0f32; 65536];
    fn fill_range(v: &mut [f32], _start: usize, len: usize) {
        const CHUNK: usize = 256;
        if len <= CHUNK {
            for x in v[..len].iter_mut() {
                *x = 1.0;
            }
        } else {
            let half = len / 2;
            let (lo, hi) = v.split_at_mut(half);
            rayon::join(
                || fill_range(lo, _start, half),
                || fill_range(hi, _start + half, len - half),
            );
        }
    }
    // warm
    fill_range(&mut v, 0, 65536);
    for i in 0..3 {
        let (_, a) = ember::alloc_counter::count_allocations(|| {
            {
                // global pool install: build once, reuse
                static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
                let pool = POOL.get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(8)
                        .build()
                        .unwrap()
                });
                pool.install(|| fill_range(&mut v, 0, 65536));
            }
        });
        eprintln!("install+join call {i}: {a} allocs");
    }
}

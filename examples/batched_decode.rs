//! Batched multi-sequence decode: real-model bit-identity check and
//! throughput benchmark.
//!
//! ```text
//! cargo run --release --no-default-features --example batched_decode -- \
//!     <model.gguf> [batch sizes, default 1,2,4,8,17] [timed steps, default 32]
//! ```
//!
//! For every batch size `N` it prefills `N` distinct prompts, checks that
//! three batched greedy steps reproduce independent single-sequence decodes
//! bit for bit (logits of every sequence), then times batched steps and the
//! same sequences decoded one at a time, and prints JSON lines with aggregate,
//! per-sequence and sequential single-decode tokens/s.

use ember::backend::CpuBackend;
use ember::kv_cache::KVCache;
use ember::llama::{DecodeBatchSequence, Llama};
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::quant_k::KStrategy;
use std::time::Instant;

const VERIFY_STEPS: usize = 3;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

fn prompt(sequence: usize) -> Vec<u32> {
    // Distinct, deterministic prompts of different lengths (ids are plain
    // vocabulary tokens of the Llama 3 tokenizer range).
    (0..3 + sequence % 5)
        .map(|index| 1000 + ((sequence * 7919 + index * 104_729) % 20_000) as u32)
        .collect()
}

struct Batch {
    caches: Vec<KVCache>,
    positions: Vec<usize>,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn prefill(model: &Llama<CpuBackend>, backend: &CpuBackend, n: usize, capacity: usize) -> Batch {
    let vocab = model.config.vocab_size;
    let mut batch = Batch {
        caches: Vec::new(),
        positions: Vec::new(),
        tokens: Vec::new(),
        logits: vec![vec![0.0; vocab]; n],
    };
    for sequence in 0..n {
        let ids = prompt(sequence);
        let mut cache = model.create_cache(backend, capacity);
        let logits =
            ForwardModel::forward_last_logits_with_cache(model, backend, &ids, &mut cache, 0)
                .expect("prefill");
        batch.tokens.push(argmax(logits.data()));
        batch.positions.push(ids.len());
        batch.caches.push(cache);
    }
    batch
}

fn batched_step(model: &Llama<CpuBackend>, backend: &CpuBackend, batch: &mut Batch) {
    let mut sequences: Vec<DecodeBatchSequence<'_>> = batch
        .caches
        .iter_mut()
        .zip(batch.logits.iter_mut())
        .enumerate()
        .map(|(index, (cache, logits))| DecodeBatchSequence {
            token_id: batch.tokens[index],
            cache,
            start_pos: batch.positions[index],
            logits,
        })
        .collect();
    model
        .forward_decode_batch(backend, &mut sequences)
        .expect("batched decode");
    for index in 0..batch.tokens.len() {
        batch.tokens[index] = argmax(&batch.logits[index]);
        batch.positions[index] += 1;
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: batched_decode <model.gguf> [sizes] [steps]");
    let sizes: Vec<usize> = args
        .get(2)
        .map_or("1,2,4,8,17", String::as_str)
        .split(',')
        .map(|size| size.parse().expect("batch size"))
        .collect();
    let steps: usize = args.get(3).map_or(Ok(32), |steps| steps.parse())?;
    let capacity = 64 + 2 * steps + VERIFY_STEPS;
    let loader = load_gguf_with_k_strategy(path, KStrategy::Auto, false)?;
    let model = Llama::from_loader_with_max_seq_len(loader, Some(capacity))?;
    anyhow::ensure!(
        model.supports_batched_decode(),
        "model is not batch-decode eligible"
    );
    let backend = CpuBackend;

    for &n in &sizes {
        // Bit identity against independent single-sequence decodes.
        let mut batched = prefill(&model, &backend, n, capacity);
        let mut single = prefill(&model, &backend, n, capacity);
        for step in 0..VERIFY_STEPS {
            batched_step(&model, &backend, &mut batched);
            for index in 0..n {
                let logits = ForwardModel::forward_last_logits_with_cache(
                    &model,
                    &backend,
                    &[single.tokens[index]],
                    &mut single.caches[index],
                    single.positions[index],
                )?;
                let same = logits
                    .data()
                    .iter()
                    .zip(&batched.logits[index])
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                anyhow::ensure!(same, "N={n} step {step} sequence {index}: logits differ");
                single.tokens[index] = argmax(logits.data());
                single.positions[index] += 1;
            }
        }

        // Throughput: warm-up step, then timed batched steps.
        batched_step(&model, &backend, &mut batched);
        let started = Instant::now();
        for _ in 0..steps {
            batched_step(&model, &backend, &mut batched);
        }
        let seconds = started.elapsed().as_secs_f64();
        let aggregate = (n * steps) as f64 / seconds;

        // Baseline: the same sequences decoded one at a time on the
        // single-sequence fast path.
        let started = Instant::now();
        for _ in 0..steps {
            for index in 0..n {
                let logits = ForwardModel::forward_last_logits_with_cache(
                    &model,
                    &backend,
                    &[single.tokens[index]],
                    &mut single.caches[index],
                    single.positions[index],
                )?;
                single.tokens[index] = argmax(logits.data());
                single.positions[index] += 1;
            }
        }
        let sequential = (n * steps) as f64 / started.elapsed().as_secs_f64();
        println!(
            "{{\"batch\":{n},\"steps\":{steps},\"bit_identical_steps\":{VERIFY_STEPS},\
             \"aggregate_tokens_per_second\":{aggregate:.2},\
             \"per_sequence_tokens_per_second\":{:.2},\"step_ms\":{:.3},\
             \"sequential_single_tokens_per_second\":{sequential:.2}}}",
            aggregate / n as f64,
            seconds * 1000.0 / steps as f64
        );
    }
    Ok(())
}

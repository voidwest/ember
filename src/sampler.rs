use alloc::vec::Vec;
use rand::Rng;

/// apply temperature scaling and sampling filters, then sample a token
/// from the resulting categorical distribution
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut impl Rng,
) -> usize {
    let mut probs: Vec<f32> = logits.to_vec();

    if temperature > 0.0 {
        for p in &mut probs {
            *p /= temperature;
        }
    }

    if let Some(k) = top_k {
        top_k_filter(&mut probs, k);
    }

    if let Some(p) = top_p {
        top_p_filter(&mut probs, p);
    }

    let dist = softmax_1d(&probs);

    categorical_sample(&dist, rng)
}

/// set all values below the k-th largest logit to -infinity
fn top_k_filter(probs: &mut [f32], k: usize) {
    if k >= probs.len() || k == 0 {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    // zero out everything below the k-th largest logit (0-indexed, so k-1)
    let threshold = indexed[k - 1].1;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = f32::NEG_INFINITY;
        }
    }
}

/// after softmax, keep only the smallest set of tokens whose cumulative
/// probability exceeds `p`, setting the rest to -infinity
fn top_p_filter(probs: &mut [f32], p: f32) {
    let soft = softmax_1d(probs);
    let cutoff = nucleus_cutoff(&soft, p);
    for (i, s) in soft.iter().enumerate() {
        if *s < cutoff {
            probs[i] = f32::NEG_INFINITY;
        }
    }
}

/// softmax a 1d slice, returning a probability distribution.
/// handles the all-masked (-inf everywhere) edge case by returning
/// a uniform distribution.
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max == f32::NEG_INFINITY {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

/// find the probability threshold for nucleus sampling:
/// sort probabilities descending, then return the smallest
/// probability in the set whose cumulative sum exceeds `p`
fn nucleus_cutoff(sorted_probs: &[f32], p: f32) -> f32 {
    let mut indexed: Vec<(f32, usize)> = sorted_probs
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let mut cum = 0.0;
    for (prob, _) in &indexed {
        cum += prob;
        if cum >= p {
            return *prob;
        }
    }
    0.0
}

/// sample from a categorical distribution using inverse cdf sampling
fn categorical_sample(dist: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in dist.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    // fallback: return the index of the largest probability
    dist.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

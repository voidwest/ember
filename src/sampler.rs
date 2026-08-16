use alloc::vec::Vec;
use rand::Rng;

/// sample a token from logits with temperature scaling, top-k, and top-p filtering.
///
/// the standard sampling pipeline:
/// 1. **temperature scaling** - divides logits by `temperature` to sharpen or flatten
///    the distribution. `0.0` means greedy argmax.
/// 2. **top-k filtering** - keeps only the `k` highest logits, sets the rest to `-inf`.
/// 3. **top-p (nucleus) filtering** - keeps the smallest set of tokens whose cumulative
///    softmax probability exceeds `p`, sets the rest to `-inf`.
/// 4. **softmax** - converts filtered logits to a probability distribution.
///    if every logit is `-inf` (fully masked), returns a uniform distribution.
/// 5. **inverse cdf sampling** - draws a token from the categorical distribution.
///
/// this is the same sampling pipeline used by llama.cpp, huggingface transformers,
/// and the openai api (holtzman et al. 2020).
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut impl Rng,
) -> usize {
    assert!(!logits.is_empty(), "cannot sample from empty logits");
    assert!(
        logits.iter().all(|value| !value.is_nan()),
        "cannot sample from logits containing NaN"
    );
    assert!(
        temperature.is_finite() && temperature >= 0.0,
        "temperature must be finite and non-negative"
    );
    if let Some(p) = top_p {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "top_p must be in [0, 1]"
        );
    }
    if temperature == 0.0 {
        return argmax_token(logits);
    }

    let mut logits: Vec<f32> = logits.to_vec();

    for l in &mut logits {
        *l /= temperature;
    }

    if let Some(k) = top_k {
        top_k_filter(&mut logits, k);
    }

    if let Some(p) = top_p {
        top_p_filter(&mut logits, p);
    }

    // single softmax at the end - top_p_filter uses its own internal softmax
    // to find the nucleus cutoff; the final distribution is computed once here.
    let dist = softmax_1d(&logits);

    categorical_sample(&dist, rng)
}

pub fn argmax_token(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "cannot take argmax of empty logits");
    assert!(
        logits.iter().all(|value| !value.is_nan()),
        "cannot take argmax of logits containing NaN"
    );
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(max_i, max_v), (i, &v)| {
            if v > max_v {
                (i, v)
            } else {
                (max_i, max_v)
            }
        })
        .0
}

/// set all values below the k-th largest logit to `-inf`.
///
/// sorts a copy of the logits in descending order, finds the k-th largest value
/// (0-indexed, so `indexed[k - 1]`), and masks every logit below that threshold.
/// a no-op when `k >= len` or `k == 0`.
fn top_k_filter(logits: &mut [f32], k: usize) {
    if k >= logits.len() || k == 0 {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    // zero out everything below the k-th largest logit (0-indexed, so k-1)
    let threshold = indexed[k - 1].1;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// nucleus sampling: keep only the tokens in the smallest set whose
/// cumulative softmax probability exceeds `p`.
///
/// computes softmax on the current logits to find the cutoff threshold,
/// then masks logits whose softmax probability falls below that threshold.
/// the caller is responsible for computing the final softmax on the
/// filtered logits - this avoids computing softmax twice.
fn top_p_filter(logits: &mut [f32], p: f32) {
    let soft = softmax_1d(logits);
    let cutoff = nucleus_cutoff(&soft, p);
    for (i, s) in soft.iter().enumerate() {
        if *s < cutoff {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// numerically stable softmax over a 1d slice of logits.
///
/// subtracts the maximum logit before exponentiating to avoid overflow (the "max trick").
/// for all-masked input (every value is `f32::NEG_INFINITY`), returns a uniform
/// distribution - this matches the behavior of `CpuTensor::softmax` and prevents
/// NaN propagation from `(-inf - -inf).exp()` per ieee 754.
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty(), "cannot take softmax of empty logits");
    assert!(
        logits.iter().all(|value| !value.is_nan()),
        "cannot take softmax of logits containing NaN"
    );
    let positive_infinities = logits
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    if positive_infinities > 0 {
        let probability = 1.0 / positive_infinities as f32;
        return logits
            .iter()
            .map(|value| {
                if *value == f32::INFINITY {
                    probability
                } else {
                    0.0
                }
            })
            .collect();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max == f32::NEG_INFINITY {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

/// find the probability threshold for nucleus sampling.
///
/// sorts probabilities descending, accumulates from the top, and returns
/// the smallest probability value in the set whose cumulative sum reaches `p`.
/// returns `0.0` if the cumulative sum never reaches `p` (shouldn't happen
/// for a valid probability distribution).
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

/// sample from a categorical distribution using inverse cdf sampling.
///
/// draws a random float in `[0, 1)`, walks the cumulative sum of probabilities,
/// and returns the index where the random value first falls below the running sum.
/// falls back to argmax if floating-point rounding causes the cdf to not reach 1.0.
fn categorical_sample(dist: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.r#gen();
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
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_shares_probability_across_positive_infinities() {
        assert_eq!(
            softmax_1d(&[f32::INFINITY, 1.0, f32::INFINITY]),
            vec![0.5, 0.0, 0.5]
        );
    }

    #[test]
    #[should_panic(expected = "containing NaN")]
    fn argmax_rejects_nan() {
        let _ = argmax_token(&[0.0, f32::NAN]);
    }
}

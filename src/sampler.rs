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

    let mut scaled: Vec<f32> = logits.to_vec();
    let mut scratch = Vec::new();

    let mut overflowed = false;
    for l in &mut scaled {
        let finite = l.is_finite();
        *l /= temperature;
        overflowed |= finite && !l.is_finite();
    }
    // A temperature small enough to push a finite logit to infinity is the
    // T -> 0 limit; sampling the overflowed values would instead be uniform
    // over every token that reached +inf (or the whole vocabulary).
    if overflowed {
        return argmax_token(logits);
    }
    let mut logits = scaled;

    if let Some(k) = top_k {
        top_k_filter(&mut logits, k, &mut scratch);
    }

    if let Some(p) = top_p {
        top_p_filter(&mut logits, p, &mut scratch);
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
/// selects the k-th largest value in scratch space without sorting the full
/// vocabulary, then masks every logit below it, retaining all threshold ties.
/// a no-op when `k >= len` or `k == 0`.
fn top_k_filter(logits: &mut [f32], k: usize, scratch: &mut Vec<f32>) {
    if k >= logits.len() || k == 0 {
        return;
    }

    scratch.clear();
    scratch.extend_from_slice(logits);
    let (_, threshold, _) = scratch.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
    });
    for l in logits.iter_mut() {
        if *l < *threshold {
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
/// filtered logits, retaining the existing floating-point normalization order.
fn top_p_filter(logits: &mut [f32], p: f32, scratch: &mut Vec<f32>) {
    let soft = softmax_1d(logits);
    let cutoff = nucleus_cutoff(&soft, p, scratch);
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
    let mut exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for value in &mut exps {
        *value /= sum;
    }
    exps
}

/// find the probability threshold for nucleus sampling.
///
/// sorts probabilities descending, accumulates from the top, and returns
/// the smallest probability value in the set whose cumulative sum reaches `p`.
/// returns `0.0` if the cumulative sum never reaches `p` (shouldn't happen
/// for a valid probability distribution).
///
/// Only the largest probabilities are ever summed, so instead of sorting the
/// whole vocabulary this selects the top `m` (doubling `m` until the
/// nucleus fits) and sorts just those. The descending prefix, and therefore
/// every partial sum, is the one a full sort produces: tied values are equal,
/// so their order cannot change a sum.
fn nucleus_cutoff(probs: &[f32], p: f32, scratch: &mut Vec<f32>) -> f32 {
    let descending = |a: &f32, b: &f32| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal);
    let mut m = probs.len().min(64);
    loop {
        scratch.clear();
        scratch.extend_from_slice(probs);
        if m < scratch.len() {
            scratch.select_nth_unstable_by(m - 1, descending);
        }
        let top = &mut scratch[..m];
        top.sort_unstable_by(descending);
        let mut cum = 0.0;
        for &prob in top.iter() {
            cum += prob;
            if cum >= p {
                return prob;
            }
        }
        if m == probs.len() {
            return 0.0;
        }
        m = (m * 2).min(probs.len());
    }
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
    use rand::{rngs::StdRng, SeedableRng};

    fn reference_softmax(logits: &[f32]) -> Vec<f32> {
        let positive_infinities = logits
            .iter()
            .filter(|&&value| value == f32::INFINITY)
            .count();
        if positive_infinities > 0 {
            return logits
                .iter()
                .map(|&value| {
                    if value == f32::INFINITY {
                        1.0 / positive_infinities as f32
                    } else {
                        0.0
                    }
                })
                .collect();
        }
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max == f32::NEG_INFINITY {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    /// Original full-sort pipeline, kept as an oracle for the selection path.
    fn reference_distribution(logits: &[f32], temperature: f32, k: usize, p: f32) -> Vec<f32> {
        let mut logits: Vec<f32> = logits.iter().map(|value| value / temperature).collect();
        let descending = |a: &f32, b: &f32| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal);
        if k > 0 && k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_by(descending);
            for value in &mut logits {
                if *value < sorted[k - 1] {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        let probabilities = reference_softmax(&logits);
        let mut sorted = probabilities.clone();
        sorted.sort_by(descending);
        let mut cumulative = 0.0;
        let cutoff = sorted
            .into_iter()
            .find(|value| {
                cumulative += value;
                cumulative >= p
            })
            .unwrap_or(0.0);
        for (value, probability) in logits.iter_mut().zip(probabilities) {
            if probability < cutoff {
                *value = f32::NEG_INFINITY;
            }
        }
        reference_softmax(&logits)
    }

    #[test]
    fn selection_preserves_seeded_full_sort_sampling_and_ties() {
        let cases = [
            vec![0.0, -0.0, 1.0, 1.0, -1.0],
            vec![f32::NEG_INFINITY; 9],
            vec![f32::INFINITY, 0.0, f32::INFINITY, f32::NEG_INFINITY],
            (0..257)
                .map(|i| ((i * 71 % 113) as f32 - 56.0) / 7.0)
                .collect(),
        ];
        for logits in cases {
            for temperature in [0.1, 0.7, 1.0, 2.0] {
                for k in [0, 1, 2, logits.len(), logits.len() + 1] {
                    for p in [0.0, 0.1, 0.5, 0.9, 1.0] {
                        let distribution = reference_distribution(&logits, temperature, k, p);
                        let mut filtered: Vec<f32> =
                            logits.iter().map(|v| v / temperature).collect();
                        let mut scratch = Vec::new();
                        top_k_filter(&mut filtered, k, &mut scratch);
                        top_p_filter(&mut filtered, p, &mut scratch);
                        assert_eq!(softmax_1d(&filtered), distribution);
                        let mut expected_rng = StdRng::seed_from_u64(41);
                        let mut actual_rng = StdRng::seed_from_u64(41);
                        for _ in 0..32 {
                            assert_eq!(
                                sample_token(
                                    &logits,
                                    temperature,
                                    Some(k),
                                    Some(p),
                                    &mut actual_rng
                                ),
                                categorical_sample(&distribution, &mut expected_rng),
                                "temperature={temperature}, k={k}, p={p}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nucleus_cutoff_matches_a_full_sort() {
        let full_sort = |probs: &[f32], p: f32| {
            let mut sorted = probs.to_vec();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            let mut cum = 0.0;
            for &prob in &sorted {
                cum += prob;
                if cum >= p {
                    return prob;
                }
            }
            0.0
        };
        let mut scratch = Vec::new();
        for len in [1usize, 5, 63, 64, 65, 1000, 40_000] {
            let logits: Vec<f32> = (0..len)
                .map(|i| ((i * 7919 % 1009) as f32 - 504.0) / 37.0)
                .collect();
            let probs = softmax_1d(&logits);
            for p in [0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.5] {
                assert_eq!(
                    nucleus_cutoff(&probs, p, &mut scratch).to_bits(),
                    full_sort(&probs, p).to_bits(),
                    "len={len}, p={p}"
                );
            }
        }
    }

    #[test]
    fn a_temperature_that_overflows_the_logits_samples_greedily() {
        let mut rng = StdRng::seed_from_u64(7);
        for logits in [[1.0f32, 3.0, 2.0], [-3.0, -1.0, -2.0]] {
            for _ in 0..16 {
                assert_eq!(sample_token(&logits, 1e-40, None, None, &mut rng), 1);
            }
        }
    }

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

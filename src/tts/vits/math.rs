use super::*;

// ---------------------------------------------------------------------------
// math primitives (torch-faithful)
// ---------------------------------------------------------------------------

pub(super) fn layer_norm(x: &mut [f32], t: usize, dim: usize, w: &[f32], b: &[f32], eps: f32) {
    for row in 0..t {
        let r = &mut x[row * dim..(row + 1) * dim];
        let mean = r.iter().sum::<f32>() / dim as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in r.iter_mut().enumerate() {
            *v = (*v - mean) * inv * w[i] + b[i];
        }
    }
}

/// transformers' default gelu (exact erf form).
pub(super) fn gelu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

pub(super) fn erf(x: f32) -> f32 {
    libm::erf(x as f64) as f32
}

pub(super) fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Depthwise dilated Conv1d over `[C, T]` (kernel 3, dilation `d`,
/// symmetric pad `d`). Output `[C, T']` with T' = T + d (same-length when
/// the reference pads by (k*d−d)//2 = d).
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
pub(super) fn depthwise_dilated(
    input: &[f32],
    c: usize,
    t: usize,
    w_taps: &[f32],
    b: &[f32],
    d: usize,
) -> Vec<f32> {
    let k = 3usize;
    let pad = d;
    let t_out = t + 2 * pad - d * (k - 1);
    let _ = pad; // symmetric pad == d by construction (k=3)
    let mut out = vec![0.0f32; c * t_out];
    out.par_chunks_mut(t_out)
        .enumerate()
        .for_each(|(ci, out_c)| {
            let wc = &w_taps[ci * k..(ci + 1) * k];
            let xc = &input[ci * t..(ci + 1) * t];
            for ti in 0..t_out {
                // stride-1 window centered at ti: taps at ti-d, ti, ti+d
                let mut acc = b[ci];
                for j in 0..k {
                    let src = ti as isize + (j as isize) * (d as isize) - d as isize;
                    if src >= 0 && (src as usize) < t {
                        acc += xc[src as usize] * wc[j];
                    }
                }
                out_c[ti] = acc;
            }
        });
    out
}

/// Transpose `[C,T]` channel-major to row-major `[T,C]`.
pub(super) fn ct_to_rows(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            out[ti * c + ci] = x[ci * t + ti];
        }
    }
    out
}

/// Transpose row-major `[T,C]` back to channel-major `[C,T]`.
pub(super) fn rows_to_ct(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ti in 0..t {
        for ci in 0..c {
            out[ci * t + ti] = x[ti * c + ci];
        }
    }
    out
}

/// groups=1 DILATED Conv1d via im2col over `[C,T]`, symmetric pad `pad`,
/// dilation `dil`. Output length = T + 2*pad - dil*(k-1).
pub(super) fn conv_dense_dilated(
    input: &CpuTensor,
    c: &DenseConv1d,
    pad: usize,
    dil: usize,
) -> CpuTensor {
    let c_in = input.shape()[0];
    let t = input.shape()[1];
    let k = c.k;
    assert_eq!(c_in, c.c_in);
    let t_pad = t + 2 * pad;
    // stride is 1 everywhere here: out = T + 2p - d*(k-1)
    let t_out = t_pad.saturating_sub(dil * (k - 1));
    let x = input.data();
    let mut xp = vec![0.0f32; c_in * t_pad];
    for ci in 0..c_in {
        xp[ci * t_pad + pad..ci * t_pad + pad + t].copy_from_slice(&x[ci * t..(ci + 1) * t]);
    }
    let feat = c_in * k;
    let mut cols = vec![0.0f32; t_out * feat];
    cols.par_chunks_mut(feat)
        .enumerate()
        .for_each(|(t_out_i, row)| {
            for ci in 0..c_in {
                let base = ci * t_pad + t_out_i; // stride-1 window start
                for j in 0..k {
                    row[ci * k + j] = xp[base + j * dil];
                }
            }
        });
    let cols_t = CpuTensor::from_data(vec![t_out, feat], cols);
    let mut out = cols_t.par_matmul(&c.w_t);
    {
        let data = out.data_mut();
        for t_out_i in 0..t_out {
            let row = &mut data[t_out_i * c.c_out..(t_out_i + 1) * c.c_out];
            for (o, slot) in row.iter_mut().enumerate() {
                *slot += c.bias[o];
            }
        }
    }
    let data = out.data().to_vec();
    let mut ct = vec![0.0f32; c.c_out * t_out];
    for i in 0..t_out {
        for o in 0..c.c_out {
            ct[o * t_out + i] = data[i * c.c_out + o];
        }
    }
    CpuTensor::from_data(vec![c.c_out, t_out], ct)
}

/// Flip along the channel axis of a `[C,T]` buffer (torch.flip(dim=1)).
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
pub(super) fn flip_channels(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t];
    for ci in 0..c {
        out[(c - 1 - ci) * t..(c - ci) * t].copy_from_slice(&x[ci * t..(ci + 1) * t]);
    }
    out
}

// ---------------------------------------------------------------------------
// relative-position attention (reference-exact index mapping)
// ---------------------------------------------------------------------------

/// Slice of the (possibly zero-padded) relative embedding table usable for
/// sequence length `t`, returned as rows `[2t-1, hd]`. Mirrors
/// `_get_relative_embeddings`: pad symmetrically by
/// `max(t-(W+1),0)`, then slice `max(W+1-t,0) .. +2t-1`.
pub(super) fn rel_slice(rel: &[f32], window: usize, hd: usize, t: usize) -> Vec<f32> {
    let w2 = 2 * window + 1;
    let pad_len = t.saturating_sub(window + 1); // reference pads by W+1
    let start = (window + 1).saturating_sub(t);
    let total_rows = w2 + 2 * pad_len;
    let end = (start + 2 * t - 1).min(total_rows);
    // build the PADDED table first (zeros beyond both ends), then slice
    let mut padded = vec![0.0f32; total_rows * hd];
    padded[hd * pad_len..hd * (pad_len + w2)].copy_from_slice(rel);
    padded[hd * start..hd * end].to_vec()
}

impl EncoderLayer {
    pub(super) fn attention(
        &self,
        x: &[f32],
        t: usize,
        heads: usize,
        hd: usize,
        window: usize,
        slope: f32,
    ) -> Vec<f32> {
        let q = self.q.apply(x, t);
        let kk = self.k.apply(x, t);
        let vv = self.v.apply(x, t);

        let rel_k = rel_slice(&self.rel_k, window, hd, t);
        let rel_v = rel_slice(&self.rel_v, window, hd, t);

        // scores[h, q, k] = (q·k)*scale + q·rel_k[slice_row(k-q)]
        // (the pad/slice/relative->absolute pipeline reduces exactly to the
        // band-offset contraction below.)
        let w = window;
        // Reference `_get_relative_embeddings`: zero-pad the (2W+1)-row table
        // by pad_len rows on BOTH ends, then slice rows
        // [start, start + 2T-1) with start = max(W+1-T, 0). A pair offset
        // o = k - q lives at padded row o + W + pad_len, i.e. SLICE row
        // o + W + pad_len - start. For long sequences (T >= W+1) start == 0
        // and the row index collapses to o + W + pad_len; for SHORT
        // sequences the -start term is load-bearing (its absence made every
        // T <= W synthesis read the wrong relative row).
        let pad_len = t.saturating_sub(w + 1);
        let slice_start = (w + 1).saturating_sub(t);
        let rel_row_base = w as isize + pad_len as isize - slice_start as isize;
        let mut scores = vec![0.0f32; heads * t * t];
        {
            let sc = &mut scores;
            sc.par_chunks_mut(t * t).enumerate().for_each(|(h, sh)| {
                for qi in 0..t {
                    for ki in 0..t {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot +=
                                q[qi * heads * hd + h * hd + d] * kk[ki * heads * hd + h * hd + d];
                        }
                        dot *= slope;
                        let off = ki as isize - qi as isize + rel_row_base;
                        if off >= 0 {
                            let r = off as usize;
                            if r < 2 * t - 1 {
                                let rk = &rel_k[r * hd..(r + 1) * hd];
                                let qp =
                                    &q[qi * heads * hd + h * hd..qi * heads * hd + (h + 1) * hd];
                                for d in 0..hd {
                                    dot += qp[d] * slope * rk[d];
                                }
                            }
                        }
                        sh[qi * t + ki] = dot;
                    }
                }
            });
        }

        softmax_rows(&mut scores, heads * t, t);

        // values + relative value bias:
        // out[q,d] += sum_k probs[q,k] · rel_v[k-q+W, d]
        let mut out = vec![0.0f32; heads * t * hd];
        out.par_chunks_mut(t * hd).enumerate().for_each(|(h, oh)| {
            for qi in 0..t {
                for ki in 0..t {
                    let p = scores[h * t * t + qi * t + ki];
                    if p == 0.0 {
                        continue;
                    }
                    let base = ki * heads * hd + h * hd;
                    for d in 0..hd {
                        oh[qi * hd + d] += p * vv[base + d];
                    }
                    let off = ki as isize - qi as isize + rel_row_base;
                    if off >= 0 {
                        let r = off as usize;
                        if r < 2 * t - 1 {
                            let rv = &rel_v[r * hd..(r + 1) * hd];
                            for d in 0..hd {
                                oh[qi * hd + d] += p * rv[d];
                            }
                        }
                    }
                }
            }
        });

        let mut merged = vec![0.0f32; t * heads * hd];
        for qi in 0..t {
            for h in 0..heads {
                for d in 0..hd {
                    merged[qi * heads * hd + h * hd + d] = out[h * t * hd + qi * hd + d];
                }
            }
        }
        self.o.apply(&merged, t)
    }
}

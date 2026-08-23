//! WavTokenizer (Vocos backbone + iSTFT head) codec DECODER.
//!
//! Decodes single-codebook codec tokens (75 tokens/s) into 24 kHz mono
//! PCM. This is the acoustic half of the OuteTTS speech architecture: the
//! LLM emits `<|{code}|>` token ids; this module turns them into audio:
//!
//! ```text
//! codes [T] -> codebook lookup [512, T]
//!           -> embed Conv1d(512->768, k7 p3)
//!           -> pos_net: ResnetBlock x2 -> time attention -> ResnetBlock x2
//!              -> GroupNorm(32, eps 1e-6)
//!           -> AdaLayerNorm(bandwidth_id = 0)
//!           -> 12 x ConvNeXt (dwconv k7 groups=C, AdaLayerNorm,
//!                             768->2304->gelu->768, layer-scale gamma)
//!           -> LayerNorm -> Linear(768->1282)
//!           -> mag = exp(x[..641]).clip(max 1e2); phase = x[641..]
//!           -> iSTFT(n_fft 1280, hop 320, hann, "same" trim) -> PCM
//! ```
//!
//! Reference semantics mirrored exactly (outetts `wav_tokenizer/decoder`):
//! GroupNorm/LayerNorm use biased variance; GELU is the exact erf form;
//! the ISTFT is the custom "same"-padded overlap-add with window-envelope
//! normalization (`decoder/spectral_ops.py::ISTFT`, padding="same").
//!
//! The inverse real FFT runs through a Bluestein chirp-z transform over an
//! f64 radix-2 FFT (N=1280 = 2^8 x 5 is not a power of two). Tables are
//! built once per decoder; per-frame cost is three M-point FFTs.

use crate::backend::{Backend, CpuBackend};
use crate::loader::load_gguf;
use crate::model::Linear;
use crate::tensor::CpuTensor;
use anyhow::{ensure, Context, Result};

// ---------------------------------------------------------------------------
// Bluestein FFT (f64) for arbitrary-size complex DFTs
// ---------------------------------------------------------------------------

/// Precomputed tables for length-N complex DFTs via Bluestein's algorithm.
///
/// X_sigma[k] = sum_n x[n] * exp(sigma * 2*pi*i*k*n / N), sigma in {-1,+1},
/// computed as a circular convolution of length M = next_pow2(2N-1):
///
/// ```text
/// kn = (n^2 + k^2 - (k-n)^2)/2
/// => W^{kn} = c^{k^2} * (x[n] c^{n^2}) (*) c^{-m^2},  c = exp(sigma*pi*i/N)
/// ```
struct Bluestein {
    n: usize,
    m: usize,
    /// bit-reversal table for the M-point radix-2 FFT
    rev: Vec<usize>,
    /// twiddle factors exp(-2*pi*i*j/M)
    tw_re: Vec<f64>,
    tw_im: Vec<f64>,
}

impl Bluestein {
    fn new(n: usize) -> Self {
        assert!(n > 0);
        let m = (2 * n - 1).next_power_of_two();
        let bits = m.trailing_zeros();
        let mut rev = vec![0usize; m];
        for (i, slot) in rev.iter_mut().enumerate() {
            let mut r = 0usize;
            for b in 0..bits {
                r |= ((i >> b) & 1) << (bits - 1 - b);
            }
            *slot = r;
        }
        let mut tw_re = vec![0.0f64; m / 2];
        let mut tw_im = vec![0.0f64; m / 2];
        for j in 0..m / 2 {
            let ang = -2.0 * std::f64::consts::PI * j as f64 / m as f64;
            tw_re[j] = ang.cos();
            tw_im[j] = ang.sin();
        }
        Self {
            n,
            m,
            rev,
            tw_re,
            tw_im,
        }
    }

    fn fft_pow2(&self, re: &mut [f64], im: &mut [f64]) {
        let len = re.len();
        debug_assert!(len.is_power_of_two());
        for i in 0..len {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= len {
            let half = size / 2;
            let step = self.m / size;
            for start in (0..len).step_by(size) {
                for j in 0..half {
                    let wr = self.tw_re[j * step];
                    let wi = self.tw_im[j * step];
                    let i0 = start + j;
                    let i1 = i0 + half;
                    let tr = re[i1] * wr - im[i1] * wi;
                    let ti = re[i1] * wi + im[i1] * wr;
                    re[i1] = re[i0] - tr;
                    im[i1] = im[i0] - ti;
                    re[i0] += tr;
                    im[i0] += ti;
                }
            }
            size *= 2;
        }
    }

    fn ifft_pow2(&self, re: &mut [f64], im: &mut [f64]) {
        for v in im.iter_mut() {
            *v = -*v;
        }
        self.fft_pow2(re, im);
        let scale = 1.0 / re.len() as f64;
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *i = -*i * scale;
        }
    }

    /// X_sigma[k] = sum_n x[n] e^{sigma*2pi*i*k*n/N}, UNNORMALIZED.
    fn dft(&self, re_in: &[f64], im_in: &[f64], sigma: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        // a[n] = x[n] * e^{sigma*pi*i*n^2/N}
        let mut a_re = vec![0.0f64; self.m];
        let mut a_im = vec![0.0f64; self.m];
        for i in 0..n {
            let ang = sigma * std::f64::consts::PI * (i as f64) * (i as f64) / n as f64;
            let (cr, ci) = (ang.cos(), ang.sin());
            a_re[i] = re_in[i] * cr - im_in[i] * ci;
            a_im[i] = re_in[i] * ci + im_in[i] * cr;
        }
        // b[m] = e^{-sigma*pi*i*m^2/N} on [0,N) and mirrored negatives
        let mut b_re = vec![0.0f64; self.m];
        let mut b_im = vec![0.0f64; self.m];
        b_re[0] = 1.0;
        for k in 1..n {
            let ang = -sigma * std::f64::consts::PI * (k as f64) * (k as f64) / n as f64;
            let r = ang.cos();
            let im_v = ang.sin();
            b_re[k] = r;
            b_im[k] = im_v;
            b_re[self.m - k] = r;
            b_im[self.m - k] = im_v;
        }
        self.fft_pow2(&mut a_re, &mut a_im);
        self.fft_pow2(&mut b_re, &mut b_im);
        for i in 0..self.m {
            let tr = a_re[i] * b_re[i] - a_im[i] * b_im[i];
            a_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
            a_re[i] = tr;
        }
        self.ifft_pow2(&mut a_re, &mut a_im);
        // multiply by c^{k^2} = e^{sigma*pi*i*k^2/N}
        let mut out_re = vec![0.0f64; n];
        let mut out_im = vec![0.0f64; n];
        for k in 0..n {
            let ang = sigma * std::f64::consts::PI * (k as f64) * (k as f64) / n as f64;
            let (cr, ci) = (ang.cos(), ang.sin());
            out_re[k] = a_re[k] * cr - a_im[k] * ci;
            out_im[k] = a_re[k] * ci + a_im[k] * cr;
        }
        (out_re, out_im)
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;

    #[test]
    fn bluestein_matches_naive_dft_both_directions() {
        let n = 40; // deliberately composite-with-non-pow2 factors
        let bl = Bluestein::new(n);
        // deterministic pseudo-random input
        let mut seed = 42u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) % 2000) as f64 / 1000.0 - 1.0
        };
        let re: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let im: Vec<f64> = (0..n).map(|_| rnd()).collect();

        for &sigma in &[-1.0f64, 1.0] {
            let (gr, gi) = bl.dft(&re, &im, sigma);
            for k in 0..n {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for (nn, (&xr, &xi)) in re.iter().zip(&im).enumerate() {
                    let ang = sigma * 2.0 * std::f64::consts::PI * k as f64 * nn as f64 / n as f64;
                    let (wr, wi) = (ang.cos(), ang.sin());
                    sr += xr * wr - xi * wi;
                    si += xr * wi + xi * wr;
                }
                assert!(
                    (sr - gr[k]).abs() < 1e-9 && (si - gi[k]).abs() < 1e-9,
                    "sigma {sigma} bin {k}: ({sr},{si}) vs ({},{})",
                    gr[k],
                    gi[k]
                );
            }
        }
    }

    /// Reference-path helper mirroring what the decoder does: rfft bins ->
    /// full hermitian spectrum -> unnormalized inverse DFT -> real part / N.
    fn irfft_from_bins(bl: &Bluestein, s_re: &[f64], s_im: &[f64]) -> Vec<f64> {
        let n = bl.n;
        let bins = n / 2 + 1;
        assert_eq!(
            s_re.len(),
            bins,
            "irfft_frame expects {bins} rfft bins for n_fft {n}, got {}",
            s_re.len()
        );
        let mut fr = vec![0.0f64; n];
        let mut fi = vec![0.0f64; n];
        fr[0] = s_re[0];
        fi[0] = 0.0; // DC is real for a real signal
        for k in 1..bins {
            fr[k] = s_re[k];
            fi[k] = s_im[k];
            if k < n - k {
                fr[n - k] = s_re[k];
                fi[n - k] = -s_im[k];
            }
        }
        let (xr, _xi) = bl.dft(&fr, &fi, 1.0);
        let scale = 1.0 / n as f64;
        (0..n).map(|t| xr[t] * scale).collect()
    }

    #[test]
    fn irfft_roundtrip_recovers_signal() {
        let n = 1280usize;
        let bl = Bluestein::new(n);
        let signal: Vec<f64> = (0..n)
            .map(|t| {
                (2.0 * std::f64::consts::PI * t as f64 * 17.0 / n as f64).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * t as f64 * 331.0 / n as f64).cos()
            })
            .collect();
        let zero = vec![0.0f64; n];
        let (fr, fi) = bl.dft(&signal, &zero, -1.0);
        // hermitian sanity of a real signal's spectrum
        for k in 1..n / 2 {
            assert!((fi[k] + fi[n - k]).abs() < 1e-7);
            assert!((fr[k] - fr[n - k]).abs() < 1e-7);
        }
        let bins = n / 2 + 1;
        let rec = irfft_from_bins(&bl, &fr[..bins], &fi[..bins]);
        let max_err = signal
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 5e-5, "max roundtrip error {max_err}");
    }

    #[test]
    fn irfft_matches_naive_inverse_definition() {
        // small N vs the textbook definition
        let n = 100usize;
        let bl = Bluestein::new(n);
        let bins = n / 2 + 1;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) % 2000) as f64 / 1000.0 - 1.0
        };
        let s_re: Vec<f64> = (0..bins).map(|_| rnd()).collect();
        let mut s_im: Vec<f64> = (0..bins).map(|_| rnd()).collect();
        // force a REAL signal's rfft: DC and Nyquist bins are real
        s_im[0] = 0.0;
        s_im[bins - 1] = 0.0;
        let got = irfft_from_bins(&bl, &s_re, &s_im);
        // build the full hermitian spectrum, then textbook inverse sum
        let mut fr = vec![0.0f64; n];
        let mut fi = vec![0.0f64; n];
        for k in 0..bins {
            fr[k] = s_re[k];
            fi[k] = s_im[k];
            if 0 < k && k < n - k {
                fr[n - k] = s_re[k];
                fi[n - k] = -s_im[k];
            }
        }
        #[allow(clippy::needless_range_loop)]
        for t in 0..n {
            let (mut sr, mut si) = (0.0f64, 0.0f64);
            for k in 0..n {
                let ang = 2.0 * std::f64::consts::PI * t as f64 * k as f64 / n as f64;
                sr += fr[k] * ang.cos() - fi[k] * ang.sin();
                si += fr[k] * ang.sin() + fi[k] * ang.cos();
            }
            assert!(si.abs() < 1e-7, "imaginary part must vanish at t={t}");
            let expect = sr / n as f64;
            assert!(
                (expect - got[t]).abs() < 5e-5,
                "t={t}: {} vs {}",
                expect,
                got[t]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WavTokenizerConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub codebook_bins: usize,
    pub latent_dim: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub convnext_layers: usize,
    pub group_norm_groups: usize,
    pub group_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub adanorm_bands: usize,
}

/// GroupNorm affine params (per-channel weight/bias).
struct GroupNorm {
    groups: usize,
    eps: f32,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// AdaLayerNorm params: LN(x) * scale[band] + shift[band].
struct AdaLayerNorm {
    eps: f32,
    /// [bands, dim]
    scale: CpuTensor,
    shift: CpuTensor,
    band: usize,
}

/// A groups=1 Conv1d with its weight pre-arranged for im2col+sgemm
/// execution (`[C_in*k, C_out]`, row-major) — built once at load time.
///
/// The hot codec convs are dominated by gather+dual-loop scalar MACs when
/// executed as direct convolution; pre-transposing lets every call run as
/// one packed sgemm over an [T', C_in*k] column matrix. Accumulation order
/// differs from the direct form (numerically equivalent, ladder-gated).
#[derive(Debug, Clone)]
struct DenseConv1d {
    /// row-major [C_in * k, C_out]; index f = ci * k + tap
    w_t: CpuTensor,
    bias: Vec<f32>,
    k: usize,
    c_in: usize,
    c_out: usize,
}

impl DenseConv1d {
    fn from_hf_weight(w: &CpuTensor, bias: Vec<f32>) -> Self {
        assert_eq!(
            w.shape().len(),
            3,
            "DenseConv1d expects HF [C_out, C_in, k]"
        );
        let (c_out, c_in, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let mut w_t = vec![0.0f32; c_in * k * c_out];
        for o in 0..c_out {
            for f in 0..c_in * k {
                w_t[f * c_out + o] = w.data()[o * c_in * k + f];
            }
        }
        Self {
            w_t: CpuTensor::from_data(vec![c_in * k, c_out], w_t),
            bias,
            k,
            c_in,
            c_out,
        }
    }
}

struct ResnetBlock {
    norm1: GroupNorm,
    conv1: DenseConv1d,
    norm2: GroupNorm,
    conv2: DenseConv1d,
}

struct TimeAttention {
    norm: GroupNorm,
    q: Linear<CpuBackend>,
    k: Linear<CpuBackend>,
    v: Linear<CpuBackend>,
    proj_out: Linear<CpuBackend>,
}

struct ConvNeXtBlock {
    dwconv_weight: CpuTensor, // [C, 1, 7] depthwise
    dwconv_bias: Vec<f32>,
    norm: AdaLayerNorm,
    pwconv1: Linear<CpuBackend>, // [dim -> intermediate]
    pwconv2: Linear<CpuBackend>, // [intermediate -> dim]
    gamma: Vec<f32>,
}

/// Progressive-validation intermediates of one decode.
#[derive(Debug, Default)]
pub struct WavTokenizerTrace {
    /// After codebook lookup, `[512, T]`.
    pub features: Option<CpuTensor>,
    /// After embed conv, `[768, T]`.
    pub embed: Option<CpuTensor>,
    /// After pos_net (incl. final group norm), `[768, T]`.
    pub pos_net: Option<CpuTensor>,
    /// After backbone-level AdaLayerNorm, `[T, 768]` row-major.
    pub adanorm: Option<CpuTensor>,
    /// Selected ConvNeXt block outputs, `[768, T]` each.
    pub convnext_blocks: Vec<(usize, CpuTensor)>,
    /// Final LayerNorm output `[T, 768]` row-major.
    pub backbone_final: Option<CpuTensor>,
    /// Magnitudes `[641, T]` and phases `[641, T]`.
    pub mag: Option<CpuTensor>,
    pub phase: Option<CpuTensor>,
    /// Overlap-add result BEFORE envelope division and trim.
    pub ola_raw: Option<CpuTensor>,
}

pub struct WavTokenizerDecoder {
    pub config: WavTokenizerConfig,
    codebook: CpuTensor, // [bins, latent]
    embed: DenseConv1d,  // embed Conv1d(latent -> dim, k7 p3)
    resnets: [ResnetBlock; 4],
    attention: TimeAttention,
    pos_group_norm: GroupNorm,
    adanorm: AdaLayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_norm_weight: Vec<f32>,
    final_norm_bias: Vec<f32>,
    head_out: Linear<CpuBackend>,
    window: Vec<f64>,
    fft: Bluestein,
}

// ---------------------------------------------------------------------------
// tensor primitives (torch-faithful)
// ---------------------------------------------------------------------------

/// Groups=1 Conv1d via im2col + packed sgemm (the codec's hot path).
/// Output `[C_out, T']`, same layout contract as [`conv1d`].
fn conv1d_dense(input: &CpuTensor, c: &DenseConv1d, pad: usize) -> CpuTensor {
    let c_in = input.shape()[0];
    let t = input.shape()[1];
    assert_eq!(c_in, c.c_in, "conv1d_dense channel mismatch");
    let k = c.k;
    let t_pad = t + 2 * pad;
    let t_out = t_pad + 1 - k; // stride 1
    let x = input.data();
    // padded channel-major scratch so each window is one contiguous gather
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
                let base = ci * t_pad + t_out_i;
                let src = &xp[base..base + k];
                let dst = &mut row[ci * k..(ci + 1) * k];
                dst.copy_from_slice(src);
            }
        });
    let cols_t = CpuTensor::from_data(vec![t_out, feat], cols);
    let mut out = cols_t.par_matmul(&c.w_t); // [T', C_out]
    {
        let data = out.data_mut();
        for t_out_i in 0..t_out {
            let row = &mut data[t_out_i * c.c_out..(t_out_i + 1) * c.c_out];
            for (o, slot) in row.iter_mut().enumerate() {
                *slot += c.bias[o];
            }
        }
    }
    // return to the module's [C_out, T'] convention
    let data = out.data().to_vec();
    let mut ct = vec![0.0f32; c.c_out * t_out];
    for i in 0..t_out {
        for o in 0..c.c_out {
            ct[o * t_out + i] = data[i * c.c_out + o];
        }
    }
    CpuTensor::from_data(vec![c.c_out, t_out], ct)
}

fn conv1d(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: &[f32],
    pad: usize,
    groups: usize,
) -> CpuTensor {
    let c_in = input.shape()[0];
    let t = input.shape()[1];
    let c_out = weight.shape()[0];
    assert_eq!(weight.shape()[1] * groups, c_in, "conv groups mismatch");
    let k = weight.shape()[2];
    let t_pad = t + 2 * pad;
    let t_out = t_pad + 1 - k; // stride 1
    let x = input.data();
    let w = weight.data();
    // channel-major padded scratch simplifies gather loops
    let mut xp = vec![0.0f32; c_in * t_pad];
    for c in 0..c_in {
        xp[c * t_pad + pad..c * t_pad + pad + t].copy_from_slice(&x[c * t..(c + 1) * t]);
    }
    let cg_out = c_out / groups;
    let cg_in = c_in / groups;
    let mut out = vec![0.0f32; c_out * t_out];
    out.par_chunks_mut(cg_out * t_out)
        .enumerate()
        .for_each(|(g, out_g)| {
            for o_local in 0..cg_out {
                let o = g * cg_out + o_local;
                let wo = &w[o * cg_in * k..(o + 1) * cg_in * k];
                for t_out_i in 0..t_out {
                    let mut acc = 0.0f32;
                    for ci_local in 0..cg_in {
                        let base = (g * cg_in + ci_local) * t_pad + t_out_i;
                        let wrow = &wo[ci_local * k..(ci_local + 1) * k];
                        let xwin = &xp[base..base + k];
                        for j in 0..k {
                            acc += xwin[j] * wrow[j];
                        }
                    }
                    out_g[o_local * t_out + t_out_i] = acc + bias[o];
                }
            }
        });
    CpuTensor::from_data(vec![c_out, t_out], out)
}

use rayon::prelude::*;

/// GroupNorm over `[C, T]`: per-group biased mean/var across (channels, T),
/// affine per channel. Variance is two-pass with f64 accumulation (the
/// naive `E[x^2] - mean^2` form loses too much precision on large-DC
/// activations and breaks the parity ladder).
fn group_norm(x: &CpuTensor, gn: &GroupNorm) -> CpuTensor {
    let (c, t) = (x.shape()[0], x.shape()[1]);
    let g = gn.groups;
    assert_eq!(
        c % g,
        0,
        "group_norm: channel count {c} not divisible by {g} groups —          check the tower config in the codec GGUF metadata"
    );
    let cg = c / g;
    let mut out = vec![0.0f32; c * t];
    for gi in 0..g {
        let mut sum = 0.0f64;
        for ci in 0..cg {
            let base = (gi * cg + ci) * t;
            sum += x.data()[base..base + t]
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>();
        }
        let n = (cg * t) as f64;
        let mean_f = sum / n;
        let mut var_acc = 0.0f64;
        for ci in 0..cg {
            let base = (gi * cg + ci) * t;
            for &v in &x.data()[base..base + t] {
                let d = f64::from(v) - mean_f;
                var_acc += d * d;
            }
        }
        let inv = 1.0 / ((var_acc / n) as f32 + gn.eps).sqrt();
        for ci in 0..cg {
            let ch = gi * cg + ci;
            let base = ch * t;
            let w = gn.weight[ch];
            let b = gn.bias[ch];
            for ti in 0..t {
                let v = x.data()[base + ti];
                out[base + ti] = ((f64::from(v) - mean_f) as f32) * inv * w + b;
            }
        }
    }
    CpuTensor::from_data(vec![c, t], out)
}

/// LayerNorm over the last dim of row-major `[rows, dim]`.
fn layer_norm_rows(x: &CpuTensor, eps: f32, weight: &[f32], bias: &[f32]) -> CpuTensor {
    let (rows, dim) = (x.shape()[0], x.shape()[1]);
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x.data()[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (d, v) in row.iter().enumerate() {
            out[r * dim + d] = (v - mean) * inv * weight[d] + bias[d];
        }
    }
    CpuTensor::from_data(vec![rows, dim], out)
}

impl AdaLayerNorm {
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let dim = self.scale.shape()[1];
        let scale = &self.scale.data()[self.band * dim..(self.band + 1) * dim];
        let shift = &self.shift.data()[self.band * dim..(self.band + 1) * dim];
        let ones = vec![1.0f32; dim];
        let zeros = vec![0.0f32; dim];
        let normed = layer_norm_rows(x, self.eps, &ones, &zeros);
        let (rows, _) = (normed.shape()[0], normed.shape()[1]);
        let mut out = normed.data().to_vec();
        for r in 0..rows {
            for d in 0..dim {
                out[r * dim + d] = out[r * dim + d] * scale[d] + shift[d];
            }
        }
        CpuTensor::from_data(vec![rows, dim], out)
    }
}

/// Swish/silu used by the ResnetBlocks.
fn swish(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

/// Reverse a GGUF-loaded tensor's dim order keeping the HF row-major
/// payload (same convention as the audio tower loader).
fn gguf_to_hf(t: &CpuTensor) -> CpuTensor {
    let mut shape = t.shape().to_vec();
    shape.reverse();
    CpuTensor::from_data(shape, t.data().to_vec())
}

impl WavTokenizerDecoder {
    /// Load from a codec GGUF produced by
    /// `tools/convert_wavtokenizer_decoder.py` (architecture
    /// `wavtokenizer-decoder`, tensor prefix `w.`).
    pub fn from_gguf(path: &std::path::Path) -> Result<Self> {
        use crate::loader::GgufValue;

        let mut loader =
            load_gguf(path).with_context(|| format!("failed to load codec {}", path.display()))?;
        let meta = &loader.metadata;
        let get_u32 = |key: &str| -> Result<usize> {
            match meta.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                Some(other) => anyhow::bail!("{key} must be U32, got {other:?}"),
                None => anyhow::bail!("codec gguf missing metadata {key}"),
            }
        };
        let get_f32 = |key: &str| -> Result<f32> {
            match meta.get(key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                Some(other) => anyhow::bail!("{key} must be F32, got {other:?}"),
                None => anyhow::bail!("codec gguf missing metadata {key}"),
            }
        };

        let config = WavTokenizerConfig {
            sample_rate: get_u32("wt.sample_rate")? as u32,
            n_fft: get_u32("wt.n_fft")?,
            hop_length: get_u32("wt.hop_length")?,
            codebook_bins: get_u32("wt.codebook_bins")?,
            latent_dim: get_u32("wt.latent_dim")?,
            dim: get_u32("wt.dim")?,
            intermediate_dim: get_u32("wt.intermediate_dim")?,
            convnext_layers: get_u32("wt.convnext_layers")?,
            group_norm_groups: get_u32("wt.group_norm_groups")?,
            group_norm_eps: get_f32("wt.group_norm_eps")?,
            layer_norm_eps: get_f32("wt.layer_norm_eps")?,
            adanorm_bands: get_u32("wt.adanorm_bands")?,
        };
        ensure!(
            config.n_fft.is_multiple_of(2),
            "n_fft {} must be even for rfft bins",
            config.n_fft
        );

        let take_vec = |l: &mut crate::loader::GgufLoader, name: &str| -> Result<CpuTensor> {
            l.take_f32(name)
                .with_context(|| format!("codec gguf missing tensor {name}"))
        };
        let take_bias = |l: &mut crate::loader::GgufLoader, name: &str| -> Result<Vec<f32>> {
            Ok(take_vec(l, name)?.data().to_vec())
        };
        let take_linear =
            |l: &mut crate::loader::GgufLoader, name: &str| -> Result<Linear<CpuBackend>> {
                // Two storage cases, both arriving HF-oriented via
                // gguf_to_hf (dim-reversed from GGUF):
                // - Conv1d(k=1):  [out, in, 1]
                // - nn.Linear:    [out, in]
                // Either way the payload is row-major (out, in) -> transpose
                // to the Linear layout [in, out].
                let w = gguf_to_hf(&take_vec(l, &format!("{name}.weight"))?);
                let (o, i) = match w.shape().len() {
                    3 if w.shape()[2] == 1 => (w.shape()[0], w.shape()[1]),
                    2 => (w.shape()[0], w.shape()[1]),
                    _ => anyhow::bail!(
                        "{name}.weight expected HF [out, in(, 1)], got {:?}",
                        w.shape()
                    ),
                };
                let m = CpuTensor::from_data(vec![o, i], w.data().to_vec());
                let b = take_bias(l, &format!("{name}.bias"))?;
                let n_bias = b.len();
                Ok(Linear::new(
                    m.transpose(),
                    Some(CpuTensor::from_data(vec![n_bias], b)),
                ))
            };
        let take_gn = |l: &mut crate::loader::GgufLoader,
                       name: &str,
                       groups: usize,
                       eps: f32|
         -> Result<GroupNorm> {
            Ok(GroupNorm {
                groups,
                eps,
                weight: take_bias(l, &format!("{name}.weight"))?,
                bias: take_bias(l, &format!("{name}.bias"))?,
            })
        };

        let codebook = take_vec(&mut loader, "w.codebook")?;
        let embed = DenseConv1d::from_hf_weight(
            &gguf_to_hf(&take_vec(&mut loader, "w.embed.weight")?),
            take_bias(&mut loader, "w.embed.bias")?,
        );

        let mut resnets: Vec<ResnetBlock> = Vec::with_capacity(4);
        for i in [0usize, 1, 3, 4] {
            resnets.push(ResnetBlock {
                norm1: take_gn(
                    &mut loader,
                    &format!("w.pos_net.{i}.norm1"),
                    config.group_norm_groups,
                    config.group_norm_eps,
                )?,
                conv1: DenseConv1d::from_hf_weight(
                    &gguf_to_hf(&take_vec(
                        &mut loader,
                        &format!("w.pos_net.{i}.conv1.weight"),
                    )?),
                    take_bias(&mut loader, &format!("w.pos_net.{i}.conv1.bias"))?,
                ),
                norm2: take_gn(
                    &mut loader,
                    &format!("w.pos_net.{i}.norm2"),
                    config.group_norm_groups,
                    config.group_norm_eps,
                )?,
                conv2: DenseConv1d::from_hf_weight(
                    &gguf_to_hf(&take_vec(
                        &mut loader,
                        &format!("w.pos_net.{i}.conv2.weight"),
                    )?),
                    take_bias(&mut loader, &format!("w.pos_net.{i}.conv2.bias"))?,
                ),
            });
        }

        let attention = TimeAttention {
            norm: take_gn(
                &mut loader,
                "w.pos_net.2.norm",
                config.group_norm_groups,
                config.group_norm_eps,
            )?,
            q: take_linear(&mut loader, "w.pos_net.2.q")?,
            k: take_linear(&mut loader, "w.pos_net.2.k")?,
            v: take_linear(&mut loader, "w.pos_net.2.v")?,
            proj_out: take_linear(&mut loader, "w.pos_net.2.proj_out")?,
        };
        let pos_group_norm = take_gn(
            &mut loader,
            "w.pos_net.5",
            config.group_norm_groups,
            config.group_norm_eps,
        )?;

        // torch stores [bands, dim]; GGUF reverses dims -> restore
        let adanorm = AdaLayerNorm {
            eps: config.layer_norm_eps,
            scale: gguf_to_hf(&take_vec(&mut loader, "w.adanorm.scale")?),
            shift: gguf_to_hf(&take_vec(&mut loader, "w.adanorm.shift")?),
            band: 0, // inference always uses bandwidth_id 0 (outetts default)
        };

        let mut convnext = Vec::with_capacity(config.convnext_layers);
        for i in 0..config.convnext_layers {
            convnext.push(ConvNeXtBlock {
                dwconv_weight: gguf_to_hf(&take_vec(
                    &mut loader,
                    &format!("w.convnext.{i}.dwconv.weight"),
                )?),
                dwconv_bias: take_bias(&mut loader, &format!("w.convnext.{i}.dwconv.bias"))?,
                norm: AdaLayerNorm {
                    eps: config.layer_norm_eps,
                    scale: gguf_to_hf(&take_vec(
                        &mut loader,
                        &format!("w.convnext.{i}.norm.scale"),
                    )?),
                    shift: gguf_to_hf(&take_vec(
                        &mut loader,
                        &format!("w.convnext.{i}.norm.shift"),
                    )?),
                    band: 0,
                },
                pwconv1: take_linear(&mut loader, &format!("w.convnext.{i}.pwconv1"))?,
                pwconv2: take_linear(&mut loader, &format!("w.convnext.{i}.pwconv2"))?,
                gamma: take_bias(&mut loader, &format!("w.convnext.{i}.gamma"))?,
            });
        }

        let final_norm_weight = take_bias(&mut loader, "w.final_layer_norm.weight")?;
        let final_norm_bias = take_bias(&mut loader, "w.final_layer_norm.bias")?;
        let head_out = take_linear(&mut loader, "w.head.out")?;
        let window: Vec<f64> = take_vec(&mut loader, "w.window")?
            .data()
            .iter()
            .map(|&v| v as f64)
            .collect();

        let fft = Bluestein::new(config.n_fft);
        ensure!(
            window.len() == config.n_fft,
            "window length {} != n_fft {}",
            window.len(),
            config.n_fft
        );
        let resnets: [ResnetBlock; 4] = resnets
            .try_into()
            .map_err(|_| anyhow::anyhow!("pos_net must contain four resblocks"))?;
        let _ = &resnets;
        Ok(Self {
            config,
            codebook,
            embed,
            resnets,
            attention,
            pos_group_norm,
            adanorm,
            convnext,
            final_norm_weight,
            final_norm_bias,
            head_out,
            window,
            fft,
        })
    }
}

// ---------------------------------------------------------------------------
// decoding
// ---------------------------------------------------------------------------

impl WavTokenizerDecoder {
    /// Decode codec token ids to mono PCM at [`Self::config.sample_rate`].
    pub fn decode(&self, backend: &CpuBackend, codes: &[u32]) -> Result<Vec<f32>> {
        let (pcm, _) = self.decode_traced(backend, codes, false)?;
        Ok(pcm)
    }

    /// [`Self::decode`] with optional progressive-validation intermediates.
    pub fn decode_traced(
        &self,
        backend: &CpuBackend,
        codes: &[u32],
        trace_on: bool,
    ) -> Result<(Vec<f32>, WavTokenizerTrace)> {
        let mut trace = WavTokenizerTrace::default();
        let cfg = &self.config;
        ensure!(!codes.is_empty(), "no codec tokens to decode");
        let t = codes.len();
        for (i, c) in codes.iter().enumerate() {
            ensure!(
                (*c as usize) < cfg.codebook_bins,
                "code {c} at {i} out of range [0, {})",
                cfg.codebook_bins
            );
        }

        // -- codebook lookup: features[c, i] = codebook[codes[i], c] --
        let latent = cfg.latent_dim;
        let mut feats = vec![0.0f32; latent * t];
        for (i, &c) in codes.iter().enumerate() {
            let row = &self.codebook.data()[c as usize * latent..(c as usize + 1) * latent];
            for d in 0..latent {
                feats[d * t + i] = row[d];
            }
        }
        let x = CpuTensor::from_data(vec![latent, t], feats);
        if trace_on {
            trace.features = Some(x.clone());
        }

        // -- embed conv k7 p3 --
        let mut x = conv1d_dense(&x, &self.embed, 3);
        if trace_on {
            trace.embed = Some(x.clone());
        }
        let dim = cfg.dim;
        let t2 = x.shape()[1];

        // -- pos_net: RB0, RB1, time-attention, RB2, RB3 (reference order) --
        {
            let run_resnet = |x: &CpuTensor, r: &ResnetBlock| -> Result<CpuTensor> {
                let h = group_norm(x, &r.norm1);
                let mut rows = h.data().to_vec();
                for v in rows.iter_mut() {
                    *v = swish(*v);
                }
                let h = conv1d_dense(&CpuTensor::from_data(vec![dim, t2], rows), &r.conv1, 1);
                let h = group_norm(&h, &r.norm2);
                let mut rows = h.data().to_vec();
                for v in rows.iter_mut() {
                    *v = swish(*v);
                }
                let h = conv1d_dense(&CpuTensor::from_data(vec![dim, t2], rows), &r.conv2, 1);
                backend.add(x, &h).map_err(|e| anyhow::anyhow!("{e}"))
            };

            x = run_resnet(&x, &self.resnets[0])?;
            x = run_resnet(&x, &self.resnets[1])?;

            // time attention over [C, T]
            {
                let a = &self.attention;
                let ht = transpose(&group_norm(&x, &a.norm)); // [T, C]
                let scale = (dim as f32).sqrt().recip();
                let q =
                    a.q.forward(backend, &ht)
                        .map_err(|e| anyhow::anyhow!("{e}"))?; // [T,C]
                let k =
                    a.k.forward(backend, &ht)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                let v =
                    a.v.forward(backend, &ht)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                // w[i,j] = <q_i, k_j> * scale, softmax over keys j —
                // one packed sgemm per product instead of scalar loops
                let kt = transpose(&k); // [C, T]
                let scores = q.par_matmul(&kt).data().to_vec();
                let mut w = vec![0.0f32; t2 * t2];
                for (acc, slot) in scores.into_iter().zip(w.iter_mut()) {
                    *slot = acc * scale;
                }
                softmax_rows(&mut w, t2);
                // out[i] = sum_j w[i,j] * v[j]
                let att = CpuTensor::from_data(vec![t2, t2], w).par_matmul(&v);
                let proj = a
                    .proj_out
                    .forward(backend, &att)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let proj_ct = transpose(&proj);
                x = backend
                    .add(&x, &proj_ct)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }

            x = run_resnet(&x, &self.resnets[2])?;
            x = run_resnet(&x, &self.resnets[3])?;
        }

        // final group norm of pos_net
        let x = group_norm(&x, &self.pos_group_norm);
        if trace_on {
            trace.pos_net = Some(x.clone());
        }

        // backbone AdaLayerNorm on [T, C]; the reference then transposes
        // back to [C, T] before the ConvNeXt stack
        let xt = transpose(&x);
        let adanorm_tc = self.adanorm.forward(&xt);
        if trace_on {
            trace.adanorm = Some(adanorm_tc.clone());
        }
        let mut x = transpose(&adanorm_tc);
        // -- ConvNeXt blocks ([C, T]) --
        let traced_blocks: &[usize] = &[0usize, cfg.convnext_layers / 2, cfg.convnext_layers - 1];
        for (bi, blk) in self.convnext.iter().enumerate() {
            let residual = x.clone();
            // depthwise conv k7 p3 groups=dim
            let dw = conv1d(&x, &blk.dwconv_weight, &blk.dwconv_bias, 3, dim);
            // norm/pwconvs operate on [T, C]
            let dwt = transpose(&dw);
            let n = blk.norm.forward(&dwt);
            let h1 = blk
                .pwconv1
                .forward(backend, &n)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let g = backend.gelu(&h1).map_err(|e| anyhow::anyhow!("{e}"))?;
            let h2 = blk
                .pwconv2
                .forward(backend, &g)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // gamma scale then transpose back and add
            let mut scaled = h2.data().to_vec();
            let (rows, cols) = (h2.shape()[0], h2.shape()[1]);
            for r in 0..rows {
                for (d, slot) in scaled[r * cols..(r + 1) * cols].iter_mut().enumerate() {
                    *slot *= blk.gamma[d];
                }
            }
            let h2_ct = transpose(&CpuTensor::from_data(vec![rows, cols], scaled));
            let sum = backend
                .add(&residual, &h2_ct)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            x = sum;
            if trace_on && traced_blocks.contains(&bi) {
                trace.convnext_blocks.push((bi, x.clone()));
            }
        }

        // final LayerNorm over C ([C,T] -> [T,C])
        let xt = transpose(&x);
        let backbone_final = layer_norm_rows(
            &xt,
            cfg.layer_norm_eps,
            &self.final_norm_weight,
            &self.final_norm_bias,
        );
        if trace_on {
            trace.backbone_final = Some(backbone_final.clone());
        }

        // head linear -> [T, 2*bins]
        let head = self
            .head_out
            .forward(backend, &backbone_final)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let bins = cfg.n_fft / 2 + 1;
        ensure!(
            head.shape()[1] == 2 * bins,
            "head output width {} != 2*{bins}",
            head.shape()[1]
        );
        // mag/phase per frame, channel layout [T, 1282]: first 641 mag
        let mut mag_t = vec![0.0f32; bins * t2];
        let mut phase_t = vec![0.0f32; bins * t2];
        for i in 0..t2 {
            let row = &head.data()[i * 2 * bins..(i + 1) * 2 * bins];
            for b in 0..bins {
                let m = row[b];
                let m_exp = m.exp();
                mag_t[b * t2 + i] = m_exp.min(1e2);
                phase_t[b * t2 + i] = row[bins + b];
            }
        }
        let mag = CpuTensor::from_data(vec![bins, t2], mag_t.clone());
        let phase = CpuTensor::from_data(vec![bins, t2], phase_t.clone());
        if trace_on {
            trace.mag = Some(mag);
            trace.phase = Some(phase);
        }

        // -- iSTFT ("same" padding semantics) --
        let pcm = self.istft(&mag_t, &phase_t, t2, trace_on, &mut trace)?;
        Ok((pcm, trace))
    }

    /// Overlap-add inverse STFT with hann window and window-envelope
    /// normalization, trimming `pad = (win-hop)/2` samples each side.
    fn istft(
        &self,
        mag: &[f32],
        phase: &[f32],
        frames: usize,
        trace_on: bool,
        trace: &mut WavTokenizerTrace,
    ) -> Result<Vec<f32>> {
        let win = self.config.n_fft;
        let hop = self.config.hop_length;
        let bins = win / 2 + 1;
        let pad = (win - hop) / 2;
        let out_len = (frames - 1) * hop + win;

        let mut y_fold = vec![0.0f64; out_len];
        let mut env = vec![0.0f64; out_len];
        // window envelope does not depend on the signal: accumulate once
        let win_sq: Vec<f64> = self.window.iter().map(|w| w * w).collect();
        for fi in 0..frames {
            let start = fi * hop;
            for (wi, &ws) in win_sq.iter().enumerate() {
                env[start + wi] += ws;
            }
        }
        for fi in 0..frames {
            // S = mag * e^{i*phase} (reference head semantics), then irfft
            let s_re: Vec<f64> = (0..bins)
                .map(|b| {
                    let m = f64::from(mag[b * frames + fi]);
                    m * f64::from(phase[b * frames + fi]).cos()
                })
                .collect();
            let s_im: Vec<f64> = (0..bins)
                .map(|b| {
                    let m = f64::from(mag[b * frames + fi]);
                    m * f64::from(phase[b * frames + fi]).sin()
                })
                .collect();
            // complex IFFT via Bluestein with 1/N scaling folded in
            let frame_full = self.irfft_frame(&s_re, &s_im);
            let start = fi * hop;
            for wi in 0..win {
                y_fold[start + wi] += frame_full[wi] * self.window[wi];
            }
        }
        if trace_on {
            // store a small slice for debugging (first 4096 samples)
            let n = out_len.min(4096);
            trace.ola_raw = Some(CpuTensor::from_data(
                vec![n],
                y_fold[..n].iter().map(|&v| v as f32).collect(),
            ));
        }
        ensure!(
            env[pad..out_len - pad].iter().all(|&e| e > 1e-11),
            "istft window envelope underflow"
        );
        let trimmed = y_fold[pad..out_len - pad]
            .iter()
            .zip(&env[pad..out_len - pad])
            .map(|(y, e)| (y / e) as f32)
            .collect();
        Ok(trimmed)
    }

    /// Real IFFT of one frame from its `n_fft/2+1` complex bins:
    /// build the hermitian spectrum, run the unnormalized inverse DFT,
    /// take the real part divided by N.
    fn irfft_frame(&self, s_re: &[f64], s_im: &[f64]) -> Vec<f64> {
        let n = self.config.n_fft;
        let bins = n / 2 + 1;
        debug_assert_eq!(s_re.len(), bins);
        let mut fr = vec![0.0f64; n];
        let mut fi = vec![0.0f64; n];
        fr[0] = s_re[0];
        fi[0] = 0.0;
        for k in 1..bins {
            fr[k] = s_re[k];
            fi[k] = s_im[k];
            if k < n - k {
                fr[n - k] = s_re[k];
                fi[n - k] = -s_im[k];
            }
        }
        // Nyquist bin must be real; clamp the imaginary residue defensively
        fi[n / 2] = 0.0;
        let (xr, _xi) = self.fft.dft(&fr, &fi, 1.0);
        let scale = 1.0 / n as f64;
        (0..n).map(|t| xr[t] * scale).collect()
    }

    pub fn output_len_for_tokens(&self, tokens: usize) -> usize {
        let win = self.config.n_fft;
        let pad = (win - self.config.hop_length) / 2;
        let total = (tokens - 1) * self.config.hop_length + win;
        total.saturating_sub(2 * pad)
    }
}

fn transpose(t: &CpuTensor) -> CpuTensor {
    let (r, c) = (t.shape()[0], t.shape()[1]);
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = t.data()[i * c + j];
        }
    }
    CpuTensor::from_data(vec![c, r], out)
}

/// Numerically stable row-major softmax over fixed-width rows.
fn softmax_rows(w: &mut [f32], width: usize) {
    for row in w.chunks_mut(width) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v as f64;
        }
        let inv = 1.0 / sum as f32;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

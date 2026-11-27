use super::*;

// ---------------------------------------------------------------------------
// DDSConv / spline / ConvFlow
// ---------------------------------------------------------------------------

impl DdsConv {
    /// Input/output channel-major `[C, T]`. No padding mask (sequences are
    /// unbatched; every frame valid) — identical to the reference under an
    /// all-ones mask.
    pub(super) fn forward(&self, x: &[f32], c: usize, t: usize, ln_eps: f32) -> Vec<f32> {
        let kernel = 3usize; // duration_predictor_kernel_size
        let mut cur = x.to_vec();
        for j in 0..self.pointwise.len() {
            // reference: dilation = kernel_size ** i (= 3^i), not 2^i
            let dil = kernel.pow(j as u32);
            let taps: Vec<f32> = self.dilated_w[j].data().to_vec(); // [C*k] ([C,1,k])
            let dw = depthwise_dilated(&cur, c, t, &taps, &self.dilated_b[j], dil);
            let tt = dw.len() / c;
            let mut rows = ct_to_rows(&dw, c, tt);
            layer_norm(&mut rows, tt, c, &self.ln1[j].0, &self.ln1[j].1, ln_eps);
            gelu_in_place(&mut rows);
            let mut pw = self.pointwise[j].apply(&rows, tt);
            layer_norm(&mut pw, tt, c, &self.ln2[j].0, &self.ln2[j].1, ln_eps);
            gelu_in_place(&mut pw);
            let pw_ct = rows_to_ct(&pw, c, tt);
            for (cv, pv) in cur.iter_mut().zip(pw_ct.iter()) {
                *cv += pv;
            }
        }
        cur
    }
}

/// Unconstrained rational-quadratic spline, reverse direction only (ember
/// runs inference backwards through the duration-predictor flows).
/// `inputs[x_len]`, parameter rows `[x_len, bins*3-1]` already divided by
/// sqrt(filter_channels). Returns transformed values.
#[allow(
    clippy::needless_range_loop,
    reason = "the index is also used in arithmetic or to index several arrays"
)]
pub(super) fn spline_reverse(
    inputs: &[f32],
    widths_unnorm: &[Vec<f32>],
    heights_unnorm: &[Vec<f32>],
    derivs_unnorm: &[Vec<f32>],
    tail_bound: f32,
    bins: usize,
) -> Vec<f32> {
    const MIN_BIN_W: f64 = 1e-3;
    const MIN_BIN_H: f64 = 1e-3;
    const MIN_DERIV: f64 = 1e-3;
    let lower = -tail_bound as f64;
    let upper = tail_bound as f64;

    // softplus
    let softplus = |x: f64| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    // constant used to pin boundary derivatives
    let constant = ((1.0 - MIN_DERIV).exp() - 1.0).ln();

    let mut out = Vec::with_capacity(inputs.len());
    for (i, &x0) in inputs.iter().enumerate() {
        let x = x0 as f64;
        if !(-tail_bound as f64..=tail_bound as f64).contains(&x) {
            out.push(x0); // identity outside
            continue;
        }
        // widths/heights: softmax -> min-scaled -> cumulative bounds
        let mut ws = vec![0.0f64; bins];
        let mut hs = vec![0.0f64; bins];
        let mut max_w = f64::NEG_INFINITY;
        let mut max_h = f64::NEG_INFINITY;
        for b in 0..bins {
            ws[b] = widths_unnorm[i][b] as f64;
            hs[b] = heights_unnorm[i][b] as f64;
            max_w = max_w.max(ws[b]);
            max_h = max_h.max(hs[b]);
        }
        let mut sw = 0.0;
        let mut sh = 0.0;
        for b in 0..bins {
            ws[b] = (ws[b] - max_w).exp();
            sw += ws[b];
            hs[b] = (hs[b] - max_h).exp();
            sh += hs[b];
        }
        // Min-scale AND stretch across the interval: reference computes
        // cumwidths = (upper-lower)*cumsum(scaled) + lower, i.e. the scaled
        // widths sum to (upper-lower), not 1.
        let span = upper - lower;
        for b in 0..bins {
            ws[b] = span * (MIN_BIN_W + (1.0 - MIN_BIN_W * bins as f64) * (ws[b] / sw));
            hs[b] = span * (MIN_BIN_H + (1.0 - MIN_BIN_H * bins as f64) * (hs[b] / sh));
        }
        let mut cumw = vec![lower; bins + 1];
        let mut cumh = vec![lower; bins + 1];
        for b in 0..bins {
            cumw[b + 1] = cumw[b] + ws[b];
            cumh[b + 1] = cumh[b] + hs[b];
        }
        // scale to span (they already do by construction of min-scaling)
        // The min-scaled widths/heights already sum to (upper-lower), so the
        // cumulative arrays span the interval exactly — no rescale needed.
        cumw[0] = lower;
        cumw[bins] = upper;
        cumh[0] = lower;
        cumh[bins] = upper;

        // Reference pads the raw internal derivatives with `constant` on
        // both sides and THEN applies min_derivative + softplus to ALL of
        // them — the boundaries become min_d + softplus(constant), not the
        // constant itself.
        let mut ds = vec![0.0f64; bins + 1];
        ds[0] = MIN_DERIV + softplus(constant);
        ds[bins] = MIN_DERIV + softplus(constant);
        for b in 0..bins - 1 {
            ds[b + 1] = MIN_DERIV + softplus(derivs_unnorm[i][b] as f64);
        }

        // bin search on cumw (input domain for reverse is cumh! careful:
        // reverse maps y->x using bin locations from cumHEIGHTS)
        let mut bin_idx = bins - 1;
        for bidx in 0..bins {
            if x < cumh[bidx + 1] {
                bin_idx = bidx;
                break;
            }
        }

        let cw0 = cumw[bin_idx];
        let w_bin = cumw[bin_idx + 1] - cw0;
        let ch0 = cumh[bin_idx];
        let h_bin = cumh[bin_idx + 1] - ch0;
        let delta = h_bin / w_bin;
        let d0 = ds[bin_idx];
        let d1 = ds[bin_idx + 1];

        // solve quadratic for theta
        let intermediate1 = d0 + d1 - 2.0 * delta;
        let intermediate2 = x - ch0;
        let intermediate3 = intermediate2 * intermediate1;
        let a = h_bin * (delta - d0) + intermediate3;
        let b_ = h_bin * d0 - intermediate3;
        let c_ = -delta * intermediate2;
        let discriminant = b_ * b_ - 4.0 * a * c_;
        let root = (2.0 * c_) / (-b_ - discriminant.sqrt());

        let y = root * w_bin + cw0;
        out.push(y as f32);
    }
    out
}

impl ConvFlow {
    /// Reverse pass. `x` channel-major `[F, S]`, F = 2*half.
    /// `cond` is the SDP main-path projection (added to the DDSConv input,
    /// exactly where the reference passes `global_conditioning`).
    pub(super) fn forward_reverse(
        &self,
        x: &[f32],
        cond: Option<&[f32]>,
        s: usize,
        cfg: &VitsConfig,
    ) -> Vec<f32> {
        let f = 2usize;
        let half = f / 2;
        let first = &x[..half * s];
        let second = &x[half * s..];
        let filter_ch = cfg.hidden_size;
        let scale = (filter_ch as f32).sqrt();

        let mut pre = conv1d_dense(
            &CpuTensor::from_data(vec![half, s], first.to_vec()),
            &self.conv_pre,
            0,
        )
        .data()
        .to_vec();
        if let Some(cond) = cond {
            for (pv, cv) in pre.iter_mut().zip(cond.iter()) {
                *pv += cv;
            }
        }
        let dds_out = self.dds.forward(&pre, cfg.hidden_size, s, cfg.ln_eps);
        let proj = conv1d_dense(
            &CpuTensor::from_data(vec![cfg.hidden_size, s], dds_out),
            &self.conv_proj,
            0,
        );

        let bins = cfg.dp_bins;

        let params_per_t = bins * 3 - 1;
        let mut out_second = second.to_vec();
        for ti in 0..s {
            for ch in 0..half {
                // proj is channel-major [half*params_per_t, S]: the param
                // row for (ch, ti) strides by S
                let row: Vec<f32> = (0..params_per_t)
                    .map(|pp| proj.data()[(ch * params_per_t + pp) * s + ti])
                    .collect();
                let row = &row[..];
                let mut widths = vec![0.0f32; bins];
                let mut heights = vec![0.0f32; bins];
                let mut derivs = vec![0.0f32; params_per_t - 2 * bins];
                widths.copy_from_slice(&row[..bins]);
                heights.copy_from_slice(&row[bins..2 * bins]);
                derivs.copy_from_slice(&row[2 * bins..]);
                // NOTE: reference divides ONLY widths and heights by
                // sqrt(filter_channels); derivatives stay unscaled.
                for v in widths.iter_mut() {
                    *v /= scale;
                }
                for v in heights.iter_mut() {
                    *v /= scale;
                }
                let y = spline_reverse(
                    &[second[ch * s + ti]],
                    &[widths],
                    &[heights],
                    &[derivs],
                    cfg.dp_tail_bound,
                    bins,
                )[0];
                out_second[ch * s + ti] = y;
            }
        }

        let mut out = x.to_vec();
        out[half * s..].copy_from_slice(&out_second);
        out
    }
}

impl Sdp {
    /// Reverse path with ZERO noise: `latents = 0` seeded, run
    /// [CF_last .. CF_1(skip CF_1? see note), Affine] per the reference's
    /// `flows[:-2] + [flows[-1]]` selection.
    ///
    /// Reference flow list = [Affine, CF1, ..., CFn]; reversed =
    /// [CFn..CF1, Affine]; `flows[:-2] + [flows[-1]]` keeps
    /// [CFn ... CF2, Affine] — CF1 is dropped. Replicated exactly.
    pub(super) fn log_duration_reverse(
        &self,
        encoder_hidden: &[f32],
        c: usize,
        t: usize,
        cfg: &VitsConfig,
    ) -> Vec<f32> {
        // main conditioning
        let pre = conv1d_dense(
            &CpuTensor::from_data(vec![c, t], encoder_hidden.to_vec()),
            &self.conv_pre,
            0,
        );
        let dds_out = self.dds.forward(pre.data(), c, t, cfg.ln_eps);
        let proj = conv1d_dense(
            &CpuTensor::from_data(vec![c, t], dds_out),
            &self.conv_proj,
            0,
        );
        let g_cond = proj.data().to_vec(); // [c, t]

        // latents = randn * noise_scale_duration = 0 (deterministic contract)
        let mut latents = vec![0.0f32; 2 * t]; // [2, t]

        let mut order: Vec<usize> = (1..=self.conv_flows.len()).rev().collect();
        order.pop(); // drop CF1 (index 1) — flows[:-2]
        order.push(0); // + Affine — [flows[-1]]

        for flow_idx in order.iter() {
            latents = flip_channels(&latents, 2, t);
            if *flow_idx == 0 {
                // ElementwiseAffine stores per-channel [C=2, 1] vectors.
                let (translate, log_scale) = self.affine.as_ref().expect("affine");
                for (li, lv) in latents.iter_mut().enumerate() {
                    let ch = li / t;
                    let tr_c = translate[ch];
                    let ls_c = log_scale[ch];
                    *lv = (*lv - tr_c) * (-ls_c).exp();
                }
            } else {
                let cf = &self.conv_flows[flow_idx - 1];
                latents = cf.forward_reverse(&latents, Some(&g_cond), t, cfg);
            }
        }
        // first half channel is log-duration
        latents[..t].to_vec()
    }
}

// ---------------------------------------------------------------------------
// prior flow (WaveNet reverse) + HiFi-GAN
// ---------------------------------------------------------------------------

impl FlowLayer {
    /// Reverse coupling layer: second_half -= mean(first_half).
    /// `module_idx` names the layer for EMBER_VITS_FLOWDUMP parity dumps
    /// (reference applies modules 3,2,1,0 in reverse mode).
    pub(super) fn forward_reverse(
        &self,
        x: &[f32],
        f: usize,
        s: usize,
        cfg: &VitsConfig,
        module_idx: usize,
    ) -> Vec<f32> {
        let half = f / 2;
        let first = x[..half * s].to_vec();
        let pre = conv1d_dense(
            &CpuTensor::from_data(vec![half, s], first.clone()),
            &self.conv_pre,
            0,
        )
        .data()
        .to_vec();

        // WaveNet (no global conditioning; speaker_embedding_size = 0)
        let dbg_dir = std::env::var("EMBER_VITS_FLOWDUMP").ok();
        if let Some(dir) = &dbg_dir {
            std::fs::create_dir_all(dir).ok();
            dump_npy(std::path::Path::new(dir), &format!("f{module_idx}_in"), x);
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x0"),
                &first,
            );
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x1"),
                &x[half * s..],
            );
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_convpre"),
                &pre,
            );
        }
        let mut cur = pre;
        let mut outputs = vec![0.0f32; cfg.hidden_size * s];
        for i in 0..self.wn_in.len() {
            // reference pads in_layers by (k*dia-dia)/2 (= 2 for k=5,dia=1)
            let pad = (self.wn_in[i].k - 1) / 2;
            let h = conv1d_dense(
                &CpuTensor::from_data(vec![cfg.hidden_size, s], cur.clone()),
                &self.wn_in[i],
                pad,
            )
            .data()
            .to_vec();
            // fused add-tanh-sigmoid-multiply with zero cond
            let mut acts = vec![0.0f32; cfg.hidden_size * s];
            for j in 0..cfg.hidden_size * s {
                let tanh_half = (h[j]).tanh();
                let sig_half = 1.0 / (1.0 + (-h[cfg.hidden_size * s + j]).exp());
                acts[j] = tanh_half * sig_half;
            }
            let rs = conv1d_dense(
                &CpuTensor::from_data(vec![cfg.hidden_size, s], acts.clone()),
                &self.wn_rs[i],
                0,
            )
            .data()
            .to_vec();
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_h"),
                    &h,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_acts"),
                    &acts,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_rs"),
                    &rs,
                );
            }
            if i < self.wn_in.len() - 1 {
                // res_skip_channels = 2H: first H residual, last H skip
                for j in 0..cfg.hidden_size * s {
                    cur[j] += rs[j];
                    outputs[j] += rs[cfg.hidden_size * s + j];
                }
            } else {
                for j in 0..cfg.hidden_size * s {
                    outputs[j] += rs[j];
                }
            }
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_cur"),
                    &cur,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("f{module_idx}_wn{i}_acc"),
                    &outputs,
                );
            }
        }

        let mean = conv1d_dense(
            &CpuTensor::from_data(vec![cfg.hidden_size, s], outputs.clone()),
            &self.conv_post,
            0,
        )
        .data()
        .to_vec();
        if let Some(dir) = &dbg_dir {
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_mean"),
                &mean,
            );
            let x1new: Vec<f32> = x[half * s..]
                .iter()
                .zip(mean.iter())
                .map(|(sv, mv)| sv - mv)
                .collect();
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_x1new"),
                &x1new,
            );
        }
        let mut out = x.to_vec();
        for (sv, mv) in out[half * s..].iter_mut().zip(mean.iter()) {
            *sv -= mv;
        }
        if let Some(dir) = &dbg_dir {
            dump_npy(
                std::path::Path::new(dir),
                &format!("f{module_idx}_out"),
                &out,
            );
        }
        out
    }
}

impl ConvTranspose1d {
    /// Input `[C_in, T]` -> output `[C_out, (T-1)*stride - 2*padding + k]`.
    #[allow(
        clippy::needless_range_loop,
        reason = "the index is also used in arithmetic or to index several arrays"
    )]
    pub(super) fn forward(&self, input: &[f32], c_in: usize, t: usize) -> Vec<f32> {
        let k = self.w.shape()[2];
        let c_out = self.w.shape()[1];
        let t_out = (t - 1) * self.stride + k - 2 * self.padding;
        let mut acc = vec![0.0f64; c_out * t_out];
        let w = self.w.data();
        for ci in 0..c_in {
            let xc = &input[ci * t..(ci + 1) * t];
            for ti in 0..t {
                let v = xc[ti] as f64;
                if v == 0.0 {
                    continue;
                }
                let obase: isize = (ti * self.stride) as isize - self.padding as isize;
                for co in 0..c_out {
                    let wc = &w[(ci * c_out + co) * k..(ci * c_out + co) * k + k];
                    for kk in 0..k {
                        let oi = obase + kk as isize;
                        if oi >= 0 && (oi as usize) < t_out {
                            acc[co * t_out + oi as usize] += v * wc[kk] as f64;
                        }
                    }
                }
            }
        }
        let mut out = vec![0.0f32; c_out * t_out];
        for co in 0..c_out {
            for ti in 0..t_out {
                out[co * t_out + ti] = acc[co * t_out + ti] as f32 + self.b[co];
            }
        }
        out
    }
}

impl ResBlock {
    /// `dump_name` prefixes EMBER_VITS_DECDUMP substep dumps ("" disables).
    pub(super) fn forward(
        &self,
        x: &[f32],
        c: usize,
        t: usize,
        slope: f32,
        dump_name: &str,
    ) -> Vec<f32> {
        let dbg_dir = std::env::var("EMBER_VITS_DECDUMP").ok();
        let mut cur = x.to_vec();
        // HiFi-GAN resblock dilation cycle: resblock_dilation_sizes[0] =
        // [1, 3, 5] for every MMS checkpoint (NOT the iteration index d+1).
        const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
        for (d, &dil1) in RESBLOCK_DILATIONS.iter().enumerate() {
            let residual = cur.clone();
            leaky_in_place(&mut cur, slope);
            let pad1 = cur_pad_for(self.c1[d].k, dil1);
            let y1 = conv_dense_dilated(
                &CpuTensor::from_data(vec![c, t], cur),
                &self.c1[d],
                pad1,
                dil1,
            );
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_c1"),
                    y1.data(),
                );
            }
            let t1 = y1.shape()[1];
            let mut tmp = y1.data().to_vec();
            leaky_in_place(&mut tmp, slope);
            let pad2 = cur_pad_for(self.c2[d].k, 1);
            let mut y2 =
                conv_dense_padded(&CpuTensor::from_data(vec![c, t1], tmp), &self.c2[d], pad2)
                    .data()
                    .to_vec();
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_residual"),
                    &residual,
                );
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_c2presid"),
                    &y2,
                );
            }
            for (cv, rv) in y2.iter_mut().zip(residual.iter()) {
                *cv += rv;
            }
            cur = y2;
            if let Some(dir) = &dbg_dir {
                dump_npy(
                    std::path::Path::new(dir),
                    &format!("{dump_name}_it{d}_out"),
                    &cur,
                );
            }
        }
        cur
    }
}

pub(super) fn leaky_in_place(v: &mut [f32], slope: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= slope;
        }
    }
}

pub(super) fn cur_pad_for(k: usize, dil: usize) -> usize {
    (k * dil - dil) / 2
}

/// groups=1 Conv1d with arbitrary symmetric padding over a borrowed buffer.
pub(super) fn conv_dense_padded(input: &CpuTensor, c: &DenseConv1d, pad: usize) -> CpuTensor {
    conv1d_dense(input, c, pad)
}

impl HifiGan {
    /// Full decode of a final spectrogram `[F, S]`.
    /// EMBER_VITS_DECDUMP=<dir> mirrors scripts/ref_decoder_dump.py.
    pub(super) fn decode(&self, spec: &[f32], f: usize, s: usize, cfg: &VitsConfig) -> Vec<f32> {
        let dbg_dir = std::env::var("EMBER_VITS_DECDUMP").ok();
        let dump = |name: &str, data: &[f32]| {
            if let Some(dir) = &dbg_dir {
                dump_npy(std::path::Path::new(dir), name, data);
            }
        };
        if let Some(dir) = &dbg_dir {
            std::fs::create_dir_all(dir).ok();
            dump("d_spec", spec);
        }
        let mut hidden = conv1d_dense(
            &CpuTensor::from_data(vec![f, s], spec.to_vec()),
            &self.conv_pre,
            3,
        )
        .data()
        .to_vec();
        dump("d_convpre", &hidden);
        let nblk = 3usize;
        for stage in 0..self.ups.len() {
            let ch_in = self.ups[stage].w.shape()[0];
            let tt = hidden.len() / ch_in;
            for v in hidden.iter_mut() {
                *v = if *v < 0.0 {
                    *v * cfg.leaky_relu_slope
                } else {
                    *v
                };
            }
            hidden = self.ups[stage].forward(&hidden, ch_in, tt);
            let ch_out = self.ups[stage].w.shape()[1];
            let to = hidden.len() / ch_out;
            dump(&format!("d_up{stage}"), &hidden);
            // average of the stage's resblocks
            let mut sum = vec![0.0f64; ch_out * to];
            for blk in 0..nblk {
                let rb = &self.resblocks[stage * nblk + blk];
                let y = rb.forward(
                    &hidden,
                    ch_out,
                    to,
                    cfg.leaky_relu_slope,
                    &format!("d_rb{stage}{blk}"),
                );
                for (sv, yv) in sum.iter_mut().zip(y.iter()) {
                    *sv += *yv as f64;
                }
                // reference accumulates in-place; the running sum is what
                // scripts/ref_decoder_dump.py names d_rb{stage}{blk}
                let running: Vec<f32> = sum.iter().map(|&v| v as f32).collect();
                dump(&format!("d_rb{stage}{blk}"), &running);
            }
            hidden = sum.iter().map(|&v| (v / nblk as f64) as f32).collect();
            dump(&format!("d_stage{stage}"), &hidden);
        }
        let c_last = self.conv_post_w.shape()[1];
        let tl = hidden.len() / c_last;
        // transformers VitsHifiGan.forward applies the FINAL pre-conv_post
        // leaky with the DEFAULT slope (0.01), not config.leaky_relu_slope
        // (0.1) used everywhere else — modeling_vits.py line
        // `nn.functional.leaky_relu(hidden_states)` before conv_post.
        const PREPOST_SLOPE: f32 = 0.01;
        for v in hidden.iter_mut() {
            *v = if *v < 0.0 { *v * PREPOST_SLOPE } else { *v };
        }
        dump("d_prepost", &hidden);
        // conv_post k7 pad3, no bias, then tanh
        let t_out = tl + 6 - 7 + 1;
        let mut wave = vec![0.0f32; t_out];
        let w = self.conv_post_w.data(); // [1, C, 7]
        let hc = &hidden;
        wave.par_chunks_mut(t_out).enumerate().for_each(|(o, ow)| {
            // every t_out = tl + 2*3 - 7 + 1 output index is a valid window
            #[allow(
                clippy::needless_range_loop,
                reason = "the index is also used in arithmetic or to index several arrays"
            )]
            for ti in 0..t_out {
                let mut acc = 0.0f64;
                for ci in 0..c_last {
                    let wc = &w[(o * c_last + ci) * 7..(o * c_last + ci) * 7 + 7];
                    let base = ti; // pad=3
                    #[allow(
                        clippy::needless_range_loop,
                        reason = "the index is also used in arithmetic or to index several arrays"
                    )]
                    for kk in 0..7 {
                        let p = base + kk;
                        if p >= 3 && p < 3 + tl {
                            acc += (hc[ci * tl + (p - 3)] as f64) * wc[kk] as f64;
                        }
                    }
                }
                ow[ti] = (acc as f32).tanh();
            }
        });
        // pre-tanh conv output (single output channel)
        let pretanh: Vec<f32> = (0..t_out)
            .map(|ti| {
                let mut acc = 0.0f64;
                for ci in 0..c_last {
                    let wc = &w[ci * 7..ci * 7 + 7];
                    #[allow(
                        clippy::needless_range_loop,
                        reason = "the index is also used in arithmetic or to index several arrays"
                    )]
                    for kk in 0..7 {
                        let p = ti + kk;
                        if p >= 3 && p < 3 + tl {
                            acc += hc[ci * tl + (p - 3)] as f64 * wc[kk] as f64;
                        }
                    }
                }
                acc as f32
            })
            .collect();
        dump("d_pretanh", &pretanh);
        dump("d_tanh", &wave);
        wave
    }
}

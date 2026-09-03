//! EmberSEC Phase V Step 2: exponent-bit sweep over REAL d values.
//!
//! Standalone crate (depends on ember by path; repo tree untouched). For
//! representative d values spanning the measured GGUF distributions x dtypes
//! {Q4_K, Q6_K, Q8_0}, flips EACH of the 16 bits of the f16 scale word in one
//! block and runs the real decode kernels (k_decode / q8_decode) via
//! ember::quant_fault. Rows go to sweep.jsonl, one JSON object per trial.

use ember::quant::{quantize_q8_0_into, QuantizedWeight};
use ember::quant_fault::{inject_bit_flip, k_decode, measure_impact, q8_decode};
use ember::quant_k::{KQuantDtype, KQuantWeight};
use std::fs::File;
use std::io::Write;

// ---- minimal f16 <-> f32 (round-to-nearest-even on the way in) ----

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as i32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as i32;
    let f = if exp == 0 {
        mant as f32 / 1024.0 * 2f32.powi(-14)
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 { -f } else { f }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() || a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a < 6.103515625e-05 {
        let m = (a / 2f32.powi(-24)).round() as u32;
        if m == 0 {
            return sign;
        }
        if m > 1023 {
            return sign | 0x0400;
        }
        return sign | (m as u16);
    }
    let mut exp = a.log2().floor() as i32;
    if exp < -14 {
        exp = -14;
    }
    if exp > 15 {
        exp = 15;
    }
    let mant = (a / 2f32.powi(exp) - 1.0) * 1024.0;
    let mut m = mant.round() as i32;
    let mut e = exp + 15;
    if m >= 1024 {
        m = 0;
        e += 1;
    }
    if m < 0 {
        m = 0;
    }
    if e >= 31 {
        return sign | 0x7C00;
    }
    sign | ((e as u16) << 10) | (m as u16)
}

fn fmt_f32(v: f32) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else {
        format!("{v:.10e}")
    }
}

// ---- deterministic fixtures (mirrors quant_fault.rs test helpers) ----

struct Xor64(u64);
impl Xor64 {
    fn next(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x & 0xFF) as u8
    }
}

const OUT: usize = 8;
const INP: usize = 256; // one K super-block per row

fn activations(seed: u64) -> Vec<f32> {
    (0..INP)
        .map(|i| (((i * 31 + seed as usize * 13) as f32) / 97.0).sin() * 0.7)
        .collect()
}

fn kweight(dtype: KQuantDtype, seed: u64, d_bits: u16) -> KQuantWeight {
    let blocks = OUT * (INP / 256);
    let block_bytes = match dtype {
        KQuantDtype::Q4K => 144usize,
        KQuantDtype::Q6K => 210usize,
    };
    let mut rng = Xor64(0x243F6A8885A308D3 ^ seed);
    let mut bytes = vec![0u8; blocks * block_bytes];
    for b in bytes.iter_mut() {
        *b = rng.next();
    }
    let d_off = match dtype {
        KQuantDtype::Q4K => 0usize,
        KQuantDtype::Q6K => 208usize,
    };
    let min_bits = f32_to_f16_bits(-0.02);
    for block in 0..blocks {
        let base = block * block_bytes;
        bytes[base + d_off..base + d_off + 2].copy_from_slice(&d_bits.to_le_bytes());
        if dtype == KQuantDtype::Q4K {
            bytes[base + 2..base + 4].copy_from_slice(&min_bits.to_le_bytes());
        }
    }
    KQuantWeight::try_new(bytes, [OUT, INP], dtype).unwrap()
}

fn q8weight(seed: u64, d_bits: u16) -> QuantizedWeight {
    let mut data = Vec::new();
    for row in 0..OUT {
        let values: Vec<f32> = (0..INP)
            .map(|i| (((i * 31 + row * 17 + seed as usize * 13) as f32) / 97.0).sin() * 0.7)
            .collect();
        let mut row_bytes = Vec::new();
        quantize_q8_0_into(&values, &mut row_bytes);
        // uniform realistic d across all blocks (mirrors the kweight helper)
        let n_blocks = INP / 32;
        for b in 0..n_blocks {
            row_bytes[b * 34..b * 34 + 2].copy_from_slice(&d_bits.to_le_bytes());
        }
        data.extend_from_slice(&row_bytes);
    }
    QuantizedWeight::try_new(data, vec![OUT, INP]).unwrap()
}

fn d_sets() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        (
            "Q4_K",
            vec![
                1.14441e-5, 3e-5, 5.4121e-5, 7.1228e-5, 8.738e-5, 9.8646e-5,
                1.15514e-4, 1.7941e-4, 2.63929e-4, 5e-4, 1.93119e-3, 1.0,
            ],
        ),
        (
            "Q6_K",
            vec![
                -2.915859e-4, -2.98023e-5, -2.11e-5, -1.42455e-5, -8.16584e-6,
                8.16584e-6, 1.40071e-5, 2.0504e-5, 2.95043e-5, 2.59876e-4,
                1.0, -1.0,
            ],
        ),
        (
            "Q8_0",
            vec![
                2.76566e-5, 1.5378e-4, 2.0957e-4, 2.78711e-4, 3.34501e-4,
                4.07457e-4, 5.84602e-4, 8.3828e-4, 2e-3, 5e-3, 9.41467e-3, 1.0,
            ],
        ),
    ]
}

fn emit(
    out: &mut File,
    dtype_name: &str,
    d_req: f32,
    d_bits: u16,
    bit: u16,
    pristine: &[f32],
    faulted: &[f32],
    faulted_bits: u16,
) {
    let impact = measure_impact(pristine, faulted);
    let (rel, mx) = if impact.logits_finite {
        (
            format!("{:.6e}", impact.rel_l2),
            format!("{:.6e}", impact.max_abs_logit_diff),
        )
    } else {
        ("null".to_string(), "null".to_string())
    };
    writeln!(
        out,
        "{{\"dtype\":\"{dtype_name}\",\"d_requested\":{d_req:.10e},\
         \"d_bits\":\"0x{d_bits:04x}\",\"d_actual\":\"{da}\",\"bit\":{bit},\
         \"faulted_bits\":\"0x{faulted_bits:04x}\",\"faulted_d\":\"{fd}\",\
         \"finite\":{fin},\"rel_l2\":{rel},\"max_abs\":{mx},\
         \"top1_flipped\":{top}}}",
        d_req = d_req,
        d_bits = d_bits,
        da = fmt_f32(f16_to_f32(d_bits)),
        bit = bit,
        faulted_bits = faulted_bits,
        fd = fmt_f32(f16_to_f32(faulted_bits)),
        fin = impact.logits_finite,
        rel = rel,
        mx = mx,
        top = impact.top1_flipped,
    )
    .unwrap();
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/opencode/phase5/sweep.jsonl".to_string());
    let mut out = File::create(&out_path).expect("create sweep output");
    let mut n = 0usize;

    for (dtype_name, ds) in d_sets() {
        for &d in &ds {
            let d_bits = f32_to_f16_bits(d);
            match dtype_name {
                "Q4_K" | "Q6_K" => {
                    let dtype = if dtype_name == "Q4_K" {
                        KQuantDtype::Q4K
                    } else {
                        KQuantDtype::Q6K
                    };
                    let d_off = if dtype == KQuantDtype::Q4K { 0 } else { 208 };
                    let src = activations(7);
                    let p =
                        k_decode(&kweight(dtype, 0x5EED, d_bits), &src, false)
                            .expect("pristine");
                    for bit in 0u16..16 {
                        let byte = usize::from(bit >= 8);
                        let b = (bit % 8) as u8;
                        let mut wf = kweight(dtype, 0x5EED, d_bits);
                        let (fb, f) = {
                            let data = wf.data_mut().expect("owned");
                            inject_bit_flip(&mut data[d_off..d_off + 2], byte, b)
                                .unwrap();
                            let fb =
                                u16::from_le_bytes([data[d_off], data[d_off + 1]]);
                            let f = k_decode(&wf, &src, false).expect("faulted");
                            (fb, f)
                        };
                        emit(&mut out, dtype_name, d, d_bits, bit, &p, &f, fb);
                        n += 1;
                    }
                }
                _ => {
                    let src8 = activations(11);
                    let p =
                        q8_decode(&q8weight(11, d_bits), &src8).expect("pristine");
                    for bit in 0u16..16 {
                        let byte = usize::from(bit >= 8);
                        let b = (bit % 8) as u8;
                        let mut wf = q8weight(11, d_bits);
                        let (fb, f) = {
                            let data = wf.data_mut().expect("owned");
                            inject_bit_flip(&mut data[0..2], byte, b).unwrap();
                            let fb = u16::from_le_bytes([data[0], data[1]]);
                            let f = q8_decode(&wf, &src8).expect("faulted");
                            (fb, f)
                        };
                        emit(&mut out, dtype_name, d, d_bits, bit, &p, &f, fb);
                        n += 1;
                    }
                }
            }
        }
    }
    println!("wrote {n} trials to {out_path}");
}

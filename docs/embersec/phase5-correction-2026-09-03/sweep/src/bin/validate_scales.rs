//! Cross-check the Python GGUF scanner against ember's own loader.
//!
//! Usage: validate_scales <model.gguf> <samples.json>
//! For each sampled (tensor, block, d_bits): load the model with ember's
//! loader (mmap + compressed-resident K strategy: no bulk RAM) and compare
//! the raw d header word with the scanner's reading.

use ember::loader::{load_gguf_with_k_strategy, LoadedTensor};
use ember::quant_k::KStrategy;
use std::fs::File;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: validate_scales <model.gguf> <samples.json>");
    let loader =
        load_gguf_with_k_strategy(&args[1], KStrategy::Auto, true).expect("load gguf");
    let mut js = String::new();
    File::open(&args[2])
        .expect("open samples")
        .read_to_string(&mut js)
        .unwrap();
    // minimal JSON parse: array of {"tensor":..,"dtype":N,"block":B,"d_bits":"0x...."}
    let samples = parse_samples(&js);
    let mut ok = 0usize;
    let mut fail = 0usize;
    for (tensor, dtype, block, want) in samples {
        let t = loader
            .tensors
            .get(&tensor)
            .unwrap_or_else(|| panic!("tensor missing: {tensor}"));
        let (data, stride, d_off) = match t {
            LoadedTensor::Q8_0(w) => (w.data(), 34usize, 0usize),
            LoadedTensor::KQuant(w) => (
                w.data(),
                if dtype == 12 { 144 } else { 210 },
                if dtype == 12 { 0 } else { 208 },
            ),
            LoadedTensor::F32(_) => panic!("sampled tensor is F32: {tensor}"),
        };
        let base = block * stride + d_off;
        let got = u16::from_le_bytes([data[base], data[base + 1]]);
        if got == want {
            ok += 1;
        } else {
            fail += 1;
            println!("MISMATCH {tensor} block {block}: scanner=0x{want:04x} ember=0x{got:04x}");
        }
    }
    println!("validate_scales: {ok} matched, {fail} mismatched");
    if fail > 0 {
        std::process::exit(1);
    }
}

// crude parser for the known samples.json shape (no serde dep on purpose)
fn parse_samples(js: &str) -> Vec<(String, u32, usize, u16)> {
    let mut out = Vec::new();
    let mut rest = js;
    while let Some(i) = rest.find("\"tensor\"") {
        rest = &rest[i..];
        let name = read_str_field(rest, "\"tensor\"");
        let dtype: u32 = read_num_field(rest, "\"dtype\"") as u32;
        let block: usize = read_num_field(rest, "\"block\"");
        let bits = read_str_field(rest, "\"d_bits\"");
        let want = u16::from_str_radix(bits.trim_start_matches("0x"), 16).unwrap();
        out.push((name, dtype, block, want));
        rest = &rest[rest.find("\"d_bits\"").unwrap() + 8..];
    }
    out
}

fn read_str_field(s: &str, key: &str) -> String {
    let i = s.find(key).unwrap();
    let after = &s[i + key.len()..];
    let q1 = after.find('"').unwrap() + 1;
    let q2 = q1 + after[q1..].find('"').unwrap();
    after[q1..q2].to_string()
}

fn read_num_field(s: &str, key: &str) -> usize {
    let i = s.find(key).unwrap();
    let after = &s[i + key.len()..];
    let c = after.find(':').unwrap() + 1;
    let num: String = after[c..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().unwrap()
}

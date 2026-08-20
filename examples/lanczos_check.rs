//! Quick LANCZOS parity check against Pillow: resize a PNG to WxH and
//! write the raw f32 CHW output to a file.
use ember::multimodal::image::{decode_rgb, resize, Resample};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = &args[1];
    let out_w: usize = args[2].parse()?;
    let out_h: usize = args[3].parse()?;
    let output = &args[4];
    let img = decode_rgb(Path::new(input))?;
    let resized = resize(&img, out_w, out_h, Resample::Lanczos)?;
    let mut bytes = Vec::new();
    for v in resized.data() {
        bytes.extend(v.to_le_bytes());
    }
    let n = bytes.len();
    std::fs::write(output, &bytes)?;
    println!(
        "wrote {} bytes ({}x{}x3) from {:?}->{}x{}",
        n,
        out_w,
        out_h,
        img.shape(),
        out_w,
        out_h
    );
    Ok(())
}

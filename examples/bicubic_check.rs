fn main() {
    // decode PNG via ember, run the exact video chain, write raw f32
    let args: Vec<String> = std::env::args().collect();
    let png = &args[1];
    let out = &args[2];
    let img = ember::multimodal::image::decode_rgb(std::path::Path::new(png)).unwrap();
    let up = ember::multimodal::image::resize(
        &img,
        2048,
        2048,
        ember::multimodal::image::Resample::Bicubic,
    )
    .unwrap();
    let down = ember::multimodal::image::resize(
        &up,
        512,
        512,
        ember::multimodal::image::Resample::Bicubic,
    )
    .unwrap();
    // normalized pixels like the processor: (x/255 - 0.5)/0.5
    let mut outv = Vec::with_capacity(3 * 512 * 512);
    for v in down.data() {
        outv.push((v / 255.0 - 0.5) / 0.5);
    }
    let bytes: Vec<u8> = outv.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(out, &bytes).unwrap();
}

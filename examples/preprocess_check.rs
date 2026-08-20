//! Full preprocessing parity check: run the SmolVLM recipe on a PNG and
//! dump the normalized tiles to a raw f32 file.
use ember::multimodal::image::{decode_rgb, preprocess, ImagePreprocessConfig};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let img = decode_rgb(Path::new(&args[1]))?;
    let config = ImagePreprocessConfig {
        resize_longest_edge: Some(2048),
        tile_size: Some(512),
        resample: ember::multimodal::image::Resample::Lanczos,
        rescale_factor: 1.0 / 255.0,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let pp = preprocess(&img, &config)?;
    let mut bytes = Vec::new();
    for v in pp.tiles.data() {
        bytes.extend(v.to_le_bytes());
    }
    std::fs::write(&args[2], &bytes)?;
    println!(
        "tiles {:?} grid {:?} resized {:?} global {}",
        pp.tiles.shape(),
        pp.tile_grid,
        pp.resized_dims,
        pp.has_global_tile
    );
    Ok(())
}

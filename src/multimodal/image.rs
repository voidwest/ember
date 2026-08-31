//! Image preprocessing, isolated from the model runtime.
//!
//! This module implements the *generic* image processing primitives a
//! vision-language model needs — decoding, RGB conversion, resizing,
//! normalization, tiling — driven by an [`ImagePreprocessConfig`]. No
//! model-specific recipe is hardcoded here and nothing in the tensor
//! runtime depends on this module.
//!
//! The LANCZOS resampler is a faithful port of Pillow's two-pass
//! fixed-point implementation (`libImaging/Resample.c`, `Image.resize`
//! with `Image.LANCZOS`), which is what HuggingFace's PIL image-processing
//! backend uses. It is bit-exact with Pillow for uint8 RGB images:
//!
//! - precompute per-output-pixel bounds and float coefficients,
//! - scale coefficients to 22-bit fixed point with Pillow's rounding,
//! - horizontal pass into a uint8 intermediate (clipped),
//! - vertical pass with the same fixed-point math.

use crate::tensor::CpuTensor;
use anyhow::{anyhow, Result};
use std::path::Path;

/// Resampling filter for [`resize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// Pillow LANCZOS (truncated sinc, 3-lobe, antialiased). Bit-exact with
    /// Pillow's fixed-point implementation.
    Lanczos,
    /// Pillow BICUBIC (cubic convolution, a = -0.5, support 2). Bit-exact
    /// with Pillow's fixed-point implementation; the SmolVLM2 video chain
    /// uses this filter for its stock resize legs (Phase 5 Track H).
    Bicubic,
}

/// A complete image preprocessing recipe.
///
/// The recipe is data, not code: the same module serves any model by
/// choosing a different config.
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    /// Resize the longest edge to this many pixels before tiling.
    /// `None` disables the initial resize.
    pub resize_longest_edge: Option<u32>,
    /// Tile size for square-crop splitting. Tiles are cropped from the
    /// resized image; a final "global" tile is the resized image scaled
    /// down to `tile_size` square. `None` disables splitting.
    pub tile_size: Option<u32>,
    /// Resample filter for all resizes.
    pub resample: Resample,
    /// Rescale factor applied to raw 0..255 pixels (typically 1/255).
    pub rescale_factor: f32,
    /// Per-channel mean for normalization.
    pub mean: [f32; 3],
    /// Per-channel std for normalization.
    pub std: [f32; 3],
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            resize_longest_edge: None,
            tile_size: None,
            resample: Resample::Lanczos,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}

/// A preprocessed image: normalized tiles plus the geometry the assembler
/// needs (tile grid, dimensions, validity mask).
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// Normalized pixel tiles, `[n_tiles, 3, tile, tile]` (CHW, 0..1 scale
    /// after rescale/normalize).
    pub tiles: CpuTensor,
    /// Validity mask `[n_tiles, tile, tile]` (1 = valid pixel).
    pub mask: CpuTensor,
    /// Original decoded dimensions `(height, width)`.
    pub original_dims: (usize, usize),
    /// Dimensions of the resized (pre-tile) image `(height, width)`.
    pub resized_dims: (usize, usize),
    /// Tile grid `(rows, cols)`; `(0, 0)` when no splitting occurred.
    pub tile_grid: (usize, usize),
    /// The global (downscaled whole-image) tile is appended last.
    pub has_global_tile: bool,
    /// Sub-stage timings (recorded by `preprocess`).
    pub timings: PreprocessTimings,
}

/// Sub-stage timings of one [`preprocess`] call (milliseconds).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PreprocessTimings {
    pub resize_ms: f64,
    pub tile_ms: f64,
    pub normalize_ms: f64,
}

/// Maximum decoded image edge (pixels) for decoders that honor strict
/// dimension limits (zune-jpeg does; the png crate does not — its output
/// is bounded by [`MAX_IMAGE_DECODE_BYTES`] instead).
pub const MAX_IMAGE_DIM: u32 = 8192;

/// Per-image decoder allocation budget. The `image` crate's own default is
/// 512 MiB and is *non-strict* for some decoders; we set an explicit
/// budget here and additionally reject anything above it. After decode,
/// [`rgb8_to_tensor`] multiplies this by four (u8 bitmap -> f32 tensor),
/// so an admission cap here is what keeps a decompression bomb from
/// turning into a multi-GiB f32 allocation on a 16 GB host.
pub const MAX_IMAGE_DECODE_BYTES: u64 = 256 * 1024 * 1024;

/// Explicit decoder limits applied to every untrusted image decode.
fn image_decode_limits() -> image::Limits {
    // `Limits` is #[non_exhaustive]; Default keeps the crate's other
    // behaviors and we tighten exactly the fields we care about.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_DECODE_BYTES);
    limits
}

/// Decode an image from memory (PNG/JPEG bytes) and return RGB pixels as
/// f32 `[3, height, width]` with values in 0..255 (channels-first).
pub fn decode_rgb_bytes(bytes: &[u8]) -> Result<CpuTensor> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes));
    reader.limits(image_decode_limits());
    let img = reader
        .with_guessed_format()
        .map_err(|e| anyhow!("failed to read image bytes: {e}"))?
        .decode()
        .map_err(|e| anyhow!("failed to decode image bytes: {e}"))?
        .to_rgb8();
    Ok(rgb8_to_tensor(&img))
}

/// Decode an image file (PNG/JPEG) and return RGB pixels as f32
/// `[3, height, width]` with values in 0..255 (channels-first).
pub fn decode_rgb(path: &Path) -> Result<CpuTensor> {
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| anyhow!("failed to open image {}: {e}", path.display()))?;
    reader.limits(image_decode_limits());
    let img = reader
        .with_guessed_format()
        .map_err(|e| anyhow!("failed to read image {}: {e}", path.display()))?
        .decode()
        .map_err(|e| anyhow!("failed to decode image {}: {e}", path.display()))?
        .to_rgb8();
    Ok(rgb8_to_tensor(&img))
}

fn rgb8_to_tensor(img: &image::RgbImage) -> CpuTensor {
    let (width, height) = img.dimensions();
    let (width, height) = (width as usize, height as usize);
    let mut data = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x as u32, y as u32);
            data[y * width + x] = p[0] as f32;
            data[height * width + y * width + x] = p[1] as f32;
            data[2 * height * width + y * width + x] = p[2] as f32;
        }
    }
    CpuTensor::from_data(vec![3, height, width], data)
}

/// Preprocess a decoded RGB image with the given recipe.
///
/// Pipeline (mirrors the HuggingFace Idefics3 processor):
/// resize longest edge -> round both edges up to whole `tile_size` multiples
/// (re-resize; the reference `resize_for_vision_encoder`) -> split into
/// `tile_size` squares + global tile -> rescale -> normalize. Returns
/// normalized tiles and geometry. Any aspect ratio is safe: the rounding
/// stage guarantees exact tile grids for every geometry.
pub fn preprocess(image: &CpuTensor, config: &ImagePreprocessConfig) -> Result<PreprocessedImage> {
    let t0 = std::time::Instant::now();
    anyhow::ensure!(
        image.shape() == [3, image.shape()[1], image.shape()[2]] && image.ndim() == 3,
        "preprocess expects CHW [3, h, w] RGB pixels"
    );
    let original_dims = (image.shape()[1], image.shape()[2]);

    // 1. resize longest edge
    let (mut h, mut w) = original_dims;
    if let Some(max_edge) = config.resize_longest_edge {
        let max_edge = max_edge as usize;
        if w >= h {
            w = max_edge;
            h = (w as f64 / aspect(original_dims)) as usize;
            if h % 2 != 0 {
                h += 1;
            }
        } else {
            h = max_edge;
            w = (h as f64 * aspect(original_dims)) as usize;
            if w % 2 != 0 {
                w += 1;
            }
        }
        h = h.max(1);
        w = w.max(1);
    }
    let mut resized = resize(image, w, h, config.resample)?;

    // 1.5 round both edges up to whole tiles when splitting is enabled
    // (the reference `resize_for_vision_encoder` stage): after this, the
    // tile grid divides exactly — no partial strips exist to drop. The
    // float arithmetic replicates the reference's f64 operations in the
    // same order so output dimensions match bit-for-bit. This is what makes
    // heterogeneous image geometry safe: every image, whatever its aspect,
    // ends in exact `tile`-multiple dimensions.
    if let Some(tile_u) = config.tile_size {
        let tile = tile_u as usize;
        let aspect_ratio = w as f64 / h as f64;
        let rounded = if w >= h {
            let nw = (w as f64 / tile as f64).ceil() as usize * tile;
            let mut nh = (nw as f64 / aspect_ratio).trunc() as usize;
            nh = (nh as f64 / tile as f64).ceil() as usize * tile;
            (nh, nw)
        } else {
            let nh = (h as f64 / tile as f64).ceil() as usize * tile;
            let mut nw = (nh as f64 * aspect_ratio).trunc() as usize;
            nw = (nw as f64 / tile as f64).ceil() as usize * tile;
            (nh, nw)
        };
        if rounded != (h, w) {
            // second resample of the already-resized image, exactly like
            // the reference's two-stage pipeline
            resized = resize(&resized, rounded.1, rounded.0, config.resample)?;
        }
        h = rounded.0.max(tile);
        w = rounded.1.max(tile);
    }
    let resized_dims = (h, w);
    let mut timings = PreprocessTimings {
        resize_ms: t0.elapsed().as_secs_f64() * 1e3,
        ..Default::default()
    };
    let t_tile = std::time::Instant::now();

    // 2. split into tiles + global tile. After stage 1.5 both edges are
    // exact tile multiples (when splitting is enabled), so the grid divides
    // with no dropped strips — same crops the reference `unfold` produces.
    let mut tiles_uint = Vec::<CpuTensor>::new();
    let mut tile_grid = (0usize, 0usize);
    let mut has_global_tile = false;
    if let Some(tile) = config.tile_size {
        let tile = tile as usize;
        if h > tile || w > tile {
            // exact because stage 1.5 rounded both edges to tile multiples
            let rows = h / tile;
            let cols = w / tile;
            tile_grid = (rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    let start_y = r * tile;
                    let start_x = c * tile;
                    tiles_uint.push(crop(&resized, start_y, start_x, tile, tile));
                }
            }
            // global tile: resized image downscaled to tile x tile
            tiles_uint.push(resize(&resized, tile, tile, config.resample)?);
            has_global_tile = true;
        } else {
            // image is exactly tile-sized after rounding: single frame, no
            // grid (reference reports splits (0, 0) in this case)
            tiles_uint.push(resized.clone());
        }
    } else {
        tiles_uint.push(resized.clone());
    }
    timings.tile_ms = t_tile.elapsed().as_secs_f64() * 1e3;
    let t_norm = std::time::Instant::now();

    // 3. rescale + normalize + pack into [n, 3, tile, tile]
    let n_tiles = tiles_uint.len();
    let tile = tiles_uint[0].shape()[1];
    let mut pixels = vec![0.0f32; n_tiles * 3 * tile * tile];
    for (n, t) in tiles_uint.iter().enumerate() {
        let (th, tw) = (t.shape()[1], t.shape()[2]);
        anyhow::ensure!(th == tile && tw == tile, "tile must be square");
        for c in 0..3 {
            for y in 0..tile {
                for x in 0..tile {
                    let raw = t.data()[c * th * tw + y * tw + x];
                    let out_idx = n * 3 * tile * tile + c * tile * tile + y * tile + x;
                    pixels[out_idx] =
                        (raw * config.rescale_factor - config.mean[c]) / config.std[c];
                }
            }
        }
    }
    let tiles = CpuTensor::from_data(vec![n_tiles, 3, tile, tile], pixels);
    let mask = CpuTensor::from_data(
        vec![n_tiles, tile, tile],
        vec![1.0f32; n_tiles * tile * tile],
    );
    timings.normalize_ms = t_norm.elapsed().as_secs_f64() * 1e3;

    Ok(PreprocessedImage {
        tiles,
        mask,
        original_dims,
        resized_dims,
        tile_grid,
        has_global_tile,
        timings,
    })
}

/// Tile grid `(rows, cols)` an image of `(h, w)` produces under `config`
/// after the longest-edge resize and tile rounding — computed without any
/// pixel work. Used by the feature cache to reconstruct assembler metadata
/// for cache hits.
pub fn tile_grid_for(dims: (usize, usize), config: &ImagePreprocessConfig) -> (usize, usize) {
    let (mut h, mut w) = dims;
    if let Some(max_edge) = config.resize_longest_edge {
        let max_edge = max_edge as usize;
        if w >= h {
            w = max_edge;
            h = (w as f64 / aspect(dims)) as usize;
            if h % 2 != 0 {
                h += 1;
            }
        } else {
            h = max_edge;
            w = (h as f64 * aspect(dims)) as usize;
            if w % 2 != 0 {
                w += 1;
            }
        }
        h = h.max(1);
        w = w.max(1);
    }
    if let Some(tile_u) = config.tile_size {
        let tile = tile_u as usize;
        let aspect_ratio = w as f64 / h as f64;
        let rounded = if w >= h {
            let nw = (w as f64 / tile as f64).ceil() as usize * tile;
            let mut nh = (nw as f64 / aspect_ratio).trunc() as usize;
            nh = (nh as f64 / tile as f64).ceil() as usize * tile;
            (nh, nw)
        } else {
            let nh = (h as f64 / tile as f64).ceil() as usize * tile;
            let mut nw = (nh as f64 * aspect_ratio).trunc() as usize;
            nw = (nw as f64 / tile as f64).ceil() as usize * tile;
            (nh, nw)
        };
        h = rounded.0.max(tile);
        w = rounded.1.max(tile);
        if h > tile || w > tile {
            return (h / tile, w / tile);
        }
    }
    (0, 0)
}

fn aspect(dims: (usize, usize)) -> f64 {
    dims.1 as f64 / dims.0 as f64
}

/// Crop a square `size x size` region starting at `(y, x)` from a CHW image.
fn crop(image: &CpuTensor, y: usize, x: usize, h: usize, w: usize) -> CpuTensor {
    let (ch, cw) = (image.shape()[1], image.shape()[2]);
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for dy in 0..h {
            for dx in 0..w {
                out[c * h * w + dy * w + dx] = image.data()[c * ch * cw + (y + dy) * cw + (x + dx)];
            }
        }
    }
    CpuTensor::from_data(vec![3, h, w], out)
}

/// Resize a CHW uint8-valued f32 image to `(out_w, out_h)` using the given
/// filter. Bit-exact with Pillow for RGB images (see module docs).
pub fn resize(
    image: &CpuTensor,
    out_w: usize,
    out_h: usize,
    resample: Resample,
) -> Result<CpuTensor> {
    anyhow::ensure!(
        image.ndim() == 3 && image.shape()[0] == 3,
        "resize expects CHW RGB"
    );
    let (in_h, in_w) = (image.shape()[1], image.shape()[2]);
    let (filter, support) = match resample {
        Resample::Lanczos => (lanczos as fn(f64) -> f64, 3.0),
        Resample::Bicubic => (bicubic as fn(f64) -> f64, 2.0),
    };

    // --- horizontal pass: in_w -> out_w, rows stay in_h ---
    let (bounds_h, kk_h) = precompute_coeffs(in_w, out_w, filter, support);
    let mut intermediate = vec![0u8; out_w * in_h * 3];
    for y in 0..in_h {
        for xx in 0..out_w {
            let (xmin, xmax) = bounds_h[xx];
            for c in 0..3 {
                let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                let row = &image.data()[c * in_h * in_w + y * in_w..];
                for x in 0..xmax {
                    ss += (row[x + xmin] as i64) * kk_h[xx][x];
                }
                let v = clip8((ss >> PRECISION_BITS) as i32);
                intermediate[(y * out_w + xx) * 3 + c] = v;
            }
        }
    }

    // --- vertical pass: in_h -> out_h, width stays out_w ---
    let (bounds_v, kk_v) = precompute_coeffs(in_h, out_h, filter, support);
    let mut out = vec![0.0f32; 3 * out_h * out_w];
    for yy in 0..out_h {
        let (ymin, ymax) = bounds_v[yy];
        for xx in 0..out_w {
            for c in 0..3 {
                let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                for y in 0..ymax {
                    let src = &intermediate[((y + ymin) * out_w + xx) * 3 + c];
                    ss += (*src as i64) * kk_v[yy][y];
                }
                let v = clip8((ss >> PRECISION_BITS) as i32);
                out[c * out_h * out_w + yy * out_w + xx] = v as f32;
            }
        }
    }
    Ok(CpuTensor::from_data(vec![3, out_h, out_w], out))
}

const PRECISION_BITS: i64 = 32 - 8 - 2; // 22, matching Pillow

/// Pillow's clip8 lookup equivalent: clamp to [0, 255].
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Pillow's truncated sinc (3-lobe Lanczos).
fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow's bicubic convolution filter (a = -0.5), support 2.
fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    const A: f64 = -0.5;
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        // a*|x|^3 - 5a*|x|^2 + 8a*|x| - 4a
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

/// Pillow's `precompute_coeffs`: per-output-pixel input bounds and
/// normalized float coefficients (rounded to 22-bit fixed point).
fn precompute_coeffs(
    in_size: usize,
    out_size: usize,
    filter: fn(f64) -> f64,
    support: f64,
) -> (Vec<(usize, usize)>, Vec<Vec<i64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv_filterscale = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut coeffs = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5).floor() as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5).floor() as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        xmax -= xmin;
        let xmin = xmin as usize;
        let xmax = xmax as usize;

        let mut k = vec![0.0f64; ksize];
        let mut ww = 0.0f64;
        for (x, slot) in k.iter_mut().enumerate().take(xmax) {
            let w = filter((x as f64 + xmin as f64 - center + 0.5) * inv_filterscale);
            *slot = w;
            ww += w;
        }
        if ww != 0.0 {
            for slot in k.iter_mut().take(xmax) {
                *slot /= ww;
            }
        }
        let mut fixed = vec![0i64; ksize];
        for x in 0..ksize {
            fixed[x] = if k[x] < 0.0 {
                (-0.5 + k[x] * (1i64 << PRECISION_BITS) as f64) as i64
            } else {
                (0.5 + k[x] * (1i64 << PRECISION_BITS) as f64) as i64
            };
        }
        bounds.push((xmin, xmax));
        coeffs.push(fixed);
    }
    (bounds, coeffs)
}

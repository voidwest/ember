//! The general multimodal request substrate.
//!
//! A request is an *ordered* list of [`ContentPart`]s: text, images, audio
//! segments, and video may be interleaved arbitrarily and repeated. This
//! module only *represents* requests; whether a combination is meaningful is
//! decided by each model adapter (the wrappers in `smolvlm`/`ultravox`),
//! which fail closed on unsupported combinations. No model-specific prompt
//! syntax lives here.
//!
//! Media inputs are not required to come from the filesystem: every modality
//! can be constructed from memory (`Bytes`, decoded `Pixels`/`Frames`,
//! PCM `Samples`). The CLI is one frontend over the same API a server, GUI,
//! microphone, or agent tool would use.

use crate::tensor::CpuTensor;
use std::path::PathBuf;

/// One part of an ordered multimodal request.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    /// Free-form text. Modality placeholders inside the text (e.g. `<image>`,
    /// `<|audio|>`) are interpreted by the model-specific assembler.
    Text(String),
    Image(ImageInput),
    Audio(crate::multimodal::audio::AudioInput),
    Video(VideoInput),
}

/// A raw image input: file path, encoded bytes (PNG/JPEG), or already-decoded
/// pixels. Decoded pixels are CHW `[3, height, width]` f32 in 0..255 — the
/// same layout [`crate::multimodal::image::decode_rgb`] produces — so callers
/// holding frames in memory never touch the filesystem.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageInput {
    File(PathBuf),
    Bytes(Vec<u8>),
    Pixels { rgb: CpuTensor },
}

impl ImageInput {
    /// Decode to RGB pixels (CHW `[3, h, w]`, 0..255). `Pixels` passes
    /// through without a copy.
    pub fn decode(&self) -> anyhow::Result<CpuTensor> {
        match self {
            ImageInput::File(path) => crate::multimodal::image::decode_rgb(path),
            ImageInput::Bytes(bytes) => crate::multimodal::image::decode_rgb_bytes(bytes),
            ImageInput::Pixels { rgb } => {
                anyhow::ensure!(
                    rgb.ndim() == 3 && rgb.shape()[0] == 3,
                    "ImageInput::Pixels expects CHW [3, h, w], got {:?}",
                    rgb.shape()
                );
                Ok(rgb.clone())
            }
        }
    }
}

/// Maximum number of image parts in one multimodal request (admission cap;
/// each image is additionally bounded by the decode limits in
/// [`crate::multimodal::image`]).
pub const MAX_IMAGES_PER_REQUEST: usize = 16;

/// Maximum number of video frames in one multimodal request (admission cap).
pub const MAX_VIDEO_FRAMES: usize = 1024;

/// Decoded image format, for provenance/logging on validated inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedImageFormat {
    Png,
    Jpeg,
    /// Already-decoded pixels supplied by the caller.
    RawPixels,
    Unknown,
}

/// A decoded, validated image: CHW `[3, h, w]` f32 pixels (0..255) plus
/// the geometry downstream stages may assume without re-validating.
///
/// This is the Phase-2 validated-state seam: raw [`ImageInput`] is decoded
/// exactly once here, under the image-crate limits, and everything after
/// (preprocess, batch encode, assembler) consumes [`ValidatedImageInput`].
#[derive(Debug, Clone)]
pub struct ValidatedImageInput {
    /// CHW `[3, height, width]`, values 0..255.
    pub rgb: CpuTensor,
    pub width: usize,
    pub height: usize,
    pub format: ValidatedImageFormat,
}

impl ValidatedImageInput {
    /// Decode (or pass through already-decoded pixels) with the image
    /// decoder limits applied. Rejects malformed `Pixels` shapes.
    pub fn decode(input: &ImageInput) -> anyhow::Result<Self> {
        match input {
            ImageInput::File(path) => {
                let format = image::ImageFormat::from_path(path)
                    .ok()
                    .map(validated_format)
                    .unwrap_or(ValidatedImageFormat::Unknown);
                let rgb = crate::multimodal::image::decode_rgb(path)?;
                Ok(Self::from_rgb(rgb, format))
            }
            ImageInput::Bytes(bytes) => {
                let format = image::guess_format(bytes)
                    .ok()
                    .map(validated_format)
                    .unwrap_or(ValidatedImageFormat::Unknown);
                let rgb = crate::multimodal::image::decode_rgb_bytes(bytes)?;
                Ok(Self::from_rgb(rgb, format))
            }
            ImageInput::Pixels { rgb } => {
                anyhow::ensure!(
                    rgb.ndim() == 3 && rgb.shape()[0] == 3,
                    "ImageInput::Pixels expects CHW [3, h, w], got {:?}",
                    rgb.shape()
                );
                Ok(Self::from_rgb(rgb.clone(), ValidatedImageFormat::RawPixels))
            }
        }
    }

    fn from_rgb(rgb: CpuTensor, format: ValidatedImageFormat) -> Self {
        let height = rgb.shape()[1];
        let width = rgb.shape()[2];
        Self {
            rgb,
            width,
            height,
            format,
        }
    }
}

fn validated_format(f: image::ImageFormat) -> ValidatedImageFormat {
    match f {
        image::ImageFormat::Png => ValidatedImageFormat::Png,
        image::ImageFormat::Jpeg => ValidatedImageFormat::Jpeg,
        _ => ValidatedImageFormat::Unknown,
    }
}

/// A raw video input: a sequence of already-decoded frames plus timing
/// metadata. Frame data uses the image layout (CHW `[3, h, w]` f32 0..255).
///
/// Encoded-video decoding (containers/codecs) is deliberately outside core
/// Ember: decode externally (ffmpeg, a browser, a camera pipeline) and pass
/// frames here. This keeps heavy codec dependencies out of the runtime while
/// giving servers/cameras/agents a zero-copy-friendly boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoInput {
    Frames(VideoFrames),
}

/// Decoded video frames with the timing metadata temporal position semantics
/// need. `timestamps_ms[i]` is the presentation timestamp of `frames[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrames {
    /// One frame each, CHW `[3, height, width]`, values 0..255.
    pub frames: Vec<CpuTensor>,
    /// Presentation timestamps in milliseconds, parallel to `frames`.
    pub timestamps_ms: Vec<f64>,
    /// Nominal source frame rate, when known (metadata only; sampling does
    /// not require it).
    pub source_fps: Option<f64>,
    /// Total source duration in seconds, when known.
    pub source_duration_s: Option<f64>,
}

impl VideoFrames {
    pub fn width(&self) -> Option<usize> {
        self.frames.first().map(|f| f.shape()[2])
    }

    pub fn height(&self) -> Option<usize> {
        self.frames.first().map(|f| f.shape()[1])
    }
}

impl ContentPart {
    /// The media kind of this part (`None` for text).
    pub fn media_kind(&self) -> Option<MediaKind> {
        match self {
            ContentPart::Text(_) => None,
            ContentPart::Image(_) => Some(MediaKind::Image),
            ContentPart::Audio(_) => Some(MediaKind::Audio),
            ContentPart::Video(_) => Some(MediaKind::Video),
        }
    }
}

/// Which modality a piece of media belongs to. Used by cache keys and batch
/// ownership metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
}

/// Stable identity for a piece of media, derived from its content bytes.
///
/// Cache keys combine this with processor/encoder configuration so features
/// are never reused across incompatible settings. Two inputs with identical
/// bytes share identity; a changed file gets a new id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MediaId(pub u64);

impl MediaId {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // length first so {b"ab", b"c"}-style ambiguities cannot collide
        bytes.len().hash(&mut h);
        h.write(bytes);
        MediaId(h.finish())
    }

    /// Identity of already-decoded sample/pixel data: shape and every f32
    /// value participate in the hash.
    pub fn from_tensor(t: &CpuTensor) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        t.shape().hash(&mut h);
        for v in t.data() {
            v.to_bits().hash(&mut h);
        }
        MediaId(h.finish())
    }

    /// Convenience for hashing several tensors in sequence (e.g. video
    /// frames): order-sensitive.
    pub fn from_tensors(tensors: &[CpuTensor]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        tensors.len().hash(&mut h);
        for t in tensors {
            t.shape().hash(&mut h);
            for v in t.data() {
                v.to_bits().hash(&mut h);
            }
        }
        MediaId(h.finish())
    }
}

/// Ownership metadata for batched work: which request and which ordered part
/// a unit of media work came from, so batched encoder outputs can be split
/// back to their owners and mismatches fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId {
    /// Caller-assigned request identity (unique within one scheduling scope).
    pub request: u64,
    /// Index of the part within that request's content parts.
    pub part: usize,
}

impl SegmentId {
    pub fn new(request: u64, part: usize) -> Self {
        Self { request, part }
    }
}

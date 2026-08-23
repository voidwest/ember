//! Video input handling: frame sampling policies and the frame geometry
//! metadata temporal position semantics need.
//!
//! Ember does not decode containers/codecs in core: callers (ffmpeg,
//! browsers, cameras, agents) hand over decoded frames as
//! [`crate::multimodal::request::VideoFrames`] and this module selects
//! which frames enter the vision tower. Every policy is explicit and
//! deterministic, and every sampled result carries enough provenance
//! (`timestamps_ms`, source fps/duration, selected count) to debug
//! downstream numerical comparisons — a sampling mismatch invalidates any
//! tensor comparison above it.

use crate::multimodal::request::VideoFrames;
use crate::tensor::CpuTensor;
use anyhow::{ensure, Result};

/// How frames are selected from a decoded video before encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameSampling {
    /// Uniformly spaced frame indices over the whole clip (inclusive of the
    /// first frame), capped at `max_frames`. Matches the reference
    /// processors' uniform sampler: indices are
    /// `floor(i * total / k)` for `i in 0..k`, `k = min(max_frames, total)`.
    Uniform { max_frames: usize },
    /// One frame every `1/fps` seconds of *source* time, capped at
    /// `max_frames`. Requires timestamps; selects the last frame whose
    /// timestamp falls in each window.
    FixedFps { fps: f64, max_frames: usize },
}

/// The result of sampling: selected frames plus full provenance.
#[derive(Debug, Clone)]
pub struct SampledVideo {
    /// Selected frames, CHW `[3, h, w]`, 0..255.
    pub frames: Vec<CpuTensor>,
    /// Presentation timestamp of each selected frame (ms).
    pub timestamps_ms: Vec<f64>,
    /// Index of each selected frame in the source sequence.
    pub source_indices: Vec<usize>,
    /// Total frames available before sampling.
    pub total_source_frames: usize,
    /// Nominal source fps when known.
    pub source_fps: Option<f64>,
    /// Source duration in seconds when known.
    pub source_duration_s: Option<f64>,
}

impl SampledVideo {
    /// Context-cost visibility: how many frames will become visual tokens.
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
}

impl FrameSampling {
    /// Apply this policy to decoded frames. Deterministic; fails closed on
    /// empty input or inconsistent metadata.
    pub fn sample(&self, input: &VideoFrames) -> Result<SampledVideo> {
        ensure!(!input.frames.is_empty(), "video input has no frames");
        ensure!(
            input.timestamps_ms.len() == input.frames.len(),
            "video frames/timestamps length mismatch: {} vs {}",
            input.timestamps_ms.len(),
            input.frames.len()
        );
        let total = input.frames.len();
        let indices: Vec<usize> = match *self {
            FrameSampling::Uniform { max_frames } => {
                let k = max_frames.min(total);
                if k == 0 {
                    Vec::new()
                } else if total <= 1 || k == 1 {
                    // single representative frame: the first, matching the
                    // floor-uniform formula at i=0
                    vec![0]
                } else {
                    // floor(i * total / k): reference uniform sampler over
                    // pre-decoded frames (integer index space)
                    (0..k).map(|i| i * total / k).collect()
                }
            }
            FrameSampling::FixedFps { fps, max_frames } => {
                ensure!(fps > 0.0, "FixedFps requires fps > 0");
                let step_ms = 1000.0 / fps;
                let mut out = Vec::new();
                let mut window = 0.0f64;
                for (i, &ts) in input.timestamps_ms.iter().enumerate() {
                    if out.len() >= max_frames {
                        break;
                    }
                    // last frame within [window, window + step)
                    if ts >= window && ts < window + step_ms {
                        let mut last_in_window = i;
                        while last_in_window + 1 < total
                            && input.timestamps_ms[last_in_window + 1] < window + step_ms
                        {
                            last_in_window += 1;
                            if out.len() + 1 > max_frames {
                                break;
                            }
                        }
                        out.push(last_in_window);
                        window += step_ms;
                    } else if ts >= window + step_ms {
                        // sparse timestamps: advance windows until covered
                        while ts >= window + step_ms && out.len() < max_frames {
                            window += step_ms;
                        }
                        out.push(i);
                        window += step_ms;
                    }
                }
                out
            }
        };
        Ok(SampledVideo {
            frames: indices.iter().map(|&i| input.frames[i].clone()).collect(),
            timestamps_ms: indices.iter().map(|&i| input.timestamps_ms[i]).collect(),
            source_indices: indices,
            total_source_frames: total,
            source_fps: input.source_fps,
            source_duration_s: input.source_duration_s,
        })
    }
}

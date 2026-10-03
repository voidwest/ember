//! Placeholders in the shape of what is loading.
//!
//! Adapted from Ely GPUI Components, `src/motion/skeleton.rs`
//! (MIT OR Apache-2.0; see `third_party/`). Changes: `Skeleton` and
//! `SkeletonText` only, as functions; theme lookups replaced by an explicit
//! colour and line height.

use gpui_kit::prelude::*;
use gpui_kit::*;
use std::{f32::consts::TAU, time::Duration};

/// One breath of a skeleton, in and out.
const BREATH: Duration = Duration::from_millis(1_600);

/// A block that breathes until the content arrives. Size it as the content.
pub(super) fn skeleton(id: impl Into<ElementId>, fill: Hsla) -> AnyElement {
    let block = div().size_full().rounded(px(4.0)).bg(fill);
    // The render harness photographs single frames; a breathing block would
    // land at an arbitrary opacity, so it holds still there.
    if cfg!(feature = "gui-tests") {
        return block.into_any_element();
    }
    block
        .with_animation(id.into(), Animation::new(BREATH).repeat(), |block, t| {
            block.opacity(0.775 + 0.225 * (TAU * t).cos())
        })
        .into_any_element()
}

/// Lines of text to come; the last runs short.
pub(super) fn skeleton_text(id: &str, lines: usize, line: f32, fill: Hsla) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .children((0..lines).map(|ix| {
            let width = if ix + 1 == lines && lines > 1 {
                0.6
            } else {
                1.0
            };
            div()
                .h(px(line))
                .w(relative(width))
                .child(div().size_full().child(skeleton(
                    SharedString::from(format!("{id}-line-{ix}")),
                    fill,
                )))
        }))
}

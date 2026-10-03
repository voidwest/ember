//! A word-sized chart of a series.
//!
//! Adapted from Ely GPUI Components, `src/data_display/spark.rs`
//! (MIT OR Apache-2.0; see `third_party/`). Changes: the theme lookups are
//! replaced by an explicit colour and size, and the bar variant is dropped.

use gpui_kit::prelude::*;
use gpui_kit::*;

/// Each value's height as a share of the series' own range; a flat series
/// sits in the middle.
fn shares(values: &[f32]) -> Vec<f32> {
    let (low, high) = values.iter().fold((f32::MAX, f32::MIN), |(low, high), v| {
        (low.min(*v), high.max(*v))
    });
    let span = high - low;
    values
        .iter()
        .map(|value| {
            if span > 0.0 {
                (value - low) / span
            } else {
                0.5
            }
        })
        .collect()
}

/// A sparkline: the line, with the area under it lightly filled. Returns
/// `None` for a series it cannot draw (fewer than two values, or a value that
/// is not finite) so a caller shows nothing rather than a broken mark.
pub(super) fn sparkline(values: &[f32], ink: Hsla, width: f32, height: f32) -> Option<Div> {
    if values.len() < 2 || !values.iter().all(|value| value.is_finite()) {
        return None;
    }
    let heights = shares(values);
    Some(
        div().flex_none().w(px(width)).h(px(height)).child(
            canvas(
                |_, _, _| {},
                move |bounds: Bounds<Pixels>, _, window, _| {
                    let stroke = (bounds.size.height / 12.0).max(px(1.0));
                    let inner = bounds.size.height - stroke * 2.0;
                    let count = heights.len() as f32;
                    let at = |ix: usize, share: f32| {
                        point(
                            bounds.left() + bounds.size.width * ix as f32 / (count - 1.0),
                            bounds.top() + stroke + inner * (1.0 - share),
                        )
                    };
                    let mut area = PathBuilder::fill();
                    area.move_to(point(bounds.left(), bounds.bottom()));
                    for (ix, share) in heights.iter().enumerate() {
                        area.line_to(at(ix, *share));
                    }
                    area.line_to(point(bounds.right(), bounds.bottom()));
                    area.close();
                    if let Ok(area) = area.build() {
                        window.paint_path(area, ink.opacity(0.16));
                    }
                    let mut line = PathBuilder::stroke(stroke);
                    for (ix, share) in heights.iter().enumerate() {
                        if ix == 0 {
                            line.move_to(at(ix, *share));
                        } else {
                            line.line_to(at(ix, *share));
                        }
                    }
                    if let Ok(line) = line.build() {
                        window.paint_path(line, ink);
                    }
                },
            )
            .size_full(),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::shares;

    #[test]
    fn values_scale_to_their_own_range() {
        assert_eq!(shares(&[2.0, 4.0, 3.0]), [0.0, 1.0, 0.5]);
        assert_eq!(
            shares(&[7.0, 7.0]),
            [0.5, 0.5],
            "a flat series sits in the middle"
        );
    }
}

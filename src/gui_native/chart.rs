//! Lightweight native research plots for the GPUI workbench.
//!
//! Metrics are reduced on the experiment worker. This module only maps the
//! cached sparse layer series into paint paths and small hit-tested points.

use super::components::{label, mono};
use super::theme::Colors;
use super::Console;
use crate::gui::LayerMetric;
use gpui_kit::prelude::*;
use gpui_kit::*;
use std::sync::Arc;

struct ChartPaint {
    grid: Vec<Path<Pixels>>,
    series: Option<Path<Pixels>>,
    reference: Option<Path<Pixels>>,
    /// A hairline at the hovered layer, so the readout has a place on the axis.
    crosshair: Option<Path<Pixels>>,
    intervention: Option<Path<Pixels>>,
    points: Vec<(usize, Point<Pixels>)>,
}

fn nice_max(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        return 0.001;
    }
    let magnitude = 10.0f64.powf(value.log10().floor());
    (value / magnitude).ceil() * magnitude
}

fn metric_label(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if value.abs() < 0.001 {
        format!("{value:.2e}")
    } else if value.abs() < 0.1 {
        format!("{value:.4}")
    } else {
        format!("{value:.3}")
    }
}

fn readout_metric_label(value: f64) -> String {
    if value != 0.0 && value.abs() < 0.0001 {
        format!("{value:.4e}")
    } else {
        format!("{value:.6}")
    }
}

/// The layers the chart marks: where the change was applied, and the layer
/// the reader selected or is hovering.
#[derive(Clone, Copy, Default)]
pub(super) struct LayerMarks {
    pub intervention: Option<usize>,
    pub selected: Option<usize>,
    pub hovered: Option<usize>,
}

pub(super) fn layer_divergence_chart(
    entity: Entity<Console>,
    metrics: Arc<[LayerMetric]>,
    // A pinned earlier result, drawn as a quiet second line so a change can be
    // judged against something.
    reference: Option<Arc<[LayerMetric]>>,
    marks: LayerMarks,
    height: f32,
    colors: &Colors,
) -> Div {
    layer_divergence_chart_with(
        entity,
        metrics,
        reference.map(|series| (series, SharedString::from("pinned reference"))),
        marks,
        height,
        colors,
    )
}

/// The chart with a named second series: the dashed line's legend says what
/// it is (a pinned reference, or the other run of a comparison).
pub(super) fn layer_divergence_chart_with(
    entity: Entity<Console>,
    metrics: Arc<[LayerMetric]>,
    reference: Option<(Arc<[LayerMetric]>, SharedString)>,
    marks: LayerMarks,
    height: f32,
    colors: &Colors,
) -> Div {
    let (reference, reference_label) = match reference {
        Some((series, name)) => (Some(series), Some(name)),
        None => (None, None),
    };
    let LayerMarks {
        intervention: intervention_layer,
        selected: selected_layer,
        hovered: hovered_layer,
    } = marks;
    let min_layer = metrics.first().map_or(0, |metric| metric.layer);
    let max_layer = metrics.last().map_or(min_layer, |metric| metric.layer);
    let y_max = nice_max(
        metrics
            .iter()
            .chain(reference.iter().flat_map(|reference| reference.iter()))
            .filter_map(|metric| metric.relative_l2_difference)
            .fold(0.0f64, f64::max),
    );
    let has_reference = reference.is_some();
    let reference_for_geometry = reference.clone();
    let active = hovered_layer
        .or(selected_layer)
        .and_then(|layer| metrics.iter().find(|metric| metric.layer == layer));
    let readout = active.map_or_else(
        || "Hover a layer for exact values \u{00b7} click to pin".to_string(),
        |metric| match (metric.relative_l2_difference, metric.cosine_distance) {
            (Some(relative_l2), Some(cosine_distance)) => format!(
                "Layer {} \u{00b7} rel L2 {} \u{00b7} cos dist {}",
                metric.layer,
                readout_metric_label(relative_l2),
                readout_metric_label(cosine_distance)
            ),
            (Some(relative_l2), None) => format!(
                "Layer {} \u{00b7} rel L2 {} \u{00b7} cos dist \u{2014}",
                metric.layer,
                readout_metric_label(relative_l2)
            ),
            _ => format!("Layer {} \u{00b7} no finite value", metric.layer),
        },
    );

    if metrics.is_empty() {
        // A quiet strip, not a chart-sized void: there is nothing to plot.
        return div()
            .h(px(88.0))
            .w_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(colors.surface_raised)
            .border_1()
            .border_color(colors.border)
            .rounded_md()
            .child(label(
                "No comparable layer captures were retained for this run.",
                13.5,
                colors.text_faint,
            ));
    }

    let grid_color = Hsla::from(colors.border).opacity(0.72);
    let line_color = colors.accent;
    // The intervention layer is an Ember-orange fact, not a caution: the
    // marker names the layer the user chose, it does not warn about it.
    let marker_color = Hsla::from(colors.accent).opacity(0.82);
    let point_color = colors.accent;
    let reference_color = Hsla::from(colors.text_muted).opacity(0.85);
    let crosshair_color = Hsla::from(colors.text_faint).opacity(0.6);
    let selected_color = colors.text;
    let metrics_for_geometry = metrics.clone();
    let metrics_for_mouse = metrics.clone();
    let chart_entity = entity.clone();
    let chart = canvas(
        move |bounds, _window, _cx| {
            let left = bounds.origin.x + px(6.0);
            let right = bounds.origin.x + bounds.size.width - px(6.0);
            let top = bounds.origin.y + px(8.0);
            let bottom = bounds.origin.y + bounds.size.height - px(8.0);
            let width = (right - left).max(px(1.0));
            let plot_height = (bottom - top).max(px(1.0));
            let layer_span = max_layer.saturating_sub(min_layer).max(1) as f32;
            let x_for = |layer: usize| {
                left + width * ((layer.saturating_sub(min_layer) as f32) / layer_span)
            };
            let y_for = |value: f64| bottom - plot_height * (value / y_max).clamp(0.0, 1.0) as f32;

            let mut grid = Vec::new();
            for step in 0..=4 {
                let y = top + plot_height * (step as f32 / 4.0);
                let mut builder = PathBuilder::stroke(px(1.0));
                builder.move_to(point(left, y));
                builder.line_to(point(right, y));
                if let Ok(path) = builder.build() {
                    grid.push(path);
                }
            }

            let points: Vec<(usize, Point<Pixels>)> = metrics_for_geometry
                .iter()
                .filter_map(|metric| {
                    metric
                        .relative_l2_difference
                        .map(|value| (metric.layer, point(x_for(metric.layer), y_for(value))))
                })
                .collect();
            let series = (points.len() >= 2)
                .then(|| {
                    let mut builder = PathBuilder::stroke(px(2.25));
                    for (index, (_, point)) in points.iter().enumerate() {
                        if index == 0 {
                            builder.move_to(*point);
                        } else {
                            builder.line_to(*point);
                        }
                    }
                    builder.build().ok()
                })
                .flatten();
            let intervention = intervention_layer
                .filter(|layer| *layer >= min_layer && *layer <= max_layer)
                .and_then(|layer| {
                    let x = x_for(layer);
                    let mut builder = PathBuilder::stroke(px(1.5)).dash_array(&[px(4.0), px(3.0)]);
                    builder.move_to(point(x, top));
                    builder.line_to(point(x, bottom));
                    builder.build().ok()
                });

            // The crosshair: a solid hairline at the layer under the pointer,
            // after the line charts in Ely GPUI Components (charts/pointer.rs).
            let crosshair = hovered_layer
                .filter(|layer| *layer >= min_layer && *layer <= max_layer)
                .and_then(|layer| {
                    let x = x_for(layer);
                    let mut builder = PathBuilder::stroke(px(1.0));
                    builder.move_to(point(x, top));
                    builder.line_to(point(x, bottom));
                    builder.build().ok()
                });
            let reference = reference_for_geometry.as_ref().and_then(|reference| {
                let mut builder = PathBuilder::stroke(px(1.5)).dash_array(&[px(2.0), px(3.0)]);
                let mut count = 0;
                for metric in reference.iter() {
                    if let Some(value) = metric.relative_l2_difference {
                        let point = point(x_for(metric.layer), y_for(value));
                        if count == 0 {
                            builder.move_to(point);
                        } else {
                            builder.line_to(point);
                        }
                        count += 1;
                    }
                }
                (count >= 2).then(|| builder.build().ok()).flatten()
            });
            ChartPaint {
                grid,
                series,
                reference,
                crosshair,
                intervention,
                points,
            }
        },
        move |bounds, paint, window, _cx| {
            // GPUI registers mouse listeners during paint, after layout is final.
            let left = bounds.origin.x + px(6.0);
            let width = (bounds.size.width - px(12.0)).max(px(1.0));
            let layer_span = max_layer.saturating_sub(min_layer).max(1) as f32;
            let event_bounds = bounds;
            let mouse_metrics = metrics_for_mouse.clone();
            let move_entity = chart_entity.clone();
            window.on_mouse_event(move |event: &MouseMoveEvent, _, _, cx| {
                let hovered = if event_bounds.contains(&event.position) {
                    let relative_x = ((event.position.x - left) / width).clamp(0.0, 1.0);
                    let target = min_layer as f32 + relative_x * layer_span;
                    mouse_metrics
                        .iter()
                        .filter(|metric| metric.relative_l2_difference.is_some())
                        .min_by(|left, right| {
                            (left.layer as f32 - target)
                                .abs()
                                .total_cmp(&(right.layer as f32 - target).abs())
                        })
                        .map(|metric| metric.layer)
                } else {
                    None
                };
                move_entity.update(cx, |console, cx| {
                    if console.hovered_layer != hovered {
                        console.hovered_layer = hovered;
                        cx.notify();
                    }
                });
            });
            let click_entity = chart_entity.clone();
            window.on_mouse_event(move |event: &MouseDownEvent, _, _, cx| {
                if event.button != MouseButton::Left || !event_bounds.contains(&event.position) {
                    return;
                }
                click_entity.update(cx, |console, cx| {
                    console.selected_layer = console.hovered_layer;
                    cx.notify();
                });
            });

            for path in paint.grid {
                window.paint_path(path, grid_color);
            }
            if let Some(path) = paint.intervention {
                window.paint_path(path, marker_color);
            }
            if let Some(path) = paint.crosshair {
                window.paint_path(path, crosshair_color);
            }
            if let Some(path) = paint.reference {
                window.paint_path(path, reference_color);
            }
            if let Some(path) = paint.series {
                window.paint_path(path, line_color);
            }
            for (layer, center) in paint.points {
                let is_selected = selected_layer == Some(layer) || hovered_layer == Some(layer);
                let radius = if is_selected { 5.5 } else { 3.25 };
                window.paint_quad(quad(
                    Bounds::new(
                        point(center.x - px(radius), center.y - px(radius)),
                        size(px(radius * 2.0), px(radius * 2.0)),
                    ),
                    px(radius),
                    if is_selected {
                        selected_color
                    } else {
                        point_color
                    },
                    px(0.0),
                    transparent_black(),
                    Default::default(),
                ));
            }
        },
    )
    .h(px(height))
    .w_full();

    let y_ticks = (0..=4)
        .map(|step| metric_label(y_max * (4 - step) as f64 / 4.0))
        .collect::<Vec<_>>();
    let mut y_tick_elements = Vec::with_capacity(9);
    for (index, value) in y_ticks.into_iter().enumerate() {
        y_tick_elements.push(mono(value, 12.5, colors.text_faint));
        if index < 4 {
            y_tick_elements.push(div().flex_1());
        }
    }

    div()
        .w_full()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .flex()
                .items_center()
                .px_2()
                .py_1()
                .bg(colors.surface_raised)
                .border_1()
                .border_color(if active.is_some() {
                    colors.border_strong
                } else {
                    colors.border
                })
                .rounded_md()
                .child(mono(
                    readout,
                    13.5,
                    if active.is_some() {
                        colors.text
                    } else {
                        colors.text_muted
                    },
                ))
                .child(div().w_full())
                .children(has_reference.then(|| {
                    mono(
                        format!("dashed: {}   ", reference_label.clone().unwrap_or_default()),
                        13.0,
                        colors.text_muted,
                    )
                }))
                .children(intervention_layer.map(|layer| {
                    mono(
                        format!("Intervention \u{00b7} L{layer}"),
                        13.0,
                        colors.accent,
                    )
                })),
        )
        .child(
            div()
                .flex()
                .gap_2()
                .w_full()
                .child(
                    div()
                        .h(px(height))
                        .w(px(56.0))
                        .flex_none()
                        .flex()
                        .flex_col()
                        .items_end()
                        .children(y_tick_elements),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.0))
                        .bg(colors.surface_raised)
                        .border_1()
                        .border_color(colors.border)
                        .rounded_md()
                        .overflow_hidden()
                        .child(chart),
                ),
        )
        .child(
            div()
                .flex()
                .items_center()
                .child(mono(format!("L{min_layer}"), 12.5, colors.text_faint))
                .child(div().w_full())
                .child(label("Transformer layer", 12.5, colors.text_faint))
                .child(div().w_full())
                .child(mono(format!("L{max_layer}"), 12.5, colors.text_faint)),
        )
        .child(
            div()
                .flex()
                .items_center()
                .child(label("Y: relative L2 difference", 12.5, colors.text_faint))
                .child(div().w_full())
                .child(mono(
                    format!("range 0 – {}", metric_label(y_max)),
                    12.5,
                    colors.text_faint,
                )),
        )
}

// ---------------------------------------------------------------------------
// SVG export
//
// `hex`, `escaped` and the page layout (background, gridlines with labels in a
// left gutter, marks, axis labels under the frame) are adapted from Ely GPUI
// Components, `src/charts/export.rs` (MIT OR Apache-2.0; see `third_party/`).
// ---------------------------------------------------------------------------

/// A colour as SVG writes it, and its opacity apart.
fn hex(color: Hsla) -> (String, f32) {
    let rgba: Rgba = color.into();
    let byte = |channel: f32| (channel.clamp(0.0, 1.0) * 255.0).round() as u8;
    (
        format!(
            "#{:02x}{:02x}{:02x}",
            byte(rgba.r),
            byte(rgba.g),
            byte(rgba.b)
        ),
        rgba.a,
    )
}

/// Text safe inside SVG.
fn escaped(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// What the exported chart shows, beyond the series itself.
pub(super) struct SvgLabels<'a> {
    pub title: &'a str,
    pub subtitle: &'a str,
    pub x_axis: &'a str,
    pub y_axis: &'a str,
}

/// The layer chart as a standalone SVG drawing, for slides and papers: the
/// series, an optional dashed second series, the intervention marker,
/// gridlines and labelled axes, in the console's current colours.
pub(super) fn layers_svg(
    metrics: &[LayerMetric],
    reference: Option<&[LayerMetric]>,
    intervention: Option<usize>,
    labels: SvgLabels<'_>,
    colors: &Colors,
) -> String {
    use std::fmt::Write;
    let (width, height) = (960.0f32, 520.0f32);
    let (left, right, top, bottom) = (72.0f32, 32.0f32, 92.0f32, 72.0f32);
    let (fx, fy, fw, fh) = (left, top, width - left - right, height - top - bottom);
    let min_layer = metrics.first().map_or(0, |metric| metric.layer);
    let max_layer = metrics.last().map_or(min_layer, |metric| metric.layer);
    let span = max_layer.saturating_sub(min_layer).max(1) as f32;
    let y_max = nice_max(
        metrics
            .iter()
            .chain(reference.into_iter().flatten())
            .filter_map(|metric| metric.relative_l2_difference)
            .fold(0.0f64, f64::max),
    );
    let x_for = |layer: usize| fx + fw * (layer.saturating_sub(min_layer) as f32 / span);
    let y_for = |value: f64| fy + fh - fh * (value / y_max).clamp(0.0, 1.0) as f32;
    let (back, _) = hex(colors.canvas.into());
    let (grid, _) = hex(colors.border_strong.into());
    let (text, _) = hex(colors.text.into());
    let (faint, _) = hex(colors.text_muted.into());
    let (accent, _) = hex(colors.accent.into());
    let (other, _) = hex(colors.text_muted.into());
    let mut out = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="-apple-system, 'SF Pro Text', 'Noto Sans', sans-serif" font-size="13">"#
    );
    let _ = write!(
        out,
        r#"<rect width="{width}" height="{height}" fill="{back}"/>"#
    );
    let _ = write!(
        out,
        r#"<text x="{fx}" y="36" font-size="20" font-weight="600" fill="{text}">{}</text>"#,
        escaped(labels.title)
    );
    let _ = write!(
        out,
        r#"<text x="{fx}" y="62" fill="{faint}">{}</text>"#,
        escaped(labels.subtitle)
    );
    for step in 0..=4 {
        let value = y_max * step as f64 / 4.0;
        let y = y_for(value);
        let _ = write!(
            out,
            r#"<line x1="{fx}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="{grid}" stroke-opacity="0.5" stroke-width="1"/>"#,
            fx + fw
        );
        let _ = write!(
            out,
            r#"<text x="{:.1}" y="{:.1}" text-anchor="end" font-family="ui-monospace, 'SF Mono', monospace" font-size="12" fill="{faint}">{}</text>"#,
            fx - 10.0,
            y + 4.0,
            escaped(&metric_label(value))
        );
    }
    let step = ((max_layer - min_layer) / 16).max(1);
    for layer in (min_layer..=max_layer).step_by(step) {
        let _ = write!(
            out,
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="ui-monospace, 'SF Mono', monospace" font-size="12" fill="{faint}">L{layer}</text>"#,
            x_for(layer),
            fy + fh + 22.0
        );
    }
    let _ = write!(
        out,
        r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" fill="{faint}">{}</text>"#,
        fx + fw / 2.0,
        height - 18.0,
        escaped(labels.x_axis)
    );
    let _ = write!(
        out,
        r#"<text transform="translate(20 {:.1}) rotate(-90)" text-anchor="middle" fill="{faint}">{}</text>"#,
        fy + fh / 2.0,
        escaped(labels.y_axis)
    );
    if let Some(layer) = intervention.filter(|layer| *layer >= min_layer && *layer <= max_layer) {
        let x = x_for(layer);
        let _ = write!(
            out,
            r#"<line x1="{x:.1}" y1="{fy}" x2="{x:.1}" y2="{:.1}" stroke="{accent}" stroke-width="1.5" stroke-dasharray="4 3"/>"#,
            fy + fh
        );
        let _ = write!(
            out,
            r#"<text x="{:.1}" y="{:.1}" fill="{accent}" font-size="12">intervention · L{layer}</text>"#,
            x + 6.0,
            fy + 14.0
        );
    }
    let polyline = |series: &[LayerMetric]| {
        series
            .iter()
            .filter_map(|metric| {
                metric
                    .relative_l2_difference
                    .map(|value| format!("{:.1},{:.1}", x_for(metric.layer), y_for(value)))
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    if let Some(reference) = reference {
        let _ = write!(
            out,
            r#"<polyline points="{}" fill="none" stroke="{other}" stroke-width="1.5" stroke-dasharray="2 3"/>"#,
            polyline(reference)
        );
    }
    let _ = write!(
        out,
        r#"<polyline points="{}" fill="none" stroke="{accent}" stroke-width="2.5" stroke-linejoin="round"/>"#,
        polyline(metrics)
    );
    for metric in metrics {
        if let Some(value) = metric.relative_l2_difference {
            let _ = write!(
                out,
                r#"<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{accent}"/>"#,
                x_for(metric.layer),
                y_for(value)
            );
        }
    }
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use super::{metric_label, nice_max};

    #[test]
    fn the_svg_export_is_a_complete_drawing() {
        let metrics: Vec<crate::gui::LayerMetric> = (0..4)
            .map(|layer| crate::gui::LayerMetric {
                layer,
                relative_l2_difference: Some(layer as f64 * 0.1),
                cosine_distance: None,
                maximum_absolute_difference: None,
                exact: false,
            })
            .collect();
        let svg = super::layers_svg(
            &metrics,
            None,
            Some(2),
            super::SvgLabels {
                title: "A & B",
                subtitle: "<sub>",
                x_axis: "Transformer layer",
                y_axis: "relative L2",
            },
            &crate::gui_native::theme::dark(),
        );
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        if let Some(path) = std::env::var_os("EMBER_SVG_DUMP") {
            std::fs::write(path, &svg).unwrap();
        }
        assert!(
            svg.contains("A &amp; B") && svg.contains("&lt;sub&gt;"),
            "labels are escaped"
        );
        assert!(svg.contains("intervention · L2"));
        assert_eq!(svg.matches("<circle").count(), 4);
    }

    #[test]
    fn chart_range_handles_zero_and_non_finite_series() {
        assert_eq!(nice_max(0.0), 0.001);
        assert_eq!(nice_max(f64::NAN), 0.001);
        assert_eq!(nice_max(0.018), 0.02);
        assert_eq!(nice_max(1.2), 2.0);
    }

    #[test]
    fn chart_readouts_keep_small_values_visible() {
        assert_eq!(metric_label(0.0), "0");
        assert!(metric_label(0.000_012).contains('e'));
        assert_eq!(metric_label(0.183), "0.183");
    }
}

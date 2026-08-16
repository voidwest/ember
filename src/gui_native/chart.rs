//! Lightweight native research plots for the GPUI workbench.
//!
//! Metrics are reduced on the experiment worker. This module only maps the
//! cached sparse layer series into paint paths and small hit-tested points.

use super::components::{label, mono};
use super::theme::Colors;
use super::Console;
use crate::gui::LayerMetric;
use gpui::prelude::*;
use gpui::*;
use std::sync::Arc;

struct ChartPaint {
    grid: Vec<Path<Pixels>>,
    series: Option<Path<Pixels>>,
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

pub(super) fn layer_divergence_chart(
    entity: Entity<Console>,
    metrics: Arc<[LayerMetric]>,
    intervention_layer: Option<usize>,
    selected_layer: Option<usize>,
    hovered_layer: Option<usize>,
    height: f32,
    colors: &Colors,
) -> Div {
    let min_layer = metrics.first().map_or(0, |metric| metric.layer);
    let max_layer = metrics.last().map_or(min_layer, |metric| metric.layer);
    let y_max = nice_max(
        metrics
            .iter()
            .filter_map(|metric| metric.relative_l2_difference)
            .fold(0.0f64, f64::max),
    );
    let active = hovered_layer
        .or(selected_layer)
        .and_then(|layer| metrics.iter().find(|metric| metric.layer == layer));
    let readout = active.map_or_else(
        || "HOVER A LAYER FOR EXACT VALUES  ·  CLICK TO PIN".to_string(),
        |metric| match (metric.relative_l2_difference, metric.cosine_distance) {
            (Some(relative_l2), Some(cosine_distance)) => format!(
                "LAYER {}  ·  REL L2 {}  ·  COS DIST {}",
                metric.layer,
                readout_metric_label(relative_l2),
                readout_metric_label(cosine_distance)
            ),
            (Some(relative_l2), None) => format!(
                "LAYER {}  ·  REL L2 {}  ·  COS DIST —",
                metric.layer,
                readout_metric_label(relative_l2)
            ),
            _ => format!("LAYER {}  ·  NO FINITE VALUE", metric.layer),
        },
    );

    if metrics.is_empty() {
        return div()
            .h(px(height))
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
                10.0,
                colors.text_faint,
            ));
    }

    let grid_color = Hsla::from(colors.border).opacity(0.72);
    let line_color = colors.accent;
    let marker_color = Hsla::from(colors.warn).opacity(0.82);
    let point_color = colors.accent;
    let selected_color = colors.text;
    let metrics_for_geometry = metrics.clone();
    let metrics_for_mouse = metrics.clone();
    let chart_entity = entity.clone();
    let chart = canvas(
        move |bounds, window, _cx| {
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
                    let mut builder = PathBuilder::stroke(px(1.75));
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
                    let mut builder = PathBuilder::stroke(px(1.0)).dash_array(&[px(4.0), px(3.0)]);
                    builder.move_to(point(x, top));
                    builder.line_to(point(x, bottom));
                    builder.build().ok()
                });

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

            ChartPaint {
                grid,
                series,
                intervention,
                points,
            }
        },
        move |_bounds, paint, window, _cx| {
            for path in paint.grid {
                window.paint_path(path, grid_color);
            }
            if let Some(path) = paint.intervention {
                window.paint_path(path, marker_color);
            }
            if let Some(path) = paint.series {
                window.paint_path(path, line_color);
            }
            for (layer, center) in paint.points {
                let is_selected = selected_layer == Some(layer) || hovered_layer == Some(layer);
                let radius = if is_selected { 4.0 } else { 2.75 };
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
        y_tick_elements.push(mono(value, 8.0, colors.text_faint));
        if index < 4 {
            y_tick_elements.push(div().flex_1());
        }
    }

    div()
        .w_full()
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
                    9.5,
                    if active.is_some() {
                        colors.text
                    } else {
                        colors.text_muted
                    },
                ))
                .child(div().w_full())
                .children(
                    intervention_layer
                        .map(|layer| mono(format!("INTERVENTION  L{layer}"), 8.5, colors.warn)),
                ),
        )
        .child(
            div()
                .flex()
                .gap_2()
                .w_full()
                .child(
                    div()
                        .h(px(height))
                        .w(px(44.0))
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
                .child(mono(format!("L{min_layer}"), 8.5, colors.text_faint))
                .child(div().w_full())
                .child(label("TRANSFORMER LAYER", 8.5, colors.text_faint))
                .child(div().w_full())
                .child(mono(format!("L{max_layer}"), 8.5, colors.text_faint)),
        )
        .child(
            div()
                .flex()
                .items_center()
                .child(label("Y: RELATIVE L2 DIFFERENCE", 8.0, colors.text_faint))
                .child(div().w_full())
                .child(mono(
                    format!("range 0 – {}", metric_label(y_max)),
                    8.0,
                    colors.text_faint,
                )),
        )
}

#[cfg(test)]
mod tests {
    use super::{metric_label, nice_max};

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

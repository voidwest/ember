use super::*;

// ---------------------------------------------------------------------------
// model discovery and the built-in sample
// ---------------------------------------------------------------------------

/// Models under the working directory, with their sizes. Blocking; run it
/// off the UI thread.
pub(super) fn scan_models() -> Vec<(String, Option<u64>)> {
    discover_models()
        .into_iter()
        .map(|path| {
            let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
            (path, size)
        })
        .collect()
}

/// The built-in sample: what a finished comparison looks like. The numbers
/// are illustrative -- shaped like a real Llama-3.2-1B run of "scale x0.5 at
/// layer 8" -- and the page says so; nothing here claims to be a measurement.
pub(super) fn sample_result() -> (RunOutput, RunOutput, ExperimentComparison, FormValues) {
    let text = " Paris. The city is known for its art, its food and its history.";
    let make = |wall_ms: f64| RunOutput {
        text: text.to_string(),
        generated_token_ids: (1..=16).collect(),
        generated_token_texts: vec![String::new(); 16],
        prompt_tokens: 6,
        generated_tokens: 16,
        bundle_dir: "sample".to_string(),
        semantic_hash: "0000000000000000".to_string(),
        payload_hash: "00000000".to_string(),
        wall_ms,
        decode_tps: Some(16.0 / (wall_ms / 1000.0)),
        events: Vec::new(),
    };
    let series: [(f64, f64); 16] = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.296, 0.045),
        (0.245, 0.031),
        (0.207, 0.024),
        (0.158, 0.015),
        (0.150, 0.013),
        (0.138, 0.011),
        (0.141, 0.011),
        (0.141, 0.011),
    ];
    let layers = series
        .iter()
        .enumerate()
        .map(|(layer, (l2, cosine))| crate::gui::LayerMetric {
            layer,
            relative_l2_difference: Some(*l2),
            cosine_distance: Some(*cosine),
            maximum_absolute_difference: None,
            exact: *l2 == 0.0,
        })
        .collect();
    let pieces = [
        " Paris", ".", " The", " city", " is", " known", " for", " its", " art", ",", " its",
        " food", " and", " its", " history", ".",
    ];
    let tokens = pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| crate::gui::TokenMetric {
            position: index + 1,
            baseline_token_id: Some(index as u32 + 1),
            intervention_token_id: Some(index as u32 + 1),
            baseline_text: Some((*piece).to_string()),
            intervention_text: Some((*piece).to_string()),
            differs: false,
        })
        .collect();
    let comparison = ExperimentComparison {
        layers,
        tokens,
        first_token_divergence: None,
        generated_tokens_equal: true,
        generated_text_equal: true,
        landmarks: crate::gui::DivergenceLandmarks {
            first_layer_divergence: Some(8),
            peak_layer: Some(8),
            peak_relative_l2: Some(0.296),
            stable_token_tail_step: None,
        },
        layer_token_grid: None,
    };
    let values = FormValues {
        model_path: String::new(),
        prompt: "The capital of France is".to_string(),
        max_tokens: "24".to_string(),
        execution: "reference".to_string(),
        site: "after-mlp".to_string(),
        layer: "8".to_string(),
        op: "scale".to_string(),
        value: "0.5".to_string(),
        source: "capture".to_string(),
        source_layer: "0".to_string(),
        token: "prompt-final".to_string(),
        span: String::new(),
    };
    (make(1_180.0), make(1_240.0), comparison, values)
}

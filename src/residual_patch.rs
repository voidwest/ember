//! Minimal centered-residual hidden-state intervention (Phase C pilot).
//!
//! Adds `alpha * delta` to selected absolute rows of one activation site
//! during a single embedding-prefill forward, then reports diagnostics.
//! Additive-only: no model, attention, KV, or decode changes. When the patch
//! is `None` (or `alpha == 0.0`, which skips the add branch structurally)
//! the hooked path performs no numeric operations beyond the stock path,
//! which the parity test verifies bit-exactly.

use crate::backend::{CpuBackend, CpuError};
use crate::experiments::LayerHooks;
use crate::llama::{llama_embed_tokens, Llama};
use crate::tensor::CpuTensor;

/// Where the residual is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchSite {
    /// Block-0 input rows (embedding-output span positions).
    EmbedSpan,
    /// Block output rows of one hidden layer.
    HiddenLayer(usize),
}

/// One centered-residual patch application.
#[derive(Debug, Clone)]
pub struct SpanPatch {
    pub site: PatchSite,
    pub rows: Vec<usize>,
    pub delta: Vec<f32>,
    pub alpha: f32,
}

/// Diagnostics recorded at the single patch firing.
#[derive(Debug, Clone, Default)]
pub struct PatchDiag {
    pub fired: bool,
    pub h_norm: f64,
    pub hp_norm: f64,
    pub cos: f64,
    pub delta_norm: f64,
}

struct PatchHook {
    patch: Option<SpanPatch>,
    embed_dim: usize,
    diag: PatchDiag,
}

impl PatchHook {
    fn maybe_apply(&mut self, layer: usize, at_embed_input: bool, values: &mut [f32]) {
        let Some(p) = self.patch.as_ref() else {
            return;
        };
        if p.alpha == 0.0 {
            return; // structural no-op: parity with the stock path
        }
        let fire = match p.site {
            PatchSite::EmbedSpan => at_embed_input && layer == 0,
            PatchSite::HiddenLayer(l) => !at_embed_input && layer == l,
        };
        if !fire {
            return;
        }
        let d = self.embed_dim;
        assert_eq!(p.delta.len(), d, "delta dim != embed_dim");
        let nrows = values.len() / d;
        assert_eq!(nrows * d, values.len(), "activation not row-major [S,d]");
        let mut h2 = 0.0f64;
        let mut hp2 = 0.0f64;
        let mut dot = 0.0f64;
        let mut dn = 0.0f64;
        for &v in p.delta.iter() {
            dn += (v as f64) * (v as f64);
        }
        for &r in p.rows.iter() {
            assert!(r < nrows, "patch row {r} out of range for {nrows} rows");
            let row = &mut values[r * d..(r + 1) * d];
            for (x, &dv) in row.iter_mut().zip(p.delta.iter()) {
                let h = *x as f64;
                let stored = *x;
                let hp = (h + (p.alpha as f64) * (dv as f64)) as f32 as f64;
                h2 += (stored as f64) * (stored as f64);
                dot += (stored as f64) * hp;
                hp2 += hp * hp;
                *x = hp as f32;
            }
        }
        self.diag.fired = true;
        self.diag.h_norm = h2.sqrt();
        self.diag.hp_norm = hp2.sqrt();
        self.diag.cos = if h2 > 0.0 && hp2 > 0.0 {
            dot / (h2.sqrt() * hp2.sqrt())
        } else {
            1.0
        };
        self.diag.delta_norm = dn.sqrt();
    }
}

impl LayerHooks<CpuTensor, CpuError> for PatchHook {
    fn before_layer(&mut self, layer_index: usize, tensor: &mut CpuTensor) -> Result<(), CpuError> {
        let d = self.embed_dim;
        let n = tensor.data().len() / d;
        assert_eq!(n * d, tensor.data().len());
        let _ = n;
        self.maybe_apply(layer_index, true, tensor.data_mut());
        Ok(())
    }
    fn after_attention(
        &mut self,
        _layer_index: usize,
        _tensor: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_mlp(&mut self, _layer_index: usize, _tensor: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_layer(&mut self, layer_index: usize, tensor: &mut CpuTensor) -> Result<(), CpuError> {
        self.maybe_apply(layer_index, false, tensor.data_mut());
        Ok(())
    }
    fn before_logits(&mut self, _tensor: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_logits(&mut self, _tensor: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
}

/// Prefill `token_ids` through the embedding path with an optional
/// single-fire residual patch. Returns last-position logits plus diagnostics.
///
/// `start_pos` must equal `cache.cursor()` (0 for a fresh cache).
pub fn prefill_embed_with_patch(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    token_ids: &[u32],
    cache: &mut crate::kv_cache::KVCache,
    start_pos: usize,
    patch: Option<SpanPatch>,
) -> anyhow::Result<(CpuTensor, PatchDiag)> {
    let embed_dim = model.config.embed_dim;
    if let Some(p) = patch.as_ref() {
        anyhow::ensure!(
            p.delta.len() == embed_dim,
            "patch delta dim {} != model embed_dim {embed_dim}",
            p.delta.len()
        );
        if let PatchSite::HiddenLayer(l) = p.site {
            anyhow::ensure!(
                l < model.blocks.len(),
                "patch layer {l} out of range ({} blocks)",
                model.blocks.len()
            );
        }
        let seq_len = token_ids.len();
        if let Some(bad) = p.rows.iter().find(|r| **r >= seq_len) {
            anyhow::bail!("patch row {bad} out of range for {seq_len} prefill rows");
        }
    }
    let embeddings = llama_embed_tokens(backend, &model.embed_tokens, token_ids, embed_dim)?;
    let mut hook = PatchHook {
        patch,
        embed_dim,
        diag: PatchDiag::default(),
    };
    let logits = model.forward_last_logits_embeddings_with_cache_hooked(
        backend,
        &embeddings,
        cache,
        start_pos,
        &mut hook,
    )?;
    Ok((logits, hook.diag))
}

struct SpanMeanHook<'a> {
    rows: &'a [usize],
    embed_dim: usize,
    means: Vec<Vec<f32>>,
}

impl LayerHooks<CpuTensor, CpuError> for SpanMeanHook<'_> {
    fn before_layer(&mut self, _: usize, _: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_attention(&mut self, _: usize, _: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_mlp(&mut self, _: usize, _: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_layer(&mut self, layer: usize, tensor: &mut CpuTensor) -> Result<(), CpuError> {
        assert_eq!(layer, self.means.len());
        let mut mean = vec![0.0; self.embed_dim];
        for &row in self.rows {
            let values = &tensor.data()[row * self.embed_dim..(row + 1) * self.embed_dim];
            for (acc, &value) in mean.iter_mut().zip(values) {
                *acc += value;
            }
        }
        for value in &mut mean {
            *value /= self.rows.len() as f32;
        }
        self.means.push(mean);
        Ok(())
    }
    fn before_logits(&mut self, _: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
    fn after_logits(&mut self, _: &mut CpuTensor) -> Result<(), CpuError> {
        Ok(())
    }
}

/// Capture block-output span means through the same cached embedding prefill
/// used by residual interventions. The observer never modifies activations.
pub fn collect_cached_span_means(
    model: &Llama<CpuBackend>,
    backend: &CpuBackend,
    token_ids: &[u32],
    rows: &[usize],
) -> anyhow::Result<Vec<Vec<f32>>> {
    anyhow::ensure!(
        !token_ids.is_empty() && !rows.is_empty(),
        "empty capture input"
    );
    anyhow::ensure!(
        rows.iter().all(|&r| r < token_ids.len()),
        "capture row out of range"
    );
    let mut unique = rows.to_vec();
    unique.sort_unstable();
    unique.dedup();
    anyhow::ensure!(unique.len() == rows.len(), "duplicate capture rows");
    let mut hook = SpanMeanHook {
        rows,
        embed_dim: model.config.embed_dim,
        means: Vec::with_capacity(model.blocks.len()),
    };
    let embeddings = llama_embed_tokens(
        backend,
        &model.embed_tokens,
        token_ids,
        model.config.embed_dim,
    )?;
    let mut cache = model.create_request_cache(backend, token_ids.len(), 3);
    model.forward_last_logits_embeddings_with_cache_hooked(
        backend,
        &embeddings,
        &mut cache,
        0,
        &mut hook,
    )?;
    Ok(hook.means)
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn span_observer_preserves_activations_and_block_indexing() {
        let mut tensor = CpuTensor::from_data(vec![3, 2], vec![1., 2., 100., 200., 3., 6.]);
        let original = tensor.data().to_vec();
        let mut hook = SpanMeanHook {
            rows: &[0, 2],
            embed_dim: 2,
            means: Vec::new(),
        };
        hook.after_layer(0, &mut tensor).unwrap();
        hook.after_layer(1, &mut tensor).unwrap();
        assert_eq!(hook.means, vec![vec![2., 4.], vec![2., 4.]]);
        assert_eq!(tensor.data(), original);
    }
}

#[cfg(test)]
mod patch_diag_tests {
    use super::*;

    /// Diagnostics must describe the stored float32 activations: the norm
    /// pair must satisfy ||h_p|| == ||h + alpha*delta|| only up to f32
    /// rounding of the stored value, and cos must stay below 1 when the
    /// pre-rounding increment is smaller than float32 resolution.
    #[test]
    fn diag_uses_stored_float32_activations_not_unrounded_intermediates() {
        // 1.0 + 1e-9 rounds back to exactly 1.0 in float32.
        let mut tensor = CpuTensor::from_data(vec![1, 2], vec![1.0, 1.0]);
        let mut hook = PatchHook {
            patch: Some(SpanPatch {
                site: PatchSite::EmbedSpan,
                rows: vec![0],
                delta: vec![1e-9, 1e-9],
                alpha: 1.0,
            }),
            embed_dim: 2,
            diag: PatchDiag::default(),
        };
        hook.maybe_apply(0, true, tensor.data_mut());
        let d = &hook.diag;
        assert!(d.fired);
        assert_eq!(
            tensor.data(),
            &[1.0, 1.0],
            "increment below f32 resolution must round away"
        );
        // Stored value is unchanged, so the recorded norms must be the
        // stored-value norms: h_norm == hp_norm exactly.
        assert_eq!(
            d.h_norm, d.hp_norm,
            "diag must use stored f32 values, not f64 intermediates"
        );
        assert!((d.cos - 1.0).abs() <= 1e-12);
        assert_eq!(tensor.data(), &[1.0, 1.0]);
    }
}

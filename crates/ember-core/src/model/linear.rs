use super::*;

/// the kind of weight backing a `Linear` layer.
///
/// `F32` is the standard path - f32 tensors loaded from gguf and
/// stored as the backend's native tensor type.  `Half` keeps f16/bf16
/// weights in their GGUF encoding and widens them inside the matmul,
/// bit-identically to an `F32` weight holding the widened values.  `Q8_0` keeps weights in
/// their raw block-compressed form and dequantizes on the fly during
/// matmul, saving ~4x memory.
pub enum WeightKind<B: Backend> {
    /// f32 weight tensor, shape [in_features, out_features]
    F32(B::Tensor),
    /// f16/bf16 weight in GGUF layout (`[out_features]` rows of
    /// `in_features` values), widened block by block during `matmul_half`.
    Half(crate::half_weight::HalfWeight),
    /// q8_0 block-compressed weight, never stored as f32.
    /// dequantized column-by-column during `matmul_q8_0`.
    Q8_0(QuantizedWeight),
    /// q4_k/q6_k super-block-compressed weight, never stored as f32.
    /// multiplied by transient Q8_K activation rows during `matmul_k`.
    KQuant(crate::quant_k::KQuantWeight),
}

/// Read-only view of a `WeightKind` for the fast decode path.
pub enum WeightKindView<'a> {
    F32(&'a CpuTensor),
    Half(&'a crate::half_weight::HalfWeight),
    Q8_0(&'a QuantizedWeight),
    KQuant(&'a crate::quant_k::KQuantWeight),
}

/// a linear (fully-connected) layer: `y = xW + b`.
/// weight must be `[in_features, out_features]`.
pub struct Linear<B: Backend> {
    /// weight matrix, shape [in_features, out_features]
    weight: WeightKind<B>,
    /// optional bias vector, shape [out_features]
    bias: Option<B::Tensor>,
    /// optional interleaved Q8_0 weight for fast decode path.
    ///
    /// Boxed: the packed layout carries two byte buffers plus shape metadata,
    /// and `Linear` is embedded in enums (e.g. Gemma-4's head) that clippy
    /// keeps size-balanced. Only built for the lm head, never read per token.
    pub interleaved: Option<Box<QuantizedWeightInterleaved>>,
    /// optional 16-output packed Q8_0 weight for batch-1 decode
    pub packed_decode: Option<QuantizedWeightVnni>,
}

impl<B: Backend> Linear<B> {
    /// create a linear layer with an f32 weight tensor.
    pub fn new(weight: B::Tensor, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::F32(weight),
            bias,
            interleaved: None,
            packed_decode: None,
        }
    }

    /// create a linear layer with an f16/bf16 weight kept in its GGUF
    /// encoding (`[in_features, out_features]` GGUF dims).
    pub fn new_half(weight: crate::half_weight::HalfWeight, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::Half(weight),
            bias,
            interleaved: None,
            packed_decode: None,
        }
    }

    /// create a linear layer with a q8_0 quantized weight.
    /// the weight stays in block-compressed form; `forward()` calls `matmul_q8_0`.
    pub fn new_q8_0(qw: QuantizedWeight, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::Q8_0(qw),
            bias,
            interleaved: None,
            packed_decode: None,
        }
    }

    /// create a linear layer with a q4_k/q6_k compressed weight.
    /// the weight stays in super-block-compressed form; `forward()` calls
    /// `matmul_k`, which packs activations to Q8_K for integer dot products.
    pub fn new_k(qw: crate::quant_k::KQuantWeight, bias: Option<B::Tensor>) -> Self {
        Self {
            weight: WeightKind::KQuant(qw),
            bias,
            interleaved: None,
            packed_decode: None,
        }
    }

    fn forward_without_bias(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let out = match &self.weight {
            WeightKind::F32(w) => backend.matmul(x, w)?,
            WeightKind::Half(w) => backend.matmul_half(x, w)?,
            WeightKind::Q8_0(qw) => backend.matmul_q8_0(x, qw)?,
            WeightKind::KQuant(qw) => backend.matmul_k(x, qw)?,
        };
        Ok(out)
    }

    pub fn forward(&self, backend: &B, x: &B::Tensor) -> Result<B::Tensor, B::Error> {
        let mut out = self.forward_without_bias(backend, x)?;
        if let Some(ref b) = self.bias {
            out = backend.add_broadcast(&out, b)?;
        }
        Ok(out)
    }

    /// [`Linear::forward`] written into `out`, reusing its storage where the
    /// backend supports it. Bit-identical to `forward`.
    pub(crate) fn forward_reusing(
        &self,
        backend: &B,
        x: &B::Tensor,
        out: &mut B::Tensor,
    ) -> Result<(), B::Error> {
        match &self.weight {
            WeightKind::F32(w) => *out = backend.matmul(x, w)?,
            WeightKind::Half(w) => *out = backend.matmul_half(x, w)?,
            WeightKind::Q8_0(qw) => backend.matmul_q8_0_reusing(x, qw, out)?,
            WeightKind::KQuant(qw) => backend.matmul_k_reusing(x, qw, out)?,
        }
        if let Some(ref b) = self.bias {
            backend.add_broadcast_in_place(out, b)?;
        }
        Ok(())
    }

    /// Apply this layer and `other` to the same input.
    ///
    /// When both weights are Q8_0 the backend can quantize the activations
    /// once and schedule both projections together.
    pub(super) fn forward_pair(
        &self,
        backend: &B,
        x: &B::Tensor,
        other: &Self,
    ) -> Result<(B::Tensor, B::Tensor), B::Error> {
        let (mut first, mut second) = match (&self.weight, &other.weight) {
            (WeightKind::Q8_0(first), WeightKind::Q8_0(second)) => {
                backend.matmul_q8_0_pair(x, first, second)?
            }
            _ => (
                self.forward_without_bias(backend, x)?,
                other.forward_without_bias(backend, x)?,
            ),
        };
        if let Some(ref bias) = self.bias {
            first = backend.add_broadcast(&first, bias)?;
        }
        if let Some(ref bias) = other.bias {
            second = backend.add_broadcast(&second, bias)?;
        }
        Ok((first, second))
    }

    /// Apply a packed pair when the backend and measured row regime support
    /// it, otherwise retain the generic Q8_0 pair as the correctness oracle.
    ///
    /// Traced runs deliberately stay generic so operation inspection and
    /// value fingerprints always follow the authoritative validation path.
    pub(crate) fn forward_packed_pair_or_generic(
        &self,
        backend: &B,
        x: &B::Tensor,
        other: &Self,
    ) -> Result<(B::Tensor, B::Tensor), B::Error> {
        if !crate::trace::is_tracing()
            && let (Some(first), Some(second)) = (&self.packed_decode, &other.packed_decode)
            && let Some(result) = backend.matmul_q8_0_packed_pair(x, first, second)?
        {
            return Ok(result);
        }
        self.forward_pair(backend, x, other)
    }

    /// Apply this layer and two peers to the same input.
    pub fn forward_triple(
        &self,
        backend: &B,
        x: &B::Tensor,
        second: &Self,
        third: &Self,
    ) -> Result<TensorTriple<B::Tensor>, B::Error> {
        let (mut first_out, mut second_out, mut third_out) =
            match (&self.weight, &second.weight, &third.weight) {
                (WeightKind::Q8_0(first), WeightKind::Q8_0(second), WeightKind::Q8_0(third)) => {
                    backend.matmul_q8_0_triple(x, first, second, third)?
                }
                _ => (
                    self.forward_without_bias(backend, x)?,
                    second.forward_without_bias(backend, x)?,
                    third.forward_without_bias(backend, x)?,
                ),
            };
        if let Some(ref bias) = self.bias {
            first_out = backend.add_broadcast(&first_out, bias)?;
        }
        if let Some(ref bias) = second.bias {
            second_out = backend.add_broadcast(&second_out, bias)?;
        }
        if let Some(ref bias) = third.bias {
            third_out = backend.add_broadcast(&third_out, bias)?;
        }
        Ok((first_out, second_out, third_out))
    }

    /// Return the byte size of the weight tensor (for trace/benchmark reporting).
    pub fn weight_bytes(&self, backend: &B) -> usize {
        match &self.weight {
            WeightKind::F32(w) => {
                let shape = backend.shape(w);
                shape.iter().product::<usize>() * 4
            }
            WeightKind::Half(w) => w.byte_len(),
            WeightKind::Q8_0(qw) => qw.byte_len(),
            WeightKind::KQuant(qw) => qw.byte_len(),
        }
    }

    pub fn in_features(&self, backend: &B) -> usize {
        match &self.weight {
            WeightKind::F32(w) => backend.shape(w)[0],
            WeightKind::Half(w) => w.in_features(),
            WeightKind::Q8_0(qw) => qw.in_features(),
            WeightKind::KQuant(qw) => qw.in_features(),
        }
    }

    pub fn out_features(&self, backend: &B) -> usize {
        match &self.weight {
            WeightKind::F32(w) => backend.shape(w)[1],
            WeightKind::Half(w) => w.out_features(),
            WeightKind::Q8_0(qw) => qw.out_features(),
            WeightKind::KQuant(qw) => qw.out_features(),
        }
    }
}

impl Linear<CpuBackend> {
    /// Return a read-only view of the underlying weight.
    pub fn weight_kind(&self) -> WeightKindView<'_> {
        match &self.weight {
            WeightKind::F32(t) => WeightKindView::F32(t),
            WeightKind::Half(w) => WeightKindView::Half(w),
            WeightKind::Q8_0(qw) => WeightKindView::Q8_0(qw),
            WeightKind::KQuant(qw) => WeightKindView::KQuant(qw),
        }
    }

    /// The optional bias tensor (qwen2.5 q/k/v projections carry F32 biases).
    pub fn bias(&self) -> Option<&<CpuBackend as crate::backend::Backend>::Tensor> {
        self.bias.as_ref()
    }

    /// Return a Q8_0 weight only when the layer needs no separate bias pass.
    pub(crate) fn q8_weight_without_bias(&self) -> Option<&QuantizedWeight> {
        if self.bias.is_some() {
            return None;
        }
        match &self.weight {
            WeightKind::Q8_0(weight) => Some(weight),
            WeightKind::F32(_) | WeightKind::Half(_) => None,
            WeightKind::KQuant(_) => None,
        }
    }

    /// Return the packed batch-1 weight only when no bias pass is needed.
    pub(crate) fn packed_q8_weight_without_bias(&self) -> Option<&QuantizedWeightVnni> {
        if self.bias.is_some() {
            return None;
        }
        self.packed_decode.as_ref()
    }

    /// Build the same-size, 16-output packed layout used by the AVX-512 VNNI
    /// batch-1 projection kernel.
    pub fn prepare_packed_decode(&mut self) {
        if self.prepare_packed_decode_without_eviction().is_some() {
            #[cfg(unix)]
            let _ = self.evict_packed_source_pages();
        }
    }

    /// As [`Self::prepare_packed_decode`], consulting an optional on-disk
    /// packed-layout cache: a validated `entry_name` replaces the repack,
    /// otherwise the freshly packed layout is recorded for the next run.
    pub fn prepare_packed_decode_cached(
        &mut self,
        cache: Option<&crate::packed_cache::PackedCache>,
        entry_name: &str,
    ) {
        if self.packed_decode.is_some() || !crate::simd::packed_q8_0_vnni_supported() {
            return;
        }
        let cached = cache.and_then(|cache| {
            let weight = self.q8_weight_without_bias()?;
            cache.get_vnni(entry_name, weight.out_features(), weight.in_features())
        });
        match cached {
            Some(packed) => self.packed_decode = Some(packed),
            None => {
                self.prepare_packed_decode();
                if let (Some(cache), Some(packed)) = (cache, self.packed_decode.as_ref()) {
                    cache.record_vnni(entry_name, packed);
                }
            }
        }
    }

    /// Build the packed decode representation while leaving source residency
    /// unchanged. Returns the new packed byte count, or `None` when the layer
    /// was already packed or is ineligible.
    pub(crate) fn prepare_packed_decode_without_eviction(&mut self) -> Option<usize> {
        if self.packed_decode.is_some() || !crate::simd::packed_q8_0_vnni_supported() {
            return None;
        }
        let WeightKind::Q8_0(weight) = &self.weight else {
            return None;
        };
        let packed = QuantizedWeightVnni::from_quantized(weight);
        let packed_bytes = packed.byte_len();
        self.packed_decode = Some(packed);
        Some(packed_bytes)
    }

    /// Evict the mapped source pages for an already-packed projection.
    ///
    /// The source object remains valid and can fault the pages back if a
    /// generic operation accesses it later.
    #[cfg(unix)]
    pub(crate) fn evict_packed_source_pages(&self) -> std::io::Result<bool> {
        if self.packed_decode.is_none() {
            return Ok(false);
        }
        let WeightKind::Q8_0(weight) = &self.weight else {
            return Ok(false);
        };
        // Safety: callers invoke this only at lifecycle phase boundaries where
        // inference is stopped and no source-weight slices can be live.
        unsafe { weight.evict_mapped_pages() }
    }

    #[cfg(not(unix))]
    pub(crate) fn evict_packed_source_pages(&self) -> std::io::Result<bool> {
        Ok(false)
    }

    /// Bytes in the optional packed representation.
    pub(crate) fn packed_decode_bytes(&self) -> usize {
        self.packed_decode
            .as_ref()
            .map_or(0, QuantizedWeightVnni::byte_len)
    }

    /// Whether this layer currently owns a packed decode representation.
    pub(crate) fn has_packed_decode(&self) -> bool {
        self.packed_decode.is_some()
    }

    /// Whether the row-contiguous source is backed by the model mmap.
    pub(crate) fn has_mapped_q8_source(&self) -> bool {
        match &self.weight {
            WeightKind::Q8_0(weight) => weight.is_mapped(),
            WeightKind::F32(_) | WeightKind::Half(_) => false,
            WeightKind::KQuant(_) => false,
        }
    }

    /// Build the decode-optimized layout for sufficiently wide Q8_0 layers,
    /// consulting an optional on-disk packed-layout cache: a validated
    /// `entry_name` replaces the repack, otherwise the freshly packed layout
    /// is recorded for the next run.
    ///
    /// Keeping this opt-in avoids duplicating every projection in memory. In
    /// practice only the LM head is wide enough to benefit.
    pub fn prepare_interleaved_cached(
        &mut self,
        min_out_features: usize,
        cache: Option<&crate::packed_cache::PackedCache>,
        entry_name: &str,
    ) {
        if self.interleaved.is_some() || !crate::simd::interleaved_q8_0_supported() {
            return;
        }
        if let WeightKind::Q8_0(weight) = &self.weight
            && weight.out_features() >= min_out_features
        {
            let cached = cache.and_then(|cache| {
                cache.get_interleaved(entry_name, weight.out_features(), weight.in_features())
            });
            match cached {
                Some(interleaved) => self.interleaved = Some(Box::new(interleaved)),
                None => {
                    let packed = QuantizedWeightInterleaved::from_quantized(weight);
                    if let Some(cache) = cache {
                        cache.record_interleaved(entry_name, &packed);
                    }
                    self.interleaved = Some(Box::new(packed));
                }
            }
        }
    }
}

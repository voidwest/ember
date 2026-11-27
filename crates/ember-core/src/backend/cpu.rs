use super::*;

impl CpuBackend {
    /// Zero-cost per-kernel timing: when `TIMED` is false the `Instant` path is
    /// eliminated at compile time and the kernel runs untouched.
    #[inline(always)]
    pub(super) fn time_kernel<const TIMED: bool>(f: impl FnOnce()) -> std::time::Duration {
        if TIMED {
            let started = std::time::Instant::now();
            f();
            started.elapsed()
        } else {
            f();
            std::time::Duration::ZERO
        }
    }

    pub(super) fn matmul_q8_0_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        rows: usize,
        w: &QuantizedWeight,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert_q8_projection_layout(
            src,
            rows,
            w.in_features(),
            w.out_features(),
            dst,
            "matmul_q8_0_into",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            quantize_projection_input(src, rows, &mut input);
            if rows == 1 {
                Self::time_kernel::<TIMED>(|| crate::simd::matmul_q8_0_decode(&input, w, dst))
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w, dst);
                std::time::Duration::ZERO
            }
        })
    }

    /// Quantize flat f32 activations `src` (shape `[rows, in_features]`) and
    /// compute `dst = src × w` using packed Q8_0 integer dots. Writes into the
    /// pre-allocated `dst` slice, which must have length `rows * w.out_features()`.
    ///
    /// This is the zero-alloc variant of `matmul_q8_0` — it reuses the
    /// thread-local quantized-activation buffer and writes directly into the
    /// caller's output slice instead of wrapping a new `Vec`.
    pub fn matmul_q8_0_into(&self, src: &[f32], rows: usize, w: &QuantizedWeight, dst: &mut [f32]) {
        self.matmul_q8_0_into_impl::<false>(src, rows, w, dst);
    }

    /// Decode MLP tail: `gated = silu(gate) * up`, quantized in the same
    /// team pass, then `dst = down · gated`.
    ///
    /// Bit-identical to [`crate::simd::silu_mul_into`] followed by
    /// [`Self::matmul_q8_0_into`] (or the packed projection when `packed` is
    /// given); it only removes a serial pass over the intermediate width.
    pub fn silu_mul_matmul_q8_0_decode_into(
        &self,
        gate: &[f32],
        up: &[f32],
        gated: &mut [f32],
        down: &QuantizedWeight,
        packed: Option<&QuantizedWeightVnni>,
        dst: &mut [f32],
    ) {
        assert_q8_projection_layout(
            gated,
            1,
            down.in_features(),
            down.out_features(),
            dst,
            "silu_mul_matmul_q8_0_decode_into",
        );
        let weight = match packed {
            Some(packed) => crate::simd::Q8DecodeWeight::Packed16(packed),
            None => crate::simd::Q8DecodeWeight::Rows(down),
        };
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::simd::silu_mul_quantize_q8_0_decode_into(gate, up, gated, &mut input);
            crate::simd::matmul_q8_0_decode_tasks(&mut [crate::simd::Q8DecodeTask::columns(
                &input, weight, 1, dst,
            )]);
        });
    }

    /// Instrumented single-row projection used only by operator profiling.
    ///
    /// The returned duration covers the matrix kernel, not activation
    /// quantization, so it is directly comparable with the pair/triple timings.
    pub fn matmul_q8_0_into_timed(
        &self,
        src: &[f32],
        w: &QuantizedWeight,
        dst: &mut [f32],
    ) -> std::time::Duration {
        self.matmul_q8_0_into_impl::<true>(src, 1, w, dst)
    }

    pub(super) fn matmul_q8_0_interleaved_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        w: &crate::quant::QuantizedWeightInterleaved,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert!(crate::simd::interleaved_q8_0_supported());
        assert_eq!(src.len(), w.in_features());
        assert_eq!(dst.len(), w.out_features());
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::simd::quantize_q8_0_decode_into(src, &mut input);
            Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_interleaved_parallel(&input, w, dst)
            })
        })
    }

    /// Q8_0 decode projection using the cache-friendly interleaved layout.
    ///
    /// The quantized activation stays borrowed inside the thread-local closure,
    /// so no fabricated lifetime or raw slice construction is required.
    pub fn matmul_q8_0_interleaved_into(
        &self,
        src: &[f32],
        w: &crate::quant::QuantizedWeightInterleaved,
        dst: &mut [f32],
    ) {
        self.matmul_q8_0_interleaved_into_impl::<false>(src, w, dst);
    }

    /// Instrumented interleaved projection used only by operator profiling.
    pub fn matmul_q8_0_interleaved_into_timed(
        &self,
        src: &[f32],
        w: &crate::quant::QuantizedWeightInterleaved,
        dst: &mut [f32],
    ) -> std::time::Duration {
        self.matmul_q8_0_interleaved_into_impl::<true>(src, w, dst)
    }

    pub(super) fn matmul_q8_0_packed_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        weight: &QuantizedWeightVnni,
        dst: &mut [f32],
    ) -> std::time::Duration {
        assert_q8_projection_layout(
            src,
            1,
            weight.in_features(),
            weight.out_features(),
            dst,
            "matmul_q8_0_packed_into",
        );
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::simd::quantize_q8_0_decode_into(src, &mut input);
            Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, weight, dst)
            })
        })
    }

    /// Batch-1 projection over a 16-output packed Q8_0 weight.
    pub fn matmul_q8_0_packed_into(
        &self,
        src: &[f32],
        weight: &QuantizedWeightVnni,
        dst: &mut [f32],
    ) {
        self.matmul_q8_0_packed_into_impl::<false>(src, weight, dst);
    }

    /// Instrumented packed projection used by the optional operator profiler.
    pub fn matmul_q8_0_packed_into_timed(
        &self,
        src: &[f32],
        weight: &QuantizedWeightVnni,
        dst: &mut [f32],
    ) -> std::time::Duration {
        self.matmul_q8_0_packed_into_impl::<true>(src, weight, dst)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub(super) fn matmul_q8_0_packed_pair_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
    ) -> [std::time::Duration; 2] {
        for (weight, dst, operation) in [
            (first, &*first_dst, "matmul_q8_0_packed_pair_into(first)"),
            (second, &*second_dst, "matmul_q8_0_packed_pair_into(second)"),
        ] {
            assert_q8_projection_layout(
                src,
                1,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::simd::quantize_q8_0_decode_into(src, &mut input);
            let mut timings = [std::time::Duration::ZERO; 2];
            if !TIMED {
                crate::simd::matmul_q8_0_decode_packed16_many(
                    &input,
                    &mut [(first, first_dst), (second, second_dst)],
                );
                return timings;
            }
            timings[0] = Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst)
            });
            timings[1] = Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst)
            });
            timings
        })
    }

    /// Two packed projections sharing one activation quantization.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_packed_pair_into(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
    ) {
        self.matmul_q8_0_packed_pair_into_impl::<false>(src, first, second, first_dst, second_dst);
    }

    /// Instrumented packed pair sharing one activation quantization.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_packed_pair_into_timed(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
    ) -> [std::time::Duration; 2] {
        self.matmul_q8_0_packed_pair_into_impl::<true>(src, first, second, first_dst, second_dst)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub(super) fn matmul_q8_0_packed_triple_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        third: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
        third_dst: &mut [f32],
    ) -> [std::time::Duration; 3] {
        for (weight, dst, operation) in [
            (first, &*first_dst, "matmul_q8_0_packed_triple_into(first)"),
            (
                second,
                &*second_dst,
                "matmul_q8_0_packed_triple_into(second)",
            ),
            (third, &*third_dst, "matmul_q8_0_packed_triple_into(third)"),
        ] {
            assert_q8_projection_layout(
                src,
                1,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::simd::quantize_q8_0_decode_into(src, &mut input);
            let mut timings = [std::time::Duration::ZERO; 3];
            if !TIMED {
                crate::simd::matmul_q8_0_decode_packed16_many(
                    &input,
                    &mut [(first, first_dst), (second, second_dst), (third, third_dst)],
                );
                return timings;
            }
            timings[0] = Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, first, first_dst)
            });
            timings[1] = Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, second, second_dst)
            });
            timings[2] = Self::time_kernel::<TIMED>(|| {
                crate::simd::matmul_q8_0_decode_packed16_parallel(&input, third, third_dst)
            });
            timings
        })
    }

    /// Three packed projections sharing one activation quantization.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_packed_triple_into(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        third: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
        third_dst: &mut [f32],
    ) {
        self.matmul_q8_0_packed_triple_into_impl::<false>(
            src, first, second, third, first_dst, second_dst, third_dst,
        );
    }

    /// Instrumented packed triple sharing one activation quantization.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_packed_triple_into_timed(
        &self,
        src: &[f32],
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
        third: &QuantizedWeightVnni,
        first_dst: &mut [f32],
        second_dst: &mut [f32],
        third_dst: &mut [f32],
    ) -> [std::time::Duration; 3] {
        self.matmul_q8_0_packed_triple_into_impl::<true>(
            src, first, second, third, first_dst, second_dst, third_dst,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub(super) fn matmul_q8_0_pair_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        rows: usize,
        w_a: &QuantizedWeight,
        w_b: &QuantizedWeight,
        dst_a: &mut [f32],
        dst_b: &mut [f32],
    ) -> [std::time::Duration; 2] {
        for (weight, dst, operation) in [
            (w_a, &*dst_a, "matmul_q8_0_pair_into(first)"),
            (w_b, &*dst_b, "matmul_q8_0_pair_into(second)"),
        ] {
            assert_q8_projection_layout(
                src,
                rows,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            quantize_projection_input(src, rows, &mut input);
            let mut timings = [std::time::Duration::ZERO; 2];
            if rows == 1 && !TIMED {
                // One decode region for both projections.
                crate::simd::matmul_q8_0_decode_rows(&input, &mut [(w_a, dst_a), (w_b, dst_b)]);
            } else if rows == 1 {
                timings[0] = Self::time_kernel::<TIMED>(|| {
                    crate::simd::matmul_q8_0_decode(&input, w_a, dst_a)
                });
                timings[1] = Self::time_kernel::<TIMED>(|| {
                    crate::simd::matmul_q8_0_decode(&input, w_b, dst_b)
                });
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w_a, dst_a);
                crate::simd::matmul_q8_0_batch(&input, rows, w_b, dst_b);
            }
            timings
        })
    }

    /// Fused dual Q8_0 projection (gate + up): quantize `src` once, compute
    /// both projections in one pass.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_pair_into(
        &self,
        src: &[f32],
        rows: usize,
        w_a: &QuantizedWeight,
        w_b: &QuantizedWeight,
        dst_a: &mut [f32],
        dst_b: &mut [f32],
    ) {
        self.matmul_q8_0_pair_into_impl::<false>(src, rows, w_a, w_b, dst_a, dst_b);
    }

    /// Instrumented fused-input pair projection used only by operator profiling.
    pub fn matmul_q8_0_pair_into_timed(
        &self,
        src: &[f32],
        w_a: &QuantizedWeight,
        w_b: &QuantizedWeight,
        dst_a: &mut [f32],
        dst_b: &mut [f32],
    ) -> [std::time::Duration; 2] {
        self.matmul_q8_0_pair_into_impl::<true>(src, 1, w_a, w_b, dst_a, dst_b)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub(super) fn matmul_q8_0_triple_into_impl<const TIMED: bool>(
        &self,
        src: &[f32],
        rows: usize,
        w_q: &QuantizedWeight,
        w_k: &QuantizedWeight,
        w_v: &QuantizedWeight,
        dst_q: &mut [f32],
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> [std::time::Duration; 3] {
        for (weight, dst, operation) in [
            (w_q, &*dst_q, "matmul_q8_0_triple_into(q)"),
            (w_k, &*dst_k, "matmul_q8_0_triple_into(k)"),
            (w_v, &*dst_v, "matmul_q8_0_triple_into(v)"),
        ] {
            assert_q8_projection_layout(
                src,
                rows,
                weight.in_features(),
                weight.out_features(),
                dst,
                operation,
            );
        }
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            quantize_projection_input(src, rows, &mut input);
            let mut timings = [std::time::Duration::ZERO; 3];
            if rows == 1 && !TIMED {
                // One decode region for all three projections.
                crate::simd::matmul_q8_0_decode_rows(
                    &input,
                    &mut [(w_q, dst_q), (w_k, dst_k), (w_v, dst_v)],
                );
            } else if rows == 1 {
                timings[0] = Self::time_kernel::<TIMED>(|| {
                    crate::simd::matmul_q8_0_decode(&input, w_q, dst_q)
                });
                timings[1] = Self::time_kernel::<TIMED>(|| {
                    crate::simd::matmul_q8_0_decode(&input, w_k, dst_k)
                });
                timings[2] = Self::time_kernel::<TIMED>(|| {
                    crate::simd::matmul_q8_0_decode(&input, w_v, dst_v)
                });
            } else {
                crate::simd::matmul_q8_0_batch(&input, rows, w_q, dst_q);
                crate::simd::matmul_q8_0_batch(&input, rows, w_k, dst_k);
                crate::simd::matmul_q8_0_batch(&input, rows, w_v, dst_v);
            }
            timings
        })
    }

    /// Fused Q/K/V decode projection: quantize `src` once, compute all three
    /// Q8_0 projections in one pass. For seq_len=1 decode this saves two
    /// activation quantization passes and their scheduling overhead.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_triple_into(
        &self,
        src: &[f32],
        rows: usize,
        w_q: &QuantizedWeight,
        w_k: &QuantizedWeight,
        w_v: &QuantizedWeight,
        dst_q: &mut [f32],
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) {
        self.matmul_q8_0_triple_into_impl::<false>(src, rows, w_q, w_k, w_v, dst_q, dst_k, dst_v);
    }

    /// Instrumented fused-input triple projection used only by operator
    /// profiling.
    #[allow(
        clippy::too_many_arguments,
        reason = "fused projection: each weight and its output buffer stay separate arguments"
    )]
    pub fn matmul_q8_0_triple_into_timed(
        &self,
        src: &[f32],
        w_q: &QuantizedWeight,
        w_k: &QuantizedWeight,
        w_v: &QuantizedWeight,
        dst_q: &mut [f32],
        dst_k: &mut [f32],
        dst_v: &mut [f32],
    ) -> [std::time::Duration; 3] {
        self.matmul_q8_0_triple_into_impl::<true>(src, 1, w_q, w_k, w_v, dst_q, dst_k, dst_v)
    }

    /// Cached causal attention into caller-owned storage.
    ///
    /// This is the allocation-free form used by the single-token Llama decode
    /// path. `q` and `out` are flat row-major buffers with width
    /// `n_heads * head_dim`.
    pub fn cached_causal_attention_into(
        &self,
        q: &[f32],
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
        out: &mut [f32],
    ) -> Result<(), CpuError> {
        let embed_dim = spec
            .n_heads
            .checked_mul(spec.head_dim)
            .ok_or_else(|| CpuError::ShapeMismatch("attention width overflow".into()))?;
        if embed_dim == 0 || !q.len().is_multiple_of(embed_dim) {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: q len {} is not divisible by width {}",
                q.len(),
                embed_dim
            )));
        }
        let seq_len = q.len() / embed_dim;
        if out.len() != q.len() {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: output len {} != q len {}",
                out.len(),
                q.len()
            )));
        }
        if spec.total_seq_len < seq_len || spec.total_seq_len > spec.max_seq_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: total_seq_len {} invalid for seq_len {} and max_seq_len {}",
                spec.total_seq_len, seq_len, spec.max_seq_len
            )));
        }
        let cache_len = spec
            .n_kv_heads
            .checked_mul(spec.max_seq_len)
            .and_then(|len| len.checked_mul(spec.head_dim))
            .ok_or_else(|| CpuError::ShapeMismatch("attention cache length overflow".into()))?;
        if cached_k.len() != cache_len || cached_v.len() != cache_len {
            return Err(CpuError::ShapeMismatch(format!(
                "cached_causal_attention: cache len mismatch, got k={} v={}, expected {}",
                cached_k.len(),
                cached_v.len(),
                cache_len
            )));
        }

        let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
        let scale = (spec.head_dim as f32).sqrt().recip();
        let cache_head_stride = spec.max_seq_len * spec.head_dim;

        out.fill(0.0);
        cached_attention_dispatch(
            q,
            cached_k,
            cached_v,
            &spec,
            seq_len,
            embed_dim,
            n_repeat,
            scale,
            cache_head_stride,
            spec.head_dim,
            None,
            true,
            qk_row,
            out,
        );
        Ok(())
    }

    /// Single-token cached attention of several independent sequences in one
    /// decode-team region, one chunk per (sequence, head).
    ///
    /// Every head is computed by the same per-head body, in the same order,
    /// as [`Self::cached_causal_attention_into`] computes it for a
    /// single-token query (whichever of its serial, Rayon or team routes runs),
    /// so each sequence's output is bit-identical to that call. Returns
    /// `Ok(false)` without computing anything when the decode team is
    /// unavailable or the sequences differ in head layout; the caller then
    /// attends each sequence on its own.
    pub(crate) fn cached_decode_attention_batch_into(
        &self,
        items: &mut [DecodeAttention<'_>],
    ) -> Result<bool, CpuError> {
        let Some(first) = items.first() else {
            return Ok(true);
        };
        let (n_heads, n_kv_heads, head_dim) = (
            first.spec.n_heads,
            first.spec.n_kv_heads,
            first.spec.head_dim,
        );
        if items.len() < 2 || !crate::decode_pool::available() {
            return Ok(false);
        }
        let embed_dim = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| CpuError::ShapeMismatch("attention width overflow".into()))?;
        let n_repeat = validate_gqa(n_heads, n_kv_heads)?;
        for item in items.iter() {
            let spec = &item.spec;
            if (spec.n_heads, spec.n_kv_heads, spec.head_dim) != (n_heads, n_kv_heads, head_dim) {
                return Ok(false);
            }
            if embed_dim == 0 || item.q.len() != embed_dim || item.out.len() != embed_dim {
                return Err(CpuError::ShapeMismatch(format!(
                    "batched decode attention: q len {} / output len {} != width {embed_dim}",
                    item.q.len(),
                    item.out.len()
                )));
            }
            if spec.total_seq_len == 0 || spec.total_seq_len > spec.max_seq_len {
                return Err(CpuError::ShapeMismatch(format!(
                    "batched decode attention: total_seq_len {} invalid for max_seq_len {}",
                    spec.total_seq_len, spec.max_seq_len
                )));
            }
            let cache_len = n_kv_heads
                .checked_mul(spec.max_seq_len)
                .and_then(|len| len.checked_mul(head_dim))
                .ok_or_else(|| CpuError::ShapeMismatch("attention cache length overflow".into()))?;
            if item.cached_k.len() != cache_len || item.cached_v.len() != cache_len {
                return Err(CpuError::ShapeMismatch(format!(
                    "batched decode attention: cache len mismatch, got k={} v={}, expected \
                     {cache_len}",
                    item.cached_k.len(),
                    item.cached_v.len()
                )));
            }
        }
        let scale = (head_dim as f32).sqrt().recip();
        let outs: Vec<crate::decode_pool::SharedMut<f32>> = items
            .iter_mut()
            .map(|item| {
                item.out.fill(0.0);
                crate::decode_pool::SharedMut::new(item.out)
            })
            .collect();
        let inputs: Vec<_> = items
            .iter()
            .map(|item| (item.q, item.cached_k, item.cached_v, item.spec))
            .collect();
        let job = |chunk: usize| {
            let (sequence, head) = (chunk / n_heads, chunk % n_heads);
            let (q, cached_k, cached_v, spec) = inputs[sequence];
            // SAFETY: chunk indices are distinct, so each (sequence, head)
            // pair, and with it this head's slice of that sequence's output,
            // is claimed by exactly one chunk; the outputs outlive the region.
            let head_out = unsafe { outs[sequence].range(head * head_dim, head_dim) };
            let max_j = spec.total_seq_len - 1;
            ATTENTION_SCORE_SCRATCH.with(|scores| {
                let mut scores = scores.borrow_mut();
                scores.resize(max_j + 1, 0.0);
                cached_attention_row_head(
                    q,
                    cached_k,
                    cached_v,
                    0,
                    head,
                    embed_dim,
                    head_dim,
                    head_dim,
                    n_repeat,
                    scale,
                    spec.max_seq_len * head_dim,
                    max_j,
                    0,
                    scores.as_mut_slice(),
                    head_out,
                );
            });
        };
        Ok(crate::decode_pool::run(items.len() * n_heads, &job))
    }
}

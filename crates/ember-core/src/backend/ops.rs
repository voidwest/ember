use super::*;

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    type Error = CpuError;

    fn zeroes(&self, shape: &[usize]) -> Result<CpuTensor, CpuError> {
        Ok(CpuTensor::zeroes(shape))
    }

    fn matmul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        // Large shapes split across the rayon pool (bit-identical to the
        // serial path; see `CpuTensor::par_matmul`). This is what makes
        // multimodal encoder prefill and long-prompt LLM prefill scale.
        Ok(a.par_matmul(b))
    }

    fn matmul_half(
        &self,
        x: &CpuTensor,
        w: &crate::half_weight::HalfWeight,
    ) -> Result<CpuTensor, CpuError> {
        if x.ndim() != 2 || w.dims().len() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_half: input and weight must be 2D, got {:?} and {:?}",
                x.shape(),
                w.dims()
            )));
        }
        let (seq_len, in_features) = (x.shape()[0], x.shape()[1]);
        if in_features != w.in_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_half: inner dims must match (got {} vs {})",
                in_features,
                w.in_features()
            )));
        }
        let output_len = seq_len.checked_mul(w.out_features()).ok_or_else(|| {
            CpuError::ShapeMismatch("matmul_half: output shape product overflow".into())
        })?;
        let mut out = vec![0.0f32; output_len];
        crate::half_weight::half_matmul_into(x.data(), seq_len, w, &mut out);
        Ok(CpuTensor::from_data(vec![seq_len, w.out_features()], out))
    }

    fn matmul_q8_0(&self, x: &CpuTensor, w: &QuantizedWeight) -> Result<CpuTensor, CpuError> {
        let (seq_len, output_len) = q8_matmul_output_len(x, w)?;
        let mut out = vec![0.0f32; output_len];
        q8_0_matmul_rows(x.data(), seq_len, w, &mut out);
        Ok(CpuTensor::from_data(vec![seq_len, w.out_features()], out))
    }

    fn matmul_k(
        &self,
        x: &CpuTensor,
        w: &crate::quant_k::KQuantWeight,
    ) -> Result<CpuTensor, CpuError> {
        let (seq_len, output_len) = k_matmul_output_len(x, w)?;
        let mut out = vec![0.0f32; output_len];
        // Decode and prefill use the same Q8_K integer-dot primitive. The
        // parallel scheduler partitions disjoint output columns, so it is
        // bit-identical to the serial entry.
        crate::k_matmul::matmul_k_into_parallel(x.data(), seq_len, w, &mut out)
            .map_err(CpuError::Kernel)?;
        Ok(CpuTensor::from_data(vec![seq_len, w.out_features()], out))
    }

    fn matmul_q8_0_reusing(
        &self,
        x: &CpuTensor,
        w: &QuantizedWeight,
        out: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        let (seq_len, _) = q8_matmul_output_len(x, w)?;
        out.reset_zeroed(&[seq_len, w.out_features()]);
        q8_0_matmul_rows(x.data(), seq_len, w, out.data_mut());
        Ok(())
    }

    fn matmul_k_reusing(
        &self,
        x: &CpuTensor,
        w: &crate::quant_k::KQuantWeight,
        out: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        let (seq_len, _) = k_matmul_output_len(x, w)?;
        out.reset_zeroed(&[seq_len, w.out_features()]);
        crate::k_matmul::matmul_k_into_parallel(x.data(), seq_len, w, out.data_mut())
            .map_err(CpuError::Kernel)
    }

    fn add_broadcast_in_place(&self, x: &mut CpuTensor, bias: &CpuTensor) -> Result<(), CpuError> {
        x.add_broadcast_in_place(bias);
        Ok(())
    }

    fn add_in_place(&self, x: &mut CpuTensor, other: &CpuTensor) -> Result<(), CpuError> {
        x.add_in_place(other);
        Ok(())
    }

    fn silu_in_place(&self, x: &mut CpuTensor) -> Result<(), CpuError> {
        x.silu_in_place();
        Ok(())
    }

    fn elemul_in_place(&self, x: &mut CpuTensor, other: &CpuTensor) -> Result<(), CpuError> {
        x.elemul_in_place(other);
        Ok(())
    }

    fn rms_norm_reusing(
        &self,
        x: &CpuTensor,
        weight: &CpuTensor,
        eps: f32,
        out: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        x.rms_norm_into(weight, eps, out);
        Ok(())
    }

    fn cached_causal_attention_reusing(
        &self,
        q: &CpuTensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
        out: &mut CpuTensor,
    ) -> Result<(), CpuError> {
        let seq_len = validate_cached_attention_inputs(q, cached_k, cached_v, &spec)?;
        out.reset_zeroed(&[seq_len, spec.n_heads * spec.head_dim]);
        cached_attention_into(
            q,
            cached_k,
            cached_v,
            &spec,
            seq_len,
            qk_row,
            out.data_mut(),
        )
    }

    fn update_in_place(
        &self,
        x: &mut CpuTensor,
        update: impl FnOnce(&mut [f32]),
    ) -> Result<(), CpuError> {
        update(x.data_mut());
        Ok(())
    }

    fn matmul_q8_0_pair(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
    ) -> Result<(CpuTensor, CpuTensor), CpuError> {
        let (seq_len, first_len) = q8_matmul_output_len(x, first)?;
        let (_, second_len) = q8_matmul_output_len(x, second)?;
        let mut first_out = vec![0.0f32; first_len];
        let mut second_out = vec![0.0f32; second_len];
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            if seq_len == 1 {
                crate::simd::matmul_q8_0_decode_rows(
                    &input,
                    &mut [(first, &mut first_out), (second, &mut second_out)],
                );
            } else {
                crate::simd::matmul_q8_0_batch(&input, seq_len, first, &mut first_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, second, &mut second_out);
            }
        });
        Ok((
            CpuTensor::from_data(vec![seq_len, first.out_features()], first_out),
            CpuTensor::from_data(vec![seq_len, second.out_features()], second_out),
        ))
    }

    fn matmul_q8_0_packed_pair(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeightVnni,
        second: &QuantizedWeightVnni,
    ) -> Result<Option<(CpuTensor, CpuTensor)>, CpuError> {
        if x.ndim() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: input must be 2D, got shape {:?}",
                x.shape()
            )));
        }
        let (rows, in_features) = (x.shape()[0], x.shape()[1]);
        if in_features != first.in_features() || in_features != second.in_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: inner dims must match (got {}, {} and {})",
                in_features,
                first.in_features(),
                second.in_features()
            )));
        }
        if first.out_features() != second.out_features() {
            return Err(CpuError::ShapeMismatch(format!(
                "matmul_q8_0_packed_pair: output dims must match (got {} and {})",
                first.out_features(),
                second.out_features()
            )));
        }

        // Real-shape A/B measurements show the packed matrix-vector schedule
        // wins for decode and very short prompts, while the generic tiled
        // batch kernel is faster from eight rows onward. Multi-row packed
        // dispatch also needs enough workers to offset its per-row scheduling.
        if rows > 6 || (rows > 1 && rayon::current_num_threads() < 4) {
            return Ok(None);
        }

        let output_features = first.out_features();
        let output_len = rows.checked_mul(output_features).ok_or_else(|| {
            CpuError::ShapeMismatch("matmul_q8_0_packed_pair: output size overflow".into())
        })?;
        let mut first_out = vec![0.0; output_len];
        let mut second_out = vec![0.0; output_len];
        let encoded_row_len = crate::quant::q8_0_encoded_len(in_features);
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            for row in 0..rows {
                let input_row = &input[row * encoded_row_len..(row + 1) * encoded_row_len];
                let output_range = row * output_features..(row + 1) * output_features;
                crate::simd::matmul_q8_0_decode_packed16_parallel(
                    input_row,
                    first,
                    &mut first_out[output_range.clone()],
                );
                crate::simd::matmul_q8_0_decode_packed16_parallel(
                    input_row,
                    second,
                    &mut second_out[output_range],
                );
            }
        });
        Ok(Some((
            CpuTensor::from_data(vec![rows, output_features], first_out),
            CpuTensor::from_data(vec![rows, output_features], second_out),
        )))
    }

    fn matmul_q8_0_triple(
        &self,
        x: &CpuTensor,
        first: &QuantizedWeight,
        second: &QuantizedWeight,
        third: &QuantizedWeight,
    ) -> Result<(CpuTensor, CpuTensor, CpuTensor), CpuError> {
        let (seq_len, first_len) = q8_matmul_output_len(x, first)?;
        let (_, second_len) = q8_matmul_output_len(x, second)?;
        let (_, third_len) = q8_matmul_output_len(x, third)?;
        let mut first_out = vec![0.0f32; first_len];
        let mut second_out = vec![0.0f32; second_len];
        let mut third_out = vec![0.0f32; third_len];
        Q8_0_DECODE_INPUT.with(|input| {
            let mut input = input.borrow_mut();
            crate::quant::quantize_q8_0_into(x.data(), &mut input);
            if seq_len == 1 {
                crate::simd::matmul_q8_0_decode_rows(
                    &input,
                    &mut [
                        (first, &mut first_out),
                        (second, &mut second_out),
                        (third, &mut third_out),
                    ],
                );
            } else {
                crate::simd::matmul_q8_0_batch(&input, seq_len, first, &mut first_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, second, &mut second_out);
                crate::simd::matmul_q8_0_batch(&input, seq_len, third, &mut third_out);
            }
        });
        Ok((
            CpuTensor::from_data(vec![seq_len, first.out_features()], first_out),
            CpuTensor::from_data(vec![seq_len, second.out_features()], second_out),
            CpuTensor::from_data(vec![seq_len, third.out_features()], third_out),
        ))
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(a.add(b))
    }

    fn softmax(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.par_softmax())
    }
    fn gelu(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.gelu())
    }

    fn gelu_tanh(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.par_gelu_tanh())
    }

    fn layer_norm(
        &self,
        x: &CpuTensor,
        weight: &CpuTensor,
        bias: &CpuTensor,
        eps: f32,
    ) -> Result<CpuTensor, CpuError> {
        Ok(x.layer_norm(weight, bias, eps))
    }
    fn index_select(&self, x: &CpuTensor, index: usize) -> Result<CpuTensor, Self::Error> {
        Ok(x.index_select(index)?)
    }
    fn row_as_2d(&self, x: &CpuTensor, index: usize) -> Result<CpuTensor, Self::Error> {
        Ok(x.row_as_2d(index)?)
    }
    fn assign_row(&self, dst: &mut CpuTensor, index: usize, src: &CpuTensor) {
        dst.assign_row(index, src);
    }
    fn assign_row_from_table(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &CpuTensor,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        let cols =
            validate_row_copy_shapes("assign_row_from_table", dst, dst_index, table, table_index)?;
        let dst_start = dst_index * cols;
        let table_start = table_index * cols;
        dst.data_mut()[dst_start..dst_start + cols]
            .copy_from_slice(&table.data()[table_start..table_start + cols]);
        Ok(())
    }
    fn assign_row_from_half(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &crate::half_weight::HalfWeight,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        if table.dims().len() != 2 {
            return Err(CpuError::ShapeMismatch(format!(
                "assign_row_from_half: table must be 2D, got {:?}",
                table.dims()
            )));
        }
        validate_quant_row_shapes(
            "assign_row_from_half",
            dst,
            dst_index,
            table.rows(),
            table.row_len(),
            table_index,
        )?;
        let cols = table.row_len();
        let start = dst_index * cols;
        table.dequantize_row(table_index, &mut dst.data_mut()[start..start + cols]);
        Ok(())
    }
    fn assign_row_from_q8_0(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &QuantizedWeight,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        validate_quant_row_shapes(
            "assign_row_from_q8_0",
            dst,
            dst_index,
            table.out_features(),
            table.in_features(),
            table_index,
        )?;
        let cols = table.in_features();
        let start = dst_index * cols;
        table.dequantize_row(table_index, &mut dst.data_mut()[start..start + cols]);
        Ok(())
    }
    fn assign_row_from_k(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        table: &crate::quant_k::KQuantWeight,
        table_index: usize,
    ) -> Result<(), Self::Error> {
        validate_quant_row_shapes(
            "assign_row_from_k",
            dst,
            dst_index,
            table.out_features(),
            table.in_features(),
            table_index,
        )?;
        let cols = table.in_features();
        let start = dst_index * cols;
        table.dequantize_row(table_index, &mut dst.data_mut()[start..start + cols]);
        Ok(())
    }
    fn assign_row_sum_from_tables(
        &self,
        dst: &mut CpuTensor,
        dst_index: usize,
        lhs_table: &CpuTensor,
        lhs_index: usize,
        rhs_table: &CpuTensor,
        rhs_index: usize,
    ) -> Result<(), Self::Error> {
        let cols = validate_row_copy_shapes(
            "assign_row_sum_from_tables",
            dst,
            dst_index,
            lhs_table,
            lhs_index,
        )?;
        validate_row_copy_shapes(
            "assign_row_sum_from_tables",
            dst,
            dst_index,
            rhs_table,
            rhs_index,
        )?;
        let dst_start = dst_index * cols;
        let lhs_start = lhs_index * cols;
        let rhs_start = rhs_index * cols;
        let dst_row = &mut dst.data_mut()[dst_start..dst_start + cols];
        let lhs_row = &lhs_table.data()[lhs_start..lhs_start + cols];
        let rhs_row = &rhs_table.data()[rhs_start..rhs_start + cols];
        crate::simd::add(lhs_row, rhs_row, dst_row);
        Ok(())
    }
    fn slice_cols(&self, x: &Self::Tensor, start: usize, end: usize) -> Self::Tensor {
        x.slice_cols(start, end)
    }
    fn shape<'a>(&self, x: &'a CpuTensor) -> &'a [usize] {
        x.shape()
    }
    fn data<'a>(&self, x: &'a Self::Tensor) -> &'a [f32] {
        x.data()
    }
    fn scale_in_place(&self, x: &mut CpuTensor, scale: f32) {
        for value in x.data_mut() {
            *value *= scale;
        }
    }
    fn softcap_in_place(&self, x: &mut CpuTensor, cap: f32) -> Result<(), CpuError> {
        for value in x.data_mut() {
            *value = (*value / cap).tanh() * cap;
        }
        Ok(())
    }
    fn load_from_cpu(&self, data: Vec<f32>, shape: &[usize]) -> Result<CpuTensor, Self::Error> {
        Ok(CpuTensor::from_data(shape.to_vec(), data))
    }

    fn into_cpu_data(&self, x: CpuTensor) -> Vec<f32> {
        x.data
    }
    fn add_broadcast(&self, x: &CpuTensor, bias: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.add_broadcast(bias))
    }

    fn causal_attention(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        spec: AttentionSpec,
    ) -> Result<CpuTensor, CpuError> {
        let seq_len = validate_attention_inputs(q, k, v, spec)?;
        let embed_dim = spec.n_heads * spec.head_dim;
        let kv_dim = spec.n_kv_heads * spec.head_dim;
        let n_repeat = validate_gqa(spec.n_heads, spec.n_kv_heads)?;
        let scale = (spec.head_dim as f32).sqrt().recip();

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // precompute per-position block start for block-diagonal masking
        let block_start: Vec<usize> = if let Some(boundaries) = spec.block_boundaries {
            let mut starts = vec![0usize; seq_len];
            let mut current_start = 0usize;
            let mut bi = 0usize;
            for (i, s) in starts.iter_mut().enumerate() {
                while bi < boundaries.len() && boundaries[bi] <= i {
                    current_start = boundaries[bi];
                    bi += 1;
                }
                *s = current_start;
            }
            starts
        } else {
            vec![0usize; seq_len]
        };
        let use_blocks = spec.block_boundaries.is_some();

        let parallel_attention =
            should_parallel_attention(spec.n_heads, seq_len, seq_len, spec.head_dim);
        if parallel_attention && seq_len > 1 {
            // Prefill rows are independent once the causal range is known.
            // Writing one complete output row per Rayon job avoids the old
            // per-head output buffers and the full-size scatter copy.
            let mut out = vec![0.0f32; seq_len * embed_dim];
            out.par_chunks_mut(embed_dim)
                .enumerate()
                .for_each(|(i, out_row)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        qk_row.resize(seq_len, 0.0);
                        let start = if use_blocks { block_start[i] } else { 0 };
                        for h in 0..spec.n_heads {
                            let head_out = &mut out_row[h * spec.head_dim..(h + 1) * spec.head_dim];
                            prefill_attention_row_head(
                                q_data,
                                k_data,
                                v_data,
                                i,
                                h,
                                embed_dim,
                                spec.head_dim,
                                kv_dim,
                                n_repeat,
                                scale,
                                start,
                                qk_row.as_mut_slice(),
                                head_out,
                            );
                        }
                    });
                });
            return Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out));
        }

        if parallel_attention {
            // seq_len == 1 here: the single output row is split across the
            // heads directly (no per-head buffers, no scatter; each head
            // writes its own disjoint columns of the one row).
            debug_assert_eq!(seq_len, 1);
            let mut out = vec![0.0f32; seq_len * embed_dim];
            out.par_chunks_mut(spec.head_dim)
                .enumerate()
                .for_each(|(h, head_out)| {
                    ATTENTION_SCORE_SCRATCH.with(|qk_row| {
                        let mut qk_row = qk_row.borrow_mut();
                        qk_row.resize(seq_len, 0.0);
                        let start = if use_blocks { block_start[0] } else { 0 };
                        prefill_attention_row_head(
                            q_data,
                            k_data,
                            v_data,
                            0,
                            h,
                            embed_dim,
                            spec.head_dim,
                            kv_dim,
                            n_repeat,
                            scale,
                            start,
                            qk_row.as_mut_slice(),
                            head_out,
                        );
                    });
                });
            return Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out));
        }

        let mut out = vec![0.0f32; seq_len * embed_dim];
        let mut qk_row = vec![0.0f32; seq_len];

        for h in 0..spec.n_heads {
            for (i, &start) in block_start.iter().enumerate() {
                let start = if use_blocks { start } else { 0 };
                let out_offset = i * embed_dim + h * spec.head_dim;
                prefill_attention_row_head(
                    q_data,
                    k_data,
                    v_data,
                    i,
                    h,
                    embed_dim,
                    spec.head_dim,
                    kv_dim,
                    n_repeat,
                    scale,
                    start,
                    &mut qk_row,
                    &mut out[out_offset..out_offset + spec.head_dim],
                );
            }
        }

        Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out))
    }

    fn cached_causal_attention(
        &self,
        q: &CpuTensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
    ) -> Result<CpuTensor, CpuError> {
        let mut qk_row = Vec::with_capacity(spec.max_seq_len);
        self.cached_causal_attention_with_scratch(q, cached_k, cached_v, spec, &mut qk_row)
    }

    fn cached_causal_attention_with_scratch(
        &self,
        q: &CpuTensor,
        cached_k: &[f16],
        cached_v: &[f16],
        spec: CachedAttentionSpec,
        qk_row: &mut Vec<f32>,
    ) -> Result<CpuTensor, CpuError> {
        let seq_len = validate_cached_attention_inputs(q, cached_k, cached_v, &spec)?;
        let embed_dim = spec.n_heads * spec.head_dim;
        let mut out = vec![0.0f32; seq_len * embed_dim];
        cached_attention_into(q, cached_k, cached_v, &spec, seq_len, qk_row, &mut out)?;
        Ok(CpuTensor::from_data(vec![seq_len, embed_dim], out))
    }

    fn rms_norm(&self, x: &CpuTensor, weight: &CpuTensor, eps: f32) -> Result<CpuTensor, CpuError> {
        Ok(x.rms_norm(weight, eps))
    }

    fn silu(&self, x: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(x.silu())
    }

    fn elemul(&self, a: &CpuTensor, b: &CpuTensor) -> Result<CpuTensor, CpuError> {
        Ok(a.elemul(b))
    }

    fn apply_rotary_emb(
        &self,
        x: &CpuTensor,
        cos: &CpuTensor,
        sin: &CpuTensor,
        start_pos: usize,
    ) -> Result<CpuTensor, CpuError> {
        Ok(x.apply_rotary_emb(cos, sin, start_pos))
    }
}

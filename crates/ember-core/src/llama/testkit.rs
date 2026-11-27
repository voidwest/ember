use super::*;
use crate::quant::{QuantizedWeight, Q8_0_BLOCK_SIZE, Q8_0_TYPE_SIZE};

pub fn q8_linear(out_features: usize, in_features: usize, seed: usize) -> Linear<CpuBackend> {
    assert!(in_features.is_multiple_of(Q8_0_BLOCK_SIZE));
    let blocks = out_features * in_features / Q8_0_BLOCK_SIZE;
    let mut data = Vec::with_capacity(blocks * Q8_0_TYPE_SIZE);
    for block in 0..blocks {
        let scale = half::f16::from_f32(0.005 + (block % 7) as f32 * 0.001);
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        for index in 0..Q8_0_BLOCK_SIZE {
            let quant = ((block * 17 + index * 13 + seed) % 31) as i8 - 15;
            data.push(quant as u8);
        }
    }
    Linear::new_q8_0(
        QuantizedWeight::try_new(data, vec![out_features, in_features]).unwrap(),
        None,
    )
}

/// Synthetic Q8_0 Llama with two query heads of `embed_dim / 2` and one
/// KV head; `embed_dim / 2` must be a multiple of the Q8_0 block size.
pub fn q8_llama(n_layers: usize, embed_dim: usize, vocab_size: usize) -> Llama<CpuBackend> {
    let head_dim = embed_dim / 2;
    let n_heads = 2;
    let n_kv_heads = 1;
    let inter_dim = 2 * embed_dim;
    let max_seq_len = 8;
    let blocks = (0..n_layers)
        .map(|layer| {
            let seed = layer * 8 + 1;
            let (rope_cos, rope_sin) =
                crate::tensor::compute_rope_freqs(max_seq_len, head_dim, 10_000.0, None);
            let attention = LlamaAttention::new(
                q8_linear(embed_dim, embed_dim, seed),
                q8_linear(head_dim, embed_dim, seed + 1),
                q8_linear(head_dim, embed_dim, seed + 2),
                q8_linear(embed_dim, embed_dim, seed + 3),
                rope_cos,
                rope_sin,
                n_heads,
                n_kv_heads,
                head_dim,
                RopeLayout::AdjacentPair,
                QkNormOrder::AfterRope,
                None,
                None,
            );
            let mlp = LlamaMlp::new(
                q8_linear(inter_dim, embed_dim, seed + 4),
                q8_linear(inter_dim, embed_dim, seed + 5),
                q8_linear(embed_dim, inter_dim, seed + 6),
            );
            LlamaBlock::new(
                CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
                attention,
                CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
                mlp,
                1e-5,
            )
        })
        .collect();
    let embedding = (0..vocab_size * embed_dim)
        .map(|index| ((index * 19 % 101) as f32 - 50.0) * 0.002)
        .collect();
    Llama {
        embed_tokens: LlamaEmbedding::F32(CpuTensor::from_data(
            vec![vocab_size, embed_dim],
            embedding,
        )),
        blocks,
        norm: CpuTensor::from_data(vec![embed_dim], vec![1.0; embed_dim]),
        head: q8_linear(vocab_size, embed_dim, 8),
        config: LlamaConfig {
            n_layers,
            n_heads,
            n_kv_heads,
            embed_dim,
            head_dim,
            max_seq_len,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            rope_layout: RopeLayout::AdjacentPair,
            qk_norm_order: QkNormOrder::AfterRope,
            vocab_size,
        },
        fast_decode_inter_dim: Some(inter_dim),
        plan_cache: OnceLock::new(),
        plan_provenance: RefCell::new((None, None, None)),
        k_decisions: BTreeMap::new(),
        head_tied: false,
        execution_mode: RefCell::new(ExecutionMode::Reference),
        decode_state: RefCell::new(None),
        packing_ns: 0,
        packing_interleaved_ns: 0,
        rope_frequency_factors: None,
    }
}

/// [`q8_llama`] with the allocation-free Q8_0 fast path disabled, so
/// decode runs the selected execution mode.
pub fn q8_llama_without_fast_decode(n_layers: usize) -> Llama<CpuBackend> {
    let mut model = q8_llama(n_layers, 32, 32);
    model.fast_decode_inter_dim = None;
    model
}

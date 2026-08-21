//! Unit tests for the multimodal modules (no real models needed).

use ember::backend::CpuBackend;
use ember::model::Linear;
use ember::multimodal::assembler::{EmbeddingAssembler, ImageFeatures, SmolVlmAssembler};
use ember::multimodal::image::{decode_rgb, preprocess, resize, ImagePreprocessConfig, Resample};
use ember::multimodal::vision::{bidirectional_attention, PixelShuffleConnector};
use ember::tensor::CpuTensor;

// ---------------------------------------------------------------------------
// image preprocessing
// ---------------------------------------------------------------------------

fn rgb_image(width: usize, height: usize) -> CpuTensor {
    // deterministic gradient CHW image with uint8-range values
    let mut data = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            // integer-valued (uint8-representable) values so PNG roundtrips
            data[y * width + x] = ((x * 255) / width.max(1)) as u8 as f32;
            data[height * width + y * width + x] = ((y * 255) / height.max(1)) as u8 as f32;
            data[2 * height * width + y * width + x] =
                ((x + y) * 255 / (width + height).max(1)) as u8 as f32;
        }
    }
    CpuTensor::from_data(vec![3, height, width], data)
}

#[test]
fn lanczos_resize_is_deterministic_and_sized() {
    let img = rgb_image(64, 48);
    let a = resize(&img, 128, 96, Resample::Lanczos).unwrap();
    let b = resize(&img, 128, 96, Resample::Lanczos).unwrap();
    assert_eq!(a.shape(), &[3, 96, 128]);
    assert_eq!(a.data(), b.data(), "resize must be deterministic");
    // uint8-range output for uint8-range input
    assert!(a.data().iter().all(|&v| (0.0..=255.0).contains(&v)));
    // downscale keeps mean approximately
    let mean_in = img.data().iter().sum::<f32>() / img.len() as f32;
    let mean_out = a.data().iter().sum::<f32>() / a.len() as f32;
    assert!((mean_in - mean_out).abs() < 3.0);
}

#[test]
fn lanczos_matches_pillow_reference_case() {
    // A tiny case whose PIL output we can reason about: identity resize
    // (same size) must reproduce the input exactly (PIL short-circuits or
    // the kernel is exact for scale 1.0).
    let img = rgb_image(16, 16);
    let same = resize(&img, 16, 16, Resample::Lanczos).unwrap();
    assert_eq!(same.data(), img.data());
}

#[test]
fn preprocess_splits_into_tiles_with_global() {
    let img = rgb_image(640, 384); // > 512 longest edge
    let config = ImagePreprocessConfig {
        resize_longest_edge: Some(1024),
        tile_size: Some(256),
        resample: Resample::Lanczos,
        rescale_factor: 1.0 / 255.0,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let pp = preprocess(&img, &config).unwrap();
    // 640x384 -> longest edge 1024: 1024x614 -> tiles (614-256)/256+1 = 2
    // rows x (1024-256)/256+1 = 4 cols (partial strips dropped, exactly the
    // reference tiling) + global tile = 9 tiles
    assert_eq!(pp.tiles.shape(), &[9, 3, 256, 256]);
    assert_eq!(pp.tile_grid, (2, 4));
    assert!(pp.has_global_tile);
    // normalized range roughly [-1, 1]
    let v = pp.tiles.data();
    assert!(v.iter().all(|&x| (-1.0 - 1e-3..=1.0 + 1e-3).contains(&x)));
    // mask all valid
    assert!(pp.mask.data().iter().all(|&m| m == 1.0));
}

#[test]
fn preprocess_small_image_no_split() {
    let img = rgb_image(64, 64);
    let config = ImagePreprocessConfig {
        resize_longest_edge: Some(128),
        tile_size: Some(256),
        resample: Resample::Lanczos,
        rescale_factor: 1.0 / 255.0,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let pp = preprocess(&img, &config).unwrap();
    assert_eq!(pp.tile_grid, (0, 0));
    assert!(!pp.has_global_tile);
    assert_eq!(pp.tiles.shape(), &[1, 3, 128, 128]);
}

#[test]
fn decode_rgb_roundtrip_via_png() {
    let path = std::env::temp_dir().join("ember_mm_decode_test.png");
    let img = rgb_image(8, 6);
    // write a PNG using the image crate and decode it back
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        use image::ImageEncoder;
        let enc = image::codecs::png::PngEncoder::new(&mut buf);
        // PngEncoder expects interleaved HWC RGB; our tensor is CHW
        let (h, w) = (6usize, 8usize);
        let mut u8data = vec![0u8; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    u8data[(y * w + x) * 3 + c] = img.data()[c * h * w + y * w + x] as u8;
                }
            }
        }
        enc.write_image(&u8data, 8, 6, image::ExtendedColorType::Rgb8)
            .unwrap();
    }
    std::fs::write(&path, buf.into_inner()).unwrap();
    let decoded = decode_rgb(&path).unwrap();
    assert_eq!(decoded.shape(), &[3, 6, 8]);
    // lossless PNG: values identical
    for (a, b) in decoded.data().iter().zip(img.data().iter()) {
        assert!(
            (a - b).abs() < 0.6,
            "PNG decode must be lossless: {a} vs {b}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// vision primitives
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::needless_range_loop)] // brute-force reference mirrors the math verbatim
fn bidirectional_attention_is_full_attention() {
    // With a constant key/value set, every query attends to all keys
    // equally: the output is the mean of v rows (scaled softmax over equal
    // scores). Compare against a brute-force reference.
    let seq = 5;
    let embed = 4;
    let n_heads = 2;
    let q = CpuTensor::from_data(
        vec![seq, embed],
        (0..seq * embed).map(|i| (i as f32) * 0.1).collect(),
    );
    let k = CpuTensor::from_data(
        vec![seq, embed],
        (0..seq * embed)
            .map(|i| ((i * 7) % 11) as f32 * 0.05)
            .collect(),
    );
    let v = CpuTensor::from_data(
        vec![seq, embed],
        (0..seq * embed)
            .map(|i| ((i * 3) % 13) as f32 * 0.1)
            .collect(),
    );
    let got = bidirectional_attention(&q, &k, &v, n_heads).unwrap();

    // brute-force per head
    let head_dim = embed / n_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut expected = vec![0.0f32; seq * embed];
    for h in 0..n_heads {
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for j in 0..seq {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q.data()[i * embed + h * head_dim + d]
                        * k.data()[j * embed + h * head_dim + d];
                }
                scores[j] = s * scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; seq];
            let mut sum = 0.0f32;
            for (e, s) in exps.iter_mut().zip(scores.iter()) {
                *e = (s - max_s).exp();
                sum += *e;
            }
            for j in 0..seq {
                let w = exps[j] / sum;
                for d in 0..head_dim {
                    expected[i * embed + h * head_dim + d] +=
                        w * v.data()[j * embed + h * head_dim + d];
                }
            }
        }
    }
    for (g, e) in got.data().iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-4, "attention mismatch: {g} vs {e}");
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // brute-force reference mirrors the HF index math
fn pixel_shuffle_matches_hf_semantics() {
    // scale=2 over a 4x4 grid: 16 tokens x 4 dims -> 4 tokens x 16 dims.
    // Hand-compute the HF permutation for one case.
    let scale = 2;
    let num_patches = 16;
    let embed = 2;
    let mut data = vec![0.0f32; num_patches * embed];
    for i in 0..num_patches * embed {
        data[i] = i as f32;
    }
    let x = CpuTensor::from_data(vec![num_patches, embed], data);
    let w = CpuTensor::from_data(
        vec![embed * scale * scale, 8],
        (0..embed * scale * scale * 8)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect(),
    );
    let proj = Linear::new(w.clone(), None);
    let connector = PixelShuffleConnector {
        scale_factor: scale,
        proj,
    };
    let backend = CpuBackend;
    let out = connector.forward(&backend, &x, num_patches).unwrap();
    assert_eq!(out.shape(), &[4, 8]);

    // brute force via the same index math as HF's pixel_shuffle:
    let side = 4;
    let s = scale;
    let tokens_per = num_patches / (s * s);
    let mut shuffled = vec![0.0f32; tokens_per * embed * s * s];
    for py in 0..side {
        for px in 0..side {
            let src_row = py * side + px;
            let ny = py / s;
            let nx = px / s;
            let dst_row = ny * (side / s) + nx;
            for e in 0..embed {
                let src_v = x.data()[src_row * embed + e];
                let dst_feature = (py % s) * (s * embed) + (px % s) * embed + e;
                shuffled[dst_row * (embed * s * s) + dst_feature] = src_v;
            }
        }
    }
    let mut expected = vec![0.0f32; tokens_per * 8];
    for r in 0..tokens_per {
        for c in 0..8 {
            let mut acc = 0.0;
            for k in 0..embed * s * s {
                acc += shuffled[r * (embed * s * s) + k] * w.data()[k * 8 + c];
            }
            expected[r * 8 + c] = acc;
        }
    }
    for (g, e) in out.data().iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-4, "pixel shuffle mismatch: {g} vs {e}");
    }
}

// ---------------------------------------------------------------------------
// assembler
// ---------------------------------------------------------------------------

#[test]
fn smolvlm_template_and_expansion() {
    let asm = SmolVlmAssembler::default();
    let text = asm.render_chat_template("What is this?", true);
    assert_eq!(
        text,
        "<|im_start|>User:What is this?<end_of_utterance>\nAssistant:"
    );
    let text2 = asm.render_chat_template("Hello", false);
    assert_eq!(
        text2,
        "<|im_start|>User: Hello<end_of_utterance>\nAssistant:"
    );

    let no_split = asm.expand_image_placeholder((0, 0));
    assert_eq!(
        no_split,
        format!(
            "<fake_token_around_image><global-img>{}<fake_token_around_image>",
            "<image>".repeat(64)
        )
    );

    let grid = asm.expand_image_placeholder((2, 3));
    // two rows of three tiles, then blank line + global
    let mut expected = String::new();
    for r in 0..2 {
        for c in 0..3 {
            expected.push_str("<fake_token_around_image>");
            expected.push_str(&format!("<row_{}_col_{}>", r + 1, c + 1));
            expected.push_str(&"<image>".repeat(64));
        }
        expected.push('\n');
    }
    expected.push('\n');
    expected.push_str("<fake_token_around_image><global-img>");
    expected.push_str(&"<image>".repeat(64));
    expected.push_str("<fake_token_around_image>");
    assert_eq!(grid, expected);
    // 2*3*64 + 6 fake + 2 row-col + 1 global + 1 fake = token count check
    assert_eq!(grid.matches("<image>").count(), 2 * 3 * 64 + 64);
}

#[test]
fn smolvlm_assembler_scatters_features() {
    // build a tiny embedding table and tokenizer-free assemble with a
    // hand-rolled tokenizer stub is overkill; instead verify the scatter
    // mechanics through the real tokenizer + a small table.
    let path = std::env::temp_dir().join("ember_mm_tokenizer_test.json");
    // minimal BPE tokenizer with the special tokens the assembler needs
    // (all 36 row/col tokens, built programmatically)
    let mut added = vec![
        serde_json::json!({"id": 1, "content": "<|im_start|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 2, "content": "<|im_end|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 3, "content": "<end_of_utterance>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 4, "content": "<fake_token_around_image>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 5, "content": "<global-img>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 6, "content": "<image>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
    ];
    let mut vocab = serde_json::Map::new();
    for (i, piece) in [
        "a", "b", "c", "d", "e", "f", "g", "User", "What", "is", "this", "?", ":",
    ]
    .iter()
    .enumerate()
    {
        vocab.insert((*piece).to_string(), serde_json::json!(i));
    }
    vocab.insert("<|im_start|>".into(), serde_json::json!(1));
    vocab.insert("<|im_end|>".into(), serde_json::json!(2));
    vocab.insert("<end_of_utterance>".into(), serde_json::json!(3));
    vocab.insert("<fake_token_around_image>".into(), serde_json::json!(4));
    vocab.insert("<global-img>".into(), serde_json::json!(5));
    vocab.insert("<image>".into(), serde_json::json!(6));
    for i in 0..36 {
        let (r, c) = (i / 6 + 1, i % 6 + 1);
        let content = format!("<row_{r}_col_{c}>");
        added.push(serde_json::json!({"id": 7 + i, "content": content, "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}));
        vocab.insert(content, serde_json::json!(7 + i));
    }
    let tok_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "post_processor": null
    });
    std::fs::write(&path, serde_json::to_string(&tok_json).unwrap()).unwrap();
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&path).unwrap();

    // embedding table with distinct rows
    let embed_dim = 4;
    let vocab = 20;
    let mut table = vec![0.0f32; vocab * embed_dim];
    for v in 0..vocab {
        for e in 0..embed_dim {
            table[v * embed_dim + e] = (v * 10 + e) as f32;
        }
    }
    let table = CpuTensor::from_data(vec![vocab, embed_dim], table);
    let table = ember::llama::LlamaEmbedding::F32(table);

    // image features: 2 tiles x 1 token each (image_seq_len = 1)
    let asm = SmolVlmAssembler {
        image_seq_len: 1,
        ..Default::default()
    };
    let features = CpuTensor::from_data(
        vec![2, embed_dim],
        vec![100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],
    );
    let backend = CpuBackend;
    // image tokens: 3 total (2 tiles + global) but features has 2 rows ->
    // the assembler must fail closed on the count mismatch
    let result = asm.assemble(
        &backend,
        &tokenizer,
        "<image>What",
        &[ImageFeatures {
            features,
            tile_grid: (1, 2),
        }],
        &table,
    );
    assert!(
        result.is_err(),
        "feature count mismatch must fail closed, got {:?}",
        result
    );

    // now with matching features (3 rows) it must succeed and scatter
    let features3 = CpuTensor::from_data(
        vec![3, embed_dim],
        vec![
            100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0, 300.0, 301.0, 302.0, 303.0,
        ],
    );
    let assembled = asm
        .assemble(
            &backend,
            &tokenizer,
            "<image>What",
            &[ImageFeatures {
                features: features3,
                tile_grid: (1, 2),
            }],
            &table,
        )
        .unwrap();
    let ids = assembled.input_ids;
    // rendered: <|im_start|>User:<fake><row_1_col_1><image><fake><row_1_col_2><image>\n\n<fake><global-img><image><fake>What<end_of_utterance>\nAssistant:
    let n_img = ids.iter().filter(|&&t| t == 6).count();
    assert_eq!(n_img, 3);
    // the three <image> positions must carry the feature rows
    let emb = assembled.embeddings;
    let img_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == 6)
        .map(|(i, _)| i)
        .collect();
    for (k, &pos) in img_positions.iter().enumerate() {
        for e in 0..embed_dim {
            let expect = [100.0, 200.0, 300.0][k] + e as f32;
            let got = emb.data()[pos * embed_dim + e];
            assert!(
                (got - expect).abs() < 1e-5,
                "feature row {k} not scattered at pos {pos}: {got} != {expect}"
            );
        }
    }
}

#[test]
fn smolvlm_assembler_binds_multiple_images_in_order() {
    // reuse a minimal tokenizer like the scatter test
    let path = std::env::temp_dir().join("ember_mm_tokenizer_multi.json");
    let mut added = vec![
        serde_json::json!({"id": 1, "content": "<|im_start|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 2, "content": "<|im_end|>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 3, "content": "<end_of_utterance>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 4, "content": "<fake_token_around_image>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 5, "content": "<global-img>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
        serde_json::json!({"id": 6, "content": "<image>", "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}),
    ];
    let mut vocab = serde_json::Map::new();
    for (i, piece) in ["a", "b", "c"].iter().enumerate() {
        vocab.insert((*piece).to_string(), serde_json::json!(i));
    }
    for (id, piece) in [
        (1usize, "<|im_start|>"),
        (2, "<|im_end|>"),
        (3, "<end_of_utterance>"),
        (4, "<fake_token_around_image>"),
        (5, "<global-img>"),
        (6, "<image>"),
    ] {
        vocab.insert(piece.into(), serde_json::json!(id));
    }
    for i in 0..36 {
        let (r, c) = (i / 6 + 1, i % 6 + 1);
        let content = format!("<row_{r}_col_{c}>");
        added.push(serde_json::json!({"id": 7 + i, "content": content, "special": true, "lstrip": false, "rstrip": false, "single_word": false, "normalized": false}));
        vocab.insert(content, serde_json::json!(7 + i));
    }
    let tok_json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "post_processor": null
    });
    std::fs::write(&path, serde_json::to_string(&tok_json).unwrap()).unwrap();
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file(&path).unwrap();

    let embed_dim = 4;
    let vocab_size = 50;
    let mut table = vec![0.0f32; vocab_size * embed_dim];
    for v in 0..vocab_size {
        for e in 0..embed_dim {
            table[v * embed_dim + e] = (v * 10 + e) as f32;
        }
    }
    let table =
        ember::llama::LlamaEmbedding::F32(CpuTensor::from_data(vec![vocab_size, embed_dim], table));

    let asm = SmolVlmAssembler {
        image_seq_len: 1,
        ..Default::default()
    };
    let backend = CpuBackend;

    // two images, one tile each (features = 1 row per image: tiles+global=...
    // grid (0,0) means no split -> exactly one <image> token per image)
    let img_a = CpuTensor::from_data(vec![1, embed_dim], vec![111.0, 112.0, 113.0, 114.0]);
    let img_b = CpuTensor::from_data(vec![1, embed_dim], vec![222.0, 223.0, 224.0, 225.0]);
    let assembled = asm
        .assemble(
            &backend,
            &tokenizer,
            "Compare <image> versus <image> ok",
            &[
                ImageFeatures {
                    features: img_a,
                    tile_grid: (0, 0),
                },
                ImageFeatures {
                    features: img_b,
                    tile_grid: (0, 0),
                },
            ],
            &table,
        )
        .unwrap();
    let ids = assembled.input_ids;
    let img_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == 6)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(img_positions.len(), 2);
    // first placeholder carries image A's rows, second carries B's
    for (value, pos) in [(111.0f32, img_positions[0]), (222.0, img_positions[1])] {
        let got = assembled.embeddings.data()[pos * embed_dim];
        assert!(
            (got - value).abs() < 1e-5,
            "image binding out of order at pos {pos}: {got} != {value}"
        );
    }

    // mismatch: two placeholders but one image must fail closed
    let result = asm.assemble(
        &backend,
        &tokenizer,
        "<image> versus <image>",
        &[ImageFeatures {
            features: CpuTensor::from_data(vec![1, embed_dim], vec![0.0; 4]),
            tile_grid: (0, 0),
        }],
        &table,
    );
    assert!(
        result.is_err(),
        "placeholder/image count mismatch must fail"
    );

    // mismatch: one placeholder but two images must fail closed
    let result = asm.assemble(
        &backend,
        &tokenizer,
        "<image> alone",
        &[
            ImageFeatures {
                features: CpuTensor::from_data(vec![1, embed_dim], vec![0.0; 4]),
                tile_grid: (0, 0),
            },
            ImageFeatures {
                features: CpuTensor::from_data(vec![1, embed_dim], vec![0.0; 4]),
                tile_grid: (0, 0),
            },
        ],
        &table,
    );
    assert!(
        result.is_err(),
        "placeholder/image count mismatch must fail"
    );
}

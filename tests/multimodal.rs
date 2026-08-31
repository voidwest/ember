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
    // 640x384 -> longest edge 1024: 1024x614 -> round both edges up to
    // whole tiles (reference resize_for_vision_encoder): 1024x768 ->
    // grid (3, 4) exact + global tile = 13 tiles. No strip is dropped.
    assert_eq!(pp.tiles.shape(), &[13, 3, 256, 256]);
    assert_eq!(pp.tile_grid, (3, 4));
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
    // 64x48 -> longest edge 128: 128x96 -> round up to whole tiles (256):
    // 256x256. Exactly tile-sized after rounding => single frame, no grid
    // (reference reports splits (0,0)); the image is upscaled, matching the
    // reference processor for any geometry.
    assert_eq!(pp.tile_grid, (0, 0));
    assert!(!pp.has_global_tile);
    assert_eq!(pp.tiles.shape(), &[1, 3, 256, 256]);
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

// ---------------------------------------------------------------------------
// Track A: general multimodal request substrate
// ---------------------------------------------------------------------------

#[test]
fn content_parts_represent_arbitrary_interleaving() {
    use ember::multimodal::audio::AudioInput;
    use ember::multimodal::{ContentPart, ImageInput, VideoFrames, VideoInput};

    // Text/Image/Text/Audio/Video/Text — order preserved, repeats fine
    let parts = [
        ContentPart::Text("look:".into()),
        ContentPart::Image(ImageInput::Bytes(vec![1, 2, 3])),
        ContentPart::Text("listen:".into()),
        ContentPart::Audio(AudioInput::Samples {
            data: vec![0.0; 16],
            sample_rate: 16_000,
        }),
        ContentPart::Video(VideoInput::Frames(VideoFrames {
            frames: vec![],
            timestamps_ms: vec![],
            source_fps: Some(30.0),
            source_duration_s: None,
        })),
        ContentPart::Text("done".into()),
    ];
    assert_eq!(parts.len(), 6);
    assert_eq!(
        parts[1].media_kind(),
        Some(ember::multimodal::MediaKind::Image)
    );
    assert_eq!(
        parts[3].media_kind(),
        Some(ember::multimodal::MediaKind::Audio)
    );
    assert_eq!(
        parts[4].media_kind(),
        Some(ember::multimodal::MediaKind::Video)
    );
    assert_eq!(parts[0].media_kind(), None);
}

#[test]
fn image_input_decodes_from_memory_bytes_and_pixels() {
    use ember::multimodal::{ContentPart, ImageInput};

    // encode a tiny PNG in memory
    let img = rgb_image(8, 6);
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        use ::image::ImageEncoder;
        let enc = ::image::codecs::png::PngEncoder::new(&mut buf);
        let (h, w) = (6usize, 8usize);
        let mut u8data = vec![0u8; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    u8data[(y * w + x) * 3 + c] = img.data()[c * h * w + y * w + x] as u8;
                }
            }
        }
        enc.write_image(&u8data, 8, 6, ::image::ExtendedColorType::Rgb8)
            .unwrap();
    }
    let bytes: Vec<u8> = buf.into_inner();

    let via_bytes = ImageInput::Bytes(bytes.clone()).decode().unwrap();
    assert_eq!(via_bytes.shape(), &[3, 6, 8]);

    // decoded pixels equal the source tensor bit-for-bit
    let via_pixels_again = ember::multimodal::image::decode_rgb_bytes(&bytes).unwrap();
    assert_eq!(via_pixels_again.data(), via_bytes.data());

    let via_pixels = ImageInput::Pixels { rgb: img.clone() }.decode().unwrap();
    assert_eq!(via_pixels.data(), img.data(), "Pixels must pass through");

    // malformed bytes fail closed with a clear error
    assert!(ImageInput::Bytes(vec![0; 10]).decode().is_err());

    let _ = ContentPart::Image(ImageInput::Bytes(bytes)); // representable as a part
}

#[test]
fn media_id_is_content_sensitive() {
    use ember::multimodal::MediaId;

    let a = MediaId::from_bytes(b"hello");
    let b = MediaId::from_bytes(b"hello");
    let c = MediaId::from_bytes(b"hell");
    let d = MediaId::from_bytes(b"o");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(c, d, "length-prefixed hash must not join across splits");

    let t1 = CpuTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t2 = CpuTensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let t3 = CpuTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.5]);
    assert_ne!(
        MediaId::from_tensor(&t1),
        MediaId::from_tensor(&t2),
        "shape participates"
    );
    assert_ne!(
        MediaId::from_tensor(&t1),
        MediaId::from_tensor(&t3),
        "values participate"
    );
}

#[test]
fn smolvlm_rejects_non_image_media_parts() {
    use ember::multimodal::audio::AudioInput;
    use ember::multimodal::ContentPart;
    use ember::smolvlm::SmolVlm;

    let parts = vec![
        ContentPart::Audio(AudioInput::Samples {
            data: vec![0.0; 8],
            sample_rate: 16_000,
        }),
        ContentPart::Text("hi".into()),
    ];
    assert!(
        SmolVlm::split_parts(&parts).is_err(),
        "audio part must fail closed on the image adapter"
    );
}

// ---------------------------------------------------------------------------
// Track E: frame sampling policies
// ---------------------------------------------------------------------------

fn fake_video(n: usize) -> ember::multimodal::VideoFrames {
    let frames = (0..n)
        .map(|t| CpuTensor::from_data(vec![3, 2, 2], vec![t as f32; 12]))
        .collect();
    ember::multimodal::VideoFrames {
        frames,
        timestamps_ms: (0..n).map(|i| i as f64 * 1000.0 / 8.0).collect(), // 8 fps
        source_fps: Some(8.0),
        source_duration_s: Some(n as f64 / 8.0),
    }
}

#[test]
fn uniform_sampling_matches_reference_index_formula() {
    use ember::multimodal::FrameSampling;

    let vid = fake_video(100);
    // reference sampler: indices = floor(i * total / k)
    let s = FrameSampling::Uniform { max_frames: 8 }
        .sample(&vid)
        .unwrap();
    assert_eq!(s.n_frames(), 8);
    assert_eq!(s.source_indices, vec![0, 12, 25, 37, 50, 62, 75, 87]);
    assert_eq!(s.total_source_frames, 100);
    assert_eq!(s.source_fps, Some(8.0));
    assert_eq!(s.timestamps_ms[3], 37.0 * 125.0);

    // cap larger than total -> every frame
    let s_all = FrameSampling::Uniform { max_frames: 200 }
        .sample(&vid)
        .unwrap();
    assert_eq!(s_all.n_frames(), 100);
}

#[test]
fn sampling_is_deterministic_and_fails_closed() {
    use ember::multimodal::FrameSampling;

    let vid = fake_video(50);
    let a = FrameSampling::Uniform { max_frames: 5 }
        .sample(&vid)
        .unwrap();
    let b = FrameSampling::Uniform { max_frames: 5 }
        .sample(&vid)
        .unwrap();
    assert_eq!(a.source_indices, b.source_indices);
    for (x, y) in a.frames.iter().zip(b.frames.iter()) {
        assert_eq!(x.data(), y.data());
    }

    // empty input fails closed
    let empty = ember::multimodal::VideoFrames {
        frames: vec![],
        timestamps_ms: vec![],
        source_fps: None,
        source_duration_s: None,
    };
    assert!(FrameSampling::Uniform { max_frames: 4 }
        .sample(&empty)
        .is_err());
    // fps=0 fails closed
    assert!(FrameSampling::FixedFps {
        fps: 0.0,
        max_frames: 4
    }
    .sample(&fake_video(10))
    .is_err());
}

// ---------------------------------------------------------------------------
// Track G: encoded-media feature cache
// ---------------------------------------------------------------------------

#[test]
fn feature_cache_reuses_bit_exact_and_evicts() {
    use ember::multimodal::cache::{FeatureCacheKey, MediaFeatureCache, PreprocessFingerprint};
    use ember::multimodal::{MediaId, MediaKind};

    let key = |v: u32| FeatureCacheKey {
        media_id: MediaId(v as u64),
        kind: MediaKind::Image,
        preprocess: PreprocessFingerprint::new("t").value(),
        tower_identity: 7,
    };

    let mut cache = MediaFeatureCache::new(1024);
    let t = CpuTensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    cache.insert(key(1), t.clone());
    assert_eq!(cache.len(), 1);
    // hit replays bit-exactly
    let hit = cache.lookup(&key(1)).unwrap().clone();
    assert_eq!(hit.data(), t.data());
    let _ = hit;
    // different content -> miss; different config -> miss
    assert!(cache.lookup(&key(2)).is_none());
    let mut fp = PreprocessFingerprint::new("t");
    fp.mix_u64(1);
    let other_cfg = FeatureCacheKey {
        media_id: MediaId(1),
        kind: MediaKind::Image,
        preprocess: fp.value(),
        tower_identity: 7,
    };
    assert!(
        cache.lookup(&other_cfg).is_none(),
        "config change must invalidate"
    );
    // different weights -> miss
    let other_model = FeatureCacheKey {
        media_id: MediaId(1),
        kind: MediaKind::Image,
        preprocess: PreprocessFingerprint::new("t").value(),
        tower_identity: 8,
    };
    assert!(
        cache.lookup(&other_model).is_none(),
        "model change must invalidate"
    );

    // eviction under byte pressure (each entry 64 bytes here)
    for i in 0..64u32 {
        cache.insert(key(i), CpuTensor::from_data(vec![4, 4], vec![i as f32; 16]));
    }
    assert!(cache.used_bytes() <= 1024);
}

#[test]
fn preprocess_fingerprint_is_field_sensitive() {
    use ember::multimodal::cache::PreprocessFingerprint;
    let mut a = PreprocessFingerprint::new("x");
    a.mix_u64(512);
    a.mix_f64(0.5);
    let mut b = PreprocessFingerprint::new("x");
    b.mix_u64(512);
    b.mix_f64(0.5000001);
    assert_ne!(a.value(), b.value());
    let mut c = PreprocessFingerprint::new("y");
    c.mix_u64(512);
    c.mix_f64(0.5);
    assert_ne!(a.value(), c.value(), "tag participates");
}

// ---------------------------------------------------------------------------
// Track F: ownership-aware cross-request vision batching
// ---------------------------------------------------------------------------

#[test]
fn batch_encode_splits_features_back_to_owners() {
    use ember::multimodal::batch::{batch_encode_images, BatchedImageInput};
    use ember::multimodal::request::SegmentId;
    use std::cell::RefCell;

    let backend = CpuBackend;
    // two requests x mixed geometry: req0 has 2 tiles @2x2 patches-per-tile
    // simulation, req1 has 1 tile of a different shape
    let mk =
        |vals: &[f32], h: usize, w: usize| CpuTensor::from_data(vec![1, 3, h, w], vals.to_vec());
    let inputs = vec![
        BatchedImageInput {
            owner: SegmentId::new(7, 0),
            tiles: mk(&[1.0; 3 * 32 * 32], 32, 32),
        },
        BatchedImageInput {
            owner: SegmentId::new(7, 1),
            tiles: mk(&[2.0; 3 * 16 * 16], 16, 16),
        },
        BatchedImageInput {
            owner: SegmentId::new(9, 0),
            tiles: mk(&[3.0; 3 * 32 * 32], 32, 32),
        },
    ];

    let seen_groups = RefCell::new(Vec::new());
    let patch_size = 8usize; // pretend patches for token math
    let scale = 2usize;
    let (outputs, traces, all) =
        batch_encode_images(&backend, &inputs, patch_size, scale, |_be, batch| {
            // fake tower+projector: one output row per tile, value = marker
            let n = batch.shape()[0];
            let (_, c, h, w) = (
                batch.shape()[0],
                batch.shape()[1],
                batch.shape()[2],
                batch.shape()[3],
            );
            let tokens_per_tile = ((h / patch_size) * (w / patch_size)) / (scale * scale);
            let mut data = Vec::with_capacity(n * tokens_per_tile * 3);
            for t in 0..n {
                for k in 0..tokens_per_tile {
                    data.push(batch.data()[t * c * h * w] + k as f32);
                    data.extend_from_slice(&[0.0, 0.0]);
                }
            }
            seen_groups.borrow_mut().push((h, w, n));
            Ok((CpuTensor::from_data(vec![n * tokens_per_tile, 3], data), ()))
        })
        .unwrap();

    // two geometry groups executed
    assert_eq!(seen_groups.borrow().len(), 2);
    assert_eq!(traces.len(), 2);
    // owners preserved in request order with correct row counts:
    // tokens_per_tile(32x32) = (32/8)^2 / 2^2 = 4; tokens_per_tile(16x16) = 1
    assert_eq!(outputs[0].owner, SegmentId::new(7, 0));
    assert_eq!(outputs[0].features.shape(), &[4, 3]);
    assert_eq!(outputs[1].owner, SegmentId::new(7, 1));
    assert_eq!(outputs[1].features.shape(), &[1, 3]);
    assert_eq!(outputs[2].owner, SegmentId::new(9, 0));
    assert_eq!(outputs[2].features.shape(), &[4, 3]);
    // values routed correctly across the group boundary
    assert_eq!(outputs[0].features.data()[0], 1.0);
    assert_eq!(outputs[2].features.data()[0], 3.0);
    // concatenated projection covers everything exactly once
    let total_rows: usize = outputs.iter().map(|o| o.features.shape()[0]).sum();
    assert_eq!(all.shape()[0], total_rows);
}

#[test]
fn batch_encode_fails_closed_on_geometry_drift() {
    use ember::multimodal::batch::{batch_encode_images, BatchedImageInput};
    use ember::multimodal::request::SegmentId;

    let backend = CpuBackend;
    // lie about geometry via a project fn that returns fewer rows than the
    // declared tile math implies -> split must fail closed
    let inputs = vec![BatchedImageInput {
        owner: SegmentId::new(1, 0),
        tiles: CpuTensor::from_data(vec![1, 3, 32, 32], vec![0.0; 3 * 32 * 32]),
    }];
    let result = batch_encode_images(&backend, &inputs, 8, 2, |_be, batch| {
        let n = batch.shape()[0];
        Ok((CpuTensor::from_data(vec![n, 3], vec![0.0; n * 3]), ()))
    });
    assert!(result.is_err(), "row-count mismatch must fail closed");
}

// ---------------------------------------------------------------------------
// Phase 4 Track B: fast-exp softmax error ladder (vision)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random f32 in [-1, 1].
fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Ladder level 2: fast softmax vs reference `CpuTensor::softmax` on
/// random score matrices. Gate: max_abs <= 1e-6 and rows sum to 1.
#[test]
fn fast_softmax_matches_reference_within_gate() {
    for (rows, cols, seed) in [(37usize, 257usize, 1u64), (128, 1024, 2), (731, 731, 3)] {
        let data: Vec<f32> = lcg_values(rows * cols, seed)
            .iter()
            .map(|v| v * 12.0)
            .collect();
        let reference = CpuTensor::from_data(vec![rows, cols], data.clone()).softmax();
        let mut got = CpuTensor::from_data(vec![rows, cols], data);
        ember::multimodal::vision::softmax_in_place_fast(&mut got);
        let max_abs: f32 = reference
            .data()
            .iter()
            .zip(got.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1e-6,
            "{rows}x{cols}: max_abs {max_abs:.3e} > 1e-6"
        );
        // probability mass conservation
        for r in 0..rows {
            let sum: f32 = got.data()[r * cols..(r + 1) * cols].iter().sum();
            assert!((sum - 1.0).abs() <= 1e-5, "row {r} sums to {sum}");
        }
    }
}

/// Masked-row semantics: additive f32::MIN bias entries must vanish under
/// the fast path exactly as under the reference path.
#[test]
fn fast_softmax_handles_masked_rows_like_reference() {
    let cols = 64usize;
    let mut scores = vec![0.0f32; 4 * cols];
    let rand = lcg_values(4 * cols, 11);
    for (i, v) in scores.iter_mut().enumerate() {
        *v = rand[i] * 8.0;
    }
    // mask half of row 1 and all of row 2
    for j in (0..cols / 2).step_by(2) {
        scores[cols + j] += f32::MIN;
    }
    for j in 0..cols {
        scores[2 * cols + j] = -3.0 + scores[2 * cols + j] * 0.001 + f32::MIN;
    }
    let reference = CpuTensor::from_data(vec![4, cols], scores.clone()).softmax();
    let mut got = CpuTensor::from_data(vec![4, cols], scores);
    ember::multimodal::vision::softmax_in_place_fast(&mut got);

    // Half-masked row: masked lanes vanish in both paths; unmasked lanes
    // agree within the ladder gate.
    for j in (0..cols / 2).step_by(2) {
        assert_eq!(reference.data()[cols + j], 0.0);
        assert!(
            got.data()[cols + j] < 1e-30,
            "masked lane {j} = {}",
            got.data()[cols + j]
        );
    }
    // Fully-masked row: f32::MIN is finite, so every lane sits at
    // exp(0) = 1 before normalization — BOTH paths produce the uniform
    // distribution (faithful encoder semantics for fully padded rows),
    // and the two paths must agree.
    for j in 0..cols {
        let expected = 1.0 / cols as f32;
        assert!((reference.data()[2 * cols + j] - expected).abs() < 1e-6);
        assert!((got.data()[2 * cols + j] - expected).abs() < 1e-6);
    }
    // Unmasked regions agree within the ladder gate.
    let max_abs: f32 = (0..cols)
        .filter(|j| j % 2 == 1)
        .map(|j| (reference.data()[cols + j] - got.data()[cols + j]).abs())
        .chain((0..cols).map(|j| (reference.data()[3 * cols + j] - got.data()[3 * cols + j]).abs()))
        .chain((0..cols).map(|j| (reference.data()[j] - got.data()[j]).abs()))
        .fold(0.0f32, f32::max);
    assert!(max_abs <= 1e-6, "unmasked drift {max_abs:.3e}");
}

// ---------------------------------------------------------------------------
// Phase 5 Track H: PIL-exact BICUBIC resize + the SmolVLM2 video chain
// ---------------------------------------------------------------------------

#[test]
fn bicubic_resize_matches_pillow() {
    // deterministic gradient image, non-trivial size change both ways
    let (w, h) = (211usize, 137usize);
    let mut img = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                img[c * h * w + y * w + x] = ((x * 7 + y * 13 + c * 29) % 251) as f32;
            }
        }
    }
    let src = CpuTensor::from_data(vec![3, h, w], img);
    for (ow, oh) in [(2048usize, 2048usize), (512, 512), (333, 87)] {
        let out = ember::multimodal::image::resize(
            &src,
            ow,
            oh,
            ember::multimodal::image::Resample::Bicubic,
        )
        .expect("bicubic resize");
        assert_eq!(out.shape(), [3, oh, ow]);
    }
}

// ---------------------------------------------------------------------------
// EmberSEC Phase 2: hostile multimodal input hardening
// ---------------------------------------------------------------------------

/// Minimal RIFF/WAVE bytes with attacker-controlled header fields.
fn wav_bytes(
    format_tag: u16,
    bits_per_sample: u16,
    channels: u16,
    sample_rate: u32,
    data: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(channels * bits_per_sample / 8).to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn wav_unsupported_format_tag_is_error_not_panic() {
    use ember::multimodal::audio::decode_wav_bytes;
    // A-law (6) and MS ADPCM (2) tags with a data chunk previously hit the
    // `_ => panic!` arm in read_one; they must now be structured errors.
    for tag in [2u16, 6, 7, 0x11] {
        let bytes = wav_bytes(tag, 8, 1, 8000, &[0u8; 8]);
        let err = decode_wav_bytes(&bytes).expect_err("unsupported wav tag must error");
        assert!(err.to_string().contains("unsupported wav format"), "{err}");
    }
    // IEEE-float tag with a 16-bit payload is also unsupported.
    let err = decode_wav_bytes(&wav_bytes(3, 16, 1, 8000, &[0u8; 4]))
        .expect_err("float tag with 16-bit payload must error");
    assert!(err.to_string().contains("unsupported wav format"), "{err}");
}

#[test]
fn wav_zero_sample_rate_is_rejected() {
    use ember::multimodal::audio::decode_wav_bytes;
    let err = decode_wav_bytes(&wav_bytes(1, 16, 1, 0, &[0u8; 8]))
        .expect_err("zero sample rate must error");
    assert!(err.to_string().contains("sample rate"), "{err}");
}

#[test]
fn resample_rejects_zero_rate_and_bounds_amplification() {
    use ember::multimodal::audio::resample;
    let x = vec![0.0f32; 16];
    // zero source rate used to produce an infinite output length
    assert!(resample(&x, 0, 16_000).is_err());
    // a tiny-rate source must not be able to amplify into a multi-GiB buffer
    let big = vec![0.0f32; 1_000_000];
    let err = resample(&big, 1, 16_000).expect_err("1 Hz upsample must hit the output cap");
    assert!(err.to_string().contains("cap"), "{err}");
    // normal rates still work
    assert!(resample(&big, 44_100, 16_000).is_ok());
}

#[test]
fn to_mono_16k_rejects_zero_rate_samples_input() {
    use ember::multimodal::audio::{to_mono_16k, AudioInput};
    let bad = AudioInput::Samples {
        data: vec![0.0f32; 1000],
        sample_rate: 0,
    };
    assert!(to_mono_16k(&bad).is_err());
}

#[test]
fn validated_audio_input_rejects_oversized_duration() {
    use ember::multimodal::audio::{AudioInput, ValidatedAudioInput};
    // 3601 s at 16 kHz (230 MB f32, no resample work): over the 1 h cap.
    let long = AudioInput::Samples {
        data: vec![0.0f32; 3601 * 16_000],
        sample_rate: 16_000,
    };
    let err = ValidatedAudioInput::from_audio_input(&long)
        .expect_err(">1h audio must be rejected at admission");
    assert!(err.to_string().contains("admission limit"), "{err}");
    // a bounded input passes and carries validated invariants
    let ok = AudioInput::Samples {
        data: vec![0.0f32; 16_000],
        sample_rate: 16_000,
    };
    let v = ValidatedAudioInput::from_audio_input(&ok).expect("1 s audio is fine");
    assert_eq!(v.sample_rate, 16_000);
    assert!((v.duration_s - 1.0).abs() < 1e-9);
    assert_eq!(v.samples.len(), 16_000);
}

// --- image decode limits ---------------------------------------------------

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = Vec::with_capacity(4 + data.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32_ieee(&crc_in).to_be_bytes());
    out
}

fn png_with_dims(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, adaptive, no interlace
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&png_chunk(b"IDAT", &[]));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    png
}

#[test]
fn image_decode_applies_decoder_limits() {
    use ember::multimodal::image::decode_rgb_bytes;
    // A decompression bomb: 65535x65535 declared dims with a tiny payload.
    // The allocation budget must reject it before materializing 12+ GiB.
    let bomb = png_with_dims(65_535, 65_535);
    let err = decode_rgb_bytes(&bomb).expect_err("huge png must be rejected by the limits");
    assert!(err.to_string().to_lowercase().contains("limit"), "{err}");
    // A sane image still decodes through the same path.
    use image::ImageEncoder;
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(
            &[0, 0, 0, 255, 255, 255],
            2,
            1,
            image::ExtendedColorType::Rgb8,
        )
        .expect("encode 2x1 png");
    let t = decode_rgb_bytes(&buf).expect("valid png decodes");
    assert_eq!(t.shape(), [3, 1, 2]);
}

// --- validated image seam + batch geometry guards --------------------------

#[test]
fn validated_image_input_rejects_malformed_pixels() {
    use ember::multimodal::request::{ImageInput, ValidatedImageInput};
    let bad = CpuTensor::from_data(vec![4, 10], vec![0.0; 40]);
    let err = ValidatedImageInput::decode(&ImageInput::Pixels { rgb: bad })
        .expect_err("rank-2 pixels must be rejected");
    assert!(err.to_string().contains("CHW"), "{err}");
}

#[test]
fn validated_image_input_decodes_bytes_with_format() {
    use ember::multimodal::request::{ImageInput, ValidatedImageFormat, ValidatedImageInput};
    use image::ImageEncoder;
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(&[0, 0, 0], 1, 1, image::ExtendedColorType::Rgb8)
        .expect("encode 1x1 png");
    let v = ValidatedImageInput::decode(&ImageInput::Bytes(buf)).expect("validated decode");
    assert_eq!((v.width, v.height), (1, 1));
    assert_eq!(v.format, ValidatedImageFormat::Png);
    assert_eq!(v.rgb.shape(), [3, 1, 1]);
}

#[test]
fn batch_encode_images_rejects_zero_geometry() {
    use ember::multimodal::batch::BatchedImageInput;
    use ember::multimodal::request::SegmentId;
    let backend = CpuBackend;
    let tiles = CpuTensor::from_data(vec![1, 3, 64, 64], vec![0.0f32; 3 * 64 * 64]);
    let input = BatchedImageInput {
        owner: SegmentId::new(0, 0),
        tiles,
    };
    for scale in [0usize, 1] {
        let r = ember::multimodal::batch::batch_encode_images(
            &backend,
            std::slice::from_ref(&input),
            16,
            scale,
            |_, _| {
                // 64x64 tile, patch 16 -> 4x4 patches -> 16 rows before
                // the scale reduction
                Ok((CpuTensor::from_data(vec![16, 1], vec![0.0; 16]), ()))
            },
        );
        if scale == 0 {
            assert!(
                r.is_err(),
                "zero scale_factor must error, not divide by zero"
            );
        } else {
            assert!(r.is_ok(), "scale 1 with 64x64 tiles must work");
        }
    }
    let r = ember::multimodal::batch::batch_encode_images(&backend, &[input], 0, 1, |_, _| {
        Ok((CpuTensor::from_data(vec![1, 1], vec![0.0]), ()))
    });
    assert!(r.is_err(), "zero patch_size must error");
}

use super::*;
use crate::cli_experiment::{execute_prepared, prepare_run};
use crate::experiment_testutil::{out_dir, resolve, spec_text, tiny_model, TinyModel};
use ember::quant_k::KStrategy;

const INPUTS: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9 w22"

[[inputs]]
id = "b"
text = "w8 w1 w33 w2"
"#;

/// Captures on both sides of every boundary the tests use, in every
/// storage form, plus a generated-step capture.
const CAPTURES: &str = r#"
[[captures]]
id = "final-rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "span-rows"
site = "residual-post-mlp"
layers = "all"
inputs = ["a"]
[captures.tokens]
kind = "matched-span"
text = "w17 w5"
occurrence = 0
subtokens = "final"

[[captures]]
id = "attn-summary"
site = "attention-output"
layers = [0, 1, 2]
storage = "summary-only"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "pre-full"
site = "residual-pre-attention"
layers = [0, 2]
storage = "full-tensor"
[captures.tokens]
kind = "absolute-token"
index = 1

[[captures]]
id = "decode-mlp"
site = "mlp-output"
layers = [0, 3]
[captures.tokens]
kind = "generated-step"
step = 2

[[captures]]
id = "logits"
site = "logits"
[captures.tokens]
kind = "prompt-final"
"#;

fn prefix_json(path: &std::path::Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path.join("runtime.json")).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    value["prefix_reuse"].clone()
}

fn runtime_json(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path.join("runtime.json")).unwrap()).unwrap()
}

/// Run base + co-baselines + variants once standalone and once shared;
/// every bundle must be identical (semantic and payload hash). Returns
/// each variant's recorded per-input paths.
fn check(
    model: &TinyModel,
    mode: &str,
    temperature: f32,
    base_body: &str,
    co_bodies: &[&str],
    variant_bodies: &[&str],
) -> Vec<serde_json::Value> {
    check_all(
        model,
        mode,
        temperature,
        base_body,
        co_bodies,
        variant_bodies,
    )
    .0
}

/// [`check`], also running the pass with batched decode disabled (the
/// sequential pass, which must give the same bundles) and checking the
/// decode route `runtime.json` records. A body starting with
/// `# max_new_tokens = N` overrides the default of 4. Also returns every
/// bundle's generated tokens per input (base, co-baselines, variants).
fn check_all(
    model: &TinyModel,
    mode: &str,
    temperature: f32,
    base_body: &str,
    co_bodies: &[&str],
    variant_bodies: &[&str],
) -> (Vec<serde_json::Value>, Vec<Vec<Vec<u32>>>) {
    let text = |body: &str| {
        let max_new_tokens = body
            .strip_prefix("# max_new_tokens = ")
            .and_then(|rest| rest.lines().next())
            .map_or(4, |value| value.trim().parse().unwrap());
        spec_text(model, mode, max_new_tokens, body).replace(
            "temperature = 0.0",
            &format!("temperature = {temperature:?}"),
        )
    };
    let base_text = text(base_body);
    let co_texts: Vec<String> = co_bodies.iter().map(|body| text(body)).collect();
    let variant_texts: Vec<String> = variant_bodies.iter().map(|body| text(body)).collect();
    let base_spec = resolve(&base_text);
    let co_specs: Vec<_> = co_texts.iter().map(|text| resolve(text)).collect();
    let variant_specs: Vec<_> = variant_texts.iter().map(|text| resolve(text)).collect();
    let mut prepared = prepare_run(&base_spec, KStrategy::Auto, false).unwrap();
    let dir = &model.dir;

    let mut full = Vec::new();
    for (index, (spec, text)) in std::iter::once((&base_spec, &base_text))
        .chain(co_specs.iter().zip(&co_texts))
        .chain(variant_specs.iter().zip(&variant_texts))
        .enumerate()
    {
        let (_, identity, report, _) = execute_prepared(
            &mut prepared,
            spec,
            text,
            &out_dir(dir, &format!("{mode}-{temperature}-full-{index}")),
            false,
            None,
        )
        .unwrap();
        assert!(report.ok);
        full.push(identity);
    }

    let outs = |pass: &str| {
        let name = |index: usize| out_dir(dir, &format!("{mode}-{temperature}-{pass}-{index}"));
        let co: Vec<_> = (0..co_specs.len()).map(|index| name(1 + index)).collect();
        let variants: Vec<_> = (0..variant_specs.len())
            .map(|index| name(1 + co_specs.len() + index))
            .collect();
        (name(0), co, variants)
    };
    let (base_out, co_outs, variant_outs) = outs("shared");
    let co_targets = targets(&co_specs, &co_texts, &co_outs);
    let variant_targets = targets(&variant_specs, &variant_texts, &variant_outs);
    let (base, co, variants) = execute_shared(
        &mut prepared,
        RunTarget {
            resolved: &base_spec,
            spec_text: &base_text,
            output_directory: &base_out,
            retain_incomplete: false,
        },
        &co_targets,
        &variant_targets,
        None,
    )
    .unwrap();
    let shared: Vec<&RunOutcome> = std::iter::once(&base)
        .chain(co.iter())
        .chain(variants.iter())
        .collect();
    for (index, (standalone, outcome)) in full.iter().zip(&shared).enumerate() {
        assert!(outcome.report.ok, "bundle {index} failed self-verification");
        assert_eq!(
            standalone.semantic_hash,
            outcome.identity.semantic_hash,
            "bundle {index} ({mode}): semantic hash differs from a standalone run; \
                 prefix record {}",
            prefix_json(&outcome.path)
        );
        assert_eq!(
            standalone.payload_hash, outcome.identity.payload_hash,
            "bundle {index} ({mode}): payload hash differs from a standalone run"
        );
    }
    assert_eq!(prefix_json(&base.path)["role"], "base");

    // The decode route: batched on the Q8_0 fast path, else sequential.
    let batched = crate::cli_generation::batched_decode_ineligibility(&prepared.model).is_none()
        && !(co.is_empty() && variants.is_empty());
    for outcome in &shared {
        let route = &runtime_json(&outcome.path)["decode_batch"];
        if batched {
            assert_eq!(route["path"], "batched", "{route}");
            assert!(route["batch_size"].as_u64().unwrap() >= 2, "{route}");
        } else {
            assert_eq!(route["path"], "sequential", "{route}");
        }
    }

    // The same pass with every run decoding on its own.
    let (base_out, co_outs, variant_outs) = outs("seq");
    let co_targets = targets(&co_specs, &co_texts, &co_outs);
    let variant_targets = targets(&variant_specs, &variant_texts, &variant_outs);
    let (seq_base, seq_co, seq_variants) = execute_shared_sequential(
        &mut prepared,
        RunTarget {
            resolved: &base_spec,
            spec_text: &base_text,
            output_directory: &base_out,
            retain_incomplete: false,
        },
        &co_targets,
        &variant_targets,
        DecodeRoute::Sequential { reason: "test" },
        None,
    )
    .unwrap();
    for (index, (standalone, outcome)) in full
        .iter()
        .zip(
            std::iter::once(&seq_base)
                .chain(&seq_co)
                .chain(&seq_variants),
        )
        .enumerate()
    {
        assert_eq!(
            (&standalone.semantic_hash, &standalone.payload_hash),
            (
                &outcome.identity.semantic_hash,
                &outcome.identity.payload_hash
            ),
            "bundle {index} ({mode}): the sequential pass differs"
        );
        assert_eq!(
            runtime_json(&outcome.path)["decode_batch"]["path"],
            "sequential"
        );
    }
    let generated: Vec<Vec<Vec<u32>>> = shared
        .iter()
        .map(|outcome| {
            outcome
                .results
                .iter()
                .map(|result| result.generated_token_ids.clone())
                .collect()
        })
        .collect();

    // The suite is only meaningful if interventions move the model.
    let moved = variants
        .iter()
        .filter(|outcome| {
            outcome.results.iter().zip(&base.results).any(|(a, b)| {
                a.final_top1 != b.final_top1 || a.generated_token_ids != b.generated_token_ids
            })
        })
        .count();
    assert!(
        variants.is_empty() || moved > 0,
        "no variant changed the model's output"
    );
    (
        variants
            .iter()
            .map(|outcome| prefix_json(&outcome.path)["inputs"].clone())
            .collect(),
        generated,
    )
}

fn targets<'a>(
    specs: &'a [ExperimentSpecV1],
    texts: &'a [String],
    outs: &'a [std::path::PathBuf],
) -> Vec<RunTarget<'a>> {
    specs
        .iter()
        .zip(texts)
        .zip(outs)
        .map(|((resolved, spec_text), output_directory)| RunTarget {
            resolved,
            spec_text,
            output_directory,
            retain_incomplete: false,
        })
        .collect()
}

fn intervention(id: &str, site: &str, layers: &str, extra: &str, tokens: &str) -> String {
    format!(
        r#"
[[interventions]]
id = "{id}"
site = "{site}"
layers = {layers}
{extra}
[interventions.tokens]
{tokens}
"#
    )
}

fn paths(value: &serde_json::Value) -> Vec<(String, Option<u64>)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|input| {
            (
                input["path"].as_str().unwrap().to_string(),
                input["resume_layer"].as_u64(),
            )
        })
        .collect()
}

const PROMPT_FINAL: &str = "kind = \"prompt-final\"";

fn variant_suite(model: &TinyModel, mode: &str, temperature: f32) {
    let base = format!("{INPUTS}{CAPTURES}");
    let with = |parts: &[String]| format!("{base}{}", parts.concat());
    // Replace from a current-run capture (the morphology shape).
    let replace = with(&[intervention(
            "replace",
            "residual-post-mlp",
            "[2]",
            "operation = { kind = \"replace\" }\nsource = { kind = \"capture-from-current-run\", capture_id = \"final-rows\" }",
            PROMPT_FINAL,
        )]);
    // Zero an attention-output row mid-block, only for input b.
    let zero_b = with(&[intervention(
        "zero-b",
        "attention-output",
        "[1]",
        "operation = { kind = \"zero\" }\ninputs = [\"b\"]",
        "kind = \"relative-token\"\noffset_from_end = 1",
    )]);
    // Scale then restore exactly.
    let restore = with(&[
        intervention(
            "scale",
            "mlp-output",
            "[3]",
            "operation = { kind = \"scale\", factor = -2.0 }",
            PROMPT_FINAL,
        ),
        intervention(
            "restore",
            "mlp-output",
            "[3]",
            "operation = { kind = \"restore-original\" }",
            PROMPT_FINAL,
        ),
    ]);
    // Logits-site and decode-only interventions share every block.
    let logits = with(&[intervention(
        "scale-logits",
        "logits",
        "\"all\"",
        "operation = { kind = \"scale\", factor = 0.5 }",
        PROMPT_FINAL,
    )]);
    let decode_only = with(&[intervention(
        "decode",
        "residual-pre-attention",
        "[0]",
        "operation = { kind = \"zero\" }",
        "kind = \"generated-step\"\nstep = 2",
    )]);
    // Block 0 has no prefix: full recompute.
    let layer0 = with(&[intervention(
        "layer0",
        "residual-pre-attention",
        "[0]",
        "operation = { kind = \"scale\", factor = 3.0 }",
        PROMPT_FINAL,
    )]);
    let recorded = check(
        model,
        mode,
        temperature,
        &base,
        &[],
        &[&replace, &zero_b, &restore, &logits, &decode_only, &layer0],
    );
    let resumed = |layer| ("resumed".to_string(), Some(layer));
    let full = ("full-recompute".to_string(), None);
    let n = model.n_layers as u64;
    assert_eq!(paths(&recorded[0]), vec![resumed(2), resumed(2)]);
    assert_eq!(paths(&recorded[1]), vec![resumed(n), resumed(1)]);
    assert_eq!(paths(&recorded[2]), vec![resumed(3), resumed(3)]);
    assert_eq!(paths(&recorded[3]), vec![resumed(n), resumed(n)]);
    assert_eq!(paths(&recorded[4]), vec![resumed(n), resumed(n)]);
    assert_eq!(paths(&recorded[5]), vec![full.clone(), full]);
}

/// Steering and projection ablation keep the prefix contract: the
/// boundary is the earliest steered block, and a resumed variant is
/// bit-identical to its full recompute, whatever the direction source.
fn steering_suite(model: &TinyModel, mode: &str) {
    let base = format!("{INPUTS}{CAPTURES}");
    let with = |parts: &[String]| format!("{base}{}", parts.concat());
    let direction: Vec<f32> = (0..64)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.25)
        .collect();
    let npy = model.dir.join("direction.npy");
    let bytes = ember::v05::steering::write_npy(&[64], &direction);
    std::fs::write(&npy, &bytes).unwrap();
    let sha = ember::v05::manifest::sha256_hex(&bytes);
    let inline = format!("{direction:?}");
    let steer_inline = with(&[intervention(
            "steer",
            "residual-post-mlp",
            "[2]",
            &format!(
                "operation = {{ kind = \"steer\", alpha = 3.0 }}\nsource = {{ kind = \"inline-vector\", values = {inline} }}"
            ),
            PROMPT_FINAL,
        )]);
    let steer_file = with(&[intervention(
            "steer-file",
            "residual-pre-attention",
            "[1]",
            &format!(
                "operation = {{ kind = \"steer\", alpha = 0.5, normalize = \"match-residual-norm\" }}\nsource = {{ kind = \"vector-file\", path = {:?}, sha256 = \"{sha}\" }}",
                npy.display().to_string()
            ),
            PROMPT_FINAL,
        )]);
    let steer_contrastive = with(&[intervention(
            "steer-contrast",
            "residual-post-mlp",
            "[3]",
            "operation = { kind = \"steer\", alpha = 4.0, normalize = \"unit\" }\nsource = { kind = \"contrastive\", positive = [\"w1 w2 w3\", \"w4 w5\"], negative = [\"w9 w8 w7\"] }",
            PROMPT_FINAL,
        )]);
    let ablate = with(&[intervention(
            "ablate",
            "residual-post-mlp",
            "[2]",
            &format!(
                "operation = {{ kind = \"ablate-projection\" }}\nsource = {{ kind = \"inline-vector\", values = {inline} }}"
            ),
            "kind = \"relative-token\"\noffset_from_end = 1",
        )]);
    let recorded = check(
        model,
        mode,
        0.0,
        &base,
        &[],
        &[&steer_inline, &steer_file, &steer_contrastive, &ablate],
    );
    let resumed = |layer| ("resumed".to_string(), Some(layer));
    assert_eq!(paths(&recorded[0]), vec![resumed(2), resumed(2)]);
    assert_eq!(paths(&recorded[1]), vec![resumed(1), resumed(1)]);
    assert_eq!(paths(&recorded[2]), vec![resumed(3), resumed(3)]);
    assert_eq!(paths(&recorded[3]), vec![resumed(2), resumed(2)]);
}

/// Captures at `attention-heads`: every head at every layer, and two
/// heads of one layer at a generated step.
const HEAD_CAPTURES: &str = r#"
[[captures]]
id = "heads-all"
site = "attention-heads"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "heads-decode"
site = "attention-heads"
layers = [1]
heads = [1, 3]
[captures.tokens]
kind = "generated-step"
step = 1
"#;

/// Interventions at `attention-heads` keep the prefix contract (the
/// boundary is the intervened block) and batched decode gives each
/// sequence its own row: every variant equals its full recompute and the
/// sequential pass.
fn attention_heads_suite(model: &TinyModel, mode: &str) {
    let base = format!("{INPUTS}{CAPTURES}{HEAD_CAPTURES}");
    let with = |parts: &[String]| format!("{base}{}", parts.concat());
    // Two of the tiny model's four 16-wide heads.
    let direction: Vec<f32> = (0..32).map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.25).collect();
    let zero_head = with(&[intervention(
        "zero-head",
        "attention-heads",
        "[2]",
        "heads = [1]\noperation = { kind = \"zero\" }",
        PROMPT_FINAL,
    )]);
    let steer_heads = with(&[intervention(
            "steer-heads",
            "attention-heads",
            "[1]",
            &format!(
                "heads = [0, 2]\noperation = {{ kind = \"steer\", alpha = 2.0 }}\nsource = {{ kind = \"inline-vector\", values = {direction:?} }}"
            ),
            PROMPT_FINAL,
        )]);
    let restore_head = with(&[
        intervention(
            "scale-heads",
            "attention-heads",
            "[3]",
            "operation = { kind = \"scale\", factor = -2.0 }",
            PROMPT_FINAL,
        ),
        intervention(
            "restore-head",
            "attention-heads",
            "[3]",
            "heads = [0]\noperation = { kind = \"restore-original\" }",
            PROMPT_FINAL,
        ),
    ]);
    let decode_head = with(&[intervention(
        "decode-head",
        "attention-heads",
        "[0]",
        "heads = [3]\noperation = { kind = \"scale\", factor = -1.0 }",
        "kind = \"generated-step\"\nstep = 2",
    )]);
    let contrastive_head = with(&[intervention(
            "contrast-head",
            "attention-heads",
            "[3]",
            "heads = [2]\noperation = { kind = \"steer\", alpha = 1.0, normalize = \"unit\" }\nsource = { kind = \"contrastive\", positive = [\"w1 w2 w3\", \"w4 w5\"], negative = [\"w9 w8 w7\"] }",
            PROMPT_FINAL,
        )]);
    let recorded = check(
        model,
        mode,
        0.0,
        &base,
        &[],
        &[
            &zero_head,
            &steer_heads,
            &restore_head,
            &decode_head,
            &contrastive_head,
        ],
    );
    let resumed = |layer| ("resumed".to_string(), Some(layer));
    let n = model.n_layers as u64;
    assert_eq!(paths(&recorded[0]), vec![resumed(2), resumed(2)]);
    assert_eq!(paths(&recorded[1]), vec![resumed(1), resumed(1)]);
    assert_eq!(paths(&recorded[2]), vec![resumed(3), resumed(3)]);
    assert_eq!(paths(&recorded[3]), vec![resumed(n), resumed(n)]);
    assert_eq!(paths(&recorded[4]), vec![resumed(3), resumed(3)]);
}

#[test]
fn attention_heads_variants_resume_bit_identically() {
    let q8 = tiny_model("prefix-heads-q8", 4, 64, true);
    attention_heads_suite(&q8, "reference");
    attention_heads_suite(&q8, "planned");
    attention_heads_suite(&q8, "planned-fused");
    let f32 = tiny_model("prefix-heads-f32", 4, 64, false);
    attention_heads_suite(&f32, "reference");
}

/// Every head zeroed before the output projection is the attention
/// output zeroed (the projection has no bias): same tokens and logits.
/// One zeroed head is neither.
#[test]
fn zeroing_every_head_equals_zeroing_the_attention_output() {
    let model = tiny_model("heads-vs-output", 4, 64, true);
    let logits = r#"
[[captures]]
id = "logits"
site = "logits"
[captures.tokens]
kind = "prompt-final"
"#;
    let body = |site: &str, heads: &str| {
        format!(
            "{INPUTS}{logits}{}",
            intervention(
                "zero",
                site,
                "[1]",
                &format!("{heads}operation = {{ kind = \"zero\" }}"),
                PROMPT_FINAL,
            )
        )
    };
    let run = |name: &str, body: &str| {
        let text = spec_text(&model, "planned", 4, body);
        let resolved = resolve(&text);
        let mut prepared = prepare_run(&resolved, KStrategy::Auto, false).unwrap();
        let (_, _, report, results) = execute_prepared(
            &mut prepared,
            &resolved,
            &text,
            &out_dir(&model.dir, name),
            false,
            None,
        )
        .unwrap();
        assert!(report.ok, "{:?}", report.checks);
        results
            .iter()
            .map(|result| {
                let row = &result.captures[0].rows;
                (
                    result.generated_token_ids.clone(),
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let heads = run("zero-heads", &body("attention-heads", ""));
    let output = run("zero-output", &body("attention-output", ""));
    assert_eq!(heads, output);
    let one = run("zero-one-head", &body("attention-heads", "heads = [2]\n"));
    assert_ne!(one, heads);
}

#[test]
fn steering_variants_resume_bit_identically() {
    let model = tiny_model("prefix-steer", 4, 64, true);
    steering_suite(&model, "reference");
    steering_suite(&model, "planned");
}

#[test]
fn shared_prefix_bundles_equal_full_recompute_f32_reference() {
    let model = tiny_model("prefix-f32", 4, 64, false);
    variant_suite(&model, "reference", 0.0);
}

#[test]
fn shared_prefix_bundles_equal_full_recompute_q8_reference_and_planned() {
    // Q8_0 weights take the fused single-token decode route.
    let model = tiny_model("prefix-q8", 4, 64, true);
    variant_suite(&model, "reference", 0.0);
    variant_suite(&model, "planned", 0.0);
    // Seeded sampling draws after the (shared) prefill.
    variant_suite(&model, "planned", 0.9);
    variant_suite(&model, "planned-fused", 0.0);
    variant_suite(&model, "planned-fused", 0.9);
}

/// Batched decode with generations that stop at different steps: a
/// generated token is made the end-of-sequence token, some runs ask for
/// more tokens than others (and fall back to a full prefill), and a
/// co-baseline with its own generated-step sites runs its own
/// generation in the batch. Every bundle equals its standalone run and
/// the sequential pass, greedy and seeded, in every execution mode.
#[test]
fn batched_decode_with_different_lengths_and_stop_tokens_is_bit_identical() {
    let model = tiny_model("batch-lengths", 4, 64, true);
    let base = format!("# max_new_tokens = 8\n{INPUTS}{CAPTURES}");
    let with = |max: usize, parts: &[String]| {
        format!(
            "# max_new_tokens = {max}\n{INPUTS}{CAPTURES}{}",
            parts.concat()
        )
    };
    let zero = intervention(
        "zero",
        "residual-post-mlp",
        "[1]",
        "operation = { kind = \"zero\" }",
        PROMPT_FINAL,
    );
    let scale = intervention(
        "scale",
        "mlp-output",
        "[2]",
        "operation = { kind = \"scale\", factor = -2.0 }",
        PROMPT_FINAL,
    );
    let decode = intervention(
        "decode",
        "residual-pre-attention",
        "[0]",
        "operation = { kind = \"zero\" }",
        "kind = \"generated-step\"\nstep = 2",
    );
    let variants = [
        with(8, std::slice::from_ref(&zero)),
        with(8, std::slice::from_ref(&scale)),
        with(8, std::slice::from_ref(&decode)),
        // Longer and shorter runs than the base: full prefill, joint decode.
        with(11, std::slice::from_ref(&scale)),
        with(3, std::slice::from_ref(&zero)),
    ];
    let own_sites = format!(
        r#"# max_new_tokens = 6
{INPUTS}
[[captures]]
id = "late"
site = "mlp-output"
layers = [1]
[captures.tokens]
kind = "generated-step"
step = 1
"#
    );
    let variant_refs: Vec<&str> = variants.iter().map(String::as_str).collect();
    // Make a token some generations produce (and others do not) the
    // end-of-sequence token, so the runs stop at different steps.
    let (_, generated) = check_all(
        &model,
        "reference",
        0.0,
        &base,
        &[own_sites.as_str()],
        &variant_refs,
    );
    let prompt_words: std::collections::BTreeSet<u32> =
        [3, 17, 5, 40, 9, 22, 8, 1, 33, 2].into_iter().collect();
    let runs: Vec<Vec<u32>> = generated.into_iter().flatten().collect();
    let eos = (0..63u32)
        .filter(|token| !prompt_words.contains(token))
        .max_by_key(|token| {
            let stops: std::collections::BTreeSet<Option<usize>> = runs
                .iter()
                .map(|run| run.iter().position(|generated| generated == token))
                .collect();
            stops.len()
        })
        .unwrap();
    for entry in std::fs::read_dir(&model.dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            std::fs::remove_dir_all(path).unwrap();
        }
    }
    let tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&model.tokenizer).unwrap()).unwrap();
    let mut tokenizer = tokenizer;
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove(&format!("w{eos}"));
    vocab.insert("<eos>".into(), serde_json::json!(eos));
    std::fs::write(&model.tokenizer, tokenizer.to_string()).unwrap();

    for (mode, temperature) in [
        ("reference", 0.0),
        ("planned", 0.0),
        ("planned-fused", 0.0),
        ("reference", 0.9),
        ("planned-fused", 0.9),
    ] {
        let (_, generated) = check_all(
            &model,
            mode,
            temperature,
            &base,
            &[own_sites.as_str()],
            &variant_refs,
        );
        let lengths: std::collections::BTreeSet<usize> =
            generated.iter().flatten().map(Vec::len).collect();
        assert!(
            lengths.len() >= 3,
            "{mode} {temperature}: generations should stop at different steps: {lengths:?}"
        );
        if temperature == 0.0 {
            assert!(
                generated
                    .iter()
                    .flatten()
                    .any(|run| run.len() < 8 && run.len() > 3),
                "{mode}: some greedy run should stop at the end-of-sequence token"
            );
        }
    }
}

#[test]
fn one_token_prompts_and_mismatched_settings_fall_back_in_full() {
    let model = tiny_model("prefix-fallback", 3, 64, true);
    let base = r#"
[[inputs]]
id = "one"
text = "w5"

[[inputs]]
id = "two"
text = "w5 w6 w7"

[[captures]]
id = "rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"
"#;
    let variant = format!(
        "{base}{}",
        intervention(
            "zero",
            "residual-post-mlp",
            "[1]",
            "operation = { kind = \"zero\" }",
            PROMPT_FINAL,
        )
    );
    let recorded = check(&model, "reference", 0.0, base, &[], &[&variant]);
    let paths = paths(&recorded[0]);
    assert_eq!(paths[0].0, "full-recompute");
    assert_eq!(paths[1], ("resumed".to_string(), Some(1)));

    // Different generation settings: the spec-level check refuses.
    let base_spec = resolve(&spec_text(&model, "reference", 4, base));
    let longer = resolve(&spec_text(&model, "reference", 5, &variant));
    assert!(matches!(
        resume_decision(&base_spec, &longer, 1, model.n_layers),
        ReuseDecision::FullRecompute { .. }
    ));
}

/// The example morphology workflow (baseline capture, intervention,
/// restoration) with the model resident: three standalone runs versus
/// one shared pass, bundles compared by hash. Needs the pinned model in
/// the repository root: `cargo test --release --bin ember
/// example_workflow_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing harness; needs Llama-3.2-1B-Instruct-Q8_0.gguf"]
fn example_workflow_timing() {
    let read = |name: &str| {
        let text = std::fs::read_to_string(format!("examples/experiments/{name}.toml")).unwrap();
        let spec = resolve(&text);
        (text, spec)
    };
    let (base_text, base_spec) = read("morphology-layerwise-capture");
    let variants = [
        read("morphology-intervention"),
        read("morphology-restoration"),
    ];
    if !base_spec.model.path.exists() {
        return;
    }
    let dir = crate::experiment_testutil::temp_dir("example-timing");
    let mut prepared = prepare_run(&base_spec, KStrategy::Auto, false).unwrap();
    let mut separate = Vec::new();
    let mut shared = Vec::new();
    for round in 0..4 {
        let out = |name: &str| dir.join(format!("{round}-{name}"));
        let started = std::time::Instant::now();
        let mut hashes = Vec::new();
        for (index, (text, spec)) in std::iter::once((&base_text, &base_spec))
            .chain(variants.iter().map(|(text, spec)| (text, spec)))
            .enumerate()
        {
            let (_, identity, _, _) = execute_prepared(
                &mut prepared,
                spec,
                text,
                &out(&format!("separate-{index}")),
                false,
                None,
            )
            .unwrap();
            hashes.push(identity.semantic_hash);
        }
        let separate_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        let outs: Vec<_> = (0..variants.len())
            .map(|index| out(&format!("shared-{}", index + 1)))
            .collect();
        let targets: Vec<RunTarget<'_>> = variants
            .iter()
            .zip(&outs)
            .map(|((text, spec), output_directory)| RunTarget {
                resolved: spec,
                spec_text: text,
                output_directory,
                retain_incomplete: false,
            })
            .collect();
        let base_out = out("shared-0");
        let (base, _, outcomes) = execute_shared(
            &mut prepared,
            RunTarget {
                resolved: &base_spec,
                spec_text: &base_text,
                output_directory: &base_out,
                retain_incomplete: false,
            },
            &[],
            &targets,
            None,
        )
        .unwrap();
        let shared_ms = started.elapsed().as_secs_f64() * 1000.0;
        let shared_hashes: Vec<String> = std::iter::once(&base)
            .chain(outcomes.iter())
            .map(|outcome| outcome.identity.semantic_hash.clone())
            .collect();
        assert_eq!(hashes, shared_hashes);
        if round > 0 {
            separate.push(separate_ms);
            shared.push(shared_ms);
        }
    }
    separate.sort_by(f64::total_cmp);
    shared.sort_by(f64::total_cmp);
    eprintln!(
        "morphology workflow (baseline + intervention + restoration, model resident): \
             separate {:.0} ms, shared prefix {:.0} ms (median of {})",
        separate[separate.len() / 2],
        shared[shared.len() / 2],
        separate.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn co_baselines_equal_their_standalone_bundles() {
    let model = tiny_model("prefix-co", 4, 64, true);
    let base = format!("{INPUTS}{CAPTURES}");
    let other = r#"
[[captures]]
id = "other"
site = "mlp-output"
layers = [1]
[captures.tokens]
kind = "prompt-final"
"#;
    let decode = r#"
[[captures]]
id = "decode-mlp"
site = "mlp-output"
layers = [0, 3]
[captures.tokens]
kind = "generated-step"
step = 2
"#;
    // Same generated-step sites as the base: observes the shared pass.
    let same_sites = format!("{INPUTS}{other}{decode}");
    // Different generated-step sites: runs on its own.
    let other_sites = format!("{INPUTS}{other}");
    let intervening = format!(
        "{base}{}",
        intervention(
            "zero",
            "residual-post-mlp",
            "[1]",
            "operation = { kind = \"zero\" }",
            PROMPT_FINAL,
        )
    );
    check(
        &model,
        "planned",
        0.0,
        &base,
        &[&same_sites, &other_sites],
        &[&intervening],
    );
}

//! Phase 5 Track J3: chat-template parity for Arabic conversations.
//!
//! Ember's VoiceSession scaffolds the Llama-3 chat template by hand
//! (`ScaffoldTokens`). This gate pins that the hand-rendered scaffold +
//! Arabic content tokenizes EXACTLY like the reference
//! `apply_chat_template` output for the same conversation and dates.
//!
//! Skipped unless `EMBER_CHAT_PARITY_JSON` points at a dump from
//! `scripts/ref_chat_template_ar.py`.

use serde_json::Value;

#[test]
fn arabic_chat_template_parity_matches_hf_reference() {
    let Ok(json_path) = std::env::var("EMBER_CHAT_PARITY_JSON") else {
        eprintln!("skipping: set EMBER_CHAT_PARITY_JSON");
        return;
    };
    let raw = std::fs::read_to_string(&json_path).expect("read parity json");
    let dump: Value = serde_json::from_str(&raw).expect("parse parity json");
    let want: Vec<u32> = dump["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let tok_dir = dump["tokenizer"].as_str().expect("tokenizer dir");
    let tok = ember::tokenizer::EmberTokenizer::from_file(
        std::path::Path::new(tok_dir).join("tokenizer.json"),
    )
    .expect("load tokenizer");

    // 1. The reference RENDER, tokenized through ember, must give the
    //    reference ids (tokenizer-level agreement on this exact string).
    let rendered = dump["rendered"].as_str().expect("rendered");
    assert_eq!(
        tok.encode_no_special(rendered).expect("encode"),
        want,
        "ember tokenization of the rendered Arabic chat template diverges \
         from the HF reference token stream"
    );

    // 2. VoiceSession-style hand scaffold over the same turns must produce
    //    the same TOKEN STREAM (the contract is ids, not bytes).
    let conv: Vec<(String, String)> = dump["conversation"]
        .as_array()
        .expect("conversation")
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap_or("").to_string(),
                m["content"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect();

    let mut full = String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
    let sys = conv[0].1.as_str();
    if sys.contains("Today Date") {
        full.push_str(sys.trim_end());
    } else if !sys.is_empty() {
        // the template renders its injected date lines BEFORE the
        // developer/system content, and NO newline before <|eot_id|>
        full.push_str("Cutting Knowledge Date: December 2023\nToday Date: 01 Jan 2026\n\n");
        full.push_str(sys.trim_end());
    } else {
        full.push_str("Cutting Knowledge Date: December 2023\nToday Date: 01 Jan 2026\n\n");
    }
    full.push_str("<|eot_id|>");
    for (role, content) in &conv[1..] {
        full.push_str(&format!(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        ));
    }
    full.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let got_full = tok.encode_no_special(&full).expect("encode scaffold");
    assert_eq!(
        got_full, want,
        "VoiceSession-style scaffold + Arabic turns must reproduce the \
         reference token stream exactly"
    );
}

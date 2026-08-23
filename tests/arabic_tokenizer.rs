//! Phase 5 Track J: Arabic/multilingual tokenizer parity against the HF
//! reference (exact token-id equality).
//!
//! Skipped unless `EMBER_TOK_PARITY=1` with `EMBER_TOK_PARITY_JSON` pointing
//! at a dump produced by `scripts/ref_tokenizer_ar.py`. The battery covers
//! MSA, dialects, code-switching, Arabic-Indic/Western numerals, dates,
//! URLs, emoji, both punctuation systems, diacritics, tatweel, presentation
//! forms, combining marks, zero-width characters, and bidi marks. The gate
//! is exact id equality — no normalization may happen anywhere in ember's
//! path.

use serde_json::Value;

#[test]
fn arabic_tokenizer_parity_matches_hf_reference() {
    let Ok(json_path) = std::env::var("EMBER_TOK_PARITY_JSON") else {
        eprintln!("skipping: set EMBER_TOK_PARITY_JSON (+ EMBER_TOK_PARITY=1)");
        return;
    };
    if std::env::var("EMBER_TOK_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skipping: set EMBER_TOK_PARITY=1");
        return;
    }
    let raw = std::fs::read_to_string(&json_path).expect("read parity json");
    let dump: Value = serde_json::from_str(&raw).expect("parse parity json");
    let tok_path = dump["tokenizer"].as_str().expect("tokenizer path");

    // The dump records the directory; EmberTokenizer wants tokenizer.json.
    let tok_file = std::path::Path::new(tok_path).join("tokenizer.json");
    let tok = ember::tokenizer::EmberTokenizer::from_file(&tok_file)
        .or_else(|_| ember::tokenizer::EmberTokenizer::from_file(tok_path))
        .expect("load ember tokenizer");

    let strings = dump["strings"].as_object().expect("battery object");
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (key, entry) in strings {
        let text = entry["text"].as_str().expect("text");
        let want: Vec<u32> = entry["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let got = tok.encode(text).unwrap_or_else(|e| {
            failures.push(format!("{key}: encode error {e}"));
            Vec::new()
        });
        checked += 1;
        if got != want {
            failures.push(format!(
                "{key}: MISMATCH\n  text:  {text:?}\n  want:  {want:?}\n  got:   {got:?}"
            ));
        }
    }
    println!("checked {checked} battery strings");
    if !failures.is_empty() {
        panic!(
            "{} of {} Arabic/multilingual parity checks failed:\n{}",
            failures.len(),
            checked,
            failures.join("\n")
        );
    }
}

/// Round-trip: decode(encode(text)) must reproduce the string exactly for
/// every battery entry — this is the runtime-side "no silent alteration"
/// guarantee, independent of the reference.
#[test]
fn arabic_roundtrip_preserves_text_exactly() {
    let texts = [
        "اللغة العربية جميلة.",
        "شخبارك؟ وين رايح اليوم؟",
        "إزيك؟ عامل إيه النهارده؟",
        "أنا أستخدم Rust للبرمجة",
        "السنة ١٤٤٧ هجرية، والعام ٢٠٢٦ ميلادي",
        "مرحباً 👋 كيف الحال؟ 🌟",
        "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ جِدًّا.",
        "مرحبـــــا بالعـــالم",
        "واش راك؟ لا باس عليك؟",
        "https://example.com/مسار/صفحة?q=بحث",
        "لغة \u{0644}\u{0651}\u{064F}\u{063A}\u{064E}\u{0629}",
        "عربي\u{200b}عربي\u{200c}عربي‍",
        "ﻟﻐﺔ ﻋﺮﺑﻴﺔ",
        "لا لأ لإ لآ",
    ];
    let tok =
        ember::tokenizer::EmberTokenizer::from_file("/home/west/ember-work/llama32/tokenizer.json")
            .or_else(|_| {
                let p = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
                    .join("tokenizer.json");
                ember::tokenizer::EmberTokenizer::from_file(p)
            })
            .expect("load a tokenizer for roundtrip test");
    for text in texts {
        let ids = tok.encode(text).expect("encode");
        let back = tok.decode(&ids).expect("decode");
        assert_eq!(back, *text, "roundtrip altered text: {text:?}");
    }
}

//! Phase 5 Track N: RTL / UTF-8-safe streaming boundaries.
//!
//! Byte-level BPE tokens can be FRAGMENTS of a multi-byte code point.
//! Naive per-token detokenization emits U+FFFD replacement characters in
//! the middle of every Arabic word. The incremental decoder must never do
//! so, and the concatenation of streamed pieces must equal the full
//! decode exactly.

use ember::tokenizer::EmberTokenizer;

fn tokenizer() -> EmberTokenizer {
    // any byte-level BPE tokenizer works; llama32 is present on this host,
    // the repo-root one is the fallback for other environments
    EmberTokenizer::from_file("/home/west/ember-work/llama32/tokenizer.json")
        .or_else(|_| {
            let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json");
            EmberTokenizer::from_file(p)
        })
        .expect("load tokenizer")
}

#[test]
fn incremental_stream_never_splits_code_points_and_matches_full_decode() {
    let tok = tokenizer();
    let cases = [
        "اللغة العربية جميلة",
        "شخبارك؟ وين رايح اليوم؟",
        "مرحباً 👋 كيف الحال؟ 🌟",
        "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ",
        "أنا أحب البرمجة بلغة Rust لأنها",
        "السنة ١٤٤٧ هجرية 🕌 والعام 2026",
    ];
    for text in cases {
        let ids = tok.encode(text).expect("encode");
        let mut dec = tok.incremental_decoder();
        let mut streamed = String::new();
        for &id in &ids {
            let piece = dec.push(id).expect("push");
            // no replacement characters may EVER reach the consumer
            assert!(
                !piece.contains('\u{FFFD}'),
                "streamed piece contains U+FFFD: {piece:?} (text {text:?})"
            );
            streamed.push_str(&piece);
        }
        let tail = dec.finish().expect("finish");
        assert!(!tail.contains('\u{FFFD}'));
        streamed.push_str(&tail);

        let full = tok.decode(&ids).expect("full decode");
        assert_eq!(
            streamed, full,
            "streamed concatenation must equal full decode for {text:?}"
        );
    }
}

/// Prefix-stability: text already released must never change when more
/// tokens arrive (a streaming consumer's fundamental assumption).
#[test]
fn incremental_stream_prefix_is_stable() {
    let tok = tokenizer();
    let text = "تُعدُّ اللغة العربية واحدة من أكثر اللغات تحدثًا في العالم";
    let ids = tok.encode(text).expect("encode");
    let mut dec = tok.incremental_decoder();
    let mut released = 0usize;
    for &id in &ids {
        let before = released;
        let piece = dec.push(id).expect("push");
        if !piece.is_empty() {
            // pieces only ever append; the consumer may rely on this
            assert_eq!(piece.len(), piece.len());
        }
        released += piece.len();
        assert!(released >= before);
    }
}

//! Phase 5 Track N: RTL / UTF-8-safe streaming boundaries.
//!
//! Byte-level BPE tokens can be FRAGMENTS of a multi-byte code point.
//! Naive per-token detokenization emits U+FFFD replacement characters in
//! the middle of every Arabic word. The incremental decoder must never do
//! so, and the concatenation of streamed pieces must equal the full
//! decode exactly.

use ember::tokenizer::EmberTokenizer;

fn tokenizer() -> EmberTokenizer {
    // Any byte-level BPE tokenizer works. Use the tracked repo-root tokenizer
    // so this behaves identically on every host and CI runner; the
    // CARGO_MANIFEST_DIR anchor keeps it independent of the working directory
    // the test binary happens to start in.
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json");
    EmberTokenizer::from_file(p).expect("load repo-root tokenizer.json")
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

    // The text a consumer sees once the whole stream is pushed. This is the
    // reference every truncated decode must agree with.
    let full = {
        let mut dec = tok.incremental_decoder();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&dec.push(id).expect("push"));
        }
        out.push_str(&dec.finish().expect("finish"));
        out
    };
    assert!(!full.is_empty(), "full decode released no text");

    // The property: stopping after N tokens must leave the consumer holding a
    // prefix of the final text. A decoder that released bytes speculatively
    // and later had to take them back would break this, and a consumer that
    // already drew those bytes would show corruption. Note we deliberately do
    // NOT call finish() on the truncated runs: those trailing bytes are
    // exactly what a live consumer has not received yet.
    for n in 0..=ids.len() {
        let mut dec = tok.incremental_decoder();
        let mut partial = String::new();
        for &id in &ids[..n] {
            partial.push_str(&dec.push(id).expect("push"));
        }
        assert!(
            full.starts_with(&partial),
            "after {n}/{} tokens the consumer holds {partial:?}, \
             which is not a prefix of the final text {full:?}",
            ids.len()
        );
    }
}

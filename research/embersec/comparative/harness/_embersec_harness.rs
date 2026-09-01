//! EmberSEC comparative evaluation harness.
//!
//! Injected into a worktree's tests/ directory by
//! research/embersec/comparative/run_eval.py (canonical copy lives in
//! research/embersec/comparative/harness/). NOT committed to the repo
//! itself; identical content is used against the baseline and current
//! trees so outcomes differ only because of the code under test.
//!
//! Contract:
//! - fixture path in EMBERSEC_FIXTURE (tests skip cleanly when unset, so
//!   `cargo test --all-targets` stays green with the file present);
//! - exit 0  = artifact accepted;
//! - exit 1  = structured rejection (error printed to stderr);
//! - panic   = propagates to libtest (exit 101);
//! - abort   = process crash (OOM / assert);
//! - HARNESS: lines on stderr identify the rejection stage.

use std::io::Cursor;

fn fixture() -> Option<std::path::PathBuf> {
    std::env::var("EMBERSEC_FIXTURE").ok().map(Into::into)
}

#[test]
fn gguf_load_check() {
    let Some(fixture) = fixture() else {
        eprintln!("HARNESS: SKIP (no EMBERSEC_FIXTURE)");
        return;
    };
    let bytes = std::fs::read(fixture).expect("read fixture");
    match ember::loader::load_gguf_from_reader(&mut Cursor::new(&bytes)) {
        Ok(_) => eprintln!("HARNESS: LOAD_OK"),
        Err(err) => {
            eprintln!("HARNESS: LOAD_REJECT: {err}");
            std::process::exit(1);
        }
    }
}

#[test]
fn gguf_model_check() {
    let Some(fixture) = fixture() else {
        eprintln!("HARNESS: SKIP (no EMBERSEC_FIXTURE)");
        return;
    };
    let bytes = std::fs::read(fixture).expect("read fixture");
    let loader = match ember::loader::load_gguf_from_reader(&mut Cursor::new(&bytes)) {
        Ok(loader) => loader,
        Err(err) => {
            eprintln!("HARNESS: LOAD_REJECT: {err}");
            std::process::exit(1);
        }
    };
    // Architecture-aware dispatch mirrors the CLI: llama/qwen3 -> Llama,
    // gemma3/gemma4 -> Gemma4, gpt2 -> Gpt2.
    let arch = match loader.metadata.get("general.architecture") {
        Some(ember::loader::GgufValue::Str(s)) => s.as_str(),
        _ => "llama",
    };
    let result = match arch {
        "gemma3" | "gemma4" => ember::gemma4::Gemma4::from_loader(loader).map(|_| ()),
        "gpt2" => ember::model::Gpt2::from_loader(loader).map(|_| ()),
        _ => ember::llama::Llama::from_loader(loader).map(|_| ()),
    };
    match result {
        Ok(_) => eprintln!("HARNESS: MODEL_OK"),
        Err(err) => {
            eprintln!("HARNESS: MODEL_REJECT: {err}");
            std::process::exit(1);
        }
    }
}

#[test]
fn tokenizer_check() {
    let Some(fixture) = fixture() else {
        eprintln!("HARNESS: SKIP (no EMBERSEC_FIXTURE)");
        return;
    };
    let bytes = std::fs::read(fixture).expect("read fixture");
    match ember::tokenizer::EmberTokenizer::from_bytes(&bytes) {
        Ok(_) => eprintln!("HARNESS: TOKENIZER_OK"),
        Err(err) => {
            eprintln!("HARNESS: TOKENIZER_REJECT: {err}");
            std::process::exit(1);
        }
    }
}

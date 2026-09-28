use std::process::Command;

#[test]
fn diff_disagreement_prints_report_and_exits_three() {
    let dir = std::env::temp_dir().join(format!("ember-verification-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("invalid.gguf");
    std::fs::write(&input, b"not a GGUF model").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ember"))
        .args(["diff", input.to_str().unwrap(), "--against", "llama.cpp"])
        .env("EMBER_LLAMACPP_BIN", dir.join("missing-runtime"))
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(output.status.code(), Some(3), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("DISAGREE"));
}

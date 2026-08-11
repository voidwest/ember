//! Build-time environment capture for execution-plan provenance.
//!
//! Records the Rust compiler, target, and workspace git commit when available.
//! Consumers use explicit `EMBER_*` names so Cargo/build-script-only variables
//! are never mistaken for variables available to crate compilation.

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Some(head_path) = command_stdout("git", &["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head_path}");
    }
    if let Some(head_ref) = command_stdout("git", &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(ref_path) = command_stdout("git", &["rev-parse", "--git-path", &head_ref]) {
            println!("cargo:rerun-if-changed={ref_path}");
        }
    }
    if let Some(packed_refs) = command_stdout("git", &["rev-parse", "--git-path", "packed-refs"]) {
        println!("cargo:rerun-if-changed={packed_refs}");
    }

    if let Some(version) = command_stdout("rustc", &["--version"]) {
        println!("cargo:rustc-env=EMBER_RUSTC_VERSION={version}");
    }
    if let Ok(target) = std::env::var("TARGET") {
        println!("cargo:rustc-env=EMBER_TARGET={target}");
    }
    if let Some(commit) = command_stdout("git", &["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=EMBER_GIT_COMMIT={commit}");
    }

    // This is build-time state only. Auditable benchmark records additionally
    // capture runtime git state because an already-built binary can outlive a
    // later working-tree edit.
    if let Some(status) = command_stdout(
        "git",
        &["status", "--porcelain", "--untracked-files=normal"],
    ) {
        println!("cargo:rustc-env=EMBER_GIT_DIRTY={}", !status.is_empty());
    } else {
        // `command_stdout` filters empty output, which is the clean-tree case.
        let clean = std::process::Command::new("git")
            .args(["status", "--porcelain", "--untracked-files=normal"])
            .output()
            .is_ok_and(|output| output.status.success() && output.stdout.is_empty());
        if clean {
            println!("cargo:rustc-env=EMBER_GIT_DIRTY=false");
        }
    }
}

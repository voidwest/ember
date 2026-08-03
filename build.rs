//! Build-time environment capture for execution-plan provenance.
//!
//! Records the Rust compiler version and the workspace git commit (when the
//! tree is a git checkout) as `EMBER_RUSTC_VERSION` / `EMBER_GIT_HASH` env
//! vars consumed by `src/plan.rs`. The git lookup is best-effort: outside a
//! git checkout both vars fall back to `unknown` at the call site.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output();
    if let Ok(output) = rustc {
        if output.status.success() {
            if let Ok(version) = String::from_utf8(output.stdout) {
                println!("cargo:rustc-env=EMBER_RUSTC_VERSION={}", version.trim());
            }
        }
    }

    let git = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();
    if let Ok(output) = git {
        if output.status.success() {
            if let Ok(commit) = String::from_utf8(output.stdout) {
                println!("cargo:rustc-env=EMBER_GIT_HASH={}", commit.trim());
            }
        }
    }
}

"""Exercise the real build script in a dependency-free temporary Cargo crate."""

import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("cargo") or not shutil.which("git"), reason="requires Cargo and Git")
def test_provenance_tracks_loose_packed_detached_and_worktree_heads(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "target"
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_NET_OFFLINE="true",
               RUSTUP_TOOLCHAIN=toolchain, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def run(args, cwd=repo):
        return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True,
                              text=True, timeout=60).stdout.strip()

    (repo / "Cargo.toml").write_text('[package]\nname="provenance-fixture"\nversion="0.0.0"\nedition="2024"\n')
    (repo / "src").mkdir()
    (repo / "src/main.rs").write_text('fn main() { println!("{}", env!("EMBER_GIT_COMMIT")); }\n')
    (repo / ".gitignore").write_text("/Cargo.lock\n")
    script = (ROOT / "build.rs").read_text()
    # Count actual build-script executions, including reruns of a cached binary.
    marker = '''
    use std::io::Write;
    let marker = std::path::PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap())
        .join("build-script-executions");
    writeln!(std::fs::OpenOptions::new().create(true).append(true).open(marker).unwrap(), "run").unwrap();
'''
    (repo / "build.rs").write_text(script.replace("fn main() {", "fn main() {" + marker, 1))
    run(["git", "init", "-b", "main"])
    run(["git", "config", "user.name", "Provenance Test"])
    run(["git", "config", "user.email", "provenance@example.invalid"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "initial"])
    initial = run(["git", "rev-parse", "HEAD"])
    assert not (repo / ".git/packed-refs").exists()

    def build(expected_runs, cwd=repo):
        run(["cargo", "build", "--offline", "--quiet"], cwd)
        assert len((target / "build-script-executions").read_text().splitlines()) == expected_runs
        assert run([str(target / "debug/provenance-fixture")], cwd) == run(["git", "rev-parse", "HEAD"], cwd)

    build(1)
    build(1)  # Missing optional packed-refs must not cause a rebuild.
    run(["git", "pack-refs", "--all", "--prune"])
    build(2)
    build(2)
    run(["git", "commit", "--allow-empty", "-m", "new loose ref"])
    build(3)
    build(3)
    run(["git", "checkout", "--detach", initial])
    build(4)
    build(4)

    worktree = tmp_path / "worktree"
    run(["git", "worktree", "add", "-b", "nested/worktree", str(worktree), "HEAD"])
    build(5, worktree)
    build(5, worktree)
    run(["git", "pack-refs", "--all", "--prune"])
    build(6, worktree)
    build(6, worktree)
    run(["git", "commit", "--allow-empty", "-m", "worktree loose ref"], worktree)
    build(7, worktree)
    build(7, worktree)

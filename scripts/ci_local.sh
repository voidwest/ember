#!/usr/bin/env bash
# Local mirror of the model-free gates in .github/workflows/ci.yml.
#
#   scripts/ci_local.sh          # quick: every lint-class gate (the pre-push set)
#   scripts/ci_local.sh full     # quick + the test suites CI runs without a model
#
# Env: PYTHON (default .venv/bin/python, else python3)
#      CARGO_BUILD_JOBS (cargo's own; e.g. 2 keeps a laptop's load moderate)
#
# "quick" only checks and lints, so it never replaces target/debug/ember.
# Every step runs even after a failure; the summary lists what failed.
#
# Keep the commands in step with ci.yml: a gate that exists only there is a
# gate that fails 25 minutes after the push instead of before it.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

mode="${1:-quick}"
case "$mode" in
quick | full) ;;
*)
    echo "usage: scripts/ci_local.sh [quick|full]" >&2
    exit 2
    ;;
esac

# Same carve-out as ci.yml: a 1.98 style lint on measured kernel loop shapes.
CLIPPY_FLAGS=(-D warnings -A clippy::chunks_exact_to_as_chunks)
# Headless MSRV; CI's arm64 job builds, tests and lints on it.
MSRV=1.92.0

failed=()
skipped=()

run() {
    local name="$1"
    shift
    printf '\n==> %s\n    %s\n' "$name" "$*"
    if ! "$@"; then
        failed+=("$name")
    fi
}

skip() {
    printf '\n==> %s: skipped (%s)\n' "$1" "$2"
    skipped+=("$1: $2")
}

run "fmt" cargo fmt --all -- --check

run "clippy (all features)" \
    cargo clippy --locked --all-targets --all-features -- "${CLIPPY_FLAGS[@]}"

run "clippy (headless)" \
    cargo clippy --locked --no-default-features --all-targets -- "${CLIPPY_FLAGS[@]}"

# The kernels are cfg(target_arch)-gated, so a lint can fail on the
# architecture this host does not compile. Lint the sibling architecture too
# when its standard library is installed (`rustup target add <triple>`).
host="$(rustc -vV | sed -n 's/^host: //p')"
case "$host" in
aarch64-*) sibling="x86_64-${host#aarch64-}" ;;
x86_64-*) sibling="aarch64-${host#x86_64-}" ;;
*) sibling="" ;;
esac
if [[ -n "$sibling" ]] && rustup target list --installed 2>/dev/null | grep -qx "$sibling"; then
    run "clippy (headless, $sibling)" \
        cargo clippy --locked --no-default-features --all-targets --target "$sibling" \
        -- "${CLIPPY_FLAGS[@]}"
else
    skip "clippy (headless, other architecture)" "rustup target add ${sibling:-<sibling triple>}"
fi

if command -v python3 >/dev/null 2>&1; then
    run "clippy (python bindings)" \
        cargo clippy --locked -p ember-python -- "${CLIPPY_FLAGS[@]}"
else
    skip "clippy (python bindings)" "python3 not found"
fi

if rustup toolchain list 2>/dev/null | grep -q "^${MSRV}-"; then
    run "headless check on the MSRV ($MSRV)" \
        cargo "+$MSRV" check --locked --no-default-features --all-targets
    if cargo "+$MSRV" clippy --version >/dev/null 2>&1; then
        # No carve-out: the lint it allows does not exist on the MSRV.
        run "clippy (headless, MSRV $MSRV)" \
            cargo "+$MSRV" clippy --locked --no-default-features --all-targets -- -D warnings
    else
        skip "clippy (headless, MSRV $MSRV)" "rustup component add clippy --toolchain $MSRV"
    fi
else
    skip "headless check on the MSRV" "rustup toolchain install $MSRV --profile minimal --component clippy"
fi

run "fuzz lockfile in sync" \
    sh -c 'cargo metadata --locked --format-version 1 --manifest-path fuzz/Cargo.toml >/dev/null'

run "docs" env RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps

# PYTHON overrides the interpreter, as in the validate_* scripts.
python="${PYTHON:-python3}"
[[ -z "${PYTHON:-}" && -x .venv/bin/python ]] && python=".venv/bin/python"
if command -v "$python" >/dev/null 2>&1; then
    run "docs site checks" "$python" scripts/check_docs.py
    run "python compile" "$python" -m compileall -q python probes stimuli scripts tests
    run "probe matrix dry run" "$python" probes/run_probe_matrix.py \
        --model smoke:dummy.gguf --generate-tokens 1 --dry-run
else
    skip "docs site checks" "python3 not found"
    skip "python smoke" "python3 not found"
fi

# One script per `bash -n`: given several paths it parses only the first.
shell_syntax() {
    local script status=0
    for script in $(git ls-files '*.sh') .githooks/pre-push; do
        bash -n "$script" || status=1
    done
    return "$status"
}
run "shell syntax" shell_syntax

if [[ "$mode" == "full" ]]; then
    run "test" cargo test --locked --all-targets
    run "doctests" cargo test --locked --doc
    run "test with the audio feature" cargo test --locked --features audio --lib
    if cargo audit --version >/dev/null 2>&1; then
        run "cargo audit" cargo audit
    else
        skip "cargo audit" "cargo install cargo-audit"
    fi
    run "native GUI interactions" \
        cargo test --locked --features gui-tests --bin ember gui_native::kit_tests
    run "K-quant kernel tier" \
        cargo test --locked --release --lib k_quant_matmul::tests
    if "$python" -c 'import pytest' >/dev/null 2>&1; then
        run "python tests" "$python" -m pytest tests probes/test_probe_workflows.py -q
    else
        skip "python tests" "pytest is not installed for $python"
    fi
fi

printf '\n'
if ((${#skipped[@]})); then
    printf 'skipped:\n'
    printf '  - %s\n' "${skipped[@]}"
fi
if ((${#failed[@]})); then
    printf 'FAILED:\n'
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
printf 'all %s gates passed\n' "$mode"

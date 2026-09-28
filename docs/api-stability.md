# Ember API stability policy

This policy describes what downstream users may rely on when embedding Ember,
using the `ember` CLI, or consuming its research artifacts. It is deliberately
conservative: numerical reproducibility matters as much as source
compatibility.

## Current status and scope

The crate reports `1.0.0` in the tree; the `v1.0.0` tag is held until the
external acceptance gates close. The headless library/CLI MSRV remains Rust
**1.92** (`--no-default-features`). The GPUI Kit migration uses
Rust **1.98.1** for GUI builds and the pinned development toolchain; CI also
checks headless builds separately on 1.92. This GUI toolchain increase is part
of the 1.0 change, not a patch. The CLI and the
schema-versioned artifact formats are the primary supported integration
surfaces. The Rust library is embeddable, but it is not yet a stable
compatibility promise: a `pub` item is source-visible, not automatically stable.

## 1.0 line commitment

The 1.0 line commits to the surfaces named in
[v0.5 research contract section 19](v05-research-contract.md): the serialized
anchors (`ember.experiment.v1`, `ember.bundle.v1`, `ember.hook.v1`,
`v04-plan/1`, run-manifest `schema_version` 2 with `execution-identity-v2`,
`signed-evidence-v1`, `ember.kv-snapshot.v1`) and the documented CLI and Python
surfaces. Within that line, normal SemVer major/minor/patch rules apply to those
surfaces. The Rust library stays experimental unless a module is explicitly
listed as stable here; diagnostic commands, kernel names, timing fields, and the
`#[doc(hidden)]` modules are not stability promises. The GPUI Kit toolchain
increase ships as the 1.0 minor change, not a patch. This section states the
commitment; it does not by itself assert that a release has been tagged.

This policy covers:

- Rust source APIs (modules, types, traits, functions, constants, and feature
  flags);
- the command-line interface documented in `README.md` and `docs/usage.md`;
- the optional Python binding (`bindings/python`): the
  `inspect`/`plan`/`diff`/`diff_corpus` function signatures and their returned
  dict schemas follow the same pre-1.0 rules as the CLI/JSON contracts above;
  the extension itself remains outside the Rust-only install path;
- serialized research contracts such as `ember.experiment.v1`,
  `ember.bundle.v1`, `ember.kv-snapshot.v1`, and `ember.agent.trace.v1`.

Model files, logits, hidden states, timings, and other numerical outputs are
not universal API promises. Each validation or research contract defines its
own tolerance, provenance, and reproducibility requirements.

## Candidate migration guidance

The [1.0 candidate migration notes](migration-to-1.0.md) identify correctness
fixes that can change scientific outputs, affected source/capture patterns, and
the preserve-rerun-compare procedure. Old bundles retain their stored identities;
verification is not a retroactive proof of correct producer execution. No
in-place scientific-artifact migration is provided.

## Pending 1.0 raw-spec migration

The candidate changes the experimental Rust `RawExperimentSpec.captures` and
`.interventions` fields to `Option<Vec<RawDefinition<T>>>`. Use `None` for an
omitted list and `Some(vec![])` for an explicit empty list; construct authored
programmatic entries with `RawDefinition::explicit(value)?`. This preserves
nested omission provenance until resolution. User TOML and the resolved
`CaptureSpec`/`InterventionSpec` types keep their wire formats. Consumers must
not assume a fixed length for the resolved spec's `defaults` array.

## Compatibility levels

Every new Rust surface should be placed in one of these levels in its module
(or item) documentation:

1. **Supported compatibility anchors.** The documented CLI behavior and
   explicitly versioned artifact identifiers are the interfaces intended for
   automation and long-lived data. Unknown schema majors fail closed; a
   change in field meaning, interpretation, or deterministic identity requires
   a new schema/contract version and a migration note.
2. **Experimental public Rust API.** A module may remain `pub` so the binary,
   examples, and research integrations can use it, while its docs say that it
   is experimental. It may change in a `0.x` minor release. The module owner
   must document invariants, error/panic behavior, feature requirements, and
   numerical/determinism expectations.
3. **Internal implementation.** Helpers and hot-path details should be
   private or `pub(crate)`. A small set of legacy/internal modules remains
   source-visible for the separate binary target and existing integrations:
   `alloc_counter`, `atomic_file`, `decode_profile`, `k_matmul`,
   `k_quant_matmul`, `model_backend`, `npy`, `planned_decode`, `quant_fault`,
   `residency`, `simd`, and `workspace`. They are marked `#[doc(hidden)]` in
   `lib.rs`. `doc(hidden)` is not a promise of compatibility; it only prevents
   accidental discovery in generated API documentation. Removing one still
   requires the versioning rules below.

All other currently public module paths are experimental Rust API by
default; none is a 1.0-stable Rust surface yet. A future stable surface will
be explicitly listed in this policy and its module docs.

Do not add a new top-level `pub mod` merely to share code with the binary.
Prefer `pub(crate)` or a deliberate re-export from a documented module. Do not
use `#[doc(hidden)]` to conceal a supported API or a breaking change.

## Semver and MSRV rules

- Ember follows [Semantic Versioning](https://semver.org/) for supported
  surfaces and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) for
  release notes.
- While the crate is below 1.0, a minor release (`0.6` -> `0.7`) is the
  planned boundary for intentional breaking Rust/CLI changes. Patch releases
  (`0.6.z`) must not knowingly break a supported surface. We still avoid
  breakage wherever practical and every intentional break needs a migration
  note. Once Ember reaches 1.0, normal major/minor/patch SemVer rules apply.
- Additive APIs, new optional CLI flags, and new optional fields are normally
  minor-release changes. Removing or renaming an item, changing a required
  argument, tightening accepted input, changing a feature's default, or
  changing an artifact's meaning is breaking even if Rust still compiles.
- A correctness or security fix may change previously incorrect behavior in a
  patch release. The changelog must call out the observable change and include
  the affected validation evidence; do not silently relabel output changes as
  refactors.
- Rust 1.92 remains supported for patch releases. Raising the MSRV requires a
  minor release before 1.0 (a major release after 1.0), a changelog entry, and
  a passing MSRV build. New dependencies must be checked against the declared
  MSRV; the lockfile alone is not an MSRV guarantee.
- Cargo features are part of the public build surface. Adding a feature is
  usually additive; removing a feature, changing its default, or moving an API
  behind a feature requires the same review as a source break. In particular,
  keep `--no-default-features` headless builds valid.

## Public API design and review

Before exposing an item, ask whether a downstream caller needs it. Public
items should have rustdoc that states ownership/lifetimes, accepted shapes and
units, error and panic conditions, feature gates, determinism, and any
quantitative compatibility envelope. Keep representation details private;
prefer constructors and non-exhaustive enums when future extension is likely.
Do not expose a type solely because an internal implementation currently uses
it.

A public API change requires an issue or design note and review of:

1. source and generated-doc diffs (including re-exports and feature matrices);
2. semver/MSRV impact and a migration path;
3. numerical, serialized, and CLI behavior (including bit-exact contracts
   where one exists);
4. focused tests at the changed boundary and updates to the relevant docs and
   changelog.

For versioned artifacts, preserve old readers/writers where the contract says
so, reject unknown major versions, and never reinterpret an existing field or
identifier in place. Put machine-dependent measurements in diagnostic fields,
not in semantic identity.

## Deprecation and removal

Deprecate before removal whenever a supported Rust API can be migrated:

```rust
#[deprecated(since = "0.7.0", note = "use `new_name`; removal is planned for 0.8")]
pub fn old_name() {}
```

The deprecation note must name the replacement and appear in the changelog and
migration documentation. Keep the old item working for at least one minor
release (or document why a security issue makes that impossible); do not add a
new deprecation and remove the item in the same release. Before 1.0, removal
belongs in the next minor release, never a patch release. Schema fields are
retired by introducing a new schema version, not by silently changing their
meaning.

## Release checklist

A release PR should:

- update `Cargo.toml`/`Cargo.lock` only as needed, `CHANGELOG.md`, and the
  version tag (`vX.Y.Z`), with a migration note for any break;
- verify the declared MSRV and both default and headless feature builds;
- run the repository gates:

  ```text
  cargo fmt --all -- --check
  cargo test --locked --all-targets
  cargo clippy --locked --all-targets --all-features -- -D warnings
  cargo doc --locked --no-deps
  cargo check --locked --no-default-features --all-targets
  ```

- run the relevant golden-logit, activation, artifact, or CLI contract tests;
- record intentional API/output changes in the release notes.

The existing rustdoc and headless-build CI checks are the current guardrail.
Once a stable Rust API is intentionally declared and a baseline release is
available, add an automated public-API diff (for example,
`cargo-semver-checks`) to release CI rather than checking an unstable baseline
now.

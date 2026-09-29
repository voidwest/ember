# Preparing experiments for the 1.0 candidate

Ember 1.0 has not shipped. The candidate still reports package version 0.6.8;
record the executable SHA-256 and source revision as well as the package version
when comparing local candidate builds. This document describes observed
correctness changes, not a promise that all release gates are complete.

## Experiments whose results may change

| Spec or workflow | Earlier behavior | Corrected behavior |
|---|---|---|
| `generated-step` capture or intervention, especially step 1 | Could also fire during prefill; the verifier rejected absolute generated positions as beyond the prompt | Fires only during decode; the verifier checks `prompt_len + step - 1` |
| `prompt-final` at `final-norm-output` or `logits` | The one-row head tensor was addressed as prompt position zero, so the selected hook could be missed | The head row represents the final prompt position |
| A current-run capture spanning multiple layers used as an intervention source | Lookup could select the first stored layer rather than the target layer | Source must match capture ID, input, semantic site, and target layer |
| A multi-row capture used by `replace`, `interpolate`, or `add-delta` | Replacement could panic; arithmetic operations could silently use only the first source row | One row broadcasts; otherwise source and selected target row counts must match and are applied in selector order |
| A cross-bundle source targeting a different layer, or multiple target layers | Explicit source layer was not compared with every target layer | Mismatched target layers are rejected before source loading |

The current-run layer correction changed the generated output of the supplied
Arabic morphology intervention. Its baseline and exact restoration still
match. Successful restoration alone therefore does **not** prove that an old
intervention used the intended source layer. Recheck the scientific conclusion
of affected interventions using a fresh run.

These are corrections to the documented addressing and compatibility rules;
they do not rename the six hook sites or rewrite the meanings of stored tensors.
An old tensor still records what its producer actually computed. Offline bundle
verification checks stored integrity and consistency; it cannot retroactively
prove that a historical producer executed the intended semantics correctly.

## Preserve, rerun, and compare

1. Keep the original bundle and its checksums unchanged. Do not edit its producer
   version, spec, captures, plan, or semantic hash to make a comparison pass.
2. Record the old and new executable hashes, model/tokenizer hashes, thread count,
   execution mode, prompt, and original spec. Prefer the same supported machine
   and numerical strategy when isolating one of these fixes.
3. Validate the authored spec with the candidate. If it now fails source
   compatibility, correct the source/target addressing explicitly and save that
   as a new spec. Cross-layer transfer is not an implicit feature of this schema.
4. Run into a **new output directory**, without overwriting the old bundle.
   Verify the new bundle, then compare it with the old bundle. Record capture and
   token differences as results, not as an artifact-format migration.
5. For restoration experiments, compare the corrected intervention and corrected
   restoration against a freshly produced baseline. Inspect source layer/site
   selection as well as output equality.

The corresponding CLI operations are `ember experiment validate`, `run`,
`verify`, and `compare`; see [the experiment guide](experiment-schema-v1.md)
and [usage](usage.md). The checked Arabic workflow is available as
`scripts/validate_morphology_example.py --workdir <new-directory>`.

A rerun is a new scientific artifact. There is no command in this candidate
that rewrites an old bundle into a scientifically equivalent corrected bundle.
Unknown contract versions fail rather than being silently reinterpreted.

## Other candidate compatibility changes

- Experimental Rust raw-spec callers must represent captures/interventions as
  `Option<Vec<RawDefinition<T>>>`. Use `None` for omission and `Some(vec![])`
  for an explicit empty list. Build explicit entries with
  `RawDefinition::explicit(value)?`. Authored TOML field names are unchanged.
- Resolved default records now retain nested omission provenance. Do not depend
  on a fixed number or array length of default records.
- Canonical JSON sorting fixes can change newly produced payload and semantic
  hashes. The verifier preserves supported historical bundle identities; it does
  not reserialize and overwrite archived bundles.
- GUI builds require the pinned Rust 1.98.1 toolchain. The documented headless
  route retains Rust 1.92. See [API stability](api-stability.md).

## Timeout errors and capture row ordering

Python `diff` and `diff_corpus` reject finite positive timeout values that cannot
fit Rust's duration representation with `ValueError`. Previously a value such
as `1e300` could panic or reach filesystem work before rejection. Callers should
handle this as an invalid argument and supply a representable timeout. Existing
positive, representable timeouts retain their behavior; zero, negative and
non-finite values remain invalid.

When capture buffers arrive out of position order, the candidate sorts token
positions and their corresponding tensor rows together. Consumers should use
the stored position-to-row mapping rather than assume buffer arrival order.
Ordinary sequential decoding already visits positions in order. The regression
checks out-of-order buffers and exact row bits; it does not claim a change to
ordinary decode outputs. Preserve historical bundles and rerun affected custom
capture workflows into a new directory rather than rewriting stored tensors.

## Evidence and limits

The [contract audit](audits/1.0-contract-audit.md) records the regression tests,
real-model comparisons, original failures, and local executable identities.
These include a six-site capture matrix, an independent lower-level observer,
72 zero/restore ordering runs, source-shape rejection, source-layer addressing,
and equivalent current-run/cross-bundle sources. Their scope is stated with the
results; they are not universal numerical equivalence claims.

The Q6 compressed-versus-eager continuation discrepancy remains unresolved;
see the [numerical investigation](audits/1.0-q6-numerics.md). Final candidate
validation, performance/memory checks, clean installation, GUI acceptance, and
external-user evidence remain governed by the [release roadmap](road-to-1.0.md).

## Run-manifest identity migration

The candidate writer now declares `execution-identity-v2` inside manifest schema
version 2. V2 sorts object keys recursively before hashing; arrays retain order.
It uses a new identity version because older development writers emitted v1
digests that depended on nested insertion order, contrary to the earlier sorting
description. The candidate verifier retains v1's stored encoding and digest.
Keep historical v1 objects in their recorded key order; do not sort, relabel,
or rehash them in place. A fresh v2 run has a new identity, not an equivalent
migration of an old scientific result. Old readers may reject v2.

Manifest verification now checks outer manifest version, identity version, and
matching inner canonical version before accepting the digest. Signed run
manifests use the same checks after signature verification; missing identity
fields cannot count as verified. Arbitrary signed JSON remains supported without
an execution-identity claim. Focused CLI tests now cover the retained v1 fixture, v2 recursive key ordering,
array-order sensitivity, malformed versions and signed-record boundaries on
Rust 1.98.1 and headless 1.92. The active performance matrix uses the earlier
release binary; these checks do not establish a frozen release candidate.

## Experiment verdict exit codes

Completed `experiment verify` failures and unsuccessful `experiment reproduce`
verdicts now use exit code 3, matching the shared verification-failure category.
They previously used codes 1 and 2 respectively. Scripts that special-case those
old codes should check code 3 and retain the emitted report for details. Usage
errors remain code 2 and execution errors remain code 1. Command handlers return
the shared verdict error so cleanup can finish before the CLI exits.

## Candidate RoPE arithmetic and kernel revision 4

The working candidate changes RoPE frequency construction to the pinned
reference's recurrence and explicitly fuses the first product in each rotation
output. Reference and planned paths use the same order; x86 vector dispatch now
requires both AVX2 and FMA, with a fused scalar fallback. This is a numerical
change, recorded as kernel revision 4. Preserve prior plan/bundle identities and
rerun into new output directories when comparing results. The measured benchmark
binaries still implement the earlier revision.

The isolated position-15 Q/K diagnostic established exact agreement using these
operations, but complete candidate validation is pending. Do not interpret this
migration note as a claim that generation parity or all release gates pass.

## Candidate ARM dot arithmetic and kernel revision 5

The working candidate extends revision 4 with a four-accumulator ARM f32 dot
reduction and reference tail ordering (separate products in groups of four,
then fused accumulation for the final one to three elements). These changes can affect attention
scores and other users of the shared dot helper. Revision 5 distinguishes this
candidate from recorded revision-4 runs. Preserve older identities and rerun
comparisons into new directories. The independent fixture demonstrated a
pre-change difference; complete candidate validation remains pending.

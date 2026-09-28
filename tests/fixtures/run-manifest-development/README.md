# Retained development run identity

`identity.json` preserves the schema version and exact identity object extracted
from `/run/cpu` of an earlier local `capture-patch-macos-verified/run-a/manifest.json`.
Other reporting fields are omitted; this is an identity fixture, not a complete
run manifest. The source artifact SHA-256 was
`18d849ffd84e9a8be9185758e73f8945afb0184fa470f1fa1c88ebe2e516181d`.
The producer was a dirty development tree based on
`85c2382f43d6976b95b5d0fe22a2312c9157f8f7`, Rust 1.98.1, aarch64-apple-darwin.
It is **not** a tagged-release compatibility fixture.

Stored digest (and compact insertion-order digest):
`9ecd501e9ac8ba978e7981cc3bf38d2d1c898d2f60ed6032ddfb54bffa225926`.
Recursively sorted digest of the same values:
`1fcec52f664c347aaee0e03ec414ed56df70f6661855bd59a2cdb98cfd43905a`.
The existing candidate CLI accepts the stored identity. Preserve its digest and
key order; do not regenerate it from a corrected writer to make tests pass.
A forthcoming verifier regression must exercise compatibility explicitly.

# EmberSEC: tokenizer JSON boundary

> **Phase I provenance:** frozen audit documentation from branch snapshot
> `e1fe6269`; the measured hardened Ember target is `3ceb7039`. Current main
> retains the applicable hardening, but implementation names and dataflow may
> have evolved. Read this as the published Phase I evidence record.

Status: audit + hardening of the tokenizer-file trust boundary
(EmberTokenizer in src/tokenizer.rs, `tokenizers` crate 0.20.4).

## 1. Dataflow

```
tokenizer.json (attacker-controlled bytes)
  -> EmberTokenizer::from_file / from_bytes
       size cap (MAX_TOKENIZER_BYTES = 256 MiB)
       UTF-8 gate (JSON is UTF-8 by spec)
       serde_json well-formedness gate (IgnoredAny, no tree allocation)
       tokenizers crate: serde_json deserialize -> BPE/Unigram/WordPiece
       model, normalizer/pre_tokenizer/decoder/post_processor configs
       (Oniguruma regex FFI compiles patterns from the file)
  -> EmberTokenizer { inner: Tokenizer }
  -> encode / decode / validate_model_vocab / eos_token_ids
```

## 2. Trust surfaces

- **serde_json** inside the tokenizers crate: safe Rust, but see the
  panic findings below.
- **Oniguruma (C FFI, `onig` crate)**: regex patterns from the file
  (`normalizer`, `pre_tokenizer`, `decoder` regexes) are compiled by C
  code at load and matched at encode time. Compile errors are reported;
  catastrophic backtracking (ReDoS) is possible at encode time on
  hostile patterns and is inherent to regex tokenizers — no match-time
  timeout is set by the crate. Inputs are user text plus tokenizer
  patterns; CPU exhaustion via crafted patterns is a theoretical
  concern, not demonstrated.
- **Model-vocab bridge**: token IDs produced by the tokenizer are
  checked against the model embedding table via
  `validate_model_vocab` (max token ID < model vocab rows) before
  embedding lookup in the CLI paths.

## 3. Existing Ember-side checks (pre-audit)

- `encode_with_offsets` validates offset monotonicity and byte bounds.
- `validate_model_vocab` checks max token ID vs model vocab.
- `eos_token_ids` handles family-specific EOS tokens.
- No size cap existed before this audit: `Tokenizer::from_file` read
  the entire file into memory.

## 4. Findings (all fixed on this branch)

1. **[confirmed crash, upstream crate]** tokenizers-0.20.4
   `src/decoders/mod.rs:90` runs
   `DecoderHelper::deserialize(deserializer).expect("Helper")` — a
   PANIC on malformed JSON instead of a serde error. Found by the
   tokenizer_json fuzz target (26-byte repro: `{"decoder": <invalid
   UTF-8>}`; also reachable with valid-UTF-8 structurally-bad values).
   In the normal binary this aborts the process on a hostile
   tokenizer.json. The crate has further `panic!`/`.expect` sites
   (models/mod.rs, added_vocabulary.rs trie build, normalizer.rs,
   truncation.rs) that are reachable from crafted files.
   Fix at the Ember boundary (the crate is not vendored):
   - **Well-formedness gate** in `EmberTokenizer::from_bytes`: explicit
     UTF-8 check + serde_json structural check (`IgnoredAny`, no tree
     allocation) so the crate only ever sees well-formed JSON. This
     works in all builds, including panic=abort fuzz builds where
     catch_unwind is dead code.
   - **catch_unwind** around `Tokenizer::from_bytes`
     (`parse_tokenizer`) converts any remaining crate panic into a
     structured `anyhow::Error` in unwind builds; the partially-built
     tokenizer is discarded, so no inconsistent state survives.
2. **[resource exhaustion]** no file-size bound before parsing; a
   multi-GB tokenizer.json was read in full. Fixed with
   `MAX_TOKENIZER_BYTES = 256 MiB` (real tokenizers are 1-35 MB),
   enforced in from_file (sparse-file test) and from_bytes.

## 5. Tests

- `test_tokenizer_hostile_payload_is_structured_error` — the fuzz
  crash bytes now return Err("tokenizer JSON ...").
- `test_tokenizer_size_cap_rejects_huge_file` — sparse 256 MiB+ file
  rejected.
- `tokenizer_json_never_panics` proptest (tests/property.rs) — 256
  arbitrary-byte payloads: Ok/Err, never panic; parsed tokenizers are
  exercised through encode/decode/vocab validation.
- fuzz/corpus/tokenizer_json seeds (valid mini BPE + malformed
  variants) verified by the corpus-sync test.
- fuzz target `tokenizer_json` (fuzz/fuzz_targets/tokenizer_json.rs):
  after the fixes, 300 s of libFuzzer ran with zero crashes.

## 6. Known upstream issues (documented, not fixable from here)

- The tokenizers crate's panic-on-malformed-JSON deserializers
  (upstream bug). Ember's boundary converts them to structured errors;
  a vendored patch would be needed to fix the crate itself.
- Oniguruma regex ReDoS at encode time: theoretical CPU-exhaustion
  concern on crafted tokenizer patterns; no demonstrated input.

# Corpus taxonomy

| id | name | input_type | origin | bug_class | format_status | comparability | coverage | size_bytes | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| gguf-001 | valid-f32 | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors | 128 | 8244581d6d03 |
| gguf-002 | valid-f16 | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors | 128 | acf41cf10268 |
| gguf-003 | valid-bf16 | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors | 128 | 0d90c368d1b1 |
| gguf-004 | valid-q8_0 | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors+quantization layout | 164 | 8dffc38ddb53 |
| gguf-005 | valid-q4_k | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors+quantization layout | 384 | 49631fcadf20 |
| gguf-006 | valid-q6_k | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors+quantization layout | 516 | c6e7f33e921d |
| gguf-007 | valid-metadata | GGUF | control | control | valid | FULLY_COMPARABLE | metadata+strings/arrays | 224 | 9fa1976aa807 |
| gguf-008 | valid-eof-exact | GGUF | control | control | valid | FULLY_COMPARABLE | extent arithmetic | 160 | c69fd55bc8b1 |
| gguf-009 | valid-two-tensors | GGUF | control | control | valid | FULLY_COMPARABLE | tensor descriptors+overlap/range | 192 | 40c054f5add9 |
| gguf-010 | bad-magic | GGUF | regression fixture | A | format-invalid | FULLY_COMPARABLE | header/count | 24 | 9d908ecfb6b2 |
| gguf-011 | unsupported-version | GGUF | regression fixture | A | format-invalid | FULLY_COMPARABLE | header/count | 24 | 69cb86ffffe1 |
| gguf-012 | huge-tensor-count | GGUF | fuzz corpus seed | G | semantically hostile | FULLY_COMPARABLE | header/count | 24 | e96aa52c3d88 |
| gguf-013 | huge-kv-count | GGUF | fuzz corpus seed | G | semantically hostile | FULLY_COMPARABLE | header/count+metadata | 24 | 3eb0a7093dca |
| gguf-014 | truncated-header | GGUF | fuzz corpus seed | A | format-invalid | FULLY_COMPARABLE | header/count | 8 | 527ee9a8eac0 |
| gguf-015 | truncated-tensor-data | GGUF | fuzz corpus seed | B | format-invalid | FULLY_COMPARABLE | extent arithmetic | 116 | 47de1b5e6059 |
| gguf-016 | offset-past-eof | GGUF | fuzz corpus seed | B | semantically hostile | FULLY_COMPARABLE | extent arithmetic | 120 | 8e8b1354dea6 |
| gguf-017 | offset-u64-max | GGUF | fuzz corpus seed | B | semantically hostile | FULLY_COMPARABLE | extent arithmetic | 72 | 92997e309809 |
| gguf-018 | dim-product-overflow | GGUF | fuzz corpus seed | B | semantically hostile | FULLY_COMPARABLE | extent arithmetic+tensor descriptors | 96 | c481af3777d5 |
| gguf-019 | rank-zero | GGUF | fuzz corpus seed | D | format-invalid | FULLY_COMPARABLE | tensor descriptors | 64 | f30ba02ed2c3 |
| gguf-020 | rank-five | GGUF | fuzz corpus seed | D | format-invalid | FULLY_COMPARABLE | tensor descriptors | 96 | 8408bf6a5450 |
| gguf-021 | zero-dimension | GGUF | fuzz corpus seed | D | format-invalid | FULLY_COMPARABLE | tensor descriptors | 96 | f200bdec8c12 |
| gguf-022 | unsupported-dtype-99 | GGUF | fuzz corpus seed | J | format-invalid | FULLY_COMPARABLE | tensor descriptors | 128 | 2675c7ae7a6c |
| gguf-023 | q4_0-unimplemented | GGUF | fuzz corpus seed | J | Ember-unsupported | PARTIALLY_COMPARABLE | tensor descriptors+quantization layout | 160 | 5553207932b2 |
| gguf-024 | q8_0-dim-misaligned | GGUF | fuzz corpus seed | D | semantically hostile | FULLY_COMPARABLE | quantization layout | 1184 | 23f857f84279 |
| gguf-025 | q4_k-dim-misaligned | GGUF | regression fixture | D | semantically hostile | FULLY_COMPARABLE | quantization layout | 384 | 15b492cfb468 |
| gguf-026 | q4_k-truncated | GGUF | fuzz corpus seed | B | format-invalid | FULLY_COMPARABLE | extent arithmetic+quantization layout | 240 | 1d5a13e98400 |
| gguf-027 | overlapping-ranges | GGUF | fuzz corpus seed | B | semantically hostile | FULLY_COMPARABLE | overlap/range | 152 | 36717bdd0ca0 |
| gguf-028 | duplicate-tensor-name | GGUF | fuzz corpus seed | A | semantically hostile | FULLY_COMPARABLE | tensor descriptors | 188 | 7452cce844d3 |
| gguf-029 | huge-string-length | GGUF | fuzz corpus seed | G | semantically hostile | FULLY_COMPARABLE | strings/arrays | 45 | 82d2c4e2f5a1 |
| gguf-030 | string-longer-than-file | GGUF | fuzz corpus seed | A | semantically hostile | FULLY_COMPARABLE | strings/arrays | 45 | ffe3f1b966c5 |
| gguf-031 | huge-array-count | GGUF | fuzz corpus seed | G | semantically hostile | FULLY_COMPARABLE | strings/arrays | 49 | 2b4e9483b0db |
| gguf-032 | deep-nested-arrays | GGUF | fuzz corpus seed | A | semantically hostile | FULLY_COMPARABLE | metadata+strings/arrays | 258 | 0f0c2f4ee5d1 |
| gguf-033 | bad-bool-metadata | GGUF | fuzz corpus seed | A | format-invalid | FULLY_COMPARABLE | metadata | 38 | f06f982b1578 |
| gguf-034 | bad-metadata-value-type | GGUF | fuzz corpus seed | A | format-invalid | FULLY_COMPARABLE | metadata | 37 | 65332c9a5bdf |
| gguf-035 | empty-metadata-key | GGUF | fuzz corpus seed | A | format-invalid | FULLY_COMPARABLE | metadata | 37 | 2cef2535ce94 |
| gguf-036 | bad-alignment | GGUF | fuzz corpus seed | A | semantically hostile | FULLY_COMPARABLE | metadata+extent arithmetic | 57 | 6e08a971b257 |
| gguf-037 | empty-tensor-name | GGUF | fuzz corpus seed | A | format-invalid | FULLY_COMPARABLE | tensor descriptors | 56 | a27c70debe46 |
| gguf-038 | offset-plus-size-overflow | GGUF | structured synthetic boundary case | B | semantically hostile | FULLY_COMPARABLE | extent arithmetic | 64 | 98f28b340cb4 |
| gguf-039 | huge-dimension-past-eof | GGUF | structured synthetic boundary case | B | semantically hostile | FULLY_COMPARABLE | extent arithmetic | 64 | 7c23c97a7150 |
| gguf-040 | tensor-count-above-abs-cap | GGUF | structured synthetic boundary case | G | semantically hostile | FULLY_COMPARABLE | header/count | 24 | 8a52817a8994 |
| gguf-041 | tiny-llama-valid | GGUF | regression fixture | control | valid | FULLY_COMPARABLE | model construction+architecture metadata | 3968 | 40538313cf33 |
| gguf-042 | llama-context-u32-max | GGUF | fuzz corpus seed | G | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata+model construction | 3968 | adac72cd8427 |
| gguf-043 | llama-block-count-1m | GGUF | fuzz corpus seed | G | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata+model construction | 3968 | ec4f299f351e |
| gguf-044 | llama-missing-tensors | GGUF | fuzz corpus seed | C | semantically hostile | EMBER_SPECIFIC | model construction | 102 | 533e6fd5ac04 |
| gguf-045 | llama-odd-key-length | GGUF | fuzz-discovered class; canonical synthetic fixture | E | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata+model construction | 3968 | 01a1ee0adc2d |
| gguf-046 | llama-1d-attn-q | GGUF | structured synthetic boundary case | E | semantically hostile | PARTIALLY_COMPARABLE | tensor descriptors+model construction | 3712 | 56423834720d |
| gguf-047 | llama-rope-product-cap | GGUF | structured synthetic boundary case | G | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata+model construction | 3968 | 484b7653d855 |
| gguf-048 | llama-unknown-architecture | GGUF | structured synthetic boundary case | C | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata | 68 | cdef83e381f0 |
| gguf-049 | llama-vocab-5m | GGUF | structured synthetic boundary case | C | semantically hostile | PARTIALLY_COMPARABLE | architecture metadata | 3968 | 9d75995c97ac |
| gguf-053 | alignment-zero | GGUF | fuzz-discovered class; canonical synthetic fixture | A | semantically hostile | FULLY_COMPARABLE | metadata+extent arithmetic | 57 | ac4c8ef93b38 |
| gguf-050 | tiny-llama-full-valid | GGUF | structured synthetic boundary case | control | valid | FULLY_COMPARABLE | model construction+architecture metadata+tokenizer JSON | 25632 | 6b4ade704210 |
| gguf-051 | tiny-qwen3-full-valid | GGUF | structured synthetic boundary case | control | valid | FULLY_COMPARABLE | model construction+architecture metadata+tokenizer JSON | 25792 | 8238be776625 |
| gguf-052 | tiny-gemma4-full-valid | GGUF | structured synthetic boundary case | control | valid | PARTIALLY_COMPARABLE | model construction+architecture metadata | 2624 | b8746d6dd14c |
| tok-001 | valid-mini-bpe | TOKENIZER_JSON | control | control | valid | TOKENIZER_ONLY | tokenizer JSON | 401 | 10a747146623 |
| tok-002 | decoder-invalid-utf8-26b | TOKENIZER_JSON | fuzz-discovered minimized artifact (reconstructed) | F | semantically hostile | TOKENIZER_ONLY | tokenizer JSON | 26 | 7ebe2a69fbca |
| tok-003 | decoder-bad-value-15b | TOKENIZER_JSON | fuzz-discovered minimized artifact (reconstructed) | F | semantically hostile | TOKENIZER_ONLY | tokenizer JSON | 15 | 60af269e6dce |
| tok-004 | decoder-invalid-utf8-nested | TOKENIZER_JSON | fuzz-discovered class; canonical synthetic fixture | F | semantically hostile | TOKENIZER_ONLY | tokenizer JSON | 35 | 0865c6dfe6fa |
| tok-005 | truncated-json | TOKENIZER_JSON | fuzz corpus seed | A | format-invalid | TOKENIZER_ONLY | tokenizer JSON | 57 | e4c010313855 |
| tok-006 | not-json | TOKENIZER_JSON | fuzz corpus seed | A | format-invalid | TOKENIZER_ONLY | tokenizer JSON | 13 | 8fde1f069125 |
| tok-007 | deep-nesting | TOKENIZER_JSON | fuzz corpus seed | A | semantically hostile | TOKENIZER_ONLY | tokenizer JSON | 466 | 12a4fe9b06b9 |
| tok-008 | bad-utf8-vocab | TOKENIZER_JSON | fuzz corpus seed | A | format-invalid | TOKENIZER_ONLY | tokenizer JSON | 60 | 1836cf372135 |
| tok-009 | valid-json-unknown-top-level-key | TOKENIZER_JSON | fuzz corpus seed | J | Ember-unsupported | TOKENIZER_ONLY | tokenizer JSON | 3062 | 8e9e175e853d |

# Token selection (v0.5)

Token selection is exact, typed, and fail-closed. Every selection
produces a machine-readable record; no selector ever guesses from
decoded strings.

## Selectors

- `prompt-final` — the final token of the complete model input (BOS
  included).
- `absolute-token { index }` — the token at a 0-based model-input
  position.
- `relative-token { offset_from_end }` — position `seq_len - 1 - offset`;
  offset 0 equals `prompt-final`.
- `generated-step { step }` — the token generated at 1-based decode step
  N, observed at the decode evaluation that processes it.
- `matched-span { text, occurrence, subtokens, normalization }` — exact
  text-span match.
- `byte-span { start, end, subtokens }` — explicit byte span into the
  prompt.

`subtokens` selects `first`, `final`, or `all` of the tokenizer's tokens
whose byte intervals intersect the span.

## Offsets and alignment

The `tokenizers` crate reports **byte offsets** relative to the original
(normalized) string. Ember records them verbatim: no char↔byte
conversion is applied, so multibyte text (Arabic, diacritics, emoji)
aligns exactly. The wrapper validates every offset against the byte
length and monotonicity before selection.

Coverage is classified:

- `exact` — the union of selected token intervals equals the span;
- `enclosing` — the union strictly contains the span; the boundary
  expansion (extra bytes pulled in by token boundaries) is recorded,
  never silently repaired;
- `none` — no token covers the span (selection fails).

## Fail-closed rules

- absent span → error naming the selector and reason;
- requested occurrence beyond the count → error;
- empty span, reversed byte spans, out-of-range positions → error;
- `generated-step 0` → error (steps are 1-based);
- ambiguous matches are resolved only by `occurrence`; without it, the
  first occurrence is selected and recorded (occurrence defaults to 0);
- normalization is never applied silently: `TextNormalization::None` is
  the default; `nfc` is an explicit opt-in that matches against an
  NFC-normalized copy (both forms recorded).

## Records

Each selection records: the selector, rule id, original and normalized
text, tokenizer IDs, pieces, byte offsets, matched byte span, selected
indices, coverage kind, boundary expansion, ambiguity status, round-trip
status, and any fallback or note.

## Arabic examples

With the pinned Llama-3.2-1B tokenizer:

```text
ember experiment tokenize --model Llama-3.2-1B-Instruct-Q8_0.gguf \
  --arch llama --tokenizer tokenizer.json \
  --text "في الجملة التالية، الكلمة المميزة هي: كِتَاب. اشرح معناها." \
  --match-span "كِتَاب"
```

reports the five subtokens covering the diacritized word (bytes
69..81 of the prompt), classified `enclosing` because the tokenizer's
first covering token also includes the preceding space.

Diacritics matter: matching `كتاب` (undiacritized) against a prompt
containing `كِتَاب` **fails** — Ember never strips combining marks.

Repeated spans: `occurrence = 0, 1, 2, ...` resolves the o-th
non-overlapping match deterministically; a missing occurrence fails.

# Qwen3-0.6B Parse Failure Manual Sample

Input file: `qwen3-0.6b.parse_failures.sample30.jsonl`

Sample: first 30 parse failures after accepting markdown-wrapped labels such as `**C**` as parseable.

| Category | Count |
| --- | ---: |
| malformed output | 28 |
| explanation only | 2 |
| no option selected | 0 |
| multiple options selected | 0 |
| invalid label | 0 |
| copied option text but no label | 0 |
| language mismatch | 0 |

| # | ID | Variant | Response excerpt | Category |
| ---: | --- | --- | --- | --- |
| 1 | 300 | arabic-instruction | `___________\n\nال` | malformed output |
| 2 | 300 | option-order-shuffle | `The correct answer` | explanation only |
| 3 | 300 | option-label-shuffle | `___________\n\nال` | malformed output |
| 4 | 300 | normalized-orthography | `___________\n\nال` | malformed output |
| 5 | 526 | arabic-instruction | `___________\n\nال` | malformed output |
| 6 | 526 | option-order-shuffle | `___________\n\nال` | malformed output |
| 7 | 526 | option-label-shuffle | `___________\n\nال` | malformed output |
| 8 | 526 | normalized-orthography | `___________\n\nال` | malformed output |
| 9 | 1724 | arabic-instruction | `___________\n\nال` | malformed output |
| 10 | 1724 | option-order-shuffle | `____\n\nالإ` | malformed output |
| 11 | 1724 | option-label-shuffle | `____\n\nالإ` | malformed output |
| 12 | 1724 | normalized-orthography | `___________\n\nال` | malformed output |
| 13 | 1868 | arabic-instruction | `___________\n\nال` | malformed output |
| 14 | 1868 | option-order-shuffle | `___________\n\nال` | malformed output |
| 15 | 1868 | option-label-shuffle | `___________\n\nال` | malformed output |
| 16 | 1868 | normalized-orthography | `____\n\nالإ` | malformed output |
| 17 | 1934 | option-order-shuffle | `____\n\nالإ` | malformed output |
| 18 | 1934 | option-label-shuffle | `____\n\nالإ` | malformed output |
| 19 | 2250 | normalized-orthography | `____\n\nالإ` | malformed output |
| 20 | 2341 | normalized-orthography | `____\n\nالإ` | malformed output |
| 21 | 2381 | option-order-shuffle | `___________\n\nال` | malformed output |
| 22 | 2381 | option-label-shuffle | `___________\n\nال` | malformed output |
| 23 | 2386 | option-order-shuffle | `___________\n\nال` | malformed output |
| 24 | 2386 | option-label-shuffle | `___________\n\nال` | malformed output |
| 25 | 2407 | option-label-shuffle | `The correct answer` | explanation only |
| 26 | 2540 | normalized-orthography | `___________\n\nال` | malformed output |
| 27 | 2614 | arabic-instruction | `___________\n\nال` | malformed output |
| 28 | 2614 | option-order-shuffle | `____\n\nالإ` | malformed output |
| 29 | 2614 | option-label-shuffle | `___________\n\nال` | malformed output |
| 30 | 2614 | normalized-orthography | `___________\n\nال` | malformed output |

Interpretation: in this sample, Qwen3-0.6B parse failures are overwhelmingly malformed or truncated outputs, not cases where the model copied an option text or emitted multiple labels.

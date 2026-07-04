# Prompt Perturbation Output Scores

- Rows: 300
- Parseable rows: 215
- Accuracy all: 0.273
- Accuracy parseable: 0.381
- Semantic accuracy all: 0.273
- Semantic accuracy parseable: 0.381
- Parse failure rate: 0.283
- Variant consistency: 0.200 (10/50)

| Variant | Rows | Acc all | Acc parseable | Semantic acc all | Semantic acc parseable | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 50 | 0.200 | 0.323 | 0.200 | 0.323 | 19 (0.380) |
| english-instruction | 50 | 0.240 | 0.240 | 0.240 | 0.240 | 0 (0.000) |
| normalized-orthography | 50 | 0.260 | 0.433 | 0.260 | 0.433 | 20 (0.400) |
| option-label-shuffle | 50 | 0.260 | 0.500 | 0.260 | 0.500 | 24 (0.480) |
| option-order-shuffle | 50 | 0.300 | 0.536 | 0.300 | 0.536 | 22 (0.440) |
| strict-output | 50 | 0.380 | 0.380 | 0.380 | 0.380 | 0 (0.000) |

## Prediction Distribution

| Variant | A | B | C | D | E | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 5 | 5 | 3 | 17 | 1 | 19 | 0 |
| english-instruction | 5 | 6 | 3 | 35 | 1 | 0 | 0 |
| normalized-orthography | 6 | 3 | 5 | 15 | 1 | 20 | 0 |
| option-label-shuffle | 1 | 9 | 5 | 11 | 0 | 24 | 0 |
| option-order-shuffle | 3 | 7 | 5 | 13 | 0 | 22 | 0 |
| strict-output | 36 | 1 | 1 | 12 | 0 | 0 | 0 |

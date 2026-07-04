# Prompt Perturbation Output Scores

- Rows: 250
- Parseable rows: 160
- Accuracy all: 0.244
- Accuracy parseable: 0.381
- Semantic accuracy all: 0.244
- Semantic accuracy parseable: 0.381
- Parse failure rate: 0.360
- Variant consistency: 0.600 (30/50)

| Variant | Rows | Acc all | Acc parseable | Semantic acc all | Semantic acc parseable | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 50 | 0.200 | 0.333 | 0.200 | 0.333 | 20 (0.400) |
| english-instruction | 50 | 0.240 | 0.240 | 0.240 | 0.240 | 0 (0.000) |
| normalized-orthography | 50 | 0.260 | 0.433 | 0.260 | 0.433 | 20 (0.400) |
| option-label-shuffle | 50 | 0.260 | 0.542 | 0.260 | 0.542 | 26 (0.520) |
| option-order-shuffle | 50 | 0.260 | 0.500 | 0.260 | 0.500 | 24 (0.480) |

## Prediction Distribution

| Variant | A | B | C | D | E | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 5 | 5 | 3 | 16 | 1 | 20 | 0 |
| english-instruction | 5 | 6 | 3 | 35 | 1 | 0 | 0 |
| normalized-orthography | 6 | 3 | 5 | 15 | 1 | 20 | 0 |
| option-label-shuffle | 1 | 9 | 4 | 10 | 0 | 26 | 0 |
| option-order-shuffle | 3 | 6 | 4 | 13 | 0 | 24 | 0 |

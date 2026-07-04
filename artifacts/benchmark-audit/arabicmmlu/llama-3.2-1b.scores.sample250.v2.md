# Prompt Perturbation Output Scores

- Rows: 250
- Parseable rows: 241
- Accuracy all: 0.304
- Accuracy parseable: 0.315
- Semantic accuracy all: 0.304
- Semantic accuracy parseable: 0.315
- Parse failure rate: 0.036
- Variant consistency: 0.140 (7/50)

| Variant | Rows | Acc all | Acc parseable | Semantic acc all | Semantic acc parseable | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 50 | 0.320 | 0.333 | 0.320 | 0.333 | 2 (0.040) |
| english-instruction | 50 | 0.300 | 0.300 | 0.300 | 0.300 | 0 (0.000) |
| normalized-orthography | 50 | 0.300 | 0.319 | 0.300 | 0.319 | 3 (0.060) |
| option-label-shuffle | 50 | 0.340 | 0.354 | 0.340 | 0.354 | 2 (0.040) |
| option-order-shuffle | 50 | 0.260 | 0.271 | 0.260 | 0.271 | 2 (0.040) |

## Prediction Distribution

| Variant | A | B | C | D | E | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 2 | 19 | 25 | 2 | 0 | 2 | 0 |
| english-instruction | 12 | 13 | 15 | 10 | 0 | 0 | 0 |
| normalized-orthography | 1 | 18 | 25 | 2 | 1 | 3 | 0 |
| option-label-shuffle | 1 | 29 | 15 | 3 | 0 | 2 | 0 |
| option-order-shuffle | 0 | 25 | 23 | 0 | 0 | 2 | 0 |

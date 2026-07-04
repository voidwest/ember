# Prompt Perturbation Output Scores

- Rows: 300
- Parseable rows: 298
- Accuracy all: 0.430
- Accuracy parseable: 0.433
- Semantic accuracy all: 0.430
- Semantic accuracy parseable: 0.433
- Parse failure rate: 0.007
- Variant consistency: 0.340 (17/50)

| Variant | Rows | Acc all | Acc parseable | Semantic acc all | Semantic acc parseable | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 50 | 0.440 | 0.449 | 0.440 | 0.449 | 1 (0.020) |
| english-instruction | 50 | 0.480 | 0.480 | 0.480 | 0.480 | 0 (0.000) |
| normalized-orthography | 50 | 0.420 | 0.420 | 0.420 | 0.420 | 0 (0.000) |
| option-label-shuffle | 50 | 0.440 | 0.440 | 0.440 | 0.440 | 0 (0.000) |
| option-order-shuffle | 50 | 0.400 | 0.408 | 0.400 | 0.408 | 1 (0.020) |
| strict-output | 50 | 0.400 | 0.400 | 0.400 | 0.400 | 0 (0.000) |

## Prediction Distribution

| Variant | A | B | C | D | E | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 10 | 11 | 15 | 12 | 1 | 1 | 0 |
| english-instruction | 13 | 14 | 13 | 10 | 0 | 0 | 0 |
| normalized-orthography | 9 | 14 | 17 | 10 | 0 | 0 | 0 |
| option-label-shuffle | 8 | 19 | 13 | 9 | 1 | 0 | 0 |
| option-order-shuffle | 8 | 19 | 13 | 9 | 0 | 1 | 0 |
| strict-output | 11 | 10 | 21 | 8 | 0 | 0 | 0 |

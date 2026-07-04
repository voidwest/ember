# Prompt Perturbation Output Scores

- Rows: 50
- Parseable rows: 46
- Accuracy all: 0.380
- Accuracy parseable: 0.413
- Semantic accuracy all: 0.380
- Semantic accuracy parseable: 0.413
- Parse failure rate: 0.080
- Variant consistency: 0.100 (1/10)

| Variant | Rows | Acc all | Acc parseable | Semantic acc all | Semantic acc parseable | Parse failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 10 | 0.500 | 0.556 | 0.500 | 0.556 | 1 (0.100) |
| english-instruction | 10 | 0.300 | 0.300 | 0.300 | 0.300 | 0 (0.000) |
| normalized-orthography | 10 | 0.400 | 0.444 | 0.400 | 0.444 | 1 (0.100) |
| option-label-shuffle | 10 | 0.400 | 0.444 | 0.400 | 0.444 | 1 (0.100) |
| option-order-shuffle | 10 | 0.300 | 0.333 | 0.300 | 0.333 | 1 (0.100) |

## Prediction Distribution

| Variant | A | B | C | D | E | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arabic-instruction | 2 | 2 | 5 | 0 | 0 | 1 | 0 |
| english-instruction | 2 | 1 | 3 | 4 | 0 | 0 | 0 |
| normalized-orthography | 1 | 3 | 5 | 0 | 0 | 1 | 0 |
| option-label-shuffle | 0 | 7 | 2 | 0 | 0 | 1 | 0 |
| option-order-shuffle | 0 | 5 | 4 | 0 | 0 | 1 | 0 |

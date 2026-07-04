# Llama-3.2-1B ArabicMMLU Option-Bias Scores

## Interpretation

This pilot tests whether strict-output ArabicMMLU accuracy is robust to option-label and position perturbations. Because the question content and answer options are preserved while only their labels/order change, instability across permutations indicates that measured accuracy may reflect label or position bias rather than stable task competence.

## Overall

- Rows: 200
- Parseable rows: 200
- Accuracy all: 0.300
- Parse failure rate: 0.000
- Label-bias score: 0.120
- Majority-label baseline: D = 0.250
- Semantic consistency: 0.160 (8/50)
- Correct-position accuracy range: 0.220

## Accuracy by Correct Label Position

| Correct label | Rows | Accuracy all |
| --- | ---: | ---: |
| A | 50 | 0.200 |
| B | 50 | 0.380 |
| C | 50 | 0.200 |
| D | 50 | 0.420 |

## Prediction Distribution

| Scope | A | B | C | D | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 29 | 57 | 40 | 74 | 0 | 0 |
| Correct=A | 10 | 14 | 10 | 16 | 0 | 0 |
| Correct=B | 4 | 19 | 10 | 17 | 0 | 0 |
| Correct=C | 10 | 10 | 10 | 20 | 0 | 0 |
| Correct=D | 5 | 14 | 10 | 21 | 0 | 0 |

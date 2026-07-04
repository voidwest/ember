# Qwen3-0.6B ArabicMMLU Option-Bias Scores

## Interpretation

This pilot tests whether strict-output ArabicMMLU accuracy is robust to option-label and position perturbations. Because the question content and answer options are preserved while only their labels/order change, instability across permutations indicates that measured accuracy may reflect label or position bias rather than stable task competence.

## Overall

- Rows: 200
- Parseable rows: 200
- Accuracy all: 0.275
- Parse failure rate: 0.000
- Label-bias score: 0.380
- Majority-label baseline: A = 0.250
- Semantic consistency: 0.120 (6/50)
- Correct-position accuracy range: 0.660

## Accuracy by Correct Label Position

| Correct label | Rows | Accuracy all |
| --- | ---: | ---: |
| A | 50 | 0.680 |
| B | 50 | 0.020 |
| C | 50 | 0.060 |
| D | 50 | 0.340 |

## Prediction Distribution

| Scope | A | B | C | D | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 126 | 4 | 9 | 61 | 0 | 0 |
| Correct=A | 34 | 0 | 2 | 14 | 0 | 0 |
| Correct=B | 31 | 1 | 2 | 16 | 0 | 0 |
| Correct=C | 30 | 3 | 3 | 14 | 0 | 0 |
| Correct=D | 31 | 0 | 2 | 17 | 0 | 0 |

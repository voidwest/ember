# ArabicMMLU Option-Bias Scores: Llama-3.2-3B

## Interpretation

This pilot tests whether strict-output ArabicMMLU accuracy is robust to option-label and position perturbations. Because the question content and answer options are preserved while only their labels/order change, instability across permutations indicates that measured accuracy may reflect label or position bias rather than stable task competence.

## Overall

- Rows: 200
- Parseable rows: 200
- Accuracy all: 0.410
- Parse failure rate: 0.000
- Label-bias score: 0.145
- Majority-label baseline: C = 0.250
- Semantic consistency: 0.260 (13/50)
- Correct-position accuracy range: 0.320

## Accuracy by Correct Label Position

| Correct label | Rows | Accuracy all |
| --- | ---: | ---: |
| A | 50 | 0.300 |
| B | 50 | 0.460 |
| C | 50 | 0.600 |
| D | 50 | 0.280 |

## Prediction Distribution

| Scope | A | B | C | D | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 25 | 62 | 79 | 34 | 0 | 0 |
| Correct=A | 15 | 14 | 12 | 9 | 0 | 0 |
| Correct=B | 4 | 23 | 17 | 6 | 0 | 0 |
| Correct=C | 3 | 12 | 30 | 5 | 0 | 0 |
| Correct=D | 3 | 13 | 20 | 14 | 0 | 0 |

# ArabicMMLU Option-Bias Pilot Summary

This pilot tests whether strict-output ArabicMMLU accuracy is robust to option-label and position perturbations. Because the question content and answer options are preserved while only their labels/order change, instability across permutations indicates that measured accuracy may reflect label or position bias rather than stable task competence.

## Setup

- Experiment: `option_bias`
- Prompt instruction: `Answer with exactly one character: A, B, C, or D. Do not explain.`
- Source items: 50 four-option ArabicMMLU questions x 4 permutations = 200 prompts
- Pilot 1 reuse: 36 of the 50 items come from the Pilot 1 source set. The remaining 14 Pilot 1 items had fewer than four choices, so the generator filled from the same local ArabicMMLU load to preserve the required A/B/C/D placement design.
- Seed: `42`

## Outputs

- `qwen3-0.6b.option_bias.prompts.sample200.jsonl`
- `qwen3-0.6b.option_bias.responses.sample200.jsonl`
- `qwen3-0.6b.option_bias.scores.sample200.md`
- `llama-3.2-1b.option_bias.prompts.sample200.jsonl`
- `llama-3.2-1b.option_bias.responses.sample200.jsonl`
- `llama-3.2-1b.option_bias.scores.sample200.md`
- `llama-3.2-3b.option_bias.prompts.sample200.jsonl`
- `llama-3.2-3b.option_bias.responses.sample200.jsonl`
- `llama-3.2-3b.option_bias.scores.sample200.md`

Low-memory scale-check runtime for Llama-3.2-3B:

- `max_tokens`: 1
- `temperature`: 0
- `max_seq_len`: 512
- batch/ubatch: Ember has no exposed batch or ubatch flags in this build; prompts were run as one prompt per process.
- output handling: prompt rows were streamed in chunks of 1 and response JSONL was flushed after each prompt. The run did not request logits or hidden states.

## Results

| Model | Accuracy all | Parse failure rate | Label-bias score | Majority-label baseline | Semantic consistency | Correct-position range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 0.275 | 0.000 | 0.380 | A = 0.250 | 0.120 | 0.660 |
| Llama-3.2-1B | 0.300 | 0.000 | 0.120 | D = 0.250 | 0.160 | 0.220 |
| Llama-3.2-3B | 0.410 | 0.000 | 0.145 | C = 0.250 | 0.260 | 0.320 |

## Accuracy by Correct Label Position

| Model | Correct=A | Correct=B | Correct=C | Correct=D |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 0.680 | 0.020 | 0.060 | 0.340 |
| Llama-3.2-1B | 0.200 | 0.380 | 0.200 | 0.420 |
| Llama-3.2-3B | 0.300 | 0.460 | 0.600 | 0.280 |

## Prediction Distribution

| Model | A | B | C | D | ParseFail | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 126 | 4 | 9 | 61 | 0 | 0 |
| Llama-3.2-1B | 29 | 57 | 40 | 74 | 0 | 0 |
| Llama-3.2-3B | 25 | 62 | 79 | 34 | 0 | 0 |

## Scale-Check Interpretation

This scale-check tests whether the prompt-format and option-position effects observed in very small decoder-only models persist, weaken, or change in a larger 3B model under the same ArabicMMLU audit setup. Because only one additional model size is added, the result should be treated as a robustness check rather than a full scaling study.

The Llama-3.2-3B option-bias run completed on this machine with the low-memory settings above. It improves `accuracy_all` to 0.410 and semantic consistency to 0.260, but it still shows measurable label and correct-position sensitivity. Its prediction distribution favors `C` and `B`, and accuracy ranges from 0.280 when the correct answer is under D to 0.600 when the correct answer is under C. This weakens the most extreme small-model bias pattern but does not remove the diagnostic effect.

## Interpretation

Strict-output prompting eliminated parse failures for all three models, but it did not remove label/position effects. Qwen3-0.6B shows a large A-label preference and strong correct-position sensitivity: accuracy is 0.680 when the correct answer is under A and 0.020 when it is under B. Llama-3.2-1B is less extreme but still not position-stable, with higher accuracy when the correct answer is under B or D. Llama-3.2-3B is more accurate overall, but it remains sensitive to answer position, with its highest accuracy when the correct answer is under C.

The low semantic consistency rates show that most source items do not receive a stable semantic answer across the four label permutations. For this Pilot 2 diagnostic, strict-output ArabicMMLU accuracy should therefore be reported with label distribution and correct-position sensitivity, not as a standalone competence estimate.

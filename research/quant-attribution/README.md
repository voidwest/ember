# Do patching results survive quantization?

People run interpretability experiments on the quantized GGUF they can fit on
a laptop and read the results as statements about the model. This study asks
whether that is safe for activation patching. It compares the effects measured
on each rung of a quantization ladder with the effects measured on the F16
model that the ladder was quantized from.

This file was written and committed before any run. The results, and any
deviation from this design, are recorded separately in `results/`.

## Models

Two families, four rungs each, all pinned by SHA-256 in
`models/v03-sources/sources.json` and `models/v03-ladder-reproduced/ladder-manifest.json`:

| Family | Layers | Rungs |
|---|---|---|
| Llama-3.2-1B-Instruct | 16 | F16 source, Q8_0, Q6_K, Q4_K_M |
| Qwen2.5-1.5B-Instruct | 28 | F16 source, Q8_0, Q6_K, Q4_K_M |

Every quantized rung was made from that F16 file by the pinned llama.cpp
quantizer, so the only difference between rungs is quantization.

## Task

Six country/capital pairs on one template, `The capital of {country} is the
city of`. The clean prompt names the first country, the corrupted prompt the
second. The metric is `logit(target) - logit(foil)` at the final position.

| Pair | Clean | Corrupted | Target | Foil |
|---|---|---|---|---|
| france-italy | France | Italy | Paris | Rome |
| germany-spain | Germany | Spain | Berlin | Madrid |
| japan-russia | Japan | Russia | Tokyo | Moscow |
| egypt-greece | Egypt | Greece | Cairo | Athens |
| poland-austria | Poland | Austria | Warsaw | Vienna |
| portugal-sweden | Portugal | Sweden | Lisbon | Stockholm |

Every name is a single token in both tokenizers, and each pair's prompts have
equal token length.

## Candidates

Every combination of three sites (`residual-pre-attention`,
`attention-output`, `mlp-output`), every layer, and every position from the
corrupted country token to the final token. Earlier positions are identical in
both prompts, so patching them changes nothing by causality. That gives 240
candidates per Llama run and 420 per Qwen run.

Every candidate is patched for real: the corrupted prompt runs with the clean
activation written at that site, layer and position. The measured effect is
`m(patched) - m(corrupted)`, and the recovered fraction divides it by the
clean-corrupted gap. Ember's direct-path estimate is recorded but is not the
object of comparison.

## Comparisons

For each family, pair and quantized rung, against the F16 run of the same pair:

1. Spearman and Pearson correlation of the measured effects over all
   candidates.
2. Largest and mean absolute difference in recovered fraction.
3. Whether the single most effective candidate is the same.
4. Overlap of the five most effective candidates.
5. Jaccard overlap of the "material" sites, those that recover at least 20% of
   the gap.
6. The clean-corrupted gap relative to F16.

## Pre-registered reading

The claim "a quantized rung gives the same patching conclusions as F16" is
supported for a family if, on every pair, the measured effects correlate with
Spearman of at least 0.9, the top candidate matches, and the material-site
Jaccard is at least 0.8. If any pair fails, the note reports which conditions
failed, on which rung, and by how much, rather than averaging the failure
away.

## Limits fixed in advance

- Two small models and one prompt template: a single fact-recall circuit, not
  patching in general.
- F16 is the reference, not ground truth: it is itself a rounding of the BF16
  training weights.
- One deterministic run per cell. Ember's reference mode is bit-reproducible on
  a given binary and host, so repeats would not add information about noise.

## Reproducing

```bash
python research/quant-attribution/run_study.py run \
  --ember target/release/ember --out runs/quant-attribution
python research/quant-attribution/run_study.py analyze \
  --out runs/quant-attribution --results research/quant-attribution/results --figures
```

`run` verifies every bundle and records its semantic and payload hashes in
`provenance.json`.

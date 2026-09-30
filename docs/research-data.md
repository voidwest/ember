# Research data out of git (plan)

Status: **prepared, not executed.** Nothing has been deleted, untracked,
moved, or published. This page describes the tooling and what a move would
save, so a maintainer can decide.

## What is tracked today

Research datasets and frozen experiment logs live in git under `data/` and
`research/`: 71.7 MiB of tracked files. Most of that is a few dozen large
files. With a 256 KiB threshold,
[`scripts/research_data_manifest.json`](../scripts/research_data_manifest.json)
lists **47 files, 63.2 MiB**, each with its path, byte count and SHA-256.
They hold only **33 distinct blobs (52.1 MiB)**: **11.0 MiB are
byte-identical copies**. No code or test names any of these paths (the
manifest's `referenced_by` is empty for all 47), so they are research
records rather than build or test inputs.

Regenerate the manifest and the report with:

```sh
python3 scripts/research_data.py manifest   # rescan tracked files (threshold: --threshold BYTES)
python3 scripts/research_data.py report     # sizes, largest directories, duplicate groups
```

`tests/test_research_data.py` checks that the committed manifest still
matches the tracked files, so editing one of these files means regenerating
the manifest in the same change.

Largest directories (files at or above the threshold):

| size | directory |
|---:|---|
| 18.1 MiB | `data/arabic_morph_real/out_disambig_padt_5000_strict` |
| 8.5 MiB | `data/arabic_morph_real/out_disambig_padt_split_strategies_strict` |
| 5.2 MiB | `data/arabic_morph_real/camel_disambig_msa_padt_5000.jsonl` |
| 4.8 MiB | `research/embersec/comparative` (diff-fuzz logs) |
| 4.5 MiB | `data/arabic_morph_real/probe_baseline_llama32_5k` |
| 4.5 MiB | `data/arabic_morph_real/probe_baseline_qwen3_5k` |
| 4.3 MiB | `data/arabic_morph_real/out_disambig_padt_1500_strict` |
| 3.1 MiB | `data/arabic_morph_sample/out_split_strategies_strict` |

## Byte-identical copies

Nine groups; storing each blob once saves 11.0 MiB:

| copies × size | saves | paths |
|---|---:|---|
| 2 × 4.5 MiB | 4.5 MiB | `data/arabic_morph_real/probe_baseline_{llama32_5k,qwen3_5k}/stimuli_ablated.json` |
| 3 × 677 KiB | 1.3 MiB | `data/arabic_morph_real/out_disambig_padt_split_strategies_strict/{concrete_pattern_heldout,lemma_heldout,lemma_random}/sft.jsonl` |
| 4 × 449 KiB | 1.3 MiB | `data/arabic_morph_sample/out_split_strategies_strict/{concrete_pattern_heldout,lemma_heldout,lemma_random,root_pattern_heldout}/sft.jsonl` |
| 3 × 561 KiB | 1.1 MiB | `data/arabic_morph_real/out_disambig_padt_split_strategies_strict/{concrete_pattern_heldout,lemma_heldout,lemma_random}/canonical.jsonl` |
| 3 × 449 KiB | 897 KiB | `data/arabic_morph_sample/out_imbalanced/sft.jsonl`, `data/arabic_morph_sample/out_imbalanced_strict/sft.jsonl`, `data/arabic_morph_sample/out_split_strategies_strict/root_heldout/sft.jsonl` |
| 2 × 677 KiB | 677 KiB | `data/arabic_morph_real/out_disambig_padt_split_strategies_strict/root_heldout/sft.jsonl`, `data/arabic_morph_real/out_disambig_padt_strict/sft.jsonl` |
| 2 × 561 KiB | 561 KiB | `data/arabic_morph_real/out_disambig_padt_split_strategies_strict/root_heldout/canonical.jsonl`, `data/arabic_morph_real/out_disambig_padt_strict/canonical.jsonl` |
| 2 × 492 KiB | 492 KiB | `data/arabic_morph_real/camel_disambig_msa_padt_{500,smoke}.jsonl` |
| 2 × 256 KiB | 256 KiB | `data/test_activations.npy`, `data/test_check_activations.npy` |

The split-strategy copies are expected (several strategies produce the same
SFT split); they are still separate files in git.

## The plan

1. **Dedupe by content.** The release asset stores each distinct blob once,
   as `blobs/<sha256>` in one deterministic tarball:

   ```sh
   python3 scripts/research_data.py pack --out dist/ember-research-data-v1.tar
   ```

   It re-hashes every file against the manifest first and prints the
   tarball's SHA-256. Current size: 52.1 MiB, well under the 2 GiB release
   asset limit.
2. **Move to a release asset (manual maintainer action).** Upload the
   tarball to a GitHub release (for example `research-data-v1`), then set
   `asset.url` and `asset.sha256` in the manifest in the same commit that
   stops tracking the files. Untrack with `git rm --cached` on exactly the
   manifest's paths and add them to `.gitignore`; do not rewrite history.
3. **Verify by SHA-256 on the way back.** Anyone who needs the data runs

   ```sh
   scripts/fetch_research_data.sh            # restore from the manifest's asset
   scripts/fetch_research_data.sh --verify   # check what is on disk
   ```

   Every restored file must match the SHA-256 in the committed manifest;
   the asset's own hash is checked when pinned. The manifest in git is the
   trust anchor, so a swapped asset cannot restore different bytes. Today,
   with the data still tracked, `fetch` reports every file already present.

## What it saves, and what it does not

- **Checkout:** 63.2 MiB fewer tracked bytes in every working tree (88% of
  `data/` + `research/`).
- **Distribution:** the asset is 52.1 MiB, since duplicates are stored once.
- **Clone size: nothing, by itself.** Git history keeps every blob ever
  committed, so a clone still downloads them. Shrinking clones needs a
  history rewrite (filter-repo or LFS migration), which is out of scope and
  would change every commit hash; do not do it as part of this move.
- **Frozen evidence:** `research/embersec/comparative/FROZEN_ARTIFACTS.md`
  pins SHA-256s of the diff-fuzz logs. Moving them keeps those hashes valid
  because the fetch step restores byte-identical files.

## Out of scope

The tokenizer JSONs at the repository root (`tokenizer*.json`, 66 MiB) are
the largest tracked files, but tests and the default tokenizer resolution
read them from the checkout, so they stay tracked. The same holds for
`tests/fixtures/` (bundle fixtures are hashed by tests).

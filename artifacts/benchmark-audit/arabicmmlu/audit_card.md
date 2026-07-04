# Benchmark Audit: ArabicMMLU

- Benchmark ID: `arabicmmlu`
- Records: 14575
- Task family: `knowledge`
- Primary format: `multiple-choice`
- Configured Arabic variety: `MSA`
- Audit risk label: `low`

## Main Audit Warnings

- no high-level audit warning triggered by implemented checks

## Dataset Shape

| Field | Unique | Top values |
| --- | ---: | --- |
| split | 2 | test (14455), dev (120) |
| source | 979 | https://www.madinaharabic.com/quiz/ (973), https://folderat.com/Reference/1649/%D8%A3%D8%B3%D8%A6%D9%84%D8%A9-%D8%AF%D9%8A%D9%86%D9%8A%D8%A9-%D9%85%D8%B9-%D8%AE%D9%8A%D8%A7%D8%B1%D8%A7%D8%AA (642), https://www.ta3lemkonline.com/2022/09/blog-post.html (509), https://www.ta3lemkonline.com/2023/09/blog-post_5.html (499), ملف ضع دائرة فصل اول (1).pdf - Google Drive (477) |
| arabic_variety | 1 | MSA (14575) |
| surface | 0 |  |
| lemma | 0 |  |
| root | 0 |  |
| pattern | 0 |  |
| subject | 21 | Islamic Studies (2222), Biology (1412), Geography (1376), Driving Test (1214), General Knowledge (1207) |
| group | 5 | Humanities (3655), Social Science (3540), STEM (3220), Other (2499), Language (1661) |
| level | 5 | High (4963), Primary (3239), Middle (1775), Univ (575), Prof (317) |
| country | 8 | Jordan (6053), Egypt (2506), Palestine (2047), Morocco (317), Lebanon (239) |

## Duplicate Audit

- Exact duplicate groups: 41
- Exact duplicate records: 82
- Exact duplicate record rate: 0.0056

## MCQ Audit

- MCQ records: 14575
- Answer position distribution: A: 0.312, B: 0.266, C: 0.202, D: 0.210, E: 0.010
- Unique normalized answer texts: 9138

## Shortcut Baselines

| Task | Classes | Majority | Random-prior | Char n-gram | Word n-gram |
| --- | ---: | ---: | ---: | ---: | ---: |
| answer | 5 | 0.312 | 0.253 | 0.295 sampled n=3000 | 0.305 sampled n=3000 |

## Split Overlap

- `surface` split value counts: `{}`
  - no split pairs available
- `lemma` split value counts: `{}`
  - no split pairs available
- `root` split value counts: `{}`
  - no split pairs available
- `pattern` split value counts: `{}`
  - no split pairs available
- `subject` split value counts: `{"dev": 21, "test": 21}`
  - dev vs test: 21 overlap (1.000 of smaller side)
- `group` split value counts: `{"dev": 5, "test": 5}`
  - dev vs test: 5 overlap (1.000 of smaller side)
- `country` split value counts: `{"dev": 5, "test": 8}`
  - dev vs test: 5 overlap (1.000 of smaller side)

## Configured First Audit Concerns

- public exam contamination
- answer-option artifacts
- source-country and subject imbalance
- prompt and option-order sensitivity

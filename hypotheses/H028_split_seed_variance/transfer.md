# H028 — Method Transfer

## Source

- **Bouthillier et al. 2021** "Accounting for Variance in Machine Learning
  Benchmarks" — split-level variance distinct from training-seed variance.
- **Reimers & Gurevych 2017** "Reporting Score Distributions" — single-split
  evaluation 의 위험성 직접 지적.

## Mechanism

H010 byte-identical, `--split_seed 42/43/44` only. 3 launches → val/OOF
mean ± stdev. cohort assignment effect isolation.

## §17.2 EXEMPT (measurement H, no mutation)

challengers 카테고리 (H022/H023 sibling, methodology framework).

## §⑤ UNI-REC alignment

mechanism / sequential / interaction reference 변경 없음. measurement 만.
primary_category = `measurement` (re-entry, H022/H023 sibling).

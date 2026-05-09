# LLM Wiki Index

> Obsidian-friendly catalog. Auto-indexed sections는 `wiki_ingest.py` hook이 갱신.
> Manual synthesis 섹션은 사람/LLM이 자유 기술.

## Overview

- [LLM Wiki 운영 개요](./overview.md) — 목적, 범위, 운영 주기
- [[../authority.md]] — 권한 / 승격 규칙
- [Auto-ingest 메커니즘](./log.md) — chronological event log

---

# Auto-indexed (catalog)

> 아래 섹션은 hook이 새 entity 발견 시 자동으로 한 줄 추가.
> 각 entity 페이지는 `## Sources` (자동) + `## Activity` (자동) + `## Notes` (수동).

## Hypotheses
- [[by_hypothesis/H051_per_pattern_user_dense]]

- [[by_hypothesis/H050_train_ratio_60]]

- [[by_hypothesis/H047_per_domain_aux]]

- [[by_hypothesis/H046_proper_dann]]

- [[by_hypothesis/H042_kld_prior_matching]]

- [[by_hypothesis/H041_cold_start_branch]]

- [[by_hypothesis/H021_per_domain_top_k]]

- [[by_hypothesis/H020_learnable_gsu]]

- [[by_hypothesis/H032_timestamp_input_features]]

- [[by_hypothesis/H031_item13_explicit_head]]

- [[by_hypothesis/H022_h010_multi_seed_variance]]

- [[by_hypothesis/H019_twin_long_seq_retrieval]]

- [[by_hypothesis/INDEX]]

- [[by_hypothesis/H018_per_user_recency_weighting]]

- [[by_hypothesis/H011_aligned_pair_encoding]]
## Experiments
- [[by_experiment/H053_item_cooccurrence]]

- [[by_experiment/H051_listwise_lambdarank]]

- [[by_experiment/H048_user_item_bilinear]]

- [[by_experiment/H047_per_domain_aux]]

- [[by_experiment/H046_proper_dann]]

- [[by_experiment/H044_dann_cohort_debias]]

- [[by_experiment/H042_kld_prior_matching]]

- [[by_experiment/H041_cold_start_branch]]

- [[by_experiment/H040_pairwise_ranking]]

- [[by_experiment/H039_no_history_baseline]]

- [[by_experiment/H036_softmax_3class]]

- [[by_experiment/H035_hstu_trunk]]

- [[by_experiment/H034_esu_2layer]]

- [[by_experiment/H033_twin_combined]]

- [[by_experiment/H021_per_domain_top_k]]

- [[by_experiment/H020_learnable_gsu]]

- [[by_experiment/H032_timestamp_input_features]]

- [[by_experiment/H031_item13_explicit_head]]

- [[by_experiment/H022_h010_multi_seed_variance]]

- [[by_experiment/H019_twin_long_seq_retrieval]]

- [[by_experiment/INDEX]]

- [[by_experiment/H018_per_user_recency_weighting]]

- [[by_experiment/H013_hyperparameter_calibration]]

- [[by_experiment/H012_multi_domain_fusion]]

- [[by_experiment/H011_aligned_pair_encoding]]

## EDA
- [[by_eda/timestamp_signal_audit]]

- [[by_eda/item_int_signal_audit]]

- [[by_eda/eda_backlog_sample1000]]

## Papers

## Governance
- [[by_governance/progress]]

- [[by_governance/CLAUDE]]

- [[by_governance/inference_lessons]]

---

# Manual synthesis

> 사람/LLM이 직접 작성. auto-ingest는 건드리지 않음.

## Concepts

- (TBD) 예: [Candidate-aware attention](./concept_candidate_attention.md)

## Comparisons

- (TBD)

## Sources (external summaries)

- (TBD)

---

## Update rule

각 페이지는 다음 frontmatter를 가져야 한다:

```yaml
---
title: "..."
type: "concept|entity|comparison|source-summary|overview"
status: "draft|reviewed"
created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
sources:
  - path: "..."
    kind: "file|url"
confidence: "low|medium|high"
promotion_state: "not-promoted|promoted"
---
```

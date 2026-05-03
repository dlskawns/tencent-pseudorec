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
- [[by_hypothesis/H022_h010_multi_seed_variance]]

- [[by_hypothesis/H019_twin_long_seq_retrieval]]

- [[by_hypothesis/INDEX]]

- [[by_hypothesis/H018_per_user_recency_weighting]]

- [[by_hypothesis/H011_aligned_pair_encoding]]
## Experiments
- [[by_experiment/H022_h010_multi_seed_variance]]

- [[by_experiment/H019_twin_long_seq_retrieval]]

- [[by_experiment/INDEX]]

- [[by_experiment/H018_per_user_recency_weighting]]

- [[by_experiment/H013_hyperparameter_calibration]]

- [[by_experiment/H012_multi_domain_fusion]]

- [[by_experiment/H011_aligned_pair_encoding]]

## EDA
- [[by_eda/eda_backlog_sample1000]]

## Papers

## Governance
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

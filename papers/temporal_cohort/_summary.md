# Temporal Cohort — Summary

> **Training procedure 카테고리** — cohort drift / distribution shift /
> recency 처리 family. mechanism axis 가 아닌 training procedure axis
> (UNI-REC 안 새 layer, deployment realism).

## Takeaway (3-5 lines)

- 신규 카테고리 first-touch (2026-05-03). 직전 9 H (H006~H014) 가 mechanism
  / envelope / hyperparameter mutation 만 시도 — training procedure axis
  미측정.
- 9 H 의 가장 강한 패턴 = **OOF AUC stable (0.857~0.860) / Platform AUC
  변동 (0.82~0.838)**. mechanism 변경이 OOF 거의 영향 없음. Platform 만
  cohort drift 로 변동. cohort drift 가 ceiling 의 진짜 정체 가능성.
- 주요 family: recency loss weighting (linear / exp decay), temporal cohort
  embedding, OOF 재정의 (label_time future-only holdout), domain adaptation,
  importance weighting (Sugiyama 2007).
- **TAAC 2026 motivation**: train (label_time < cutoff) vs platform eval
  (label_time > cutoff or different period) 의 distribution shift 처리.
- Sample-scale viability: loss weighting form 은 params 추가 0 + mean weight
  보존 → §10.6 budget 영향 없음. embedding form 은 params 추가 medium.

## Entries

- (cold-start, H015 first-touch) — Pan & Yang 2010 survey, Gama et al. 2014
  concept drift, Sugiyama 2007 importance weighting 외 production CTR
  practice.
- 향후 추가 후보: TWIN 의 lifelong handling, OneTrans 의 mixed causal mask
  (timestamp 기반), session-based recency weighting.

## Carry-forward rules (post-H015 시작)

- **Rule TC-1**: loss weighting 은 mean weight = 1.0 보존 (loss scale →
  lr/optim 영향 없음). non-conservative weighting (e.g., [1.0, 5.0]) 은
  separate sub-H.
- **Rule TC-2**: per-batch normalization 우선 (implementation 단순, shuffle
  무관). per-dataset normalization 은 sub-H.
- **Rule TC-3**: OOF 재정의 (label_time future-only holdout) 은 separate
  H (paired Δ baseline 깨짐, careful comparison 필요).
- **Rule TC-4**: training procedure axis 의 효과는 mechanism axis 와 orthogonal
  → 둘 다 PASS 시 stacking 가능 (단 single mutation 룰 위반 가능성, careful
  H 분리).

## Candidate sources to fetch later

- TWIN (Pan et al. RecSys 2024) — lifelong behavior modeling, target-aware
  retrieval. recency 와 다른 axis.
- DLRM-v2 또는 후속 — production CTR sample handling 진화.
- Time-aware sequential recommendation surveys.
- DANN (Domain-Adversarial Neural Networks) — adversarial domain adaptation,
  cohort 별 invariant representation.

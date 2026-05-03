# H016 — Problem Statement

## What we're trying to explain

9 H 누적 ceiling 0.82~0.838. 4-layer ceiling diagnosis 종료 — L2 (cohort drift)
가 마지막 가설.

L2 의 두 form:
- (a) **train 측 처리** — H015: recency loss weighting → train 이 platform
  분포에 closer.
- (b) **OOF 측 처리** — H016: OOF 정의 변경 → OOF 가 platform proxy 됨,
  measurement 정합성 자체 회복.

H016 = (b) form. 9 H 의 OOF AUC stable (0.857~0.860) / Platform 변동 패턴이
"OOF measure 가 platform 과 다른 것" 이라면, OOF 자체 재정의가 더 fundamental
fix.

**측정 가능한 gap**: OOF 정의 변경 (random 10% user → label_time future-only)
시 Platform AUC vs H010 corrected 비교. paired Δ baseline 일부 깨짐 (OOF
정의 다름) but Platform 비교는 valid.

## Why now

H015 와 동시 launch (triple-H setup) — L2 의 두 form 동시 검증. cohort drift
의 진짜 정체 결정.

직전 H carry-forward:
- H014 verdict (L4 retire, 4-layer 종료, paradigm shift inevitable).
- H011 F-5 ~ H014 F-2/F-3: cohort drift hard ceiling 가설 누적.
- H015 와 같은 axis, 다른 form (Frame C in H015 challengers).

§10.7 rotation: temporal_cohort 카테고리 H015 first-touch + H016 sub-form
(같은 카테고리 sub-form). §10.7 의 "변형/sub-H 패턴" 정합 — mechanism class
같음 (cohort drift), form 만 다름.

§17.2 룰: dataset.py + train.py 변경 (mechanism mutation). 단일 concern
(OOF 정의).

§0 north star alignment: training/measurement procedure axis (deployment
realism).

## Scope

- **In**:
  - dataset.py `_RowFilter` 확장 (oof_cutoff field 추가).
  - dataset.py `split_parquet_by_label_time` + `get_pcvr_data_v2` 에 oof_split_type
    분기.
  - train.py CLI flag `--oof_split_type {random_user, future_label_time}`.
  - run.sh: `--oof_split_type future_label_time`.
- **Out**:
  - model.py / infer.py / trainer.py / utils.py: 변경 없음.
  - OOF user_ratio 변경 (H010 default 0.1 유지 — quantile 0.9 = 상위 10%).
  - 다른 OOF 정의 변형 (e.g., user-time hybrid, recency window) — 별도 sub-H.

## UNI-REC axes

H015 와 동일 axis (temporal_cohort, training/measurement procedure).
mechanism stack (NS xattn + DCN-V2) byte-identical.

## Success / Failure conditions

- **Success — L2 confirmed (Frame C form)**:
  - Δ Platform vs H010 corrected (0.837806) ≥ +0.005pt → strong, OOF 재정의
    가 measurement 정합성 회복 → mechanism 진짜 효과 가시화.
  - 또는 measurable [+0.001, +0.005pt] → partial.
- **Failure (REFUTED)**:
  - noise (−0.001, +0.001pt] → OOF 재정의도 cohort drift 못 풀림. **L2 다른
    form 도 실패** (H015/H017 와 함께 보면 L2 retire confirmed). paradigm
    shift mandatory.
  - degraded < −0.001pt → 새 OOF 정의가 train cohort 변경 (label_time 기반
    cutoff 가 random user holdout 보다 작은 train set) → 학습 자원 부족.

## Frozen facts referenced

- §3.4 label_time + 9 H verdicts.
- §3.5 sequence length 분포 (변경 없음).
- H010 corrected anchor (Platform 0.837806).
- 9 H OOF stable / Platform 변동 일관 패턴.
- `competition/dataset.py:split_parquet_by_label_time` baseline 코드.

## Inheritance from prior H

- H010 mechanism + envelope: byte-identical.
- H015 와 paired sibling (같은 axis, 다른 form).
- H011~H014 cohort drift 가설 누적 → L2 직접 검증.
- prior H 들 OOF AUC 직접 비교 invalid (OOF 정의 다름) — Platform 만 비교.

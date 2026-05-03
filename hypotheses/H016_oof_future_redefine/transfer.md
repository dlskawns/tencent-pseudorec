# H016 — Method Transfer

## ① Source

- **Time-based holdout** (production CTR engineering standard) — train/eval
  split 의 future-only holdout 패턴. label_time 기반 quantile cutoff.
- **Pan & Yang 2010** (transfer learning survey) — source-target distribution
  alignment 의 measurement 측면.
- **Sugiyama et al. 2007** (importance weighting) — covariate shift 의 평가
  metric 정합성.
- 카테고리 family: `temporal_cohort/` (H015 first-touch). H016 = same category,
  same mechanism class (cohort drift mitigation), 다른 form (measurement
  procedure 측).

## ② Original mechanism

**Time-based holdout** (1단락):

학습-평가 split 시 user-based random split 대신 label_time 기반 cutoff 사용.
train = past, valid = recent past (val cutoff 이후), holdout = future (oof
cutoff 이후). future holdout 이 production deployment 의 진짜 측정값에 가장
가까움 (deployment 시 모델은 미래 sample 평가).

**우리 적용**:
- `quantile(label_time, 0.9)` 이상 = OOF (10%).
- `quantile(label_time, 0.85) ~ 0.9` = valid (5%).
- `< quantile(0.85)` = train (85%).
- prior H 들 (random_user) 와 비교 시 train cohort 작아짐 (~10% 감소).

## ③ What we adopt

- **Mechanism class**: time-based OOF. `_RowFilter` 의 oof_cutoff field.
- **변경**: dataset.py + train.py + run.sh.
- **CLI**: `--oof_split_type future_label_time`.

## ④ What we modify (NOT a clone)

- **default 유지 가능 (random_user)**: H016 만 future_label_time 사용. backward
  compat.
- **Single mutation**: OOF 정의만 변경. mechanism stack byte-identical.
- **paired Δ baseline 깨짐 인정**: prior H 들 OOF AUC 비교 invalid (OOF 정의
  다름). Platform 비교는 valid.

## ⑤ UNI-REC alignment

H015 와 동일 axis (temporal_cohort).

## ⑥ Sample-scale viability

- params 추가 0 (model byte-identical).
- train cohort ~10% 감소 (label_time 기반 cutoff). full-data 에서 영향 작음
  (10% 감소 대비 cohort 정합성 효과 더 클 가능성).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm**: 변경 없음.
- **§10.6 budget**: 변경 없음.
- **§10.7 rotation**: temporal_cohort 카테고리 H015 first-touch + H016 sub-form.
- **§10.9 attn entropy**: 변경 없음.
- **§17.2 single mutation**: OOF 정의만.
- **§17.3 binary**: Δ Platform ≥ +0.001pt 또는 +0.005pt.
- **§17.5 sample-scale**: cloud full-data 결과로만 결정.
- **§17.6 cost cap**: ~3-4h, 누적 ~40h (triple-H 동시 launch 시 ~4h wall clock).
- **§18 inference**: byte-identical.
- **H010 F-1**: NS-only enrichment safe pattern → mechanism stack 변경 0.
- **H011~H014 cohort drift 가설 누적**: H016 의 핵심 동기.

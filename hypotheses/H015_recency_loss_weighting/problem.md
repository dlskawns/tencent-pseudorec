# H015 — Problem Statement

## What we're trying to explain

9 H 누적 (H006~H014) Platform AUC ceiling 0.82~0.838. 4-layer ceiling diagnosis
종료 — L1 (hyperparameter, H013 REFUTED), L3 (NS xattn sparse, H011/H012/H013/H014
누적 retire), L4 (truncate, H014 REFUTED). **L2 (cohort drift) 가 마지막
unexplored 가설**.

가장 강한 데이터 패턴 (9 H 일관):
- OOF AUC 변동: 0.857~0.860 (variance 0.23pt) — 거의 일관.
- Platform AUC 변동: 0.82~0.838 (variance 1.8pt) — **8× variance**.
- OOF-Platform gap: 1.88~2.59pt 모두 양수.
- → 모델이 OOF cohort 의 capacity 한계 hit. **Platform 만 cohort drift 로 변동**.

→ mechanism 변경이 OOF 에 거의 영향 없음. cohort drift 처리 안 하면 어떤
mechanism 도 ceiling 못 깨.

**측정 가능한 gap**: H015 = recency-aware loss weighting (per-batch linear
[0.5, 1.5] by label_time). 학습 sample 의 recency 에 따라 loss weight 부여
→ train 이 platform eval distribution (미래 시점) 에 closer. Δ vs H010
corrected anchor 결과로 L2 가설 검증.

## Why now

직전 H carry-forward:
- H014 verdict (L4 retire): dense long-seq expansion 효과 없음, 4-layer
  diagnosis 종료, paradigm shift inevitable.
- H014 F-2: OOF-Platform gap 2.59pt = 9 H 중 가장 큼 → cohort drift
  강한 confirm.
- H011 F-5 + H012 F-3 + H013 F-4 + H014 F-3: cohort drift hard ceiling
  가설 누적 강화.

§10.7 rotation: 직전 2 H = H013 (no category) + H014 (envelope_expansion).
H015 = `temporal_cohort` 신규 카테고리 first-touch (FREE).

§17.2 룰: mechanism mutation (loss computation 변경). 단일 concern (cohort
drift 처리). category 신규 first-touch.

§0 north star alignment: training procedure axis 강화 — production CTR 시스템
의 cohort handling 표준 (deployment realism).

## Scope

- **In**:
  - dataset.py: batch dict 에 `label_time` 노출 (mechanism 학습 안 함, trainer
    에서 weight 계산용).
  - trainer.py: `_train_step` 의 BCE/focal loss 계산 시 per-batch linear
    weight 곱셈.
  - train.py: CLI flags `--use_recency_loss_weighting`, `--recency_weight_min`,
    `--recency_weight_max`.
  - run.sh: H010 mechanism + envelope baked + 3 recency flags.
  - mean weight = 1.0 (loss scale 보존, lr/optim 영향 없음).
- **Out**:
  - model.py / infer.py / utils.py / make_schema.py / local_validate.py /
    ns_groups.json / requirements.txt: 변경 없음.
  - per-dataset weighting (전체 train min/max) — sub-H 후보.
  - exponential decay weighting — sub-H 후보.
  - quadratic / non-linear weighting — sub-H 후보.
  - OOF 재정의 (label_time future-only holdout) — 별도 H 후보.

## UNI-REC axes

- **Sequential**: 변경 없음 (H010 NS xattn + per-domain encoder 그대로).
- **Interaction**: 변경 없음 (H008 DCN-V2 fusion 그대로).
- **Bridging mechanism**: 변경 없음 (mechanism stack 그대로).
- **Training procedure**: NEW axis — recency 가 loss gradient 강조 → 모델이
  recent sample 에 더 fit. cohort drift mitigation.

## Success / Failure conditions

- **Success — L2 confirmed**:
  - Δ vs H010 corrected (0.837806) ≥ +0.005pt (strong) → Platform ≥ 0.8428.
    Cohort drift = ceiling 의 진짜 정체. paradigm 안에서 ceiling 깰 수 있음.
  - 또는 measurable [+0.001, +0.005pt] → L2 partial confirmed.
- **Failure (REFUTED)**:
  - noise (−0.001, +0.001pt] → L2 도 가설 약함. **마지막 layer 도 retire
    → paradigm shift mandatory** (backbone replacement 또는 retrieval).
  - degraded < −0.001pt → recency weighting 이 학습 disrupt.

## Frozen facts referenced

- §3.4 label_time (verified, sibling cite from `tencent-cc/eda/out/`).
- §3.5 sequence length 분포 (verified).
- 9 H verdicts: OOF stable / Platform 변동 일관 패턴.
- H010 corrected anchor (Platform 0.837806).
- H014 verdict F-2: OOF-Platform gap 2.59pt = 9 H 중 가장 큼 (cohort drift
  강한 신호).
- linear scaling rule 적용 안 함 (mean weight = 1.0 보존).

## Inheritance from prior H

- H010 mechanism + envelope: byte-identical (NS xattn + DCN-V2 fusion +
  per-domain encoder + seq 64-128).
- H010 corrected (0.837806) = anchor.
- H011 F-5 + H012 F-3 + H013 F-4 + H014 F-2/F-3: cohort drift hard ceiling
  가설 → H015 의 핵심 동기.
- 4-layer diagnosis 종료 → H015 = L2 만 남음, paradigm 안에서 마지막 시도.

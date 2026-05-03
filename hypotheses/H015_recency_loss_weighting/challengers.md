# H015 — Challenger Frames

> CLAUDE.md §10.1, §10.7. primary_category = `temporal_cohort` (신규 first-touch
> — FREE). 직전 2 H = H013 (no category) + H014 (envelope_expansion). 카테고리
> 충돌 없음. **마지막 layer (L2) 검증, paradigm shift trigger 가능성**.

## Frame A (default — what we're proposing)

- **Claim**: 9 H 의 OOF stable (0.857~0.860) / Platform 변동 (0.82~0.838)
  패턴 = train/eval distribution shift (cohort drift) 가 ceiling 의 진짜
  정체. recency-aware loss weighting (per-batch linear [0.5, 1.5] by
  label_time) 시 train cohort 가 platform eval 분포 (미래 시점) 에 closer
  → Δ vs H010 ≥ +0.001pt (measurable) 또는 ≥ +0.005pt (strong).
- **Mechanism**: dataset.py 가 batch 에 label_time 노출, trainer.py 가 per-
  batch min-max → percentile → linear weight. mean weight = 1.0 보존.
- **Why this could be wrong**:
  1. **Per-batch normalization 의 noise**: batch (2048) 안의 label_time
     spread 가 작으면 weight 거의 1.0 일관 → 효과 없음.
  2. **Cohort drift 가 hard ceiling 이 아닌 다른 무엇**: distribution shift
     의 본질이 recency 가 아닌 다른 factor (user behavior change, content
     drift, system change).
  3. **Loss weighting 이 model capacity 한계 못 깸**: OOF 0.86 가 model
     capacity 한계면 weighting 만으로 platform 도 한계 그대로.

## Frame B (counter-frame — paradigm shift mandatory)

- **Claim**: 4-layer diagnosis 모두 retire 후 paradigm 안 mutation 모두 ceiling.
  **backbone replacement** 또는 **다른 model class** (LLM-based, generative)
  가 진짜 답. recency weighting 도 marginal.
- **What evidence would prove this?**:
  - H015 결과 noise → L2 도 가설 약함 → **paradigm shift mandatory confirm**.
- **Distinguishing experiment**: H015 자체. PASS → A confirmed. noise →
  B confirmed → H016 = backbone replacement.

## Frame C (orthogonal frame — OOF 재정의)

- **Claim**: 진짜 fix 는 loss weighting 이 아닌 **OOF 재정의** (label_time
  future-only holdout). 현재 random 10% user holdout = OOF 와 platform
  분포 다름. label_time 기반 future-only holdout 시 OOF 가 platform proxy
  됨 → 측정 정합성 자체 회복.
- **Mechanism**: dataset.py 의 OOF user 선정 로직 변경 (random 10% → label_time
  cutoff 이후 user 만).
- **Cost vs A**: medium-large. OOF 재정의 시 모든 prior H paired Δ 비교
  base 깨짐. H015 = loss weighting 이 더 conservative (data 자체 안 건드림).

## Decision

**A 선택** 이유:
1. **단일 mutation, mean weight 보존**: lr/optim 영향 없음. 가장 conservative.
2. **Cost 작음**: dataset.py + trainer.py + train.py 만 수정. ~5h 학습 (H010
   envelope).
3. **L2 직접 attack**: 9 H 의 가장 일관된 패턴 (cohort drift) 직접 검증.
4. **paradigm 안 마지막 시도**: A 결과로 paradigm 안 ceiling 가설 결정.
5. **temporal_cohort 신규 카테고리 first-touch** (§10.7 FREE).

**B/C 미루기 조건 (즉, B/C 로 돌아갈 trigger)**:
- A 결과 noise (Δ ≤ +0.001pt) → **B confirmed (paradigm shift mandatory)**.
  H016 = backbone replacement (OneTrans full / HSTU trunk / InterFormer).
- A 결과 measurable + OOF-Platform gap 줄지 않음 → C 후보 (OOF 재정의가
  더 fundamental). H016 sub-H.
- A 결과 strong PASS → recency variants (exp decay, larger range, per-dataset).

## (조건부) Re-entry justification

해당 없음. `temporal_cohort` 신규 카테고리 first-touch.

§10.3 challenger rule (3회 연속 mutation REFUTED 후 강제 challenger) 정합:
- H011 / H012 / H013 / H014 = 4 H 모두 H010 anchor 위 mutation REFUTED.
- H015 = **새 카테고리 (temporal_cohort)**, 다른 axis (training procedure).
  challenger 사고 정합. **paradigm 안 마지막 카테고리 시도**.

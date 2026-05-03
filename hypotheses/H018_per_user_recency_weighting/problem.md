# H018 — Problem (per-user recency loss weighting)

## Background — what we already learned

- 9 H 누적 (H006–H014) 모두 H010 anchor 위 mutation 시도. H010 (NS→S xattn,
  Platform 0.8408 prior / 0.837806 corrected) 가 4-layer ceiling diagnosis
  의 단일 champion.
- 4-layer diagnosis 종료 (H014 verdict): L1 hyperparameter / L3 NS xattn
  sparse / L4 dense long-seq 모두 retire. **L2 (cohort drift hard
  ceiling) 만 남음**.
- Triple H setup (H015/H016/H017) 가 L2 직접 attack:
  - **H015** (per-batch linear recency loss [0.5, 1.5]): Platform 0.83805
    vs corrected anchor 0.837806 = Δ +0.00024pt. **§17.3 binary 임계
    +0.5pt 미달 → 형식상 REFUTED**. 그러나 **부호 positive + carry-forward
    pattern (recency direction 작동)** 신호.
  - **H016** (OOF future-only redefine): Platform 0.83192 vs anchor =
    Δ −0.00589pt → REFUTED for model lift, **PASS for OOF infra** (gap
    −0.0036pt ≈ 0, redefined OOF 가 platform 분포에 align).
  - **H017** (recency exp decay form): submission lost, no verdict.
- 5 H 모두 OOF (legacy random-user) saturated 0.858~0.860. local
  의사결정 신호 죽음.

## Core observation

H015 의 marginal +0.00024pt 는 random noise 인지 weak signal 인지 구분
어렵지만, **per-batch coarse weighting** 의 알려진 약점:
- batch 내 label_time spread 작으면 weight 거의 1.0 일관 → 효과 무력.
- batch 평균 가까운 sample 들 거의 동일 weight → temporal granularity
  손실.
- inactive user 와 active user 가 같은 weight (per-batch 만 본 결과).

**Per-user time-decay weighting** 은 동일 mechanism class 안 finer
granularity:
- 각 user 의 own event history 기준 time gap (`days_since_last_event`)
  계산.
- weight = `exp(-gap / tau)` (tau ≈ 14 일).
- 같은 user 의 recent event 는 큰 weight, old event 는 작은 weight.
- 다른 user 와 무관 (per-batch coarse 가 아닌 per-user fine).

**Falsifiable claim**: per-user time-decay weighting 이 per-batch linear
weighting (H015) 위 추가 lift Δ ≥ +0.5pt (§17.3 binary cut, OOF redefined
H016 default) 또는 Δ ∈ [+0.1, +0.5pt] (measurable but sub-threshold —
mechanism class 가 marginal 인 confirmation, paradigm shift mandatory).

## Why per-user, not per-batch (mechanism reasoning)

1. **Granularity mismatch**: per-batch min/max 는 batch composition 의존
   (shuffle 로 변동). 같은 sample 이 다른 batch 에서 다른 weight 받음.
   per-user 는 stable.
2. **Cohort drift 의 진짜 source**: user 마다 last interaction 시점이
   다름 → user-level recency 가 distribution shift 의 근본 변수.
   batch-level 평균은 user-level signal 을 dilute.
3. **Production CTR standard**: Tencent / Meta / Google production
   시스템에서 per-user time-decay 가 standard (논문 부재, infra
   know-how). per-batch 는 minimum viable form 의 약식.

## Constraint-aware framing

- **§17.6 cost cap**: 누적 cost 압박 (H006–H017 ~40h). H018 = T2.4 ~3-4h
  envelope (H015 동일) 가능. paradigm shift (TWIN/OneTrans) 의 large cost
  대안.
- **§17.2 single mutation**: granularity change (per-batch → per-user)
  단 1개. mechanism stack (NS xattn + DCN-V2 + recency loss class)
  byte-identical.
- **§17.4 rotation**: temporal_cohort 4번째 entry. challengers.md 에
  재진입 정당화 (§17.4 cite + §10.7 audit).

## Falsifiable predictions

- **PASS (strong)**: Δ vs H015 corrected ≥ +0.5pt → mechanism class
  saved, per-user granularity 가 cohort drift 핵심 lever 검증.
- **PASS (measurable)**: Δ ∈ [+0.1, +0.5pt] → per-user 효과 있지만
  saturated. mechanism class retire 권고, paradigm shift 정당.
- **REFUTED (degraded)**: Δ < 0 → per-user weighting 학습 disrupt (high
  variance per-sample weight 가 gradient noise 증폭). granularity 가설
  REFUTED, recency mechanism class 전체 retire.
- **REFUTED (noise)**: Δ ∈ [0, +0.1pt] → per-user 가 per-batch 위 무
  effect → granularity 가설 REFUTED, **recency mechanism class 전체
  retire, paradigm shift mandatory** (TWIN / OneTrans / HSTU).

## Decision tree (post-result)

| Outcome | Δ vs H015 | Action |
|---|---|---|
| strong | ≥ +0.5pt | H018 = new anchor, sub-H = tau sweep, per-user variants. mechanism class saved. |
| measurable | [+0.1, +0.5pt] | per-user 약 effect, saturated. mechanism class retire 권고. H019 = paradigm shift candidate (TWIN). |
| noise | [0, +0.1pt] | granularity 무 effect → recency mechanism class 전체 REFUTED → H019 = paradigm shift mandatory. |
| degraded | < 0 | per-user gradient noise. H018-sub = tau sweep (smoother decay) 또는 mechanism class retire. |

## Out of scope

- tau (decay constant) sweep — sub-H 후보 (post H018 PASS).
- Hybrid per-batch × per-user — sub-H 후보.
- Sample weighting in cross-domain transfer — separate H.
- OneTrans / TWIN paradigm shift — H018 REFUTED 시 H019 후보.

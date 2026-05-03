# H016 — Challenger Frames

> primary_category = `temporal_cohort` (H015 first-touch + H016 sub-form).

## Frame A (default — what we're proposing)

- **Claim**: 9 H 의 OOF stable / Platform 변동 패턴 = OOF 정의 자체가 platform
  과 다른 distribution 측정. OOF 재정의 (random 10% user → label_time
  future-only) 시 OOF 가 platform proxy 됨, mechanism 의 진짜 효과 가시화 →
  Δ Platform ≥ +0.005pt (strong) 또는 ≥ +0.001pt (measurable).
- **Mechanism**: dataset.py 의 OOF 분리 로직 변경. user-based → time-based.
- **Why this could be wrong**:
  1. OOF 새 정의가 train cohort 도 변경 (label_time cutoff 가 random user
     보다 train set 작음) → 학습 자원 감소로 effect 작음.
  2. cohort drift 가 OOF measurement 의 문제가 아닌 진짜 distribution shift
     → OOF 재정의해도 train-eval gap 그대로.
  3. 새 OOF 가 platform 과 정확히 align 안 됨 (정확히 어떤 distribution 인지
     unknown).

## Frame B (counter-frame — paradigm shift mandatory)

- **Claim**: H015 / H016 / H017 셋 다 noise → cohort drift 가 paradigm 안
  ceiling 의 진짜 정체이지만 **paradigm 안에서 못 깸**. backbone replacement
  mandatory.
- **Distinguishing experiment**: triple-H 셋 다 결과로 결정.

## Frame C (orthogonal frame — H015 train-side approach)

- **Claim**: H015 의 train-side recency weighting 이 OOF 재정의보다 더 효과적.
- **Distinguishing experiment**: H015 vs H016 paired Δ.

## Decision

A 선택 — H015 와 다른 axis (measurement procedure 측면) 의 cohort drift attack.
같이 launch 시 두 form 의 효과 비교 가능.

**B/C 미루기 조건**:
- A 결과 noise + H015 / H017 도 noise → B confirmed (paradigm shift mandatory).
- A 결과 noise + H015 strong → C 가 더 효과적, train-side approach 우선.
- A 결과 strong + H015 noise → A 가 더 효과적, OOF 재정의 우선.

## (조건부) Re-entry justification

`temporal_cohort` 카테고리 H015 first-touch + H016 sub-form (다른 axis).
§10.7 의 "변형/sub-H 패턴" 정합 — 같은 카테고리 내 같은 mechanism class 의
다른 form. triple-H setup (H015/H016/H017) 의 sibling 관계.

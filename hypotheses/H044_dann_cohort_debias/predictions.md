# H044 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.003pt platform vs H019): 15% — cohort drift 가 ceiling 의
  진짜 정체 + GRL 이 본 form 으로 충분히 attack.
- Measurable (+0.001~+0.003pt): 30% — cohort-invariance 부분 작동.
- Noise (-0.001~+0.001pt]: 30% — timestamp 가 cohort proxy 로 부족 또는
  GRL effect 가 다른 mechanism 들로 흡수됨.
- Degraded (<-0.001pt): 25% — DANN training 의 known instability (lambda
  tuning 잘못 시 학습 destabilize), 또는 timestamp 가 task 와 entangled
  (시간 모르면 main task 도 안 됨).

## Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | cohort-attack axis main lever. sub-H = lambda sweep (0.1/0.5/1.0/2.0) + cohort label 다양화 (label_time bins, user activity quartiles). |
| [+0.001, +0.003pt] | additive 약 effect. sub-H 가치 — cohort signal 의 다른 form 추가. |
| (-0.001, +0.001pt] | timestamp 가 cohort proxy 로 부족 또는 다른 mechanism 들이 이미 cohort 학습 중. 다른 cohort signal (split_seed bin, user activity) 시도. |
| < -0.001pt | DANN training instability 또는 axis dead. lambda↓ retry 또는 retire. |

추가 paired 비교:
- Δ vs H038 platform (0.839071): aux MSE positive (H038) vs adversarial (H044) 의 정확한 effect size. ≥ +0.001pt 면 GRL 의 specific contribution.

## Risk
- DANN training 의 known instability — gradient reversal 이 backbone 의
  주 task BCE 와 fight 하면 학습 destabilize.
- timestamp 가 *task-relevant* signal 일 가능성 — 시간 정보 자체가 prediction
  에 도움이라면 forget 강제 시 main task 도 약화. 그러면 platform 도 drop.
- F-A 패턴 의 진짜 정체가 cohort drift 가 아닐 가능성 — sample size 작음
  (1000 rows), random variance 가 dominant 면 attack 무용.
- §0.5 step 2 정확도 — "timestamp 가 cohort 의 main proxy" 가설이 진짜
  signal 매핑 인지 검증 안 됨 (H041 의 R2 reframe wrong frame 사례 같은
  risk).

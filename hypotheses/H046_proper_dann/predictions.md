# H046 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.003pt): 15% — F-A 패턴 진짜 cohort drift 면 hit 가능.
- Measurable (+0.001~+0.003pt): 30% — partial transfer.
- Noise (-0.001~+0.001pt]: 35%.
- Degraded (<-0.001pt): 20% — DANN training instability (lambda 너무 큼) 또는 timestamp cohort proxy 부족.

## Decision tree

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | cohort attack main lever, lambda sweep |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | cohort proxy 부족, 다른 cohort label |
| < -0.001pt | DANN setup destabilize, lambda↓ retry 또는 retire |

## Risk
- DANN training instability — lambda=0.1 보수적이지만 여전 발산 가능성.
- timestamp 가 task-relevant signal 일 수도 — forget 시 main task 도 약화.
- F-A 진짜 정체가 cohort drift 가 아니면 attack 무용.

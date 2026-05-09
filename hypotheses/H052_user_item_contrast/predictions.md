# H052 — predictions.md

## Outcome distribution
- Strong (≥+0.003pt): 18% — SSL representation supervision 본 데이터 작동 시.
- Measurable (+0.001~+0.003pt): 30% — additive.
- Noise (-0.001~+0.001pt]: 32% — in-batch negatives 부족 또는 BCE 와 redundant.
- Degraded (<-0.001pt): 20% — contrast 가 main BCE 와 fight.

## Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | contrastive axis main lever, lambda + memory bank sub-H |
| [+0.001, +0.003pt] | additive, sub-H 가치 |
| (-0.001, +0.001pt] | in-batch 부족, harder negative mining |
| < -0.001pt | retire |

## Risk
- B=1024 → 1023 in-batch negatives. SimCLR 본 paper 는 4K-32K negatives 필요 보고.
- contrast_temperature=0.1 너무 sharp → loss saturation 위험. 0.5 milder sub-H 가능.
- user_repr 와 item_repr 의 representation space 다름 — 학습 초기 alignment 어려움.

# H041 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.005pt): 15% — gate bimodal 학습 + cold_clsfier specialize.
- Measurable (+0.001~+0.005pt): 30% — 약 effect.
- Noise (-0.001~+0.001pt]: 35% — gate 안 학습됨 (init 0.88 그대로).
- Degraded (<-0.001pt): 20% — dual branch destabilize.

## Decision tree (post-result)

| Δ vs H019 (Val 0.8372 / OOF 0.8611) | Action |
|---|---|
| ≥ +0.005pt | cold-start lever 진짜, sub-H = explicit novelty signal (history-candidate sim) |
| [+0.001, +0.005pt] | 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | gate 안 학습, single classifier 충분 |
| < -0.001pt | dual branch destabilize, R2 axis retire |

## Risk
- gate init 0.88 → 학습이 0.88 부근에서 안 움직이면 H019 의 noise 만 추가.
- §3.5 target_in_history=0.4% 의 해석이 "0.4% familiar + 99.6% novel split"
  이 아닌 "100% novel regime" 일 가능성 — 이 경우 dual branch 의 split boundary
  자체가 데이터에 없음 → R2 reframe wrong.

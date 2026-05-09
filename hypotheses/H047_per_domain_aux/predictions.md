# H047 — predictions.md

## Outcome distribution
- Strong (≥+0.003pt): 10%
- Measurable (+0.001~+0.003pt): 30%
- Noise (-0.001~+0.001pt]: 40%
- Degraded (<-0.001pt): 20% — aux 가 main 과 fight, capacity 분배 잘못.

## Decision tree

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | multi-task axis main lever, sub-H weight + head depth |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | main head 가 이미 per-domain signal 충분, axis exhausted |
| < -0.001pt | aux fight main, weight↓ 또는 retire |

## Risk
- aux weight 0.25 가 너무 높을 가능성 — main BCE 와 비등 → backbone capacity 분산.
- per-domain pooled (mean) 가 너무 coarse — fine-grained signal 손실.
- domain d (frac_empty 8%) 에서 aux head 가 informative 안 함.

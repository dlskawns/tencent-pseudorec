# H048 — predictions.md

## Outcome distribution
- Strong (≥+0.003pt): 20% — user × item axis 가 진짜 lever 면 H019 magnitude.
- Measurable (+0.001~+0.003pt): 35% — additive.
- Noise (-0.001~+0.001pt]: 30% — bilinear form 부족, 다른 form 필요.
- Degraded (<-0.001pt): 15% — overfit / capacity 분배 잘못.

## Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | user × item main lever, sub-H = stacked layers + 다른 cross form (additive + multiplicative) |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | bilinear 부족, multi-layer 또는 retire |
| < -0.001pt | retire |

## Risk
- W matrix (D×D=4096 params) overparameterize — sample-scale 1000 rows 에서
  overfit risk.
- Gate init 0.12 → 학습 초기 거의 OFF, gate 학습 안 되면 H019 byte-identical.
- TWIN candidate Q 와 redundant 가능 — TWIN 이 이미 candidate-aware 하므로
  추가 cross 가 marginal.

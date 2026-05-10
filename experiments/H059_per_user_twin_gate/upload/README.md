# H059_per_user_twin_gate — Technical Report

> H057 SUCCESS TEMPLATE-DRIVEN: H057 (TWIN aggregator mean → per-user attention)
> Platform +0.0017pt = 14 H 만의 첫 ANCHOR break. H059 = same template at gate
> layer: scalar twin_gate → per-sample sigmoid(Linear(output)).

## 1. Hypothesis & Claim
- Hypothesis: H019 의 scalar twin_gate (=0.12 init, 전체 batch 동일) 가 모든 user
  에 동일 weight TWIN signal 적용. user 별 history 양/질 다르면 (§3.5 bimodal,
  imbalance) per-user gate 가 더 informative.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation
```python
# H019: gate = sigmoid(twin_gate)  # scalar
# H059:
if use_per_user_twin_gate:
    per_user_gate = sigmoid(self.twin_per_user_gate_proj(output))  # (B, 1)
    twin_residual = (twin_residual / scalar_gate) * per_user_gate
output = output + twin_residual
```
- bias init = -2.0 → sigmoid ≈ 0.12 (H019 등가, drift 없는 init)
- 학습 시 per-user 0~1 사이 adaptive

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | H059 anchor candidate, sub-H = gate input richer |
| [+0.001, +0.003pt] | additive, H057+H059 stacking 시도 |
| (-0.001, +0.001pt] | scalar gate 충분 |
| < -0.001pt | per-user gate noisy, retire |

## 4. Files
| File | H019 대비 |
|---|---|
| model.py | +1 __init__ param, +twin_per_user_gate_proj init, + per-user gate apply |
| train.py | +1 argparse +1 plumbing |
| infer.py | +1 cfg.get (H043 방지) |
| run.sh | + flag |
| 외 .py | byte-identical |

trainable params 추가: ~64.

## 5. Carry-forward
- §17.2 single mutation: scalar gate → per-user gate.
- §17.4: per_user_adaptive_gating axis (H057 success template, gate layer).
- §10.10 σ(-2)≈0.12 mandate 통과 (bias init).
- **infer.py flag parity** verified.

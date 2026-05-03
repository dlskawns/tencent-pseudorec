# H017 — Predictions

> H015 sub-form variant. **Triple-H setup with H015/H016 동시 launch**.

## P1 — Code-path success
- PASS expected. trainer.py exp branch + auto-normalize.

## P2 — Primary lift (vs anchor + vs H015)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H010 corrected (0.837806) | strong | ≥ +0.005pt |
| Δ vs H010 corrected | measurable | [+0.001, +0.005pt] |
| Δ vs H010 corrected | noise | (−0.001, +0.001pt] |
| Δ vs H010 corrected | degraded | < −0.001pt |
| **Δ vs H015** | exp form 효과 | ≥ +0.001pt |
| Δ vs H015 | form 변경 효과 없음 | (−0.001, +0.001pt] |

## P3 — NS xattn entropy
H010 baseline [0.8127, 0.8133]. 변화 미세 expected.

## P4 — §18 PASS

## P5 — val ↔ platform

## P6 — OOF-Platform gap
H015 vs H017 비교: 어느 form 이 gap 더 줄이는지.

## Decision tree

| Result | Implication | H018 |
|---|---|---|
| Δ vs H010 strong + Δ vs H015 ≥ +0.001pt | exp form 가 진짜 효과 | H018 = exp form variants (range tuning, per-dataset). |
| Δ vs H015 noise | form 변경 효과 없음, mechanism class marginal | (H015/H016 결과와 종합) backbone replacement. |
| Δ vs H015 degraded | exp 가 학습 disrupt | H018 = linear with wider range. |

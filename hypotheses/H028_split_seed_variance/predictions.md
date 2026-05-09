# H028 — Predictions

## P1 — Code-path success (3 runs)
- 각 launch NaN-free. §18.8 SUMMARY block emit. OOF eval 작동 (H023 fix carry-forward).

## P2 — Variance estimate (primary)

| split_seed | best val_auc | best oof_auc | platform AUC |
|---|---|---|---|
| 42 (control) | TBD | TBD | TBD |
| 43 | TBD | TBD | TBD |
| 44 | TBD | TBD | TBD |

**σ_val** = stdev of 3 best_val_auc.
**σ_oof** = stdev of 3 best_oof_auc.
**σ_platform** = stdev of 3 platform AUC (if 회수).

## Cut

| σ_val | cohort hypothesis | next action |
|---|---|---|
| ≤ 0.001pt | REFUTED — cohort 무관 | H029/H030 결과로 root cause 좁힘 |
| (0.001, 0.005pt] | partial — measurement noise | additional split_seed sweep 또는 다른 axis |
| > 0.005pt | confirmed — cohort saturation | future H 모두 multi-split (≥3) 의무 |

## Falsification

> "9 H 의 val_auc 0.832~0.836 ceiling 이 split_seed=42 cohort 의존한다."

거짓 (σ_val ≤ 0.001pt) → cohort 가설 retire, 다른 root cause (Keskar/loss/Bayes).

# H029 — Predictions

## P1 — Code-path success
- NaN-free. §18.8 SUMMARY block. OOF eval 작동.

## P2 — val_auc / Platform AUC

vs H023 (bce explicit, batch=1024) baseline best_val 0.8334:

| outcome | val_auc | Keskar |
|---|---|---|
| strong | > 0.840 | confirmed (4pt+ lift) |
| measurable | [0.836, 0.840] | partial — sub-linear sweep |
| noise | (0.834, 0.836] | weak signal (within H023 noise band) |
| flat | ≤ 0.834 | REFUTED |

## P3 — train_loss scale + convergence rate

batch=256 → 8x more steps per epoch vs batch=2048. epoch-level convergence
~8x slower per epoch but per-step optimization 더 정밀 expected. trainer
의 epoch print 가 best_epoch 늦은지 (예: 8/10) 확인.

## P5/P6 — val ↔ platform / OOF ↔ platform

H016 framework — val_auc → platform 변환 추정 (val 0.840+ 면 platform
0.843+ expected).

## Falsification

> "batch=256 + lr=1e-4 (original default) 가 val_auc 0.836 ceiling 깨고
> 0.840+ 도달한다."

거짓 → Keskar gap 가설 REFUTED. ceiling 이 batch/lr 무관 → cohort/loss/Bayes
hypothesis 가 root cause.

## Cost

T2.4 batch=256 → wall ~5-7h (8x steps but per-step faster — 1.7x net).
~$5-7 cost (per-job cap audit).

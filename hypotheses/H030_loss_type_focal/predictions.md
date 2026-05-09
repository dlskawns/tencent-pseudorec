# H030 — Predictions

## P1 — Code-path success
- NaN-free. focal loss numerical stability OK (logits clamped). §18.8
  SUMMARY block. OOF eval 작동.

## P2 — val_auc vs H023 (bce explicit baseline 0.8334)

| Δ vs H023 | classification | loss_type effect |
|---|---|---|
| > +0.005pt | strong focal lift | future default focal |
| (+0.001, +0.005pt] | measurable | sub-H γ sweep 가능 |
| (-0.001, +0.001pt] | noise | loss type 무관 |
| < -0.001pt | degraded | bce default 유지 |

## P3 — train_loss scale

focal expected ~0.12 (12% pos rate × focal γ=2). 만약 0.16~0.32 면
focal 작동 안 하는 (numerical issue) 신호.

## P5/P6 — val ↔ platform

H016 framework — val 0.84 → platform 0.84 expected mapping.

## Falsification

> "focal loss (α=0.25, γ=2.0) 가 bce (H023 baseline) 위 val_auc Δ ≥
> +0.005pt 추가 lift 만든다."

거짓 → focal 가 bce 위 추가 가치 없음 + train_loss 0.12 (focal-like) 의
다른 origin 검토 mandatory.

## Cost

T2.4 batch=1024 → wall ~3.5h. ~$5.

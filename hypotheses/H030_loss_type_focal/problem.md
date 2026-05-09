# H030 — Problem (loss_type focal, loss isolation diagnostic)

## Background

9 H train_loss scale 다양:
- H015: ~0.32 (bce-like)
- H011/H012/H013/H018/H025: ~0.12 (focal-like)
- H023/H026: ~0.16 (bce explicit, batch=1024)
- H022: ~0.21 (bce default, batch=2048, anomaly)

run.sh 에 `--loss_type` explicit 한 H 는 H023 (bce) + H030 (focal).
나머지는 default (bce). 그런데 train_loss 0.12 = focal characteristic →
**사용자가 "$@" override 로 focal 적용했을 가능성**. H030 = explicit focal
처음 측정 (control = H023 bce explicit).

## Falsifiable claim

H030 (focal explicit) val_auc vs H023 (bce explicit) baseline:
- **|Δ| > 0.005pt** → loss_type 효과 큼 → prior H 들 loss_type 다름 가능성
  → cross-H val 비교 invalid carry-forward.
- **|Δ| ≤ 0.001pt** → loss_type 무관, train_loss 차이 다른 origin.

## Measurement only — single mutation = loss_type swap

H010 mechanism + envelope byte-identical EXCEPT `--loss_type focal
--focal_alpha 0.25 --focal_gamma 2.0` (paper default).

## Note on H005

H005 = focal loss attempt 도 prior. REFUTED (Δ vs anchor = 무효 — pre-
correction era, heuristic fallback). H030 = first valid focal measurement
on corrected eval data.

## Decision tree

| Δ vs H023 (bce) | loss_type 효과 | next |
|---|---|---|
| > +0.005pt | focal > bce 명확 | future H 모두 focal default 권장 |
| ∈ (-0.005, +0.005pt) | 무관 | train_loss 차이 다른 원인 (weighting, scaling) |
| < -0.005pt | bce > focal | H023 bce regime confirm |

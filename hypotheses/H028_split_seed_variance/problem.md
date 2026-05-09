# H028 — Problem (split_seed variance, methodology validation)

## Background

9 H 누적 (H011~H018, H022, H023, H025, H026) val_auc 0.832~0.836 narrow
band (0.4pt = 0.5% relative variation). OOF (legacy) 0.857~0.860 더 좁은
saturation. mechanism 변경 무관 → measurement framework 자체 의문.

모든 prior H run.sh 가 `--split_seed 42` 공유 → 같은 val/OOF cohort
fit → val saturation 자연스러울 수 있음.

## Falsifiable claim

`split_seed` 만 42→43→44 변경 시 val_auc 가:
- **변동 > 0.005pt** → cohort saturation hypothesis (#1) confirmed.
  9 H 의 val 비교 invalid carry-forward, future H 모두 multi-split 의무.
- **변동 < 0.001pt** → cohort 가 진짜 ceiling 아님. 다른 hypothesis (#2-#4)
  로 root cause 이동.

## Measurement only — NO model mutation

H010 mechanism (NS xattn + DCN-V2) byte-identical. envelope (10ep × 30%
× patience=3) byte-identical. `--seed 42` (training seed) byte-identical
across 3 launches. `--split_seed` 만 변경.

## Decision tree (post-result)

| 산출 | Action |
|---|---|
| σ_val > 0.005pt | cohort saturation confirmed → H10+ 모든 H multi-split |
| σ_val ≤ 0.001pt | cohort 무관 → H029 (Keskar) / H030 (loss) 결과로 root cause 좁힘 |
| σ_val ∈ (0.001, 0.005pt] | partial — measurement noise band 안 |

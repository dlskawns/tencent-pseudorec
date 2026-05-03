# E_H013 — Training Result Intake

## Submission meta
- submitted_at: ~2026-05-02 (log paste 2026-05-03 retroactive intake).
- platform: Taiji (Tencent Angel) `ams_2026_1029735554728163481`.
- gpu: TBD.
- wall_time_train: **4시간 8분 13초** (H010 3:44:54 +10%).
- wall_time_infer: **95.57초** (H010 297초 −68%).
- cost_usd: TBD (Taiji 가격 미공개).
- **Mutation**: `lr 8e-4 + batch 2048` (Linear scaling rule applied to H010).

## Headline metrics (log paste)

```
eval auc: 0.834376
inference time 95.57s
OOF AUC: 0.8573
OOF LogLoss: 0.2327
attn_entropy_per_layer ≈ [0.8138, 0.8140]
```

## metrics.json blob
TBD — 사용자 raw paste 시 verbatim 저장.

## Falsification check

| P | Predicted | Measured | Status |
|---|---|---|---|
| P1 (code-path) | NaN-free 완주 | Training complete | **PASS** |
| P2 (primary lift) | Δ vs H010 anchor ≥ +0.001pt | −0.0064pt | **REFUTED — degraded (Frame A REFUTED)** |
| P2 strong | Δ ≥ +0.005pt | 미달 | — |
| P3 (entropy) | 변화 미세 expected | [0.8138, 0.8140] | **PASS** |
| P4 (§18 인프라) | PASS | eval auc 0.834376 ≠ 0.5 | **PASS** |
| P5 (val↔platform) | gap ≤ 0.05 | −0.0028 | **PASS** |
| P6 (OOF-platform gap) | ≤ 2pt | 2.29pt | **경계 미달** |

## §18.8-style SUMMARY (retroactive — log paste 2026-05-03)

> User pasted training log 9 epochs + best/OOF/eval. emit_train_summary()
> 미적용된 H — 수동 reconstruct.

```
==== TRAIN SUMMARY (H013_hyperparameter_calibration, RETROACTIVE) ====
git=TBD cfg=TBD seed=42 ckpt_exported=best
epoch | train_loss | val_auc | val_logloss | oof_auc
  1   |    N/A     |  N/A    |    N/A      | (final 0.8573)
  2   |    N/A     | 0.8287  |   0.2870    | —
  3   |   0.1217   | 0.8300  |   0.2874    | —
  4   |   0.1201   | 0.8313  |   0.2872    | —
  5   |   0.1208   | 0.8302  |   0.2835    | —
  6   |   0.1200   | 0.8316  |   0.2851    | —  (best)
  7   |   0.1204   | 0.8309  |   0.2855    | —
  8   |   0.1198   | 0.8310  |   0.2831    | —
  9   |   0.1211   | 0.8300  |   0.2853    | (final 0.8573)
best=epoch6  val=0.8316  oof=0.8573 (legacy)
last=epoch9  val=0.8300  oof=0.8573
overfit=+0.0016 (best_val − last_val)
calib pred=N/A label=N/A ece=N/A
==== END SUMMARY ====
```

## Trajectory analysis (post log paste)

| metric | value | 비고 |
|---|---|---|
| best_epoch | 6 / 9 | mid (H011/H012 의 4 보다 늦음 — large lr 효과) |
| best_val_AUC | **0.8316** | H011/H012 보다 낮음 (0.8336/0.8331) |
| last_val_AUC | 0.8300 | epoch 9 |
| overfit_gap | **+0.0016** | 거의 없음 (lr 큰 데도 overfit 표시 안 됨) |
| OOF (legacy) | 0.8573 | **8 H 중 처음 0.86 미달** (Keskar large-batch generalization gap) |
| platform AUC | **0.8344** (uncorrected eval) | INDEX prior |
| **val ↔ platform gap** | **−0.0028** | val UNDER platform (H011/H012/H015 와 같은 부호) |
| **OOF ↔ platform gap** | **+0.0229** | OOF saturated OVER platform |

**Caveat (uncorrected eval)**: platform 0.8344 은 organizer 2026-05-02~03
eval data correction **이전** 측정. corrected anchor 0.837806 와 직접 비교
invalid. −0.003pt heuristic shift 적용 시 corrected ≈ 0.8314 → §17.3
binary REFUTED 강화 (lr 8e-4 가 학습 자체 disrupt).

## Notes / surprises

- **OOF AUC 0.8573 = 8 H 중 처음 0.86 미달**: 그동안 OOF 0.858~0.860
  saturated 였는데 H013 만 drop. lr 8e-4 + batch 2048 (linear scaling rule)
  이 학습 자체를 less optimal regime 으로 밀어냄. **Keskar et al. 2017**
  "On Large-Batch Training" 의 generalization gap 직접 manifest.
- **best_epoch=6 (H011/H012 의 4 보다 늦음)**: large lr 가 빠른 convergence
  보다는 늦은 plateau 야기 — large-batch 의 noisy gradient noise 때문.
- **train wall +10%, infer wall −68%**: 학습 더 오래 (epoch 9 vs 7) 하지만
  infer 매우 빠름 (params 변경 없음, 다른 H 들 IO bound). H012 의 IO bound
  가설 추가 evidence.

## Carry-forward (verdict.md F-N 카피)

- **F-1**: Linear scaling rule (lr 1e-4 → 8e-4 for batch 2048) REFUTED.
  Δ vs H010 anchor −0.0064pt = clearly degraded.
- **F-2**: Keskar large-batch generalization gap manifest (OOF drop 0.0016pt).
- **F-3**: 4-layer ceiling diagnosis 갱신 — L1 (hyperparameter) ❌ retire.
- **L1 retire confirmed** with corrected anchor estimate (still −0.006pt).

## Next actions

- H013 verdict.md 이미 REFUTED — log paste 로 trajectory 만 보강.
- L1 retire 결정 변동 없음.
- Carry-forward 적용 (H014~H018).

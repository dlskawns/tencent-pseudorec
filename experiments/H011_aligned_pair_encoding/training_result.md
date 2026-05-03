# E_H011 — Training Result Intake

## Submission meta
- submitted_at: 2026-05-01 13:41 (UTC) — log timestamp `20260501134147`.
- platform: Taiji (Tencent Angel) `ams_2026_1029735554728163481`.
- gpu: TBD (사용자 paste 시 보완).
- wall_time_train: **2시간 46분 54초** (`2:46:54`, H010 3:44:54 대비 **−25%**).
- wall_time_infer: **178.31초** (H010 297초 대비 **−40%**).
- cost_usd: TBD (Taiji 가격 미공개).

## Headline metrics (log paste)

```
2026-05-01 16:29:23 - 2:46:54 - OOF AUC: 0.8589, OOF LogLoss: 0.2342
2026-05-01 16:29:24 - 2:46:54 - attn_entropy_per_layer=[0.8130203485488892, 0.8132437467575073], threshold=5.6531, violation=False
2026-05-01 16:29:24 - Metrics dumped to /apdcephfs_fsgm2/.../ckpt/metrics.json
2026-05-01 16:29:24 - Training complete!
2026-05-01 16:29:27 [run.sh] training complete (aligned-pair-encoding); metrics at .../ckpt/metrics.json
eval auc: 0.834699
inference time 178.31s
```

## metrics.json blob
TBD — 사용자 paste 시 verbatim 저장 (`metrics.json`):
- best_step, best_val_AUC, best_oof_AUC.
- seed=42, git_sha (TBD), config_sha256 (TBD).
- mutation flags (`use_aligned_pair_encoding=true`, `aligned_pair_fids=[62,63,64,65,66,89,90,91]`, `use_ns_to_s_xattn=true`, `fusion_type=dcn_v2`, `dcn_v2_num_layers=2`, `dcn_v2_rank=8`, `ns_xattn_num_heads=4`).
- split_meta (rows, cutoff, oof_user_count).

## Falsification check

| P | Predicted | Measured | Status |
|---|---|---|---|
| P0 (audit) | mapping verified | aligned_audit.json PASS | **PASS** (pre-train) |
| P1 (code-path) | NaN-free 완주 | Training complete | **PASS** |
| P2 (primary lift) | Δ vs anchor ≥ +0.001pt | −0.0061pt vs H010 | **REFUTED (degraded)** |
| P2 strong | Δ ≥ +0.005pt | 미달 | — |
| P3 (entropy) | sparse / 미세 / uniform | [0.8130, 0.8132] = 변화 미세 | **변화 미세 (Frame B 신호)** |
| P4 (§18 인프라) | PASS | eval auc 0.834699 ≠ 0.5 | **PASS** |
| P5 (val↔platform) | gap ≤ 0.05 | TBD | TBD |
| P6 (OOF-platform gap) | ≤ 2pt | 2.42pt | **미달 (cohort overfit)** |

## Notes / surprises
- OOF 거의 보존 (−0.0007) + Platform 큰 폭 하락 (−0.0061) = **classic
  overfit signature 재발** (H009 패턴). 두 mechanism (H009 candidate
  prepend / H011 input embedding) 위치 다른데 같은 signature → cohort
  drift 가 platform 일반화의 hard ceiling 일 가능성.
- attn_entropy 거의 변화 없음 (Δ ~0.001) → H010 의 selective routing 이
  H011 input variance 흡수. predictions.md "Frame B" 신호 (NS-level
  binding 충분).
- Training/inference wall 큰 폭 단축 (−25% / −40%) → params 추가 0 +
  simpler weighted-mean 효과. positive carry-forward.
- OOF-platform gap 2.42pt = H006 3.5 → H010 1.88 narrow → H011 2.42
  역행. cohort effect 재증폭.

## Sanity gate (verify-claim §1)

| Check | Status |
|---|---|
| config_sha256 ↔ card.yaml | TBD (raw metrics.json paste 후) |
| git_sha ↔ card.yaml | TBD |
| split_meta ↔ card.yaml expected_split | TBD |
| §4.5 메타 (seed/git_sha/config_sha256) | TBD (raw paste 후 검증) |

현재 REFUTED 분류는 **provisional** — raw metrics.json 도착 후 sanity gate
재검증. Δ 부호와 분류 (degraded) 는 platform AUC 0.8347 만으로도 robust.

## Next actions
- 6 artifact 갱신 완료 (이 turn).
- H012 후보 추천: **multi_domain_fusion (MMoE/PLE)** — 신규 카테고리
  first-touch (§10.7 FREE). 4 도메인 expert routing, H010 anchor 위
  single mutation.
- 사용자 raw metrics.json paste 시:
  - sanity gate 재검증.
  - val_AUC + best_step 보강 → P5 채움.
  - config_sha256 INDEX 에 기록.

## §18.8-style SUMMARY (retroactive — log paste 2026-05-03)

> User pasted training log 7 epochs + best/OOF/eval. emit_train_summary()
> 미적용된 H 라 marker 없음 — 수동 reconstruct. Future H 는 §18.8 emit
> mandatory.

```
==== TRAIN SUMMARY (H011_aligned_pair_encoding, RETROACTIVE) ====
git=TBD cfg=TBD seed=42 ckpt_exported=best
epoch | train_loss | val_auc | val_logloss | oof_auc
  1   |    N/A     | 0.8261  |   0.2915    | (final 0.8589)
  2   |   0.1233   | 0.8308  |   0.2887    | —
  3   |   0.1225   | 0.8323  |   0.2864    | —
  4   |   0.1218   | 0.8336  |   0.2872    | —  (best)
  5   |   0.1210   | 0.8332  |   0.2873    | —
  6   |   0.1204   | 0.8319  |   0.2887    | —
  7   |   0.1205   | 0.8323  |   0.2892    | (final 0.8589)
best=epoch4  val=0.8336  oof=0.8589 (legacy)
last=epoch7  val=0.8323  oof=0.8589
overfit=+0.0013 (best_val − last_val)
calib pred=N/A label=N/A ece=N/A   (not computed during prior run)
==== END SUMMARY ====
```

## Trajectory analysis (post log paste)

| metric | value | 비고 |
|---|---|---|
| best_epoch | 4 / 7 | 중간 (early-mid convergence) |
| best_val_AUC | **0.8336** | 신규 (was TBD) |
| last_val_AUC | 0.8323 | epoch 7 |
| overfit_gap | **+0.0013** | best − last, 거의 없음 |
| OOF (legacy) | 0.8589 | saturated 0.858~0.860 range |
| platform AUC | **0.8347** (uncorrected eval) | INDEX prior |
| **val ↔ platform gap** | **−0.0011** | best_val UNDER platform (consistent direction across H011/H012/H013/H015) |
| **OOF ↔ platform gap** | **+0.0242** | legacy OOF dominantly OVER platform (saturation effect) |

**Caveat (uncorrected eval)**: platform AUC 0.8347 은 organizer
2026-05-02~03 eval data correction **이전** 측정. corrected anchor
0.837806 와 직접 비교 invalid. −0.003pt heuristic shift estimate
적용 시 corrected ≈ 0.8317. 어떤 경우든 §17.3 binary REFUTED.

**Carry-forward signal (corrected)**: H011 verdict (REFUTED — degraded)
변동 없음. trajectory data 만 보강.

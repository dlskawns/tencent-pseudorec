# E_H010 — Training Result Intake

## Submission meta
- submitted_at: 2026-04-30 16:14 (UTC) — log timestamp `20260430161426`.
- platform: Taiji (Tencent Angel) `ams_2026_1029735554728163481`.
- gpu: TBD (사용자 paste 시 보완).
- wall_time_train: **3시간 44분 54초** (`3:44:54` 로그).
- wall_time_infer: **297.02초**.
- cost_usd: TBD (Taiji 가격 미공개).

## Headline metrics (log paste)

```
2026-04-30 20:00:09 - 3:44:54 - OOF AUC: 0.8596, OOF LogLoss: 0.2323
2026-04-30 20:00:10 - 3:44:54 - attn_entropy_per_layer=[0.8126923441886902, 0.8133374452590942], threshold=5.6531, violation=False
2026-04-30 20:00:10 - Metrics dumped to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029735554728163481/angel_training_ams_2026_1029735554728163481_20260430161426_9b100cf6/95d1b1c49dca22ef019ddd742d662abe/ckpt/metrics.json
2026-04-30 20:00:10 - Training complete!
2026-04-30 20:00:13 [run.sh] training complete (ns-to-s-xattn); metrics at .../ckpt/metrics.json, eval auc: 0.840771
inference time: 297.02s
```

## metrics.json blob
TBD — 사용자 paste 시 verbatim 저장 (`metrics.json`):
- best_step, best_val_AUC, best_oof_AUC.
- seed=42, git_sha (TBD), config_sha256 (TBD).
- mutation flags (`use_ns_to_s_xattn=true`, `fusion_type=dcn_v2`,
  `dcn_v2_num_layers=2`, `dcn_v2_rank=8`, `ns_xattn_num_heads=4`).
- split_meta (rows, cutoff, oof_user_count).

## Falsification check

| P | Predicted | Measured | Status |
|---|---|---|---|
| P1 (code-path) | NaN-free 완주 | Training complete | **PASS** |
| P2 (primary lift) | Δ vs anchor ≥ +0.5pt | +0.7~+1.1pt (anchor 0.83X) | **PASS** |
| P2-sub (paired vs H008) | additive ∈ [+0.001, +0.005pt] | +0.0021pt | **additive** |
| P3 (attn entropy) | < 5.65 | [0.8127, 0.8133] | **PASS** (sparse) |
| P4 (§18 인프라) | PASS | eval auc 0.840771 ≠ 0.5 | **PASS** |
| P5 (val↔platform) | gap ≤ 0.05 | TBD | TBD |
| P6 (NS routing) | nontrivial spread | entropy 0.81 selective | **indirect PASS** |

## Notes / surprises
- Entropy 0.81 = threshold 의 14% — predicted (H004 baseline ~3.5–3.9)
  보다 4× 더 sparse. NS-tokens 이 384 S tokens 중 ~2 tokens 만 attend
  하는 hard routing 패턴.
- OOF gap (vs platform): 1.88pt — H006 3.5pt → H007 ~2.5pt → H008 1.98pt
  → H009 2.31pt 역행 → H010 **1.88pt** 다시 narrow. 통합 위치 선택이
  cohort fit 에도 영향.
- Inference wall 297초 (H008 220초 +35%, H009 259초 +14%) — NSToSCrossAttention
  layer 추가 latency cost.

## Sanity gate (verify-claim §1)

| Check | Status |
|---|---|
| config_sha256 ↔ card.yaml | TBD (raw metrics.json paste 후) |
| git_sha ↔ card.yaml | TBD |
| split_meta ↔ card.yaml expected_split | TBD |
| §4.5 메타 (seed/git_sha/config_sha256) | TBD (raw paste 후 검증) |

현재 PASS 분류는 **provisional** — raw metrics.json 도착 후 sanity gate
재검증. Δ 부호와 분류는 platform AUC 0.8408 만으로도 robust (provisional
verdict 인정).

## Next actions
- 6 artifact 갱신 완료 (이 turn).
- H011 후보 추천: **aligned `<id, weight>` pair encoding** (orthogonal
  axis, interference 위험 0, CLAUDE.md §3 / §4.8 mandate).
- 사용자 raw metrics.json paste 시:
  - sanity gate 재검증.
  - val_AUC + best_step 보강 → P5 채움.
  - config_sha256 INDEX 에 기록.

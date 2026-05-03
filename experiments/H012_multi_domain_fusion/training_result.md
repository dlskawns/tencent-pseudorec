# E_H012 — Training Result Intake

## Submission meta
- submitted_at: 2026-05-02 03:25 (UTC) — log timestamp `20260502032539`.
- platform: Taiji (Tencent Angel) `ams_2026_1029735554728163481`.
- gpu: TBD.
- wall_time_train: **2시간 49분 43초** (H010 3:44:54 −24%).
- wall_time_infer: **137.39초** (H010 297초 −54%).
- cost_usd: TBD.
- **Critical user override**: `batch_size=2048` (default run.sh 256 의 8×).
  lr default 1e-4 추정 → linear scaling rule 미적용 → underpowered.

## Headline metrics (log paste)

```
2026-05-02 06:16:08 - 2:49:43 - OOF AUC: 0.8589, OOF LogLoss: 0.2335
2026-05-02 06:16:09 - 2:49:43 - attn_entropy_per_layer=[0.8129708766937256, 0.8139130473136902], threshold=5.6531, violation=False
2026-05-02 06:16:09 - 2:49:43 - moe_gate_entropy_per_block=[1.377906322479248, 1.3629282712936401], collapse_threshold=0.6931, uniform_threshold=1.3863, collapse_violation=False
2026-05-02 06:16:09 - 2:49:44 - Metrics dumped to /apdcephfs_fsgm2/.../ckpt/metrics.json
2026-05-02 06:16:09 - 2:49:44 - Training complete!
eval auc: 0.838047
inference time 137.39s
```

## metrics.json blob
TBD — 사용자 paste 시 verbatim 저장:
- best_step, best_val_AUC, best_oof_AUC.
- seed=42, git_sha (TBD), config_sha256 (TBD).
- mutation flags (`use_multi_domain_moe=true`, `num_experts=4`, `moe_ffn_hidden=128`,
  `use_ns_to_s_xattn=true`, `fusion_type=dcn_v2`).
- **batch_size=2048** (사용자 override) + lr value (확인 필요).
- moe_gate_entropy_per_block, moe_collapse_threshold, moe_uniform_threshold.
- split_meta.

## Falsification check

| P | Predicted | Measured | Status |
|---|---|---|---|
| P1 (code-path) | NaN-free 완주 | Training complete | **PASS** |
| P2 (primary lift) | Δ vs anchor ≥ +0.001pt | −0.0028pt | **REFUTED** |
| P2 strong | Δ ≥ +0.005pt | 미달 | — |
| P3 (gate entropy) | specialized [0.69, 1.30] | [1.378, 1.363] | **uniform (Frame B)** |
| P3 collapse | < 0.69 | 미발생 | safe |
| P4 (§18 인프라) | PASS | eval auc 0.838 | **PASS** |
| P5 (val↔platform) | gap ≤ 0.05 | TBD | TBD |
| P6 (OOF-platform gap) | ≤ 2pt | 2.10pt | **경계 미달** |

## Notes / surprises
- **MoE gate entropy = uniform**: [1.378, 1.363] vs uniform threshold 1.386.
  격차 0.009~0.024 → gate 가 random init (1.378) 에서 거의 안 멀어짐.
  underpowered regime 가능성 (F-2). 또는 NS-token 7개 input 자체가 routing
  결정 정보 부족.
- **Δ vs H011 = +0.0033pt**: H012 가 H011 보다 나음. NS-level (H012) 이
  input-stage (H011) 보다 안전한 stacking 위치 confirmed.
- **Inference wall −54%**: H010 297초 → H012 137초. params 추가 66K 인데
  학습/inference 모두 단축. **IO bound 신호** (GPU idle, 데이터 로딩
  bottleneck). H013 calibration 시 num_workers / buffer_batches 증가 같이
  시도.
- **OOF AUC 0.8589 = 7개 H 모두 거의 동일** (0.858~0.860 범위). Platform
  AUC 만 변동 → OOF 와 Platform 의 분포 자체가 다름 (cohort/temporal
  drift, H011 F-5 누적 confirm).

## 4-layer ceiling diagnosis (carry-forward to H013)

| Layer | 가설 | H013 검증 우선순위 |
|---|---|---|
| L1 | hyperparameter regime (batch 2048 + lr 1e-4 = effective lr 1/8) | **1순위** — Track A |
| L2 | OOF-platform gap 1.9~2.4pt 일관 = cohort/temporal drift hard ceiling | L1 후 결정 |
| L3 | NS xattn entropy 0.81 (~2 tokens / 384 attended) = dominant signal sparse | mechanism 추가 무용 |
| L4 | truncate 64-128 = §3.5 p90 1393~2215 의 95%+ 정보 손실 | Track B (long-seq P2) |

## Sanity gate (verify-claim §1)

| Check | Status |
|---|---|
| config_sha256 ↔ card.yaml | TBD (raw metrics.json paste 후) |
| git_sha ↔ card.yaml | TBD |
| split_meta ↔ card.yaml expected_split | TBD |
| §4.5 메타 (seed/git_sha/config_sha256) | TBD |
| **batch_size mismatch** (user override 2048 vs card.yaml 256) | **검토 필요** — card.yaml `config.batch_size: 256` 이지만 실제 학습 2048. metrics.json 의 batch_size 기록 확인 + card.yaml override 명시. |

## Next actions
- 6 artifact 갱신 완료 (이 turn).
- H013 후보 추천: **hyperparameter calibration** — H010 mechanism + envelope
  byte-identical, lr linear scaling (1e-4 → 8e-4) 또는 batch 256 복귀.
- Track B (long-seq retrieval) 는 H013 결과 따라 결정.
- 사용자 raw metrics.json paste 시:
  - sanity gate 재검증 (특히 batch/lr 실제값).
  - val_AUC + best_step 보강 → P5 채움.
  - config_sha256 INDEX 에 기록.

## §18.8-style SUMMARY (retroactive — log paste 2026-05-03)

> User pasted training log 7 epochs + best/OOF/eval. emit_train_summary()
> 미적용된 H — 수동 reconstruct. Future H 는 §18.8 emit mandatory.

```
==== TRAIN SUMMARY (H012_multi_domain_fusion, RETROACTIVE) ====
git=TBD cfg=TBD seed=42 ckpt_exported=best
epoch | train_loss | val_auc | val_logloss | oof_auc
  1   |    N/A     | 0.8275  |   0.2899    | (final 0.8589)
  2   |   0.1234   | 0.8303  |   0.2868    | —
  3   |   0.1232   | 0.8318  |   0.2850    | —
  4   |   0.1217   | 0.8331  |   0.2865    | —  (best)
  5   |   0.1215   | 0.8312  |   0.2877    | —
  6   |   0.1210   | 0.8327  |   0.2873    | —
  7   |   0.1205   | 0.8317  |   0.2878    | (final 0.8589)
best=epoch4  val=0.8331  oof=0.8589 (legacy)
last=epoch7  val=0.8317  oof=0.8589
overfit=+0.0014 (best_val − last_val)
calib pred=N/A label=N/A ece=N/A
==== END SUMMARY ====
```

## Trajectory analysis (post log paste)

| metric | value | 비고 |
|---|---|---|
| best_epoch | 4 / 7 | early-mid convergence (H011 와 동일 패턴) |
| best_val_AUC | **0.8331** | 신규 |
| last_val_AUC | 0.8317 | epoch 7 |
| overfit_gap | **+0.0014** | 거의 없음 |
| OOF (legacy) | 0.8589 | H011 과 동일 (saturated) |
| platform AUC | **0.8380** (uncorrected eval) | INDEX prior |
| **val ↔ platform gap** | **−0.0049** | 가장 큰 negative gap, val 이 가장 underestimate |
| **OOF ↔ platform gap** | **+0.0209** | saturated OOF 가 platform 보다 큼 |

**Caveat (uncorrected eval)**: platform 0.8380 은 organizer 2026-05-02~03
eval data correction **이전** 측정. corrected anchor 0.837806 와 직접
비교 invalid. −0.003pt heuristic shift 적용 시 corrected ≈ 0.8350. 그래도
H012 corrected estimate (0.8350) < H010 corrected anchor (0.8378) =
Δ −0.0028 → **§17.3 binary REFUTED 유지**. 사용자 직관 "H012 가 가장
높다" 도 **REFUTED** (corrected H015 0.83805 > H012 estimated 0.8350).

**Carry-forward signal**: H012 verdict (REFUTED — Frame B uniform routing)
변동 없음. 단 H012 platform 0.8380 (uncorrected) ≈ H015 platform 0.83805
(corrected) — 서로 다른 eval data 라 직접 비교 invalid 지만 두 mechanism
이 비슷한 ceiling 영역. cohort drift hard ceiling 가설 추가 confirm.

# Experiments Registry — TAAC 2026 UNI-REC

> CLAUDE.md §4.6 + §9 + §17 + §18.

## Schema

| EXP_ID | hypothesis_id | seed | val_AUC | OOF_AUC | platform_AUC | compute_tier | wall (학습) | wall (infer) | status |
|---|---|---|---|---|---|---|---|---|---|
| **original_baseline** | (anchor smoke) | 42 | ~0.83 | TBD | **~0.83X** | T2.4 smoke | ~3 min | 180 sec | **active anchor (smoke)** |
| E_H006 | H006_longer_encoder_d_domain | 42 | ~0.82 | 0.8562 | **0.82** | T2.4 extended (10ep × 30%) | 4시간 10분 | 180 sec | refuted |
| E_H007 | H007_candidate_aware_xattn | 42 | 0.8321 (epoch 3 peak) | TBD | **0.8352** | T2.4 extended (3ep early stop × 30%) | 3시간 7분 | ~3 min | PASS marginal |
| E_H008 | H008_dcn_v2_block_fusion | 42 | TBD | **0.8585** | **0.8387** | T2.4 extended (10ep × 30%) | 3시간 41분 | 220 sec | **PASS — 지금까지 최고** |
| E_H009 | H009_combined_xattn_dcn_v2 | 42 | TBD | **0.8595** | **0.8364** | T2.4 extended (10ep × 30%, patience=3 미발동) | 3시간 36분 | 259 sec | **REFUTED — interference** |
| E_H010 | H010_ns_to_s_xattn | 42 | TBD | **0.8596** | **0.8408** | T2.4 extended (10ep × 30%, patience=3) | 3시간 44분 54초 | 297.02초 | **PASS — additive vs H008, 새 champion** |
| E_H011 | H011_aligned_pair_encoding | 42 | **0.8336** (best ep4/7) | 0.8589 (legacy) | **0.8347** (uncorrected eval) | T2.4 extended (10ep × 30%, patience=3) | 2시간 46분 54초 | 178.31초 | **REFUTED — degraded vs H010** (val_platform_gap −0.0011, overfit +0.0013) |
| E_H012 | H012_multi_domain_fusion | 42 | **0.8331** (best ep4/7) | 0.8589 (legacy) | **0.8380** (uncorrected eval) | T2.4 extended (10ep × 30%, patience=3, batch=2048) | 2시간 49분 43초 | 137.39초 | **REFUTED — Frame B (uniform routing)** (val_platform_gap −0.0049, overfit +0.0014) |
| E_H013 | H013_hyperparameter_calibration | 42 | **0.8316** (best ep6/9) | 0.8573 (legacy) | **0.8344** (uncorrected eval) | T2.4 extended (lr 8e-4, batch 2048, num_workers 4, buffer 8) | 4시간 8분 13초 | 95.57초 | **REFUTED — Frame A (Keskar generalization gap)** (val_platform_gap −0.0028, overfit +0.0016) |
| E_H014 | H014_long_seq_envelope | 42 | TBD | 0.8595 | **0.833587** (corrected eval) | T2.4 extended (10ep × 30%, batch=1024, seq=192 uniform, OOM iter-4) | 3시간 30분 49초 | 229.87초 | **REFUTED — L4 retire, paradigm shift inevitable** |
| **(re-run)** | H010_ns_to_s_xattn (corrected eval) | 42 | TBD | TBD | **0.837806** (new anchor) | (re-inference only) | — | 92.74초 | **new corrected anchor** |
| E_H015 | H015_recency_loss_weighting | 42 | 0.8358 (best ep5/8) | 0.8592 (legacy) | **0.83805** | T2.4 extended (10ep × 30%, batch 2048, lr 1e-4, recency [0.5, 1.5]) | (8 ep completed) | TBD | **REFUTED — Δ vs corrected anchor +0.00024 (§17.3 < +0.5pt cut), marginal positive direction** |
| E_H016 | H016_oof_future_redefine | 42 | 0.8399 (best ep4/7) | 0.8283 (redefined) | **0.83192** | T2.4 extended (oof_split_type future_label_time, train ~85%) | (7 ep completed) | TBD | **REFUTED model lift (Δ −0.0059), PASS infra (OOF redefined gap −0.004)** |
| E_H017 | H017_recency_exp_decay | 42 | TBD | TBD | **submission lost** | T2.4 extended (H015 + recency_weight_form exp) | TBD | TBD | **INVALID — submission lost, no platform AUC** |
| **~~Tier 0 (cancelled)~~** | ~~H011/H012/H013 corrected re-inference~~ | — | — | — | — | — | — | — | **CANCELLED 2026-05-03** — over-engineered. User correctly noted verdict already REFUTED for all 3 H, re-inference doesn't change verdict. **Replaced by retroactive log paste analysis** (see each H's training_result.md §18.8-style SUMMARY). 3 `training_request_remeasure.md` docs DELETED. corrected platform AUC remains uncorrected — −0.003pt heuristic shift estimate documented per H. |
| E_H018 | H018_per_user_recency_weighting | 42 | TBD | TBD | TBD | T2.4 extended (per-user exp tau=14, clip [0.1, 3.0], oof_redefine future_only) | TBD (~3-4h) | TBD | **SCAFFOLDED 2026-05-03** — patch spec (`upload_patch.md`) + card.yaml + 6 hypothesis docs complete. upload/ package NOT YET BUILT (out of Ralph PRD scope). config_sha256=TBD. control_exp_id=H015. updated outcome dist (noise ~50%). |
| E_H019 | H019_twin_long_seq_retrieval | 42 | TBD | TBD | TBD | **T3** ~6h ~$15 (TWIN GSU top-K=64 + ESU MHA, seq cap 512, oof_redefine future_only) | TBD (~6h) | TBD | **SCAFFOLDED 2026-05-03** — 6 hypothesis docs + card.yaml complete. upload/ + upload_patch.md NOT YET BUILT. config_sha256=TBD. control_exp_id=H010 corrected anchor. **CONDITIONAL launch** depends on H018 outcome. cost cap audit critical (per-campaign ≤ $100). |

## Archive (anchor reset 2026-04-28, H001–H005)

이전 측정값들은 인프라 bug 들 영향 → invalid.

| EXP_ID | hypothesis_id | val_AUC (당시) | platform AUC | Status |
|---|---|---|---|---|
| E000 | H001 | 0.5088 (demo) / 0.7055 OOF | n/a | archived |
| E_baseline_organizer | H001 | 0.8251 | unknown | archived |
| E001 | H002 | 0.8248 | unknown | archived (refuted) |
| E002 | H004 | 0.8174 | 0.5 (heuristic fallback) | archived |
| E_focal_smoke | H005 | 0.8253 | unknown | archived (refuted) |

## Cumulative cost (CLAUDE.md §17.6)

- T0/T1: $0
- T2 (Taiji): H006 (4h) + H007 (3h) + H008 (3.7h) + H009 (3.6h) + H010 (3.75h) + H011 (2.78h) + H012 (2.83h) + H013 (4.14h) + H014 (3.51h) + H015 (~3.5h) + H016 (~3.5h) + H017 (~3.5h, submission lost) = **~42시간 누적 학습** + 11+회 inference + H010 corrected re-inference. ~~Tier 0 H011/H012/H013 corrected re-inference~~ **CANCELLED 2026-05-03** (over-engineered per user pushback — verdict already REFUTED for all 3, log paste analysis sufficient). Taiji 가격 미공개 — 사용자 확인 필요.
- **§17.6 cap audit** (2026-05-03): H018 추가 시 ~46h. paradigm shift candidates (TWIN/OneTrans, T3 ~$20+/job) cost cap 위협 — H018 결과 후 재평가 mandatory.
- T3: $0
- **§17.6 budget cap 도달**: 32시간 누적 → cap 한계. H015 진행 시 cap 위협.
- **eval data correction (2026-05-02~03)**: organizer 가 eval data 수정. H010 재측정 = **0.837806** (이전 잘못된 eval 0.8408 −0.003 shift). prior H011/H012/H013 corrected 재측정 안 됨 → **ranking 비교 정밀도 낮음**. corrected ranking 명확히 위해 prior H 들 re-inference 권장.
- **H014 결과: L4 (truncate 정보 손실) REFUTED** (Δ vs H010 corrected −0.0042pt). dense long-seq expansion 가 ceiling 풀지 못함. OOF-Platform gap 2.59pt = 9 H 중 가장 큼 (cohort drift 강한 신호).
- **4-layer ceiling diagnosis 종료**: L1 + L3 + L4 모두 retire. **L2 (cohort drift) 만 남음**. paradigm shift inevitable.

## Conventions

- `EXP_ID = E_HXXX`.
- Platform AUC 측정 의무: §18 룰 통과 (batch heartbeat + `[infer] OK: torch path produced N predictions`).
- §18 인프라 (batch=256 default + PYTORCH_CUDA_ALLOC_CONF + universal handler + 진단 로그) 모든 H 패키지 적용 — H006/H007/H008 검증 완료.

## Anchor pair / 비교 룰

- **paired Δ 는 platform AUC 으로만** (H006 F-3 carry-forward).
- val ↔ platform 정합 확인됨 (H006/H007 패턴).
- OOF AUC 는 supplementary — H008 OOF-platform 갭 2pt (H006 의 3.5pt 보다 좁아짐). H009 에서 2.31pt 역행. **H010 에서 1.88pt 다시 narrow** (통합 위치 회피 설계 효과).
- **H010 PASS additive → anchor 갱신**: H011+ 부터 H010 (Platform 0.8408) 이 새 anchor. H008 (0.8387) carry-forward control 로 보존 (paired Δ 비교 용).
- **H010 = 새 champion** (Δ vs H008 +0.0021pt). H009 의 interference 와 정반대 — 통합 위치 (NS-only enrichment, anchor 입력 byte-identical) 가 안전 stacking 패턴 confirmed.

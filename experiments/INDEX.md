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
- T2 (Taiji): H006 (4h) + H007 (3h) + H008 (3.7h) + H009 (3.6h) + H010 (3.75h) = **~18시간 누적 학습** + 7+회 inference. Taiji 가격 미공개 — 사용자 확인 필요.
- T3: $0
- **§17.6 budget cap 압박**: 18시간 누적 → cap 근접. H011 부터 patience=3 + 가능하면 fp16/batch=512 으로 wall 절반.

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

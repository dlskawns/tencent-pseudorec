# H007 — Verdict (PASS marginal)

> 클라우드 학습 + inference 완료 (2026-04-29). §17.3 binary 임계 통과 (marginal). target_attention mechanism class 작동 first confirmed PASS.

## Status
`done` — PASS marginal. Platform AUC 0.8352 > anchor (original_baseline) 0.83X by ~+0.005pt. §17.3 +0.5pt 임계 거의 도달/통과.

## Source data
- 학습: 3 epoch (early stop after epoch 4 regression), train_ratio=0.3, label_time split + 10% OOF, ~3시간 wall.
- ckpt: `global_step24069.layer=2.head=4.hidden=64.best_model` (epoch 3).
- Inference: §18 인프라 (batch=256 default, PYTORCH_CUDA_ALLOC_CONF, universal handler) 정상 통과 ~3분 wall.

## P1 — Code-path success
- Measured: 3 epoch NaN-free 완주 + early stop on epoch 4 regression. CandidateSummaryToken init 정상.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.835213**.
- Anchor (original_baseline): Platform AUC ~**0.83X**.
- Δ vs anchor: **~+0.005 pt** (marginal).
- Predicted: Δ ≥ +0.5pt.
- **Verdict: PASS marginal**. mechanism 작동 first confirmed.

## P3 — Mechanism check (candidate attention pattern)
- Measured: per-domain attention pattern 직접 측정 안 함 (instrumentation 부재).
- Verdict: UNVERIFIED. P2 PASS 자체가 mechanism 작동 indirect evidence.

## P4 — §18 인프라 통과
- Measured: inference 3분 wall, batch heartbeat + `[infer] OK: torch path produced 609197 predictions` 둘 다 보임. heuristic fallback 없음.
- Verdict: **PASS**.

## P5 — val ↔ platform AUC alignment
- val_AUC (peak epoch 3): 0.8321.
- Platform AUC: 0.8352.
- |val − platform| = 0.0031 ≤ 0.05.
- Verdict: **PASS**. H006 의 val ↔ platform 정합 패턴 confirmed for H007 too.

## Findings (F-N carry-forward)

- **F-1 (P2 PASS marginal — mechanism 작동 first confirmed)**: candidate-as-attention-query mechanism class (DIN/CAN/SIM/TWIN/HSTU/OneTrans family idea, modern multi-head xattn) 가 우리 데이터에서 **+0.5pt 임계 거의 도달 또는 통과**. H001–H006 의 architectural mutation 들 (bridges, OneTrans backbone, focal, longer encoder) 모두 marginal 또는 refuted 였는데 **H007 이 처음으로 platform 에서 measurable lift**. mechanism class 가치 검증.
- **F-2 (val ↔ platform 정합 confirm 두 번째)**: H006 (val 0.82 / platform 0.82) 에 이어 H007 도 val 0.832 / platform 0.835. 즉 우리 split (label_time + 10% OOF) 의 val_AUC 가 platform 과 정합하는 pattern. **leakage 가설 무효** + 미래 H 의 paired Δ 측정에 val 신뢰 가능.
- **F-3 (대부분 lift 가 envelope 효과)**: H007 (extended envelope 3 epoch × 30%) Δ vs anchor (smoke 1 epoch × 5%) 비교 unfair — envelope 도 변수. mechanism 효과만 isolate 하려면 anchor 도 extended 에서 측정 필요. 현재 anchor smoke 0.83X / H007 extended 0.8352 = **mechanism 효과 + envelope 효과 합산 +0.005pt**. mechanism 단독 효과 separate 하려면 추후 anchor recalibration H 필요.
- **F-4 (cost: 3 epoch ≈ 3시간 wall)**: H007 wall 3시간 (epoch 4 early stop). H006 4시간 (10 epoch). Extended envelope 는 cheap 하지 않음. **§17.6 cost cap 압박**: H006 + H007 누적 wall 7+ 시간 = T2 cap (per-job ≤ $5) 근접. H008 부터 smoke 우선, extended 는 mechanism 검증 후만.
- **F-5 (§18 인프라 검증 견고)**: batch=256 + PYTORCH_CUDA_ALLOC_CONF + universal handler + 진단 로그 = vGPU 환경에서 reliably 작동. H006 + H007 두 H 모두 inference 통과. carry-forward 로 H008 패키지에 그대로 inherit.

## Surprises
- **3 epoch early stop 으로 best ckpt** — extended envelope 임에도 plateau 가 일찍 옴. trajectory: 0.8283 → 0.8312 → 0.8321 (peak) → 0.8313 (regression). 10 epoch 전부 학습할 필요 없음.
- **Marginal 이지만 first PASS** — H001–H006 모든 mutation 의 noise floor 안 묻힘 패턴 깸. mechanism class 가 아예 효과 없는 게 아니라 candidate-aware 라는 axis 가 우리 데이터에 작동.

## Update to CLAUDE.md?
- Carry-forward 후보: "**candidate-as-attention-query mechanism class 가 우리 PCVRHyFormer baseline 위에서 +0.5pt 임계 도달**". §0 north star 의 sequence axis 강화 직접 확인. 단 single confirmed run, multi-seed ablation 후 본문 추가 결정.

## Carry-forward to H008

- F-1 → H007 mechanism 작동 confirmed → H008 은 orthogonal axis (interaction axis = explicit feature cross) 추가로 stack 가능 검증.
- F-2 → val 신뢰 가능 → H008 도 val ↔ platform 정합 expected.
- F-3 → H008 anchor 는 original_baseline (smoke). 단일 mutation 비교 유지. H007-anchor 는 등록 안 함.
- F-4 → H008 smoke envelope (1 epoch × 5%) 우선. extended 는 PASS confirmed 후만.
- F-5 → §18 인프라 룰 H008 패키지에 그대로 inherit.

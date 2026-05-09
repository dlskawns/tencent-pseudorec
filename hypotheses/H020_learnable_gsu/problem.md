# H020 — Problem (Learnable GSU, TWIN sub-H of H019)

## Background — what we already learned

- H019 (TWIN long-seq retrieval, paradigm shift first entry) BUILT 2026-05-05, cloud-ready.
- Local sweep around H019 (top_k 32/64/128, seq 256→384 single/uniform, gate -2→-1, batch 1024→512): **모든 axis saturated**. top_k 64 ≈ 128 (동급, OOF 0.8612 vs 0.8611), top_k 32 −0.0037pt, seq expansion (uniform/single-domain) 모두 H019 미만 (H014 dense expansion 패턴 재현), gate -1 ↓, batch 512 ↓.
- F-G ceiling (val_auc 0.832~0.836, 12 H 누적) — H019 가 best_val 0.8372 로 ceiling 직전 ~ 약간 위. mechanism 진위 확인 단계 종료.

## Why Learnable GSU (H019 mechanism 내부 1단계 깊이)

H019 의 GSU = **parameter-free inner product**:
```
score(history_token, candidate) = history_emb · candidate_emb
```
즉 candidate-relevance scoring 자체가 학습 안 됨. 이는:
1. backbone 이 학습한 embedding space 가 retrieval 에도 잘 맞는다는 가정.
2. candidate 와 history 가 *같은 metric space* 에서 비교 가능하다는 가정.
TWIN paper 의 GSU 는 learnable scorer (별도 projection) — H019 는 §10.6 sample budget 친화 위해 단순화.

**H020 mutation**:
```
score = (W_q · candidate_emb) · (W_k · history_emb)
W_q, W_k: nn.Linear(d_model, d_model // 4)
```
즉 **scoring projection 1쌍 추가** — "어떤 history token 이 candidate 와 관련 있는가" 자체를 학습.

## Core observation

- H019 sweep saturation 신호 = **K (pool size) 와 capacity (seq cap) 는 더 이상 lever 아님**. retrieval 의 진짜 bottleneck 은 *어떤 token 을 가져오는가* (selection policy) 일 가능성.
- learnable scorer 는 backbone embedding space 와 retrieval space 분리 → backbone 이 "다음 item 예측" 에 optimize 된 representation 을 retrieval-task-specific 으로 재투사.
- params 추가 ~64K (4 도메인 × 16K = W_q + W_k per domain) — TWIN 자체 71K 가 ~135K 로 늘지만 total 161M 의 0.084%, sample budget 영향 적음 (cloud full-data 측정 의존).

## Why now, not before

- H019 직후 "TWIN 진위 확인" 단계 = 주변 sweep 으로 saturation 확인 완료.
- 현재 = TWIN mechanism class 안 1단계 깊이로 들어가는 cheap-and-meaningful zone.
- 다음 paradigm shift (HSTU trunk / OneTrans full / cohort attack) 는 cost cap 위협 + 측정 비용 큼. learnable GSU 는 +64K params, 추가 복잡도 작고 결과 해석 깔끔.

## Constraint-aware framing

- **§17.2 single mutation**: GSU scoring function 의 단일 변경 (parameter-free → learnable projection). top_k / seq_max_lens / gate / batch / num_heads / aggregation 전부 H019 byte-identical.
- **§17.4 rotation**: `retrieval_long_seq` re-entry. 사유 = H019 mechanism class 안 sub-H (paper-faithful form 직접 검증). H019 PASS 이전 sub-H 진입은 §10.7 strict reading 위반 가능 — 그러나 H019 result 회수 전 이미 sweep 으로 mechanism 진위 partial confirm + 사용자 paradigm shift 채택 → re-entry 정당.
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7 (H019 동급). cumulative cost cap 친화.

## Falsifiable predictions

- **PASS (strong)**: Δ vs H019 ≥ +0.003pt → learnable scoring 이 retrieval policy lift, sub-H = H021 per-domain top_k stack.
- **PASS (measurable)**: Δ ∈ [+0.001, +0.003pt] → 약 effect, A2 (per-domain K) 와 stack 가능성 검증.
- **REFUTED (noise)**: Δ ∈ (−0.001, +0.001pt] → scoring 학습이 inner product 와 다르지 않음 → retrieval bottleneck 이 selection 이 아니라 ESU (token 처리 capacity) 일 가능성. H022′ = ESU 2-layer (A3) 또는 H019 + cohort attack 으로 pivot.
- **REFUTED (degraded)**: Δ < −0.001pt → projection 이 backbone embedding 정보 손실 (rank reduction d_model → d_model//4) 또는 학습 instability. sub-H = projection dim 증가 (d_model//2) 또는 retire.

H019 anchor 로 비교하는 이유 = paired Δ 정합성 (H010 corrected anchor 위 H019 가 이미 sub-H 깊이 들어감, control = H019 가 mechanism 영향 isolation 깔끔).

## Decision tree (post-result)

| Outcome | Δ vs H019 | Action |
|---|---|---|
| strong | ≥ +0.003pt | anchor = H020. H021 = per-domain top_k stack on H020. retrieval scoring class 영구 confirm. |
| measurable | [+0.001, +0.003pt] | H021 단독 (per-domain top_k on H019 base) → H020 vs H021 paired 비교 후 stack 결정. |
| noise | (−0.001, +0.001pt] | scoring 자체 무 effect → H022′ = ESU 2-layer (A3) 또는 cohort drift attack (C). retrieval mechanism class 의 selection axis retire. |
| degraded | < −0.001pt | projection 구현 issue. sub-H = d_model//2 또는 W_k only (W_q identity) 또는 retire. |

## Out of scope

- Per-domain top_k — H021 (next H, separate single mutation).
- ESU 2-layer / pre-norm — H022′ 후순위 (A3, scoring 개선 결과 후).
- cohort drift attack — H020/H021 모두 fail 시 결정.
- top_k > 128 또는 < 32 추가 sweep — saturation 확인 완료, 가치 없음.
- seq_max_lens 변경 — H014 + H019 sweep 모두 retire.

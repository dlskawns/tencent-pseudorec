# H034 — Problem (ESU 2-layer, TWIN sub-H of H019, capacity axis)

## Background — what we already learned

- H019 cloud measurable PASS (+0.001868pt). retrieval mechanism class lever confirmed.
- H020 (scoring quality) BUILT, H021 (scoring quantity) BUILT, H033 (stacking) BUILT — 모두 *scoring* 또는 그 stacking. **token 처리 capacity 는 미검증**.
- H019 의 ESU = 1-layer MultiheadAttention. paper TWIN 의 ESU 는 multi-layer 가능 (paper 명시 안 함, 보통 transformer block-style).

## Why ESU 2-layer (capacity axis sub-H)

H020/H021 모두 noise 일 때 자연스러운 다음 H = "가져온 token 의 *처리* 가 lever 인가?":
- H020 = 어떤 token 을 가져오는가 (scoring quality).
- H021 = 얼마나 많은 token 을 가져오는가 (scoring quantity).
- H034 = 가져온 token 을 얼마나 깊게 처리하는가 (ESU capacity).

3 axis 가 직교 → H020/H021 NOOP 시 H034 만 PASS 가능.

## Mutation

`TWINBlock.esu` = 1-layer → 2-layer MHA stack:
- num_layers=1 (H019): `out = norm(candidate_q + esu(candidate_q, topk_history, topk_history))`
- num_layers=2 (H034): 각 layer 후 `x = ln(x + layer(x, topk_history, topk_history))` 누적.

per-domain trainable params 추가: ESU MHA 1-layer ≈ 16K params × 4 도메인 = +64K.

## Core observation

ESU 의 1-layer 가 충분한지 미검증:
- candidate Q (1 token) × topk_history KV (64 tokens) = 단순 attention pooling.
- 2-layer = candidate-history interaction 의 비선형 변환 (첫 layer 가 weighted sum, 두 번째 layer 가 그 결과 위 추가 attention).
- 만약 1-layer 가 sufficient → noise (1-layer ESU 가 retrieval signal 의 effective limit).
- 만약 2-layer 가 lift → capacity bottleneck confirmed.

## Why now, not before

H020/H021 결과 회수 후 build 시 round-trip 1회 추가. 사용자 = pre-build insurance.

## Constraint-aware framing

- §17.2 single mutation: ESU layer 수의 단일 변경. GSU / top_k / aggregator / gate 전부 H019 byte-identical.
- §17.4 rotation: retrieval_long_seq 4회 연속. RE_ENTRY_JUSTIFIED (H020/H021/H033 carry + capacity axis 추가 사유).
- §17.6 cost cap: T2.4 ~3.5h × $5-7. H019 동급.

## Falsifiable predictions

| Outcome | Δ vs H019 | 다음 액션 |
|---|---|---|
| strong | ≥ +0.003pt | capacity bottleneck confirmed, anchor = H034, sub-H = num_layers=3 |
| measurable | [+0.001, +0.003pt] | 약 effect, sub-H 후보 |
| noise | (-0.001, +0.001pt] | 1-layer ESU sufficient, retrieval capacity axis dead → cohort/HSTU pivot |
| degraded | < -0.001pt | over-capacity (학습 instability), retire |

## Out of scope

- ESU num_layers ≥ 3 (sub-H 후순위).
- Cohort attack (H020/H021/H033/H034 모두 noise 시).
- Backbone replacement (HSTU/OneTrans full).

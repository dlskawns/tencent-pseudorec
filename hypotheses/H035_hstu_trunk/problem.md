# H035 — Problem (HSTU trunk replacement, paradigm shift)

## Background — what we already learned

- H019 cloud measurable PASS (+0.001868pt). retrieval mechanism class lever confirmed.
- **H019 data_ratio=1.0 → eval AUC 0.837785** (vs ratio=0.3 의 0.839674). 더 많은 데이터가 도움 안 됨 → **F-G ceiling 의 강한 confirm**, capacity bottleneck 아님 → cohort drift / mechanism class 자체가 진짜 limit.
- H020/H021/H033/H034 (in-flight 또는 pre-built) 모두 retrieval class 안 internal axis. 0.85 까지 +0.012pt 갭은 internal axis 합으로 못 감.
- §17.4 rotation: retrieval_long_seq 4회 연속 → backbone_replacement 강제 transition.

## Why HSTU trunk replacement (paradigm shift class)

**HSTU** (Meta 2024 ICML, "Actions Speak Louder than Words", arXiv:2402.17152):
- 기존 transformer self-attention (softmax) → pointwise silu-attention.
- Gated linear unit (U projection 으로 V 를 multiplicative gate).
- Trillion-parameter scale 검증, 추천 task 에 paper 가 SOTA 보고.

H019 baseline 의 per-domain seq encoder = standard `TransformerEncoder` (softmax MHA + GELU FFN). H035 = HSTU 로 swap → **mechanism class 자체 변경**:
- softmax-attention 의 sparse routing 가정 ↔ silu-attention 의 dense pointwise.
- FFN-style nonlinearity ↔ gate-based nonlinearity.

→ F-G ceiling 이 mechanism class 한계라면 HSTU 가 lever 가능.
→ ceiling 이 cohort drift 라면 HSTU 도 못 풂 (그럼 cohort 직접 attack 필수).

## Mutation

`--seq_encoder_type hstu` 단일 flag. 4 도메인 × 2 hyformer block = 8 encoder 가 HSTU 로 swap.

다른 모든 부분 H019 byte-identical:
- TWIN GSU+ESU per-domain retrieval (gate=-2.0, top_k=64)
- NS xattn + DCN-V2 fusion (interaction axis)
- seq_max_lens 256/256/256/256, batch=1024, lr=1e-4

## Core observation — HSTU 의 차이

| 측면 | TransformerEncoder (H019) | HSTUEncoder (H035) |
|---|---|---|
| Attention | softmax(QK^T/√d) | silu(QK^T/√d) / L |
| Routing | sparse (top tokens dominant) | dense (all tokens contribute pointwise) |
| Gating | FFN nonlinearity (GELU) | U projection (silu-gated multiplicative) |
| Positional | RoPE | implicit (silu attention 자체) |
| Params per layer (D=64, h=4) | ~48K | ~21K (HSTU 가 절반) |

**HSTU 가 더 작음** → sample-scale 친화. capacity 가 진짜 lever 라면 오히려 H035 가 underperform 할 수도. 그러나 우리는 **capacity 가 lever 아님을 data_ratio=1 결과로 검증** → mechanism class 변경이 진짜 lever.

## Why now, not before

- H019 cloud measurable PASS + data_ratio=1 ceiling confirm = retrieval class 위로 가려면 mechanism class 전환 필수.
- H020/H021 in-flight + H033/H034 pre-built = retrieval class 안 internal axis 다 hedge 됨. paradigm shift class 의 다른 후보 (HSTU / cohort) 가 다음 자연스러운 단계.
- HSTU = paper-faithful trunk replacement, paradigm shift class 의 strongest hedge (data 추가/내부 axis 와 직교 mechanism).

## Constraint-aware framing

- **§17.2 single mutation**: per-domain seq encoder 의 type 단일 변경. 다른 모든 부분 byte-identical.
- **§17.4 rotation**: backbone_replacement 카테고리 NEW first-touch (retrieval_long_seq 4회 연속 후의 강제 rotation).
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. 단 HSTU 가 unstable 하면 epoch 수 늘 수 있음 → cost cap 위협 시 사용자 confirm.
- **§10.6 sample budget**: HSTU 가 transformer 보다 params 더 작음. budget 영향 적음.
- **§10.9 attn entropy**: HSTU 의 silu-attention 은 softmax 와 다른 metric — entropy threshold 의 의미 재정의 필요 (분포가 실제로 prob distribution 아님). 측정만 carry-forward, abort threshold 적용 안 함 (§10.9 룰 명시 예외).

## Falsifiable predictions

| Outcome | Δ vs H019 | 다음 액션 |
|---|---|---|
| **strong** | ≥ +0.005pt | mechanism class 전환이 진짜 lever, anchor = H035, sub-H = HSTU + per-domain top_k stack |
| **measurable** | [+0.001, +0.005pt] | 약 effect, sub-H 후보 |
| **noise** | (-0.001, +0.001pt] | mechanism class 도 못 풂 → cohort drift 가 진짜 last hope, H036_cohort_embed 강제 |
| **degraded** | < -0.001pt | HSTU 가 본 데이터에 안 맞음, retire (paper claim 본 환경 외 일반화 안 됨) |

§17.3 binary cut: paradigm shift first-class entry 라 H019 의 +0.005pt 임계 (sub-H 깊이 들어가는 변경의 +0.003pt 보다 높음).

## Decision tree (post-result)

```
H035 strong → anchor = H035, sub-H = HSTU + retrieval stack + cohort variant
H035 measurable → sub-H 시도 (HSTU 의 hidden_mult 확장 또는 multi-layer)
H035 noise → mechanism class lever 도 dead → cohort drift (L2) 강제 attack
            → H036_cohort_embed 우선
H035 degraded → HSTU 본 데이터 부적합 → retire, OneTrans full backbone 또는 cohort 으로
```

## Out of scope

- HSTU + relative attention bias (paper full form). 본 H = minimum viable (silu + gate).
- HSTU + causal masking (recommendation autoregressive form). 본 데이터 binary classification 에는 비대상.
- Full OneTrans-style trunk replacement (NS xattn + DCN-V2 도 같이 변경). H035 PASS 시 sub-H 후보.
- Cohort drift attack (L2). H035 noise 시 다음 H.

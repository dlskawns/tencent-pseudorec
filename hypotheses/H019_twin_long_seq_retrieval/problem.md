# H019 — Problem (TWIN long-seq retrieval, paradigm shift)

## Background — what we already learned

- 4-layer ceiling diagnosis 종료 (H014 verdict, 2026-05-03): L1 + L3 + L4 all retire, **L2 (cohort drift) only remaining**.
- L2 attack via temporal_cohort 4 H attempts (H015 marginal +0.0002, H016 model REFUTED / infra PASS, H017 INVALID, H018 SCAFFOLDED): **mechanism class 가 ceiling 위 lift 못 만듦** strong indication.
- Recent Findings F-D: H012 corrected estimate ≈ H015 corrected = ceiling tied. 다른 mechanism class (MoE / recency) 가 같은 영역에서 멈춤.
- §17.4 rotation: H015~H018 모두 temporal_cohort. H019 mandatory rotation. 미경험 카테고리: `retrieval_long_seq`, `backbone_replacement`, `debiasing`.

## Why TWIN long-seq retrieval (paradigm shift class 1순위)

- §3.5 강한 정량 motivation: 도메인 a/b/c/d 의 p90 seq length = 1393 / 1393 / 887 / 2215. 현 envelope (truncate 64-128) = **p90 의 95%+ 정보 손실**.
- L4 (truncate 정보 손실) 의 dense form 은 H014 가 retire (uniform 192 → −0.0042pt). 단 **retrieval form** (top-K 동적 선택) 은 미시도. L4 의 retrieval form = open lever.
- **TWIN** (Tencent, 2024 RecSys, "TWo-stage Interest Network for Long-term User Behavior Modeling"): GSU (general search unit) → ESU (exact search unit) 로 long-seq 에서 candidate-relevant top-K 만 attend. dense computation 회피하면서 long-seq 정보 활용.
- **SIM** (Alibaba, 2020, "Search-based User Interest Modeling"): TWIN 의 prior — soft / hard search → top-K extract → attention.
- **HSTU** (Meta, 2024, "Actions Speak Louder than Words" — generative recommender): trunk replacement 형태, sequence axis 강화.
- 본 H 의 1순위 = **TWIN GSU+ESU minimum viable form** (다른 두 paper 는 H020+ backlog).

## Core observation

cohort drift hard ceiling 가설이 (a) L2 직접 attack 의 4 H 누적 marginal/REFUTED 결과 + (b) F-D (다른 mechanism class 도 같은 영역 ceiling) → cohort drift 자체가 진짜 ceiling 이 아닐 가능성. 진짜 source 는 **truncate 정보 손실 + dense form 한계의 조합**:
- truncate (현 64-128) 가 user 의 historical interest 의 95%+ 손실.
- dense expansion (H014 192 uniform) 은 모든 token 동등 처리 → noise 증폭.
- **retrieval (top-K candidate-relevant 동적)** 가 신호/노이즈 비 회복.

**Falsifiable claim**: TWIN GSU+ESU (top-K=64 from per-domain seq L=512 cap) 가 corrected H010 anchor (0.837806) 위 Δ ≥ +0.005pt 추가 lift 만든다. 만약 거짓 → cohort drift 가 진짜 hard ceiling, retrieval 도 못 풂 → H020 = backbone replacement (HSTU trunk) 또는 multi-modal 데이터 도착 (P3 phase) 까지 plateau 인정.

## Why now, not before

- 4-layer 종료 + temporal_cohort 4 attempts 누적 → paradigm shift 정당화 충분.
- 본 PRD F-F: cost cap audit 기반 (T2 누적 ~46h after H018, T3 추가 ~$15 estimate). T3 위협 인지 + accept (paradigm shift 비용).
- §3.5 frac_empty (도메인 d 8% empty user) + p90 ≫ 100 quantitative motivation 강함.

## Constraint-aware framing

- **§17.6 cost cap critical**: T3 (Lambda/RunPod ~$15/job) per-job cap 위협. 본 H REFUTED 시 paradigm shift 다른 form (HSTU/OneTrans) 추가 비용 어려움 → **H019 가 retrieval form 의 minimum viable test**.
- **§17.2 single mutation**: TWIN GSU+ESU 모듈 추가가 1 mechanism class 변경. NS xattn / DCN-V2 stack byte-identical 유지.
- **§17.4 rotation justified**: 미경험 카테고리 (retrieval_long_seq) first-touch.

## Falsifiable predictions

- **PASS (strong)**: Δ vs H010 corrected ≥ +0.005pt → retrieval form 검증, L4 retrieval branch open. anchor = H019.
- **PASS (measurable)**: Δ ∈ [+0.001, +0.005pt] → retrieval 약 effect, sub-H 후보 (top-K sweep, GSU/ESU variant).
- **REFUTED (noise)**: Δ ∈ [−0.001, +0.001pt] → retrieval 도 cohort hard ceiling 못 풂. backbone replacement 또는 P3 phase 대기.
- **REFUTED (degraded)**: Δ < −0.001pt → TWIN 구현 detail 문제 (top-K too small / GSU mismatch). sub-H = top-K sweep 또는 ESU detail 변경.

## Decision tree (post-result)

| Outcome | Δ vs H010 corrected | Action |
|---|---|---|
| strong | ≥ +0.005pt | anchor = H019, sub-H = TWIN top-K sweep / GSU variant. retrieval mechanism class 영구 confirm. |
| measurable | [+0.001, +0.005pt] | retrieval 약 effect. H020 = sub-H sweep 또는 SIM (TWIN 의 prior) re-attempt. |
| noise | [−0.001, +0.001pt] | retrieval 무 effect → cohort drift 가 진짜 hard ceiling 강한 confirm. H020 = backbone replacement (HSTU trunk) — cost cap 적용 후. |
| degraded | < −0.001pt | TWIN 구현 issue. sub-H = top-K=128 (larger) 또는 ESU multi-head 추가. |

## Out of scope

- HSTU trunk replacement — H020 후보 (REFUTED 시).
- OneTrans full single-stream — H020 후보.
- SIM original form — TWIN 이 SIM 의 superset 이라 TWIN 우선.
- Multi-modal (image/text) data — P3 phase, organizer 미공개.
- top-K sweep (32, 64, 128, 256) — sub-H, H019 PASS 시.

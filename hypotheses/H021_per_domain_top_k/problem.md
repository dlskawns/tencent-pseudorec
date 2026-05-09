# H021 — Problem (Per-domain top_k, TWIN sub-H of H019, quantity axis)

## Background — what we already learned

- **H019 cloud measurable PASS**: platform AUC 0.839674 vs H010 corrected 0.837806 = Δ **+0.00187pt**. retrieval mechanism class 가 ceiling 깰 수 있는 first family confirmed.
- H019 직후 local sweep: top_k 32 (val 0.8335, **−0.0037pt**), top_k 64 (H019 baseline 0.8372), top_k 128 (OOF 0.8612 ≈ H019 0.8611, **flat**). **uniform K=64 sweet spot, K↓ 위험, K↑ 무해**.
- H020 SCAFFOLDED 2026-05-06 (TWIN sub-H scoring axis, learnable GSU projection). H021 = 동시 scaffold 의 quantity axis sub-H — H020 과 직교 axis.

## Why per-domain top_k (quantity axis sub-H)

H019 의 top_k=64 는 4 도메인 모두 동일. 그러나 §3.5 도메인별 p90 seq length 분포는 매우 비대칭:

| 도메인 | p50 | p90 | max | K=64 의 비율 (vs p90) |
|---|---|---|---|---|
| a | 577.5 | 1562.1 | 1888 | 4.1% |
| b | 405.0 | 1393.0 | 1952 | 4.6% |
| c | 322.0 | 887.3 | 3894 | **7.2%** |
| d | 1035.5 | 2215.3 | 3951 | **2.9%** |

→ **domain d 는 K=64 가 top 2.9% 만 retrieve** (under-collecting 가능성). domain c 는 K=64 가 top 7.2% 이미 충분.

uniform K 의 saturation 신호 = "평균 도메인 에서 K=64 가 충분, K=128 추가 token 은 noise". 그러나 이는 *평균* 신호 — domain-specific mismatch 가능성 미검증.

**H021 mutation**: per-domain top_k = `{a:64, b:64, c:64, d:96}`. 단일 도메인 (d) 만 K 확장 — known-flat zone (K=128) 까지 50% 거리. K=64 floor 유지 (K=32 known-bad zone 회피).

## Core observation

- uniform K sweep (32/64/128) 의 신호 = "K↓ 위험, K↑ 무해, sweet spot 64". 이는 *모든 도메인 동시 변경* 의 평균 효과 — **domain d 만 고립 확장 효과 미검증**.
- §3.5 quantitative motivation 강함: domain d p90=2215 은 K=64 가 top 2.9% 만 cover. paper TWIN K/L = 1.3% 보다 높지만, 4 도메인 중 가장 under-served.
- 보수적 변경 (한 도메인만 50% 확장) → §17.2 single-mutation 룰 깔끔. paired Δ 해석 명확.

## Why now, not before

- H019 cloud measurable PASS = retrieval mechanism class lever 가 진짜임 confirm. 다음은 mechanism 안 다른 axis (scoring + quantity) 동시 검증.
- H020 (scoring axis) 와 H021 (quantity axis) 가 직교 → 동시 scaffold + 결과 보고 stack/replace 결정.
- 학습 시간 길어서 pre-build 가치 큼 (사용자 명시 요청).

## Constraint-aware framing

- **§17.2 single mutation**: top_k policy 의 uniform → domain-aware. TWINBlock class **byte-identical** (이미 top_k 를 init param 으로 받음). PCVRHyFormer wiring 만 변경 (`twin_top_k` int → per-domain dict).
- **§17.4 rotation**: `retrieval_long_seq` re-entry. H019 same category 3 회 연속 — H020 의 RE_ENTRY_JUSTIFIED 사유 carry-forward (sweep saturation, paper-faithful 검증, cost-effective, falsification value).
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. H019/H020 동급, campaign cap 친화.

## Falsifiable predictions

- **PASS (strong)**: Δ vs H019 ≥ +0.003pt → per-domain K 가 진짜 lever (특히 domain d under-collecting hypothesis confirmed). H022 = 정량안 (a:96 d:128 stack) sub-H.
- **PASS (measurable)**: Δ ∈ [+0.001, +0.003pt] → 약 effect, H020 결과와 paired stack 가능성.
- **REFUTED (noise)**: Δ ∈ (−0.001, +0.001pt] → domain d 도 K=64 가 충분 (Frame B confirmed). retrieval quantity axis dead, ESU capacity (A3) 또는 cohort attack pivot.
- **REFUTED (degraded)**: Δ < −0.001pt → K=96 가 noise injection (Frame C). sub-H = K=80 (더 보수) 또는 retire.

## Decision tree (post-result)

| Outcome | Δ vs H019 | Action |
|---|---|---|
| strong | ≥ +0.003pt | H020 + H021 paired 비교 후 anchor 결정. H022 = 정량안 (a:96 d:128) sub-H. retrieval quantity axis 영구 confirm. |
| measurable | [+0.001, +0.003pt] | H020 결과와 paired 비교. 둘 다 measurable+ 면 H022 = stack (H020 + H021). 한 쪽만 measurable+ 면 그쪽 anchor. |
| noise | (−0.001, +0.001pt] | quantity axis dead. retrieval class 의 quantity axis 도 saturation. ESU 또는 cohort pivot. |
| degraded | < −0.001pt | K=96 too aggressive. sub-H = K=80 (gentle) 또는 K=128 (paper-faithful) 한 번 더 또는 retire. |

## Out of scope

- 정량안 (a:96 d:128) — H022 후순위 (PASS 이후).
- domain c 의 K↓ 변경 — sweep K=32 known-bad. 회피.
- top_k > 128 — sweep saturation 확인, 무용.
- ESU mechanism 변경 — A3, H020/H021 모두 fail 시.
- cohort drift attack — A1/A2 retrieval axis 모두 saturation 후.

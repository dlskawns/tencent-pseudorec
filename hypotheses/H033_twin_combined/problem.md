# H033 — Problem (TWIN combined: H020 ∘ H021 stacking sub-H)

## Background — what we already learned

- H019 cloud measurable PASS (Δ vs H010 corrected +0.001868pt). retrieval mechanism class lever confirmed.
- H020 (scoring quality axis, learnable GSU) BUILT 2026-05-06, cloud submit ready.
- H021 (scoring quantity axis, per-domain top_k seq_d:96) BUILT 2026-05-06, cloud submit ready.
- 두 sub-H 의 paired 비교 framework — predictions.md 에 4 outcome 별 next-H 결정 명시.

## Why H033 (stacking sub-H)

H020 + H021 모두 PASS 시 자연스러운 다음 H = 두 axis 동시 mutation 의 stacking effect 측정. 사전 build = insurance:
- 둘 다 PASS 확인 후 즉시 cloud submit (round-trip 절약).
- 한 쪽만 PASS 시: stacking 결과는 PASS axis + NOOP axis 의 합산 효과 측정으로 해석.
- 둘 다 NOOP 시: H033 도 NOOP — retrieval class 전체 saturation 4 H 누적 evidence.

Mutation:
- `--twin_learnable_gsu` (H020) + `--twin_top_k_per_domain "64,64,64,96"` (H021) 동시.
- 다른 모든 부분 H019 byte-identical.
- 추가 params = H020 (8K) + H021 (0) = +8K total.

## Core observation

직교 axis stacking 의 가능한 결과:
1. **Super-additive**: Δ > H020 Δ + H021 Δ → 두 axis 시너지 (예: learnable scoring 이 더 긴 K 의 추가 token 들에서 더 잘 작동).
2. **Additive**: Δ ≈ H020 Δ + H021 Δ → axis 독립.
3. **Sub-additive**: Δ < max(H020, H021) → axis 간 간섭 (H009 패턴 — 위치 충돌).
4. **NOOP**: Δ ≈ 0 → 둘 다 saturated 인 강한 evidence.

## Why now, not before

H020/H021 둘 다 결과 회수 후 build 시 round-trip 1회 추가 비용. 사용자 요청 = 학습 시간 길어서 pre-build insurance.

## Constraint-aware framing

- **§17.2 strict reading 위반**: 2 axis 동시 mutation. 정당화 = challengers.md 에 "H020+H021 모두 PASS 가정 stacking H" 명시. *발견* (single-mutation) → *통합* (stacking) 단계 구분.
- **§17.4 rotation**: retrieval_long_seq 4회 연속 (H019/H020/H021/H033). RE_ENTRY_JUSTIFIED — H020/H021 의 4 사유 carry + stacking 정보 가치.
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. H019 동급.

## Falsifiable predictions

| Outcome | Δ vs H019 | 다음 액션 |
|---|---|---|
| super-additive | ≥ max(H020, H021) + 0.003pt | anchor = H033, 두 axis 시너지 confirm |
| additive | ≈ H020 Δ + H021 Δ | axis 독립, 향후 stacking 일반화 가치 |
| interference | < max(H020, H021) | H009 패턴 — axis 위치 충돌, sub-H = sequential apply |
| noise | (-0.001, +0.001pt] | 둘 다 NOOP 강한 confirm, retrieval class saturation, ESU(H034) 또는 cohort pivot |

## Out of scope

- ESU capacity axis (H034 별도 sub-H).
- Cohort drift attack (H020/H021/H033/H034 모두 NOOP 시 후순위).
- Backbone replacement (HSTU/OneTrans full).

## Conditional cloud submit (timing)

H020/H021 cloud actual 결과 회수 후:
- 둘 다 PASS measurable+ → H033 cloud submit (즉시).
- 한 쪽만 PASS → H033 결과 해석 ambiguous, 우선순위 낮춤.
- 둘 다 noise → H033 cloud submit 불필요 (NOOP 4 H 누적이면 이미 결정), H034 또는 cohort pivot.

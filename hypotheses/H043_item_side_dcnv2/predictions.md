# H043 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.003pt platform): 15% — item-side가 진짜 lever, user-side DCN-V2
  와 비슷한 magnitude.
- Measurable (+0.001~+0.003pt): 30% — additive 약 effect.
- Noise (-0.001~+0.001pt]: 35% — modest mechanism 으론 부족, 더 강한 form
  필요.
- Degraded (<-0.001pt): 20% — item-side cross 가 noise / overfit.

## Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | item-side axis main lever 강한 confirm. sub-H = layers↑ (4) / rank↑ (8) sweep + item_dense_tok 도 cross 추가. |
| [+0.001, +0.003pt] | additive 작동. sub-H 가치 — item-side axis 깊이 1회 더. |
| (-0.001, +0.001pt] | modest mechanism noise — 다음 form (item-side cross-domain attention 또는 GNN-style item co-occurrence) 또는 axis pivot. |
| < -0.001pt | item-side cross 본 데이터 안 맞음 (overfit 또는 capacity 분배 잘못), retire. |

추가 비교:
- H042 (output-dist) 와의 paired 비교 — 두 axis 의 platform 결과로 axis priority 결정.
- 둘 다 fail (각 −0.001pt 이하) → next-paradigm (SSL pretraining 또는 generative) 강제 진입.

## Risk
- DCN-V2 form 이 user-side 에선 작동했지만 item-side 토큰은 num_item_ns=2 만
  (vs user num_user_ns=5+1+1=7) — 더 적은 토큰 → cross interaction 이 학습할
  signal 적음. mechanism magnitude 작을 가능성.
- item_int feature 의 13 scalar 가 NS-token 화 거치면서 entanglement 손실.
  cross 효과 mild 가능.
- §0.5 step 2 (diagnose root cause) — "item-side mechanism 부재 가 진짜 ceiling
  원인인지" 직접 검증 안 됨. 만약 ceiling 의 진짜 원인이 cohort drift (F-A
  패턴) 면 H043 도 OOF→Platform transfer 실패 (H038 처럼).

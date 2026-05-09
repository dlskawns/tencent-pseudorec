# H045 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.003pt platform): 15% — cross-domain axis lever, jaccard signal
  진짜 attack 가능.
- Measurable (+0.001~+0.003pt): 30% — additive 약 effect.
- Noise (-0.001~+0.001pt]: 35% — mean-pool form 의 cross-domain 정보 압축
  부족, 또는 도메인 간 jaccard 0.7~10% 가 너무 작아 transfer learning signal
  미약.
- Degraded (<-0.001pt): 20% — bridge 가 noise/gate 학습 안 됨, capacity
  분배 잘못.

## Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | cross-domain axis main lever, sub-H = num_heads↑ + stacked bridge layers |
| [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 — per-token cross-attn 또는 cross-domain GSU |
| (-0.001, +0.001pt] | mean-pool form 부족, 더 강한 form (도메인 토큰 expand 또는 per-position cross-attn) 또는 retire |
| < -0.001pt | bridge noise/destabilize, retire |

추가 paired 비교: H042 (output-dist) / H043 (item-side) / H044 (cohort-attack)
의 4 axis 결과 종합으로 ceiling-breaking axis priority 결정.

## Risk
- mean-pool 이 sequence-level information 의 information loss form. cross-
  domain bridge 가 이 압축된 token 위 attention 만 → original sequence 의
  fine-grained signal 안 전달. mechanism magnitude 작을 가능성.
- gate init 0.12 → 학습 초기 거의 OFF, gate 가 학습 안 되면 H019 byte-identical.
- §3.5 jaccard 0.7~10% 가 너무 작아 cross-domain transferable signal 미약
  가능성. 진짜 disjoint 하면 bridge 무용.
- 4 도메인 토큰만으론 cross-attn 학습 어려움 (T=4 매우 짧음). num_heads=4
  로 4 토큰 사이 attention → effective capacity 적음.

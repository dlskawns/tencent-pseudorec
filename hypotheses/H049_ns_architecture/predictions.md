# H049 — predictions.md

## Outcome distribution
- Strong (≥+0.003pt): 12% — NS architecture 가 진짜 capacity bottleneck 면 hit.
- Measurable (+0.001~+0.003pt): 30% — partial.
- Noise (-0.001~+0.001pt]: 40% — backbone 이 이미 충분 학습.
- Degraded (<-0.001pt): 18% — item_ns 6 expansion 으로 capacity 분배 잘못.

## Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | NS axis lever, sub-H = item_ns 더 expand (10) or per-domain NS |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | structure 만 부족, 다른 form |
| < -0.001pt | retire |

## Risk
- d_model%T 제약: T = 2×4 + 13 = 21, d_model=64 % 21 ≠ 0 → 'full' rank_mixer
  mode 면 raise. 'half' mode 사용 가정 (H019 default).
- item_ns_tokens 6 으로 RankMixer split 이 작아짐 (slot 당 capacity ↓) — 
  too thin 가능성.
- type embedding 이 학습 안 되면 H019 등가 (gate 없는 단순 add 라 학습 보장 약함).

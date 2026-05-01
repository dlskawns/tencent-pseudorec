# HXXX — Predictions

> 반증 가능해야 함. "X가 Y보다 좋을 것이다" 만으로는 부족, 효과 크기 + 신뢰 임계치 + OOF 재현 조건 명시.

## P1 — Primary prediction
- Quantity: (예: ROC-AUC on label_type=2 in OOF)
- Direction & magnitude: (예: control 대비 ≥ +0.5 pt)
- Confidence: (n_seeds × n_folds, p-threshold 또는 paired Δ 부호)
- Falsification: 본 조건 미달 시 verdict = `refuted`.

## P2 — Secondary / mechanism check
- (P1이 통과해도 메커니즘이 의도대로 작동하는지 별도 측정 — 예: bridge_active_share, attn_entropy_per_layer, cross_std)
- Threshold: ...
- Falsification: ...

## P3 — Negative control (optional but recommended)
- (작동하면 안 되는 ablation — 예: NS-token을 random shuffle로 대체했을 때 효과 사라짐)

## Reproducibility
- compute_tier: T?
- seeds: [42, 1337, 2026]
- splits: label_time 기준 train/val + 10% OOF
- expected wall: ~XXX min on tier

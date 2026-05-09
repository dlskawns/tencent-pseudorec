# H030 — Method Transfer

## Source

- **Lin et al. 2017** "Focal Loss for Dense Object Detection" — class
  imbalance mitigation via (1-p)^γ down-weighting easy examples.
- **H005 archived REFUTED** (pre-correction) — re-measurement on corrected
  eval data.

## Mechanism

H010 byte-identical EXCEPT loss path:
- bce (H023): `BCE(logits, label)`
- focal (H030): `α (1-p)^γ BCE(logits, label)` with α=0.25, γ=2.0.

12.4% conversion rate → focal upweights minority class implicitly.

## §17.2 EXEMPT (measurement H, no mechanism mutation)

loss path swap = 1 mutation. mechanism stack 그대로.
`measurement` re-entry (methodology framework).

## §⑤ UNI-REC alignment

mechanism / sequential / interaction 변경 없음. loss-axis 만.

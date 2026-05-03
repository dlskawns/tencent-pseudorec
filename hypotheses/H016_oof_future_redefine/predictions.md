# H016 — Predictions

> H015 sibling H — OOF 재정의 form. **Triple-H setup with H015/H017 동시
> launch** (L2 multi-form attack).

## P0 — Sanity gate
- OOF 새 정의 dataset 가 학습 가능 row 수 충분 — train ~85% × dataset size.
- Predicted: PASS (분포 단순 quantile cut).

## P1 — Code-path success
- PASS expected. dataset.py + train.py 변경, mechanism stack byte-identical.

## P2 — Primary lift (Platform AUC vs H010 corrected 0.837806)

| Result | Implication |
|---|---|
| strong (Δ ≥ +0.005pt) | OOF 재정의 가 measurement 정합성 회복, mechanism 진짜 효과 가시화. anchor = H016. |
| measurable [+0.001, +0.005pt] | partial. OOF 효과 측정 가능. |
| noise (−0.001, +0.001pt] | OOF 재정의도 효과 없음. cohort drift 가 paradigm 안 ceiling 의 hard ceiling. |
| degraded (< −0.001pt) | train cohort 감소 (10%) 가 학습 disrupt. |

## P3 — NS xattn entropy
H010 baseline [0.8127, 0.8133]. 변화 미세 expected (mechanism stack 변경 0).

## P4 — §18 PASS

## P5 — val ↔ platform
- val_AUC 가 새 정의 (label_time 기반 cutoff)인 점 인지.

## P6 — OOF-Platform gap (핵심 진단 — 다른 의미)
- **prior H 들과 비교 invalid** — OOF 정의 다름.
- 단일 측정값으로 새 OOF 정의 하의 gap 확인:
  - gap ≤ 1pt: OOF 가 platform proxy 잘 됨 (재정의 성공).
  - gap > 1.5pt: 새 OOF 도 platform 과 다른 분포 (재정의 효과 약함).

## P7 — train cohort 변화
- 새 정의로 train 약 10% 감소. learning curve 차이 monitor.

## Decision tree (post-result)

| Result | + H015 + H017 결과 | Next H |
|---|---|---|
| Δ ≥ +0.005pt + (H015 or H017 PASS) | L2 confirmed (multi-form) | H018 = OOF + recency combo, 또는 anchor = H016. |
| Δ ≥ +0.005pt + H015/H017 noise | OOF approach 가 train-side 보다 효과적 | H018 = OOF variants (different cutoff quantile). |
| noise + H015 noise + H017 noise | **L2 retire confirmed (Frame B)** | H018 = backbone replacement (paradigm shift). |
| degraded | train cohort 감소 disrupt | sub-H = oof_user_ratio 5% (작은 cutoff). |

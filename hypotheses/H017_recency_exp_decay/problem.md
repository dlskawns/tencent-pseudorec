# H017 — Problem Statement

## What we're trying to explain

H015 = recency loss weighting linear [0.5, 1.5]. linear curve 가 cohort drift
처리에 적정한 form 인지 검증 안 됨. exp curve (geometric) 가 더 강한 recent
emphasis 가능 — 적정 form 분리 측정.

## Why now

H015 와 동시 launch (triple-H setup). 같은 envelope, 같은 mechanism, 같은
range — **form 변경의 효과만 isolated 측정**. cohort drift 처리의 적정 curve
결정.

## Scope

- **In**: trainer.py 의 weight 함수 — linear → exp (geometric, auto-normalize
  to mean=1.0).
- **Out**: range 변경 (H015 와 동일 [0.5, 1.5]), per-dataset normalization
  (sub-H 후보), other curves (quadratic, sigmoid).

## UNI-REC axes

H015 와 동일 — training procedure axis (loss function shape).

## Success / Failure conditions

- **Success**: Δ vs H015 ≥ +0.001pt (form 효과) 또는 Δ vs H010 corrected
  ≥ +0.005pt (strong, exp curve 가 cohort drift 더 잘 처리).
- **Failure**:
  - Δ vs H015 ∈ (−0.001, +0.001pt]: form 변경 효과 없음, linear 와 동일.
  - Δ < −0.001pt vs H015: exp 가 학습 disrupt (asymmetry 너무 큼).

## Frozen facts referenced

- H015 problem.md / verdict.md (sibling H — 같은 mechanism class, 다른 form).
- §3.4 label_time + 9 H verdicts (cohort drift 가설 누적).

## Inheritance from prior H

- H015 mechanism + envelope byte-identical 외 form parameter.
- 4-layer ceiling diagnosis 의 L2 (cohort drift) sub-form 검증.
- H015 / H016 / H017 triple-H 동시 launch — multi-form L2 attack.

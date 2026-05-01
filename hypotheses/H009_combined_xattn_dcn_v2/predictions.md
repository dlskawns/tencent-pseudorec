# H009 — Predictions

> Pre-registered before run.

## P1 — Code-path success
- Quantity: train.py 1 epoch (early stop 시) 또는 10 epoch 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite val/OOF AUC.
- Falsification: NaN abort, OOM, 두 mechanism dispatch 라우팅 오류 → REFUTED.

## P2 — Primary lift (§17.3 binary)
- Quantity: extended envelope **platform AUC** vs anchor (original_baseline) ~0.83X.
- Predicted: Δ ≥ **+0.5pt** AND additivity sub-criterion:
  - **additive**: Δ ∈ [+0.005, +0.010pt] (H007 +0.0035 + H008 +0.0035 = +0.007pt 근처).
  - **super-additive**: Δ > +0.010pt.
  - **sub-additive**: Δ ∈ [+0.0035, +0.005pt].
  - **interference**: Δ < +0.0035pt (한 쪽 mechanism 약화).
- Falsification: Δ < +0.5pt → REFUTED. 양쪽 mechanism 동시 적용이 baseline 보다 못함 — 디버그 필요.

## P3 — Mechanism 두 쪽 작동 검증
- Quantity:
  - H007 candidate xattn weights nontrivial.
  - H008 DCN-V2 cross weight norms nontrivial.
- Predicted: 둘 다 학습됨 (i.e., not degenerate to identity).
- Falsification 아님: P2 결과와 별개. mechanism 작동 sanity check.

## P4 — §18 인프라 통과
- Quantity: inference 시 §18 룰 모두 충족.
- Predicted: PASS (H007 + H008 패키지 둘 다 §18 검증 통과).

## P5 — Additivity 정량 측정 (보너스)
- Quantity: combined Δ vs (H007 Δ + H008 Δ).
- Predicted analysis:
  - additive 가정 검증 → linear independence of mechanisms.
  - sub-additive → mechanism interference (Frame 1 risk 발현).
  - super-additive → multiplicative synergy (paper-grade).
- 의미: §0 north star 두 축 통합의 정량 효과 first measurement.

## Reproducibility
- compute_tier: T2.4 extended (10 epoch × 30%, patience=3 with H008 F-4 carry-forward).
- seed: 42.
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H006 (4h) ~ H008 (3.7h) 범위. patience=3 + 가능하면 batch=512 적용 시 ~2-2.5h.
- code: `experiments/H009_combined_xattn_dcn_v2/upload/` (12 파일, run.sh 가 두 flag baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 또는 sub-additive 시:

- **P2 fail with Δ ∈ (−0.005, +0.5pt)** (REFUTED but not <0): 단일 mutation 보다 못함 — 두 mechanism 간 interference. **carry-forward**: candidate summary token 통합 위치 변경 (prepend → separate stream) sub-H, 또는 lr scaling H.
- **P2 sub-additive with Δ ∈ [+0.0035, +0.005pt]**: 한 쪽 mechanism dominant. **carry-forward**: ablation H — H007 단독 vs H008 단독 vs combined 의 정량 비교 분석.
- **P2 super-additive with Δ > +0.010pt**: paper-grade 발견. **carry-forward**: multi-seed × 3 ablation H 로 super-additive confirm.
- **P3 fail (양쪽 mechanism degenerate)**: 학습 dynamics 문제. lr 또는 init scale 조정 sub-H.
- **P4 fail**: §18 회귀 — 두 cfg key 동시 read-back 검증.

→ 모든 negative-result interpretable.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.5pt + additive (Δ ∈ [+0.005, +0.010pt]) + P3/P4 PASS | H009 PASS additive 가정 검증. **anchor 갱신**: H009 가 새 anchor. H010 = 다른 axis (multi_domain_fusion / external_inspirations / aligned pair encoding) 위 single mutation. |
| Δ super-additive | paper-grade 발견. multi-seed 검증 H 우선. |
| Δ sub-additive | mechanism interference. ablation H — H007/H008 단독 다시 측정 후 통합 위치 재검토. |
| Δ < anchor | combined 가 baseline 보다 못함 — interference 강함. lr scaling 또는 통합 위치 변경 H. |
| P4 fail | §18 회귀. infer.py 디버깅 우선. |

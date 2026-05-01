# H008 — Predictions

> Pre-registered before run.

## P1 — Code-path success
- Quantity: train.py 1 epoch 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite val_AUC.
- Falsification: NaN abort, OOM, fusion dispatch 라우팅 오류 → REFUTED.

## P2 — Primary lift (§17.3 binary)
- Quantity: smoke (또는 extended) **platform AUC** vs anchor (original_baseline) ~0.83X.
- Predicted: Δ ≥ **+0.5 pt**.
- Falsification: Δ < +0.5pt → REFUTED. sparse_feature_cross 카테고리 일시 archive.

## P3 — Mechanism check (cross block 작동)
- Quantity: 가능하면 cross layer 의 weight 분포 (`||W₀||`, `||W₁||`) 측정.
- Predicted: weights 가 nontrivial (즉 ‖W‖ ≫ 0) — cross 가 학습됨.
- Falsification 아님: REFUTED 와 무관하게 mechanism 작동 여부 separate signal.

## P4 — §18 인프라 통과
- Quantity: inference 시 §18 룰 모두 충족 (batch heartbeat + `[infer] OK` 로그 + platform AUC ≠ 0.5).
- Predicted: PASS.

## P5 — val ↔ platform alignment (보너스)
- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05 (H006/H007 패턴 재현).

## Reproducibility
- compute_tier: T2.4 smoke 우선. extended (3ep × 30%, H007 envelope) 는 marginal/REFUTED 시 retry.
- seed: 42.
- split: label_time + 10% OOF (anchor 동일).
- expected wall: smoke ~5분, extended ~3시간.
- code: `experiments/H008_dcn_v2_block_fusion/upload/` (12 파일, run.sh 가 `--fusion_type dcn_v2` baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P2 fail with Δ ∈ (−0.005, +0.5pt)** (noise): DCN-V2 cross 가 RankMixer 와 비슷한 capacity 또는 우리 데이터에 marginal. **carry-forward**: layer 수 (2→4), rank (8→16) tuning sub-H. 또는 다른 explicit cross (FwFM, AutoDis).
- **P2 fail with Δ < 0** (악화): RankMixer 가 우리 데이터에 더 적합. swap mutation 자체가 information loss. **carry-forward**: RankMixer + DCN-V2 cross 병행 (separate H, parallel arm).
- **P3 fail (cross weights ≈ 0)**: cross 가 학습 안 됨 — degenerate to identity (xₗ₊₁ ≈ xₗ). lr 또는 init scale 조정 필요. sub-H.
- **P4 fail (인프라 회귀)**: §18 룰 inheritance 깨짐. infer.py 의 cfg.get 추가 부분 검증.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.5pt + P3/P4 PASS | H008 PASS. **H009 = combined H007 + H008** (sub-H, candidate xattn + DCN-V2 cross stack 검증). additive 가정 검증. |
| Δ ∈ [+0.0, +0.5pt) + P4 PASS at smoke | weak signal. extended envelope retry → 재평가. |
| Δ < 0 | swap 자체가 information loss. RankMixer 와 DCN-V2 cross 병행 H 후보. |
| P4 fail | §18 회귀. infer.py 디버깅 우선. |

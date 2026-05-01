# H001 — Predictions

## P1 — Code-path success (가장 강한 검증)
- Quantity: train.py이 1 epoch 끝까지 NaN-free 완주.
- Direction & magnitude: `metrics.json` 의 `best_val_AUC` 가 finite, ∈ [0, 1].
- Confidence: single seed=42 (sample-scale anchor — multi-seed는 cloud 전환 시).
- Falsification: NaN abort, OOM, schema mismatch, label leakage 검출 → REFUTED + 결함 분류 carry-forward.

## P2 — Mechanism check (unified block 실제 동작 검증)
- Quantity: `infer.py` 가 학습된 ckpt를 로드한 결과의 **value 분포**가 prior fallback과 다름.
- Threshold: predictions의 standard deviation ≥ 0.01 (prior fallback은 std=0). 즉 모델이 user별로 다른 점수를 내야 함.
- Falsification: std < 0.01 ⇒ 모델이 사실상 prior만 출력 ⇒ unified block forward path가 작동 안 함 ⇒ REFUTED.

## P3 — Negative control (overfit 검증)
- Quantity: train AUC vs valid AUC 차이.
- Threshold: train AUC가 valid AUC 보다 ≥ 0.05 pt 높음 (1000 rows에서는 거의 확실히 overfit이 발생, 안 발생하면 그게 더 이상함).
- Interpretation: train AUC ≈ valid AUC 라면 (a) 모델이 학습이 안 됐거나 (b) data leakage 있거나 둘 중 하나. 둘 다 verdict = REFUTED.

## P4 — Submission round-trip (§13 contract)
- Quantity: `submission/local_validate.py` G1–G6 통과 카운트.
- Threshold: 5/5 PASS (G1 signature, G2 env-only, G3+G4 run+coverage, G5 determinism, G6 no-internet).
- Falsification: 1개라도 실패 → REFUTED + §13.7 항목 매핑.

## P5 — OOF generalization (가장 약한 검증, anchor 자격용)
- Quantity: OOF AUC (10% user holdout, seed=42).
- Threshold: ≥ 0.50 (random보다 나쁘지 않음). 0.55–0.65 정도면 anchor로 충분.
- Falsification: < 0.50 ⇒ 모델이 random보다 나쁨 ⇒ pipeline 결함 의심 ⇒ REFUTED.

## Reproducibility
- compute_tier: T1.2 (M1 Pro MPS 또는 CPU — sample-scale code-path verification).
- seeds: [42] 단일 (cloud 전환 시 [42, 1337, 2026] 으로 확장).
- splits: `label_time` 90th percentile cutoff + 10% user OOF (seed=42).
- expected wall: ≤ 30분 on M1 (1000 rows × 4–8 epochs, batch=256, num_workers=0).
- code: `experiments/E000_unified_baseline_demo/run.sh` (래퍼).

## Pre-registration
본 predictions.md 는 실험 실행 전 commit 됨. 실행 후 verdict.md 의 measured 값과 비교.

# H002 — Predictions

> Pre-registered before run. 실행 후 verdict.md 의 measured 값과 비교.

## P1 — Code-path success
- Quantity: train.py 1 epoch 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite best_val_AUC.
- Falsification: NaN abort, OOM, schema mismatch → REFUTED.

## P2 — Primary lift (§17.3 binary success)
- Quantity: val_AUC vs E_baseline_organizer (control: 0.8251).
- Predicted: **Δ ≥ +0.5 pt** (즉 val_AUC ≥ 0.8301).
- Confidence: single seed=42, 같은 split (organizer row-group, 100 valid RGs), 같은 train_ratio=0.05.
- Falsification: Δ < +0.5 pt → REFUTED. InterFormer bridge 방향 retire.

## P3 — Mechanism check (bridge gate 학습 신호)
- Quantity: 12 bridges 의 final `sigmoid(gate)` 분포.
- Predicted: 학습 후 평균 gate ≥ 0.20 (init 0.119 의 +0.08+).
- 의미: 학습이 진행되면 bridge 가 useful 이라고 판단해 gate 가 grow. 변화 없음 = bridge 미사용.
- Falsification: 평균 gate < 0.15 (5% 변화 미만) → 모델이 bridge 신호 안 쓰는 것. **P2 lift 와 무관하게 mechanism 미작동 신호** → verdict 에 noted (lift 가 다른 데서 왔거나 lift 자체 없음).

## P4 — Submission round-trip
- Quantity: 다운로드된 ckpt 로 `submission/local_validate.py` 5/5 PASS.
- Predicted: G1–G6 모두 통과 (organizer baseline 과 동일 contract).
- Falsification: 1개라도 fail → REFUTED + §13.7 매핑.

## P5 — Variance reduction (paper claim 보너스 측정)
- Quantity: bridge ON 의 train-loss-per-step variance vs OFF.
- Predicted: paper 주장 30–50% variance 감소 → 본 run 의 step-loss std 가 baseline 대비 ≤ 0.85x.
- Falsification 아님: 측정 실패해도 H002 supported 유지 (P5 는 보너스). 대신 noted in verdict.

## Reproducibility
- compute_tier: Taiji organizer mode (smoke run.sh).
- seed: 42 (single — paired Δ 의도, multi-seed 는 H002 통과 후 별도 ablation).
- split: organizer row-group, train_ratio=0.05, valid 100 RGs.
- seq_max_lens: seq_a:64,seq_b:64,seq_c:128,seq_d:128 (anchor 와 동일).
- expected wall: ~30–45분 (anchor 와 거의 동일, bridge overhead +5%).
- code: `experiments/H002_interformer_domain_bridges/upload/` (12 파일, run.sh 가 `--enable_inter_domain_bridges --bridge_rank 4 --bridge_gate_init -2.0` baked).

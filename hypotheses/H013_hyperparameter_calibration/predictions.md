# H013 — Predictions

> Pre-registered before run. 반증 가능. **Measurement H** — primary
> objective 는 ceiling diagnostic.

## P1 — Code-path success

- Quantity: train.py NaN-free 완주, `metrics.json` 생성.
- Predicted: NaN 0건. lr 8e-4 가 batch 2048 와 함께 stable expected (Linear
  scaling rule).
- Falsification: NaN abort, OOM, gradient explosion → REFUTED + sub-H
  (lr 4e-4 절반 scaling 또는 warmup 추가).

## P2 — Primary lift (§17.3 binary, sample-scale relaxed)

- Quantity: extended envelope **platform AUC** vs anchor (H010 0.8408).
- Predicted classifications:
  - **strong** (Δ ≥ +0.005pt): **ceiling 은 hyperparameter artifact**.
    Platform ≥ 0.8458. 모든 prior H paired Δ 재해석 의무.
  - **measurable** (Δ ∈ [+0.001, +0.005pt]): 부분적 hyperparameter 효과,
    부분적 mechanism ceiling.
  - **noise** (Δ ∈ (−0.001, +0.001pt]): mechanism ceiling 진짜. lr 변경
    효과 없음 → Track B (long-seq P2) 또는 cohort drift 우선.
  - **degraded** (Δ < −0.001pt): lr 8e-4 너무 큼 → divergence/instability.
    sub-H lr 4e-4.
- Falsification (binary): Δ < +0.001pt → §17.3 measurable 미달, mechanism
  ceiling confirmed direction.

## P3 — NS xattn entropy 변화 (mechanism interpretation)

- Quantity: H010 NS xattn `attn_entropy_per_layer` (baseline [0.8127, 0.8133]).
- Predicted classifications:
  - **변화 미세** (|Δ| < 0.05): selective routing pattern 동일 → lr 효과
    가 mechanism 표현에 미세 영향. dominant signal sparse capture 그대로.
  - **더 sparse** (entropy < 0.7): proper lr 시 더 강한 specialization.
  - **더 uniform** (entropy > 1.0): lr 너무 커서 attention pattern 흐트러짐.
    P2 degraded 와 함께 발현 시 발산 신호.
- Falsification 아님 — mechanism interpretation diagnostic.

## P4 — §18 인프라 통과

- Quantity: inference §18 룰 모두 충족.
- Predicted: PASS (mechanism 변경 0, infer.py 변경 0).
- Falsification: P4 fail → §18 회귀 (단 가능성 매우 작음).

## P5 — val ↔ platform 정합 (보너스)

- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05.

## P6 — OOF-platform gap (보너스, cohort drift 모니터)

- Quantity: OOF AUC − Platform AUC.
- 비교 baseline: H010 1.88 → H011 2.42 → H012 2.10. 7개 H 평균 ~2pt.
- Predicted classifications:
  - **gap 크게 줄어듦 (≤ 1.5pt)**: hyperparameter calibration 가 cohort
    drift 도 일부 완화 (예상 외, positive surprise).
  - **gap 유지 (1.8~2.2pt)**: cohort drift 가 hyperparameter 와 독립적.
    Frame C (cohort hard ceiling) 강한 신호.
  - **gap 더 벌어짐 (> 2.5pt)**: lr 너무 커서 OOF cohort 만 fit, platform
    악화 (overfit 패턴).
- Falsification 아님 — cohort drift 가설 분리 측정.

## P7 — IO/wall efficiency (보너스, F-5 신호 검증)

- Quantity: 학습 wall + GPU utilization (가능 시).
- 비교 baseline: H010 3:44:54, H011 2:46:54, H012 2:49:43. wall 단축 패턴
  = IO bound 신호 (H012 F-5).
- Predicted: H013 wall ≤ H010 (3:44) — num_workers 4 + buffer_batches 8
  로 IO 완화 시 GPU idle 시간 감소.
- Falsification 아님 — efficiency diagnostic.

## Reproducibility

- compute_tier: T2.4 extended (10 epoch × 30%, patience=3).
- seed: 42.
- batch_size: 2048 (explicit).
- lr: 8e-4 (linear scaling).
- num_workers: 4.
- buffer_batches: 8.
- split: label_time + 10% OOF.
- expected wall: H010 envelope ~3-3.5h (lr 큰 효과로 epoch 빨리 수렴 가능
  → patience=3 trigger 빠를 수 있음).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P1 fail (NaN abort or divergence)**: lr 8e-4 가 batch 2048 + this model
  에서 너무 큼. **carry-forward**: H013-sub = lr 4e-4 (절반 scaling) 또는
  lr 8e-4 + warmup 100 step 추가. NaN 발생 위치 (epoch / batch step) 기록.
- **P2 fail with Δ ∈ (−0.001, +0.001pt) (noise)**: **Frame B confirm —
  mechanism ceiling 진짜**. carry-forward: H014 = long-seq retrieval (P2
  entry, TWIN/SIM/HSTU). 현재 truncate 64-128 vs §3.5 p90 1393~2215 의
  95%+ 정보 손실 직접 motivation.
- **P2 fail with Δ < −0.001pt (degraded)**: lr 너무 커서 generalization
  악화. carry-forward: H013-sub = lr 4e-4 또는 batch 256 복귀.
- **P3 더 uniform + P2 degraded**: attention pattern 흐트러진 발산 신호.
  lr 4e-4 sub-H.
- **P6 gap 더 벌어짐**: cohort overfit. carry-forward: cohort H 우선
  (Frame C confirm).

## Decision tree (post-result)

| Result | Implication | Next H |
|---|---|---|
| Δ vs H010 ≥ +0.005pt + P1/P3/P4 PASS | **Ceiling = hyperparameter artifact (Frame A confirm)**. 모든 prior H ranking 재해석. anchor = H013 (새 baseline). | H014 = mechanism H (NS xattn sub-H 또는 long-seq P2 진입). prior H 들 paired Δ 재해석 보고. |
| Δ ∈ [+0.001, +0.005pt] + P3 변화 미세 | 부분적 hyperparameter 효과. anchor 갱신 검토. | H014 = long-seq P2 entry 또는 cohort H. |
| Δ ∈ (−0.001, +0.001pt) (noise) + P3 변화 미세 | **Mechanism ceiling 진짜 (Frame B confirm)**. lr 적정. | H014 = long-seq P2 (truncate 95%+ 정보 손실 motivation 강함). |
| Δ < −0.001pt (degraded) | lr 너무 큼. | H013-sub = lr 4e-4 또는 batch 256 복귀. |
| P1 NaN abort | lr 8e-4 발산. | H013-sub = lr 4e-4 또는 warmup 추가. |
| P6 gap 더 벌어짐 | cohort overfit. | cohort H 우선 (Frame C confirm). |
| P7 wall 단축 안 됨 | IO bound 가설 약화. | num_workers 변경 효과 없음 confirm. |

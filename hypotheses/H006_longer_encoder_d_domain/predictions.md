# H006 — Predictions

> Pre-registered before run.

## P1 — Code-path success
- Quantity: train.py 1 epoch 완주, `metrics.json` 생성, `model.pt` save.
- Predicted: NaN 0건, finite val_AUC, finite logloss.
- Falsification: NaN abort, OOM, encoder type 라우팅 오류 → REFUTED. LongerEncoder 가 organizer 코드라 가능성 매우 낮음.

## P2 — Primary lift (§17.3 binary)
- Quantity: smoke val_AUC vs original_baseline-anchor.
- Predicted: Δ ≥ **+0.5 pt**.
- Confidence: single seed=42, 같은 split (label_time + 10% OOF), 같은 envelope. **Encoder type 만 변수**.
- Falsification: Δ < +0.5 pt → REFUTED. long_seq_retrieval 카테고리 일시 archive.

## P3 — D-도메인 vs A/B/C 비대칭 (메커니즘 검증)
- Quantity: 가능하면 (per-domain attention 통계 logging 가능 시) — D vs A/B/C 의 attention pattern 차이 측정.
- Predicted: D 의 representation 변화량 (또는 attention spread) 이 A/B/C 보다 의미 있게 큼. 가설 mechanism 검증.
- Falsification: A/B/C 에서도 같은 lift → 메커니즘이 D-tail 활용이 아니라 다른 부수 효과 (예: top-K masking 자체의 implicit regularization). 본 H 의 specific motivation 약화 — 단 P2 lift 자체는 retain 가능.
- 의미: 본 P3 는 **direct measurement 어려움** (current code 가 per-domain attn 통계 dump 안 함). Smoke run 에서 best effort, 구체 측정 안 되면 noted.

## P4 — Submission round-trip
- Quantity: 다운로드된 ckpt 로 §18 룰 준수한 inference. 핵심:
  - `[infer] starting inference loop` 로그 + **batch heartbeat (`[infer] batch 50/100/...`)** + **`[infer] OK: torch path produced N predictions`**.
  - heuristic fallback 신호 없음 (`[infer] FALLBACK: ...` 미출현).
  - predictions.json 의 분포가 uniform (≈ 0.124) 아님 — 모델이 실제로 추론.
  - Platform AUC ≠ 0.5 (chance level 아님).
- Predicted: §18 룰 준수 → 5/5 PASS.
- Falsification: heuristic fallback 발생 또는 platform AUC 0.5 → infrastructure 회귀 → REFUTED on P4.

## P5 — Platform AUC vs val_AUC alignment (보너스)
- Quantity: smoke val_AUC ↔ platform AUC 차이.
- Predicted: |val_AUC − platform_AUC| ≤ 0.10 (anchor + H006 둘 다). 만약 두 H 모두 platform AUC 가 val 보다 한참 낮으면 leakage 가설 carry-forward.
- Falsification 아님: 정렬 안 되어도 본 H 의 P2 success 와 무관. 단 **이 측정 자체가 leak 가설 검증** = future H 들의 anchor 신뢰성 결정.

## Reproducibility
- compute_tier: T2.4 smoke (Taiji + leak-fix smoke envelope).
- seed: 42 (single — paired Δ 의도, multi-seed 는 H006 PASS 후 별도 ablation).
- split: label_time-aware + 10% user OOF holdout (anchor 와 동일).
- seq_max_lens: seq_a:64,seq_b:64,seq_c:128,seq_d:128 (anchor 와 동일).
- num_epochs: 1.
- expected wall: ~5 분 (anchor ~3분 + LongerEncoder top-K compression overhead 2분).
- code: `experiments/H006_longer_encoder_d_domain/upload/` (12 파일, run.sh 가 `--seq_encoder_type longer` baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H 가 REFUTED 인 경우 학습 가능한 정보:

- **P2 fail with Δ ∈ (−0.001, +0.5)** (소폭 lift 또는 noise): LongerEncoder 의 top-K=50 selection 이 우리 envelope (seq_max_lens 64–128) 에선 effective benefit marginal. **정보 가치 = 큼** — paper 의 long-seq retrieval claim 이 우리 sample-scale + 짧은 envelope 에선 발현 안 함. carry-forward: seq_max_lens 확장 (별도 H, compute cost 큼) + top-K tuning (별도 H) + candidate-aware retrieval (target_attention 카테고리, 별도 H) 후보 정렬.
- **P2 fail with Δ < 0** (악화): top-K=50 selection 이 우리 envelope 에서 정보 손실. K=50 < L=128 (D 도메인) 인 경우 일부 token 잘림. carry-forward: K tuning 으로 K ≥ L 만든 ablation 후보.
- **P3 fail (D vs A/B/C 비대칭 없음)**: 메커니즘 가설 약화. 본 H 의 specific lift 영역 (D long-tail 활용) 가 아니라 다른 부수 효과 (top-K masking 의 regularization 등) 가능성.
- **P4 fail**: §18 인프라 회귀. 즉시 진단 + fix.

→ 모든 negative-result 가 interpretable. malformed experiment 아님.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.5pt + P3/P4 PASS | H006 PASS. anchor 갱신 또는 H006-anchor 등록. H007 = target_attention (candidate-aware retrieval, paper-grade upgrade). |
| Δ ∈ [+0.0, +0.5pt) + P3/P4 PASS | Weak signal. long_seq_retrieval 일시 archive, retry on extended envelope (train_ratio=0.3, num_epochs=3) 별도 H 후보. |
| Δ < 0 + P3 PASS | Top-K=50 가 envelope 에서 정보 손실. K tuning H 후보. |
| P4 fail | §18 회귀. 인프라 우선. |

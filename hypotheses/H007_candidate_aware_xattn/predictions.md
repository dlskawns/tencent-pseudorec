# H007 — Predictions

> Pre-registered before run.

## P1 — Code-path success
- Quantity: train.py 1 epoch 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite val_AUC, finite logloss.
- Falsification: NaN abort, OOM, attention dim mismatch, candidate token shape mismatch → REFUTED.

## P2 — Primary lift (§17.3 binary)
- Quantity: smoke (또는 extended) **platform AUC** vs anchor (original_baseline) ~0.83X.
- Predicted: Δ ≥ **+0.5 pt**.
- Falsification: Δ < +0.5pt → REFUTED. target_attention 카테고리 일시 archive.

## P3 — Mechanism check (candidate attention pattern)
- Quantity: 가능하면 candidate token 의 attention probability mass distribution 측정 (per-domain × random batch).
- Predicted: candidate-relevant events 로 pattern concentration (uniform 아님). 즉 `attn_entropy_per_layer` < 0.95·log(L_d) — H004 와 동일 §10.9 룰.
- Falsification: uniform collapse → candidate-aware mechanism 미작동, lift 와 무관하게 mechanism 자체 fail.

## P4 — §18 인프라 통과
- Quantity: inference 시 §18 룰 모두 충족:
  - `[infer] starting inference loop` + batch heartbeat 보임.
  - `[infer] OK: torch path produced N predictions`.
  - heuristic fallback 없음.
  - platform AUC ≠ 0.5.
- Predicted: PASS (anchor 패키지에서 inherit 한 인프라).
- Falsification: 1개라도 fail → §18 회귀.

## P5 — val ↔ platform alignment (보너스)
- Quantity: |val_AUC − platform_AUC| (anchor 와 H007 둘 다).
- Predicted: ≤ 0.05 (H006 처럼 정합 예상). OOF 는 supplementary, paired 비교에 사용 안 함.

## Reproducibility
- compute_tier: T2.4 smoke 우선. extended (10ep × 30%) 는 smoke PASS or marginal 시 retry.
- seed: 42.
- split: label_time-aware + 10% user OOF (anchor 동일).
- seq_max_lens: seq_a:64,seq_b:64,seq_c:128,seq_d:128 (anchor 동일).
- num_epochs: 1 smoke / 10 extended.
- expected wall: smoke ~5분 (anchor 3분 + cross-attention ~2분), extended ~4–5시간.
- code: `experiments/H007_candidate_aware_xattn/upload/` (12 파일, run.sh 가 `--use_candidate_summary_token` baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H 가 REFUTED 인 경우 학습 가능한 정보:

- **P2 fail with Δ ∈ (−0.005, +0.5pt)** (noise 영역): candidate-aware mechanism 효과 marginal at smoke envelope. **정보 가치 = 큼** — extended envelope retry 필요. mechanism 자체 retire 가 아님.
- **P2 fail with Δ < 0** (악화): candidate token 구성 (mean pool) 또는 prepend 위치 문제. **carry-forward**: candidate token alternative (first token, learnable weighted, item_id direct) ablation H 후보.
- **P3 fail (uniform attention collapse)**: cross-attention layer 가 학습 안 됨. sample-scale 47k rows 로 50K params 부족 가능성. extended envelope 시도.
- **P4 fail (인프라 회귀)**: §18 룰 inheritance 깨짐. infer.py 의 cfg.get 추가 부분 검증.
- **smoke + extended 양쪽 모두 fail**: target_attention 카테고리 retire. CAN co-action / HSTU hierarchical / item_id direct embedding 후보 정렬.

→ 모든 negative-result 가 interpretable. malformed experiment 아님.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.5pt + P3/P4 PASS (smoke 또는 extended) | H007 PASS. anchor 갱신 또는 H007-anchor 등록. H008 = candidate-aware 변형 (CAN co-action 또는 multi-layer target attention) 또는 다른 카테고리. |
| Δ ∈ [+0.0, +0.5pt) + P3/P4 PASS (smoke) | weak signal at smoke. extended retry → 재평가. |
| Δ ≥ +0.5pt + P3 fail | mechanism 작동 안 했는데 lift 있음 — 수상한 신호. 부수 효과 (token shape variance regularization 등) 의심. 별도 진단. |
| Δ < 0 + P3 PASS | candidate token 구성 또는 prepend 위치 문제. ablation H. |
| 양쪽 모두 fail | target_attention 카테고리 archive. CAN/HSTU 변형 H 또는 다른 mechanism. |
| P4 fail | §18 회귀. infer.py 의 새 cfg 추가 부분 디버깅 우선. |

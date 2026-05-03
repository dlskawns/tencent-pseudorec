# H012 — Predictions

> Pre-registered before run. 반증 가능.

## P1 — Code-path success

- Quantity: train.py NaN-free 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite val/OOF AUC. MultiDomainMoEBlock dispatch +
  H010 NS xattn + DCN-V2 fusion 모두 정상.
- Falsification: NaN abort, OOM, gate softmax 발산 → REFUTED + scale 디버깅.

## P2 — Primary lift (§17.3 binary, sample-scale relaxed)

- Quantity: extended envelope **platform AUC** vs anchor (H010 0.8408).
- Predicted classifications:
  - **strong PASS** (Δ ≥ +0.005pt): explicit expert routing 효과 큼. Platform ≥ 0.8458.
  - **measurable** (Δ ∈ [+0.001, +0.005pt]): mechanism 작동 confirmed. Platform ∈ [0.8418, 0.8458].
  - **noise** (Δ ∈ [−0.001, +0.001pt]): H010 implicit routing 충분 (Frame B). expert routing redundant.
  - **degraded** (Δ < −0.001pt): expert routing 이 H010 selective routing 흔듦. REFUTED.
- Falsification (binary): Δ < +0.001pt → §17.3 sample-scale relaxed 임계 미달.

## P3 — Expert utilization 분포 (mechanism check)

- Quantity: gate softmax routing 의 per-expert utilization (per-batch averaged).
- Threshold: utilization entropy = `−Σ p_i log p_i` (over 4 experts).
  - Uniform routing entropy = log(4) ≈ 1.386.
  - Single-expert collapse entropy = 0.
  - Reasonable specialization entropy ∈ [0.5, 1.2] (1-2 dominant experts but
    not collapse).
- Predicted classifications:
  - **specialized** (entropy ∈ [0.5, 1.2]): 도메인별 routing 학습. Frame A
    confirmed.
  - **uniform** (entropy > 1.3): expert specialization 안 일어남. gate
    학습 실패. Frame B 신호 (NS-token 표현이 이미 mixed).
  - **collapse** (entropy < 0.5): 1 expert dominant. §10.9 abort 신호 +
    expert mechanism 무용.
- Falsification 아님 — mechanism 작동 진단.
- Threshold: §10.9 룰 collapse abort threshold = 0.5 × log(4) ≈ 0.69 미만
  시 경고.

## P4 — §18 인프라 통과

- Quantity: inference §18 룰 모두 충족 (batch heartbeat + `[infer] OK` 로그
  + platform AUC ≠ 0.5).
- Predicted: PASS (H010 패키지 inherit + 3 cfg key 추가).
- Falsification: P4 fail → §18 회귀.

## P5 — val ↔ platform 정합 (보너스)

- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05.

## P6 — OOF-platform gap (보너스, cohort drift 모니터)

- Quantity: OOF AUC − platform AUC.
- 비교 baseline: H006 3.5 → H010 1.88 → H011 2.42 (cohort drift 재발).
- Predicted: ≤ 2pt (H010 패턴 회복, H011 의 cohort drift 안 따라가기).
- Falsification 아님 — H011 F-5 hypothesis (cohort drift hard ceiling)
  검증 신호.

## Reproducibility

- compute_tier: T2.4 extended (10 epoch × 30%, patience=3) ~3-3.5시간.
- seed: 42.
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H010 (3.7h) ~ H011 (2.8h) 사이. ~33K params 추가로 약간
  슬로우 가능.

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P1 fail (NaN)**: gate softmax 발산 — temperature scaling 또는 LayerNorm
  on gate 입력 sub-form. **carry-forward**: H013 = MoE numerical stability
  변형.
- **P2 fail with Δ ∈ (−0.001, +0.001pt) (noise) + P3 uniform**: Frame B
  강한 confirm. H010 NS xattn 이 이미 mixed routing 학습 → explicit MMoE
  redundant. **carry-forward**: H013 = NS xattn sub-H (multi-layer) 또는
  cohort 처리 (H011 F-5 motivation, recency-aware OOF 재정의).
- **P2 fail with Δ < −0.001pt (degraded) + P3 specialized**: expert routing
  자체는 작동했지만 H010 selective routing 흔듦. interference 패턴 (H009
  와 같은 구조). **carry-forward**: H013 = MMoE 통합 위치 변경 (NS xattn
  전 또는 DCN-V2 후) 또는 hard routing 변형.
- **P3 collapse (entropy < 0.5)**: §10.9 abort. expert collapse — sample-
  scale 한계. **carry-forward**: H013 = num_experts ≤ 2 또는 hard routing
  변형 (top-1 with straight-through).
- **P4 fail**: §18 회귀.
- **P6 fail (gap > 2pt)**: cohort drift 재발 (H011 F-5 confirm). **carry-
  forward**: cohort 처리 H 별도 (recency-aware OOF, temporal weighting).

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ vs anchor (H010) ≥ +0.005pt + P3 specialized + P4 PASS | H012 strong PASS. **anchor 갱신**: H012 = 새 baseline. H013 = orthogonal axis (NS xattn sub-H 또는 input-stage sub-form, rotation 따라). |
| Δ ∈ [+0.001, +0.005pt] + P3 specialized | H012 PASS (measurable). anchor 갱신 검토. H013 = MMoE variants (PLE progressive, num_experts) 또는 다른 axis. |
| Δ ∈ (−0.001, +0.001pt) (noise) + P3 uniform | Frame B 채택. H013 = NS xattn sub-H 또는 cohort 처리 (H011 F-5). |
| Δ < −0.001pt (degraded) + P3 specialized | interference. H013 = MMoE 통합 위치 변경 sub-H. |
| P3 collapse (entropy < 0.69) | §10.9 abort. num_experts ≤ 2 또는 hard routing sub-H. |
| P4 fail | §18 회귀 디버깅. |
| P6 fail (gap > 2pt) | cohort drift hard ceiling confirmed (H011 F-5). cohort 처리 별도 H 우선. |

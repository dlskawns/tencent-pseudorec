# H015 — Predictions

> Pre-registered before run. **Cohort drift mitigation H** — paradigm 안
> 마지막 layer (L2) 검증.

## P1 — Code-path success
- Quantity: train.py NaN-free 완주, `metrics.json` 생성.
- Predicted: PASS. mean weight = 1.0 보존, loss scale 영향 없음.
- Falsification: NaN abort → loss reduction='none' + weighted mean 의 numerical
  issue (가능성 매우 작음).

## P2 — Primary lift (§17.3 binary, sample-scale relaxed)
- Quantity: extended envelope **platform AUC** vs anchor (H010 corrected
  0.837806).
- Predicted classifications:
  - **strong** (Δ ≥ +0.005pt, Platform ≥ 0.8428): **L2 confirmed**. Cohort
    drift = ceiling 의 진짜 정체. paradigm 안 ceiling 깰 수 있음.
  - **measurable** (Δ ∈ [+0.001, +0.005pt]): L2 partial.
  - **noise** (Δ ∈ (−0.001, +0.001pt]): **L2 도 가설 약함**. 마지막 layer
    retire → **paradigm shift mandatory** (Frame B confirmed).
  - **degraded** (Δ < −0.001pt): recency weighting 학습 disrupt.

## P3 — NS xattn entropy 변화 (mechanism interpretation)
- Quantity: H010 NS xattn `attn_entropy_per_layer` (baseline [0.8127, 0.8133]).
- Predicted classifications:
  - **변화 미세** (|Δ| < 0.05): selective routing pattern 유지 (loss 가중치만
    영향).
  - **더 sparse / 더 uniform**: 가중치 effect on attention learning. 기대
    effect 작음.
- Falsification 아님 — mechanism diagnostic.

## P4 — §18 인프라 통과
- Quantity: inference §18 룰 모두 충족.
- Predicted: PASS (infer.py 변경 0, mechanism byte-identical).

## P5 — val ↔ platform 정합 (보너스)
- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05.

## P6 — OOF-platform gap (보너스, 핵심 진단)
- Quantity: OOF AUC − Platform AUC.
- 9 H baseline: 1.88 ~ 2.59pt.
- Predicted classifications:
  - **gap 크게 줄어듦 (≤ 1.5pt)**: recency weighting 가 train cohort 를
    platform 분포에 closer 만듦. **L2 strong confirm**.
  - **gap 유지 (1.8~2.2pt)**: cohort drift 가 weighting 으로 못 풀림.
    Frame B 신호.
  - **gap 더 벌어짐 (> 2.5pt)**: weighting 이 train cohort 더 specialize
    → platform 더 멀어짐. negative.
- Falsification 아님 — L2 가설 직접 검증.

## P7 — OOF AUC 변화 (보너스)
- Quantity: OOF AUC.
- 9 H baseline: 0.857~0.860 (variance 0.23pt).
- Predicted classifications:
  - **OOF 약간 하락** (예: 0.855): recent sample 만 emphasized → OOF cohort
    은 random user holdout 이라 fit 약화. 단 platform 향상이 더 크면 net
    positive.
  - **OOF 유지** (~0.858): weighting 영향 작음.
  - **OOF 향상**: 예상 외, recent pattern 이 OOF cohort 에도 transfer.

## Reproducibility
- compute_tier: T2.4 extended (10 epoch × 30%, patience=3) ~3-4h.
- seed: 42, batch 2048, lr 1e-4, seq 64-128 (H010 envelope).
- mechanism: H010 NS xattn + H008 DCN-V2 fusion byte-identical.

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P2 fail with Δ ∈ (−0.001, +0.001pt) (noise) + P6 gap 유지**: cohort drift
  hard ceiling = paradigm 안 mutation 모두 못 풀림. **carry-forward**:
  H016 = paradigm shift mandatory. backbone replacement (OneTrans full /
  HSTU trunk / InterFormer 3-arch) 또는 OOF 재정의 (Frame C).
- **P2 fail with Δ < −0.001pt (degraded) + P7 OOF 하락**: weighting 이 학습
  disrupt. **carry-forward**: H015-sub = weight range 좁힘 [0.7, 1.3] 또는
  exp decay (less aggressive) 또는 per-dataset normalization.
- **P6 gap 더 벌어짐**: weighting 이 train cohort 더 specialize, platform
  더 멀어짐. negative carry-forward — recency 자체가 wrong direction.
- **P3 더 uniform**: 가중치 effect on routing. 단순 noise 추가 신호.

## Decision tree (post-result)

| Result | Implication | Next H |
|---|---|---|
| Δ ≥ +0.005pt + P6 gap 줄어듦 | **L2 strong confirmed**. cohort drift = ceiling. | H016 = recency variants (exp decay, larger range, per-dataset). |
| Δ ∈ [+0.001, +0.005pt] | L2 partial. | H016 = recency + cohort embedding combo 또는 OOF 재정의 (Frame C). |
| Δ ∈ (−0.001, +0.001pt] (noise) | **L2 retire → paradigm shift mandatory**. | H016 = backbone replacement (OneTrans full / HSTU / InterFormer). |
| Δ < −0.001pt | weighting disrupt. | H015-sub = weight range 좁힘 또는 exp decay. |
| P6 gap 더 벌어짐 | recency direction 잘못. | H016 = OOF 재정의 (Frame C) 또는 backbone replacement. |

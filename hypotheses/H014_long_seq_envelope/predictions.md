# H014 — Predictions

> Pre-registered before run. **Envelope mutation H** — single change
> (seq_max_lens 4×).

## P1 — Code-path success

- Quantity: train.py NaN-free 완주, `metrics.json` 생성, OOM 없음.
- Predicted: PASS. mechanism 변경 0, dataset.py byte-identical.
- Falsification:
  - **OOM** → batch 2048 + seq 512 메모리 위협. sub-H 1 (seq 256 uniform)
    또는 sub-H 3 (batch 256 복귀).
  - NaN abort → 긴 seq 의 padding/mask issue (가능성 낮음, dataset.py
    byte-identical).

## P2 — Primary lift (§17.3 binary, sample-scale relaxed)

- Quantity: extended envelope **platform AUC** vs anchor (H010 0.8408).
- Predicted classifications:
  - **strong** (Δ ≥ +0.005pt, Platform ≥ 0.8458): **L4 confirmed**.
    truncate 정보 손실이 진짜 ceiling. P2 phase entry 정당.
  - **measurable** (Δ ∈ [+0.001, +0.005pt]): L4 partial. dense expansion
    효과 측정 가능 lift, retrieval 추가로 더 큰 lift 가능.
  - **noise** (Δ ∈ (−0.001, +0.001pt]): L4 도 ceiling 아님. dense self-
    attention 의 한계 — retrieval/compression 없이는 long-seq 효과 0.
    또는 cohort drift L2 가 진짜 ceiling.
  - **degraded** (Δ < −0.001pt): 긴 seq 가 noise 추가, attention 분산되어
    dominant signal 희석.

## P3 — NS xattn entropy 변화 (mechanism interpretation)

- Quantity: H010 NS xattn `attn_entropy_per_layer` (baseline [0.8127, 0.8133]).
- 새 threshold: §10.9 룰 = 0.95 × log(L_total) where L_total = 256+256+512+512
  = 1536. **threshold = 0.95 × log(1536) ≈ 6.97** (이전 5.65).
- Predicted classifications:
  - **변화 미세** (|Δ| < 0.05): selective routing pattern 유지. NS xattn
    여전히 ~2 tokens 만 attend (이전 384 중 2 → 1536 중 2, 더 sparse 비율).
    + Δ ≥ +0.005pt 시 expansion 자체 효과 (encoder 표현 더 풍부) → **strong
    PASS + sparse routing 유지** = best case.
  - **더 sparse** (entropy < 0.5): proper longer context 시 더 명확한 dominant
    routing.
  - **더 uniform** (entropy > 1.5): 긴 seq 분산 → routing 흐트러짐. P2
    degraded 와 함께 발현 시 dense expansion 한계.
- Falsification 아님 — mechanism interpretation diagnostic.

## P4 — §18 인프라 통과

- Quantity: inference §18 룰 모두 충족.
- Predicted: PASS (mechanism 변경 0, infer.py 변경 0). 단 inference 도
  seq_max_lens 영향 받음 → wall 증가 예상.

## P5 — val ↔ platform 정합 (보너스)

- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05.

## P6 — OOF-platform gap (보너스, cohort drift 모니터)

- Quantity: OOF AUC − Platform AUC.
- 비교 baseline: H010 1.88 / H011 2.42 / H012 2.10 / H013 2.29.
- Predicted classifications:
  - **gap 줄어듦** (≤ 1.5pt): long-seq 가 cohort drift 일부 완화 (예상
    외, positive surprise). 단 가능성 낮음.
  - **gap 유지** (1.8~2.2pt): cohort drift 가 truncate 와 독립적. **Frame
    B 신호** (cohort = hard ceiling).
  - **gap 더 벌어짐** (> 2.5pt): long-seq 가 OOF cohort 만 fit, Platform
    악화. **Frame B 강한 confirm**.
- Falsification 아님 — Frame B (cohort) vs A (truncate) 분리.

## P7 — Wall efficiency (보너스)

- Quantity: 학습 wall.
- 비교 baseline: H010 3:44:54.
- Predicted: 2-4× 증가 (attention O(L²) — seq 4× → compute 16×, 단 batch
  / IO 영향 일부 흡수). **6-15시간 범위**.
- Falsification 아님 — efficiency monitor.

## Reproducibility

- compute_tier: T2.4 extended (10 epoch × 30%, patience=3) ~6-15h (long
  seq overhead).
- seed: 42.
- batch_size: H010 default 256 (사용자 override 2048 시).
- lr: 1e-4 (default).
- num_workers / buffer_batches: H010 default 2 / 4.
- split: label_time + 10% OOF.

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P1 OOM**: batch 2048 + seq 512 메모리 위협. **carry-forward**: H014-sub
  = (a) seq 256 uniform, (b) seq 128/128/256/256 conservative, (c) batch 256
  복귀.
- **P2 fail with Δ ≤ +0.001pt (noise)**: dense self-attention expansion
  효과 없음. retrieval/compression (TWIN/SIM/HSTU) 없이는 long-seq 가치
  0. **carry-forward**: H015 = TWIN/SIM (target-aware retrieval) 또는
  HSTU trunk. 단 paradigm shift inevitable — backbone replacement 도 후보.
- **P2 fail with Δ < −0.001pt (degraded)**: 긴 seq 가 noise 추가, dominant
  signal 희석. **carry-forward**: H015 = retrieval-based (denoising 효과)
  또는 cohort H 우선.
- **P6 gap > 2.5pt**: cohort drift 더 벌어짐. **Frame B 강한 confirm**.
  **carry-forward**: H015 = cohort H 우선 (recency-aware loss / temporal
  cohort embedding). long-seq 더 시도 무의미.
- **P3 더 uniform + P2 degraded**: attention 분산 + 학습 악화. seq 줄임 sub-H.

## Decision tree (post-result)

| Result | Implication | Next H |
|---|---|---|
| Δ ≥ +0.005pt + P3 sparse 유지 | **L4 confirmed (strong)**. anchor = H014. | H015 = TWIN/SIM (target-aware retrieval) for further gain. P2 phase entry. |
| Δ ∈ [+0.001, +0.005pt] | L4 partial. | H015 = retrieval combo 또는 더 큰 expansion (1024). |
| Δ ∈ (−0.001, +0.001pt] | L4 도 ceiling. dense expansion 효과 없음. | H015 = TWIN/SIM (retrieval mandatory) 또는 cohort H. |
| Δ < −0.001pt | 긴 seq noise. | sub-H = seq 256 uniform 또는 retrieval. |
| OOM | memory 한계. | sub-H = seq 256 uniform 또는 batch 256. |
| P6 gap > 2.5pt | cohort drift 강화. | cohort H 우선 (Frame B). |

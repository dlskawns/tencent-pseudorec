# H010 — Predictions

> Pre-registered before run. measurement objective — §17.3 binary 임계 미적용.

## P1 — Code-path success
- Quantity: train.py 10 epoch (또는 patience=3 early stop) NaN-free 완주,
  `metrics.json` 생성.
- Predicted: NaN 0건, finite val/OOF AUC.
- Falsification: NaN abort, OOM, original_baseline 코드 회귀 → REFUTED (drum-
  beat 대신 인프라 디버깅 H).

## P2 — Anchor extended ground truth (primary measurement)
- Quantity: extended envelope **platform AUC** for original_baseline
  (mechanism mutation 0).
- **§17.3 binary 임계 미적용** (measurement objective).
- **시나리오 분기 (사전 등록)**:
  - **A. envelope 효과 작음** (anchor extended ∈ [0.825, 0.840]):
    smoke anchor 0.83X 와 비슷. mechanism 효과 dominant 결론. H007/H008/H009
    의 paired Δ 거의 변경 없음. H011+ 우선순위 변경 없음.
  - **B. envelope 효과 양수 large** (anchor extended ∈ [0.840, 0.850]):
    smoke 보다 ≥ +0.5pt 위. mechanism 효과 일부가 envelope. H007 (0.8352) 가
    extended anchor 보다 **아래** 일 가능성 → H007 mechanism 효과 reassessment.
    H008 (0.8387) 도 marginal 또는 아래. **결론 대규모 재정렬**. H011+ 우선순위
    급격 변경 (mechanism class 자체 의문).
  - **C. envelope 효과 음수** (anchor extended < 0.825):
    smoke 보다 낮음. extended envelope 자체가 overfit 또는 sparse reinit 패턴
    이슈. envelope 룰 자체 재검토 필요. H011+ 부터 envelope 정의 다시 정립.
  - **D. envelope 효과 양수 small** (anchor extended ∈ [0.835, 0.840]):
    가장 likely 시나리오. H007 (0.8352) 가 anchor 와 거의 동급, H008 (0.8387)
    만 measurable lift. H011+ 부터 H008 anchor 위 single mutation 진행.
- Falsification 아님 — 4 시나리오 모두 interpretable.

## P3 — val ↔ platform 정합 (보너스)
- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05 (H006/H007 패턴 재현). val_AUC 가 platform 의 useful proxy
  검증 세 번째.
- Falsification 아님.

## P4 — §18 인프라 통과
- Quantity: inference 시 §18 룰 모두 충족 (batch heartbeat + `[infer] OK` 로그,
  no fallback).
- Predicted: PASS (original_baseline 패키지가 §18 인프라 룰 inherit).
- Falsification: P4 fail → §18 회귀 디버깅 우선.

## P5 — OOF-platform 갭 측정 (보너스)
- Quantity: |OOF_AUC − platform_AUC|.
- Predicted: ~2-3pt (H006 3.5pt → H008 1.98pt → H009 2.31pt 사이).
- Interpretation:
  - 갭 ~3.5pt → cohort effect baseline (H006 시점 동일).
  - 갭 ~2pt → mechanism 강화 없이도 cohort effect 줄어 있음 — H006-H008 의
    cohort 갭 좁아짐이 envelope 효과일 가능성.
  - 갭 ~1pt → smoke anchor 의 cohort 갭이 이미 작았음 (현재 미측정).
- Falsification 아님.

## Reproducibility
- compute_tier: T2.4 extended (10 epoch × 30%, patience=3).
- seed: 42.
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H006 (4h) ~ H008 (3.7h) 범위. patience=3 + plateau early →
  ~2-3.5시간 추정.
- code: `experiments/H010_anchor_recalibration_extended/upload/` (12 파일,
  original_baseline 코드 byte-identical + run.sh 만 envelope 변경).

## Negative-result interpretation (§17.7 falsification-first)

본 H 는 measurement objective — REFUTED 기준 미적용. 다만 부수 게이트:

- **P1 fail**: original_baseline 코드 회귀 (envelope 변경만 했는데 NaN 발생) →
  envelope 자체의 NaN 트리거 (overflow, lr × longer training 인터랙션 등) →
  envelope 디버깅 sub-H.
- **P3 fail (|val − platform| > 0.05)**: val 신뢰 안 됨. anchor 의 cohort effect
  큼. 후속 H 에서 val 의 useful proxy 가정 재검토 필요.
- **P4 fail**: §18 회귀 — original_baseline 패키지의 인프라 룰 깨짐. 인프라
  디버깅 sub-H.

## Decision tree (post-result)

| Scenario | Anchor extended | Next action |
|---|---|---|
| **A** | [0.825, 0.840] | H007/H008/H009 paired Δ 거의 변경 없음. H011 = NS→S xattn 또는 aligned `<id, weight>` pair encoding (mechanism class rotation). |
| **B** | [0.840, 0.850] | mechanism 효과 reassessment. H007 (0.8352) marginal/aligned with anchor → mechanism class 자체 의문. H011 = sub-H 로 H007/H008 단독 paired re-measure 또는 mechanism 효과 sub-decomposition. |
| **C** | < 0.825 | envelope 룰 재검토. H011 = envelope 정의 sub-H (train_ratio=0.5, num_epochs=15 등). |
| **D** | [0.835, 0.840] | most likely. H007 ≈ anchor, H008 only measurable lift. H011 = H008 anchor 위 single mutation (NS→S xattn, multi_domain_fusion, aligned pair encoding 중 하나). |

## What we won't do (single mutation discipline)

- mechanism mutation 추가 안 함.
- envelope 의 patience 외 hyperparameter (lr, dropout, batch_size) 변경 안 함.
- train_ratio 외 sampling 변경 안 함.
- val_AUC 측정만 추가, 새 metric 추가 안 함.

# H032 — Predictions

> Pre-registered before run.

## P0 — Audit gate (pre-train)

- **Quantity 1**: `dataset.py` 의 `_convert_batch` 가 4 도메인 ts column 에서 max_seq_ts 추출 가능 (`competition/dataset.py:633-648` 의 ts 처리 로직 재사용).
- **Quantity 2**: training data 의 timestamp non-null (CLAUDE.md §3 — timestamp `nullable=False` 명시). inference data 의 timestamp null 처리 = `fill_null(0)` 후 default bucket.
- **Quantity 3**: ModelInput NamedTuple 확장 후 모든 forward 호출 (train, infer, predict) 에서 일관 shape.
- Predicted: PASS — 모든 quantity 가 코드 수정만으로 충족.
- Falsification: training timestamp null 발견 → schema 재확인. P0 fail → INVALID.

## P1 — Code-path success

- Quantity: T1 1-batch forward NaN-free + 5-step train loss decreasing + cloud train 완주 (`metrics.json` 생성).
- Predicted: PASS. 새 Embedding 3개 + Linear 1개 + LayerNorm 1개 추가만.
- Falsification: NaN abort → time_emb sum scale 또는 LayerNorm 위치 issue.

## P2 — Primary lift (§17.3 binary)

- **Quantity**: extended envelope **platform AUC** vs control (TBD: H023 corrected baseline 또는 H010 corrected anchor 0.837806).

| Δ vs control | classification | mechanism interpretation |
|---|---|---|
| ≥ +0.005pt | **strong PASS** | temporal context input 이 기존 trunk 로 못 잡던 signal capture |
| [+0.001, +0.005pt] | **measurable** | 작동 marginal, sub-H (week_of_year, multi-resolution time bucket) 정당화 |
| (−0.001, +0.001pt) | **noise** | seq 내부 time bucketing + per-batch optimizer 로 implicit 학습 충분 |
| < −0.001pt | **degraded** | temporal feature cohort overfit (H011 패턴) — retract |

## P3 — Mechanism check: time_token gradient norm

- Quantity: `time_token.grad.norm(dim=-1).mean()` over training — logging.
- Predicted classifications:
  - **active** (norm > 0.5 × mean(other NS-tokens grad norm)): time signal 학습.
  - **inactive** (norm < 0.1): time embedding 학습 못 함 → P2 noise 와 일관 → mechanism dispatch fail.
- Falsification 아님 — diagnostic.

## P4 — §18 인프라 통과

- Quantity: inference 시 §18 rules 충족 + `timestamp.fill_null(0)` (§18.7 룰 준수).
- Predicted: PASS (H010 패키지 inherit + dataset-inference-auditor 서브에이전트 invoke).
- Falsification: §18 회귀 → infer.py / dataset.py 의 timestamp 처리 디버깅.

## P5 — val ↔ platform gap (보너스)

- Quantity: `best_val_AUC − platform_AUC`.
- Predicted: ∈ [−0.005, +0.005pt] (H010 ~−0.003pt 패턴).
- Falsification 아님.

## P6 — OOF (redefined future-only) ↔ platform gap

- Quantity: redefined OOF AUC − platform AUC.
- Predicted: ∈ [−0.005, +0.005pt] (H016 framework 검증).
- Falsification 아님.

## P7 — verify-claim §18.8 SUMMARY parser dry-run

- Quantity: train.py 끝 SUMMARY block 정규식 파싱 PASS.
- Predicted: PASS (H018 framework inherit).

## Reproducibility

- compute_tier: T2.4 extended (10ep × 30%, batch=1024 OOM safety, patience=3) ~3-4시간.
- seed: 42 (H028 결과 보고 multi-seed 결정).
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H010 (3.7h) 동급 또는 약간 길어짐 (forward path 1 token 추가).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P2 noise**: seq 내부 time bucketing 이 implicit 학습 충분. **input-axis temporal signal retire**. carry-forward: H015~H018 결과와 합쳐 "temporal axis 전반 retire" 가능성.
- **P2 degraded**: cohort overfit (H011 패턴). retract — sub-H = time embedding dimension 8 로 축소 또는 LayerNorm 강도 증가.
- **P2 measurable**: partial PASS — sub-H (week_of_year, gap_to_label_time as inference-safe form) 진행. anchor 갱신 보류.
- **P3 inactive + P2 noise**: time_token gradient flow 막힘. init scale / dropout 검토 후 sub-H.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.005pt + P0/P1/P3/P4 PASS | **strong PASS — anchor 갱신**: H032 = 새 baseline. H031 (item_13 head) 와 stack 또는 multi-resolution time. |
| Δ ∈ [+0.001, +0.005pt] | measurable PASS. sub-H (week_of_year). anchor 갱신 검토. |
| Δ ∈ (−0.001, +0.001pt) | noise. anchor 유지. **input-axis temporal signal retire**. |
| Δ < −0.001pt | degraded. retract — sub-H (smaller dim, stronger LN). |
| P0 fail | INVALID. timestamp 처리 재확인. |
| P3 inactive | mechanism dispatch issue. init scale / gradient flow 디버깅 후 sub-H. |

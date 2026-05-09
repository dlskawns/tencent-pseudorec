# H046 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- Sequential: H019 의 TWIN GSU+ESU + per-domain encoder carry.
- Interaction: NS→S xattn + DCN-V2 carry.
- Cohort-invariance (NEW): proper DANN — backbone forced to encode features
  invariant to temporal cohort. 같은 unified backbone graph 안 main BCE +
  adversarial cohort task 동시 gradient 공유.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- F-A 패턴 9 H 누적 + H019/H038/H039 platform 비교가 정량 confirm:
  cohort drift = transfer 실패 root cause.
- H044 fail 의 분석에서 design 결함 발견 → proper DANN structure 로 retry.
- mechanism 정당성: 본 데이터 9 H signal + 표준 DANN structure 정확성.

## What's NOT a clone
- 외부 DANN paper 의 image domain adaptation 과 다름 — temporal cohort
  (label_time) 사용, image domain 아님.
- H044 와 다름 — proper structure (separate cohort_head + GRL between
  backbone and cohort_head, not on raw_logits).

## Carry-forward
- §17.2 single mutation: cohort_head + GRL.
- §17.4: cohort_drift_attack axis (H044 retry).
- §10.6: trainable params +65.
- §0.5: F-A 패턴 직접 attack.

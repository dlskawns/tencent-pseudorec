# H044 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- **Sequential axis**: H019 의 TWIN GSU+ESU + per-domain encoder 그대로 carry.
- **Interaction axis**: NS→S xattn + DCN-V2 stack 그대로 carry.
- **Adversarial axis (NEW)**: backbone output 이 두 task 동시 학습 — main
  BCE (positive grad) + aux timestamp MSE (negative grad via GRL). 같은
  unified backbone graph 에서 두 task 가 gradient 공유, sequential ×
  interaction 통합 위에 cohort-invariance constraint 부과.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- Signal: F-A 패턴 (9 H 누적, OOF over Platform 1.88~2.59pt) + H019/H038
  비교가 cohort drift 가 platform transfer 실패의 근본 원인 정량 confirm.
- Diagnose: H038 의 aux MSE positive direction 이 OOF +0.0012 → Platform
  −0.0006 (cohort overfit). H044 = 역방향 (forget timestamp = forget
  cohort proxy).
- Mechanism design: GradReverse autograd Function (Ganin & Lempitsky 2015
  DANN form), 본 데이터의 F-A 패턴 signal 직접 attack. 외부 paper transplant
  아니라 — 본 9 H 의 OOF/Platform divergence 측정이 정당화.
- Validation chain: T0 sanity 5 test PASS (forward identity + grad reversal
  math + mse integration + lambda=0/1 boundary).

## What's NOT a clone
- 외부 DANN paper 의 cross-domain image classification (MNIST → MNIST-M)
  과 다름 — 본 case 은 *temporal* cohort (label_time 기반), domain 분류
  아님.
- Domain Generalization 의 일반 form 과 다름 — explicit domain label 없이
  timestamp 를 cohort proxy 로 사용. F-A 패턴 의 직접 측정에서 도출된
  cohort 정의.
- 외부 paper transplant 0. mechanism 정당성은 본 9 H 의 F-A 패턴 누적
  signal + H038 transfer fail 의 직접 데이터 facts 에서 옴.

## Carry-forward
- §17.2 single mutation: trainer.py 에 GRL + branch 추가, model graph
  byte-identical to H038, infer.py byte-identical to H038.
- §17.4 rotation: cohort_drift_attack axis NEW first-touch, AUTO_JUSTIFIED.
- §10.6 sample budget: trainable params +0 (loss path 만 변경, no new module).
- §0.5 data-signal-driven: F-A 패턴 직접 attack, 본 데이터 signal 의 진짜
  ceiling 원인 추정 attack.

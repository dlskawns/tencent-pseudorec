# H047 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- Sequential: H019 의 TWIN GSU+ESU + per-domain encoder carry.
- Interaction: NS→S xattn + DCN-V2 carry.
- Multi-task (NEW): main + 4 per-domain aux head 가 같은 backbone 위 gradient
  공유. backbone 이 4 도메인 모두에 useful 한 representation 강제.

## Why this mechanism for THIS data
- §3.5 4 도메인 disjoint vocab + per-domain seq length 분포 imbalanced.
- 14 H 모두 single-head supervision → balanced per-domain signal 부재 가능.
- H012 MoE (uniform routing) 와 다름 — explicit prediction supervision per
  domain, routing 학습 안 함.

## What's NOT a clone
- 외부 multi-task learning paper 의 hard parameter sharing 과 다름 — 모든
  task 가 같은 label 예측 (다른 task 아님). per-domain head 가 단순히 더
  정밀한 supervision provider.
- 외부 paper transplant 0.

## Carry-forward
- §17.2 single mutation: per-domain heads.
- §17.4: multi_task_per_domain axis NEW first-touch.
- §10.6: trainable params +260.
- §0.5: §3.5 도메인 imbalance signal direct attack.

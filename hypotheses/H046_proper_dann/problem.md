# H046 — problem.md

## Trigger
H044 (GRL on raw_logits[:, 1]) **design 결함 by me** — clsfier-aux head + backbone
둘 다 reverse → no actual discriminator → loss 발산 (epoch 1: 4.93, epoch 5: 545).
표준 DANN 은 separate cohort_head (positive grad) + backbone (reversed grad).
H046 = proper DANN structure 로 H044 retry.

## Data signal (CLAUDE.md §3.4 + F-A 9 H)
- F-A 패턴 9 H 누적 (OOF over Platform 1.88~2.59pt) + H019/H038/H039 platform
  비교가 cohort drift = transfer 실패 root cause confirm.
- timestamp = cohort proxy (label_time-based split).

## Hypothesis
backbone 이 timestamp 를 못 예측하도록 표준 DANN (separate discriminator + GRL)
로 강제 → cohort-invariant feature → platform transfer 향상.

## Mutation
- model.py 에 _GradReverse(autograd.Function) 추가.
- 별도 cohort_head = nn.Linear(d_model, 1).
- forward: GRL(output) → cohort_head → cohort_pred (tuple return).
- trainer: isinstance(out, tuple) 체크 후 cohort MSE 추가.
- dann_cohort_lambda=0.1 (H044 0.5 의 1/5, 보수적).
- 65 trainable params 추가.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → DANN cohort attack 작동. 미달
시 timestamp cohort proxy 부족 (다른 cohort label 필요) 또는 DANN form 본
setup 에 안 맞음.

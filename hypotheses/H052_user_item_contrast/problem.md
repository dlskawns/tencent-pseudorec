# H052 — problem.md

## Trigger
14 H 모두 pointwise BCE 단독 supervision. H040 BPR (logit pairwise) REFUTED.
representation-level contrastive (SSL form) 0회 시도 — paradigm-shift candidate.

## Data signal
- §3.5 target_in_history=0.4% — user 가 본 적 없는 item 추천. user_repr 의
  discriminative quality 가 cold-item generalization 핵심.
- in-batch negatives (B=1024) → 1023 negative per positive, 충분한 contrastive signal.

## Hypothesis
user_repr 와 item_repr 의 InfoNCE alignment 가 user representation 을 다른
user 의 item 으로부터 discriminative 학습 → platform AUC lift.

## Mutation
- model.py: forward tuple return (logits, user_repr=backbone_output, item_repr=item_ns mean).
- trainer.py: InfoNCE auxiliary loss = cross_entropy on (B, B) sim matrix, target=diag.
- contrast_lambda=0.1, contrast_temperature=0.1.
- 0 trainable params 추가.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → SSL contrastive supervision lever.

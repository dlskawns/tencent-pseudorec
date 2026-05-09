# H052 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- Sequential / Interaction: H019 carry.
- **Contrastive aux (NEW)**: backbone output 위 SSL-style InfoNCE supervision.

## Why this mechanism
- 14 H 동안 contrastive supervision 0회.
- H040 BPR REFUTED 는 logit-level pairwise — H052 는 representation-level
  in-batch contrastive (SimCLR/CLIP form).
- §3.5 cold-item task = discriminative user representation 핵심.

## What's NOT a clone
- 외부 SimCLR/CLIP 와 다름 — augmentation 없는 단일 view 의 in-batch
  contrastive (CLIP-lite).
- H040 BPR 와 다름 — representation level (after backbone), pairwise 가 아닌
  multi-class softmax (in-batch all negatives).

## Carry-forward
- §17.2: model + trainer 둘 다 변경 (single mechanism).
- §17.4: representation_contrastive axis NEW.
- §10.6: trainable params 0.
- **infer.py flag parity** verified.

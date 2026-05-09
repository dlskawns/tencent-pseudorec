# H047 — problem.md

## Trigger
14 H 동안 main classifier 1개 만 사용, per-domain prediction multi-task 0회.
backbone 이 4 도메인 정보를 *대표 1개 logit* 로 압축 → per-domain signal
mixing 손실 가능. H012 MoE (uniform routing REFUTED) 와 다른 form — explicit
per-domain prediction supervision.

## Data signal (CLAUDE.md §3.5)
- 4 도메인 disjoint vocab + per-domain seq length imbalance (frac_empty:
  a 0.5%, b 1.2%, c 0.2%, d 8% bimodal).
- 각 도메인이 독립적 prediction 신호 가질 가능성, single-head architecture
  로는 balanced supervision 안 됨.

## Hypothesis
4 per-domain heads 가 각자 같은 main label 예측 → backbone 이 per-domain
prediction balanced 학습 → cold-domain (e.g., domain d) 에서도 prediction
유지 → platform AUC 향상.

## Mutation
- model.py 에 4 per_domain_aux_heads ModuleList 추가 (각 nn.Linear(d_model, 1)).
- forward: per-domain seq tokens 의 masked mean-pool → per-domain head → aux_logit.
  tuple (logits, per_domain_aux_list) return.
- trainer: tuple 처리 + per-domain BCE 합 / 4, weight 0.25 (4 × 0.25 = 1.0 vs main).
- 260 trainable params 추가.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → per-domain multi-task supervision
가 lever. 미달 시 main head 가 이미 per-domain signal 충분 학습 또는 aux
weight 잘못.

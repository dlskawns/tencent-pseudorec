# H045 — problem.md

## Trigger
14 H 동안 4 도메인 cross-domain feature transfer 0회. TWIN aggregator 는
per-domain 독립 처리 후 단순 mean. per-domain encoder 도 독립. 도메인 간
정보 흐름 mechanism 부재.

## Data signal (CLAUDE.md §3.5)
- 4 도메인 jaccard intersection: a-b 7.9%, a-d 10.0%, b-d 2.2%, c-d 1.6%, a-c
  0.7%, b-c 1.0% — disjoint 같지만 0.7~10% overlap.
- 0.4% target_in_history (any domain) — user 가 본 적 없는 item 추천 task,
  cross-domain semantic transfer 가 cold-item generalization 의 lever 가능성.

## Hypothesis
4 도메인 encoder output 사이 explicit MultiheadAttention bridge → 도메인 간
feature transfer 학습 → user 의 다른 도메인 활동 정보가 prediction 에 활용
→ platform AUC lift.

## Mutation
- model.py 에 CrossDomainBridge class 추가 (4-head MHA on (B, 4, D) 도메인
  토큰).
- per-domain mean-pool → stack → MHA → mean → Linear+LN+gated residual ADD.
- gate init sigmoid(-2)≈0.12 (§10.10 InterFormer pattern).
- TWIN 직전에 적용 → TWIN candidate attention 이 bridge-aware base 위 동작.
- trainer.py / dataset.py / infer.py / utils.py byte-identical to H019.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → cross-domain axis lever 작동.
미달 시 mean-pool 이 cross-domain 정보 압축 한계, sub-H 형식 (per-token
cross-attn 또는 stacked bridge) 또는 axis pivot.

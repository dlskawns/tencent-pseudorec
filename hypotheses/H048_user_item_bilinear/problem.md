# H048 — problem.md

## Trigger
14 H 분석 결론: PASS mechanism (H010 NS→S xattn +0.0024pt, H019 TWIN candidate
Q +0.0019pt) 모두 *새 정보 흐름 추가* 패턴. 변조/regularization (H011-H018,
H038, H041, H042, H044) = 모두 NOOP. **새 pathway = lever**.

14 H 동안 user representation × item representation 사이 explicit bilinear
interaction **0회**. TWIN 은 item → user_history (한 방향). DCN-V2 는
within-side cross. user × item DIRECT interaction NEW pathway.

## Data signal
- §3.5 4 도메인 disjoint vocab + target_in_history=0.4% — user 가 본 적 없는
  item 추천 task → user × item explicit cross 가 전체 task 핵심.
- 14 H 모두 user representation 만 modeling, item 은 input 단계 entanglement
  뿐 → top-level cross 부재.

## Hypothesis
backbone output (user_repr) 와 item_repr (item_ns mean) 사이 element-wise
bilinear cross 가 학습 → user × item 직접 신호 추가 → platform AUC lift.

## Mutation
- model.py 에 UserItemBilinearCross class (FM-style cross) 추가.
- forward + predict 에서 backbone output 위 residual ADD (TWIN 직전).
- ~8K trainable params 추가.
- gate init sigmoid(-2)≈0.12 (§10.10).
- trainer.py / dataset.py / utils.py byte-identical to H019.
- infer.py flag 추가 (H043 사고 방지).

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → user × item axis lever. 미달
시 bilinear form 이 부족 (multi-layer 또는 다른 form 필요) 또는 axis 무용.

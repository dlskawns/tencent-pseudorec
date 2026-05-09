# H049 — problem.md

## Trigger
14 H 동안 NS architecture 자체 0회 변경. item_ns_tokens 모든 H 에서
default=2, NS slot type 차별 embedding 0개. 사용자가 직접 지적: "왜 s token,
ns token 은 안 바꾸냐". 합리적 axis — backbone input level 변경, mutation
효과가 모든 후속 layer 에 propagate.

## Data signal
- §3.5 4 도메인 disjoint vocab + target_in_history=0.4% — item 이 prediction
  target 인데 NS 단계에서 2 토큰 만으로 representation = slot 부족 가설.
- §0 P1 의 "두 축 동시 같은 backbone gradient 공유" 룰 — NS structure 가
  바로 그 통합 지점, refinement 가 직접 영향.

## Hypothesis
NS architecture refinement (item_ns_tokens 2→6 + 4-type slot embedding) →
backbone 이 type-aware NS processing + 풍부한 item representation 학습 →
platform AUC lift.

## Mutation
- model.py: ns_type_emb (4-class Embedding) + register_buffer ns_type_ids +
  _build_token_streams 에서 broadcast ADD.
- run.sh: --item_ns_tokens 6 + --use_ns_slot_type_emb.
- ~16K trainable params 추가.
- trainer.py / dataset.py / utils.py byte-identical to H019.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → NS architecture axis lever.
미달 시 NS structure 변경 만으로 부족 또는 더 강한 form 필요.

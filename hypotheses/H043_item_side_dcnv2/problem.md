# H043 — problem.md

## Trigger
H038 platform 0.839071 vs H019 0.839674 = Δ −0.000603pt. OOF +0.0012 → Platform
−0.0006 (F-A 패턴 재발). output-distribution-supervision axis 가 platform 으로
안 transfer → 다음 H 는 truly orthogonal axis 필요.

14 H 의 axis map 전수조사:
- user-side input / temporal / hyperparameter / loss / retrieval / output-dist
  모두 시도됨.
- **item-side modeling 0회** — item_int 13 scalar + 1 array 가 nn.Embedding
  lookup 후 NS-token 으로 직행, feature interaction 한 번도 explicit 안 함.

## Data signal (CLAUDE.md §3.5)
- 4 도메인 disjoint vocab (jaccard 0.7~10%): item 자체의 cross-domain semantic
  modeling 부재.
- target_in_history=0.4%: 99.6% prediction 이 user 가 본 적 없는 item — item
  representation 의 수준이 cold case generalization 의 lever.
- user-side DCN-V2 (H008) PASS +0.0035pt — 같은 mechanism 의 item-side first
  application 의 정당성.

## Hypothesis
item_ns 토큰 (B, num_item_ns=2, D=64) 에 DCN-V2 polynomial cross 적용 →
item feature 들 사이의 high-order interaction 학습 → user-side modeling 만으론
부족한 cold-item representation 강화 → platform AUC lift.

## Mutation
- model.py PCVRHyFormer __init__ 에 use_item_side_cross 분기 추가.
- DCNV2CrossBlock(d_model=64, num_cross_layers=2, rank=4) 빌드.
- _build_token_streams 에서 item_ns_tokenizer call 직후 cross 적용.
- 1,280 trainable params 추가.
- trainer.py / dataset.py / infer.py / utils.py byte-identical to H019.

## Falsifiable
Δ vs H019 platform (0.839674) ≥ +0.001pt → item-side axis 가 platform 으로
transfer 되는 lever. 미달 시 modest mechanism 으론 안 됨, 더 강한 form
(item-side cross-domain attention 또는 SSL pretraining) 또는 axis pivot.

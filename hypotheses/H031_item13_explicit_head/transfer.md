# H031 — Method Transfer

## Sources

- **Wang et al. 2017** "Deep & Cross Network for Ad Click Predictions" — explicit cross at feature level via $x_0 x_l^T$.
- **Cheng et al. 2016** "Wide & Deep Learning for Recommender Systems" — wide path 가 strong categorical feature 에 explicit 가중치 학습.
- **Zhou et al. 2018** "Deep Interest Network for Click-Through Rate Prediction" — candidate item × user history attention. 본 H031 은 candidate item 의 single strongest categorical (item_13) 을 user-side small-vocab features 와 outer-product → DIN 의 fine-grain 변형.
- **OneTrans (arXiv:2510.26104) NS-token framework** — 본 프로젝트 anchor (H010). NS-token enrichment 가 anchor 입력 byte-identical 유지 시 safe stacking (H010 F-1 carry-forward).

## Mechanism — single mutation

H010 anchor (NS xattn + DCN-V2 + RankMixer NS tokenizer) byte-identical EXCEPT:

1. **item_int_feats_13 dedicated embedding**: `nn.Embedding(10, 32)`. RankMixerNSTokenizer 의 item-side group 에서 fid=13 contribution 제거 (해당 dim 의 weight 0 으로 mask 또는 group 재구성).
2. **user_int dedicated embeddings (5개)**: 각 fid 별 `nn.Embedding(vs_fid, 16)` for fid ∈ {1, 97, 58, 49, 95}. vs ≤ 6 이라 총 ~384 params.
3. **Outer-product cross**: 각 user_int_k embed_k (16-dim) 와 item_13 embed (32-dim) 의 outer product `embed_k.unsqueeze(-1) * item_13_embed.unsqueeze(-2)` → mean over (vs_k, 32) → 32-dim. 5 user features → 5 outer-products → mean-stack → 1 cross-token (32-dim).
4. **Extra NS-token 합류**: `_make_extra_ns_token = Linear(32, d_model)`. user_ns + item_ns + cross_token concat → DCN-V2 input (NS-token 1개 추가).

## §17.2 single mutation compliance

1 mutation = "item_13 + 5 user_int 의 explicit fid-level cross block 추가". 다른 mechanism (NS xattn / DCN-V2 / TransformerEncoder / seq) 변경 없음.

## §⑤ UNI-REC alignment (P1+ 의무)

- **Sequential axis**: 변경 없음 (TransformerEncoder + MultiSeqHyFormer 그대로).
- **Interaction axis**: **fid-level explicit cross 신규 추가** (block-level H008 DCN-V2 위 stacking).
- **Bridging mechanism**: 추가 cross-token 이 NS xattn (H010) → DCN-V2 → 분류 head 의 같은 trunk 합류 → seq encoder 출력 (NS-tokens) 과 **같은 cross block 에서 gradient 공유** (P1 룰 ✅).
- **What's not a clone**: DCN-V2 의 block-level cross 와 다름 — fid-level pre-cross 후 NS-token 형태로 합류. DIN 의 candidate-history attention 과도 다름 — history 가 아닌 user_int categorical 과 cross.

## §10 Anti-bias rules audit

- **§10.5 LayerNorm on x_0**: cross-token 합류 직전 `nn.LayerNorm(d_model)` 적용. ✅
- **§10.6 sample-scale param budget ≤ N/10 = 100~200 params**: 추가 params ~ 720 (32×10 + 16×(6+5+4+4+5) + Linear 32×d_model). **위반 가능** — 1000-row sample 에서 trainable params hard cap 200. **Action**: T1 sanity skip, 직접 T2 cloud (1.5M+ rows) 에서만 학습. local sample run 금지.
- **§10.7 category rotation**: 직전 2 H primary_category — H030 (measurement), H029 (measurement). H031 = `feature_engineering` (또는 `interaction_explicit`) — 신규 first-touch.
- **§10.8 continuous-scouting**: 본 transfer.md 가 DCN-V2 + Wide&Deep + DIN 3 paper 동시 인용 → multiple cat scouting 충족.

## Data-grounding (왜 이 user_int 5개)

이번 세션 EDA — hot_item_13 (i13 ∈ {4,7,8}) vs cold (else) 분류 univariate AUC 가장 큰 user_int (small vocab only):

| fid | vs | hot_AUC | |Δ| |
|---|---|---|---|
| 1 | 6 | 0.5617 | 0.0617 |
| 97 | 5 | 0.5552 | 0.0552 |
| 58 | 4 | 0.4546 | 0.0454 |
| 49 | 4 | 0.5425 | 0.0425 |
| 95 | 5 | 0.5346 | 0.0346 |

5개 모두 schema_vs ≤ 6 → outer-product cross matrix size 작음 (총 ~24 cross cells × 32 cross_dim ≈ 768 params).

## Implementation risk

- **High**: model.py 의 `RankMixerNSTokenizer.forward` 수정 필요 (item-side group 에서 fid=13 제외 + user-side group 에서 5 fids 제외 + 새 Embedding/Linear 추가). 1714-line model.py 의 forward path 변경 → side-effect risk.
- **Mitigation**: H019 / H018 처럼 upload/ build 는 별도 세션에서 careful local sanity (T0 1000-row forward pass NaN-free + param count match) 후 진행. 본 scaffold 단계는 hypothesis docs + card.yaml 만.

## Local sanity test plan (upload build 시 mandatory)

1. T0: `.venv-arm64/bin/python -c "from model import PCVRHyFormer; m=PCVRHyFormer(...); print(sum(p.numel() for p in m.parameters()))"` — params 추가량 확인 (~1.5K 예상)
2. T1: 1-batch forward pass → output shape (B, 1) NaN-free
3. T1: backward pass → grad on new embeddings non-zero
4. T1: 5-step train → loss decreasing
5. dataset-inference-auditor 서브에이전트 invoke (CLAUDE.md §19)

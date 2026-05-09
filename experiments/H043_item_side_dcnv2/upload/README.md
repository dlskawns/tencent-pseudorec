# H043_item_side_dcnv2 — Technical Report

> CLAUDE.md §0.5 data-signal-driven: 14 H 동안 **item-side modeling 0회**.
> item_int 13 scalar + 1 array 가 nn.Embedding lookup 후 NS-token 으로
> 직행, explicit feature interaction 한 번도 안 함. user-side DCN-V2 (H008
> PASS +0.0035pt) 의 mechanism 을 item-side 에 first 적용. §3.5 4 도메인
> disjoint vocab + target_in_history=0.4% 가 item 측 cross-domain semantic
> modeling 의 데이터 motivation.

## 1. Hypothesis & Claim
- Hypothesis: item_ns 토큰 (B, num_item_ns, D) 에 DCN-V2 polynomial cross
  를 explicit 적용 시, item-side feature 사이의 high-order interaction 이
  학습되어 lift. 14 H 모두 user/global side 만 mechanism 추가 — item
  representation 깊이가 진짜 lever 일 가능성.
- Falsifiable: Δ vs H019 (Val 0.83720 / OOF 0.8611 / Platform 0.839674) ≥
  +0.001pt → item-side axis 가 lever, sub-H 가치.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — DCN-V2 cross on item_ns

```python
# In _build_token_streams, after item_ns_tokenizer call:
item_ns = self.item_ns_tokenizer(inputs.item_int_feats)  # (B, num_item_ns=2, D=64)
if self.use_item_side_cross:
    item_ns = self.item_side_cross(item_ns)              # H043 NEW
ns_parts.append(item_ns)                                  # 기존 흐름
```

`item_side_cross = DCNV2CrossBlock(d_model=64, num_cross_layers=2, rank=4, dropout)`.

DCN-V2 form: `x_l+1 = x_0 * (V_l(U_l(x_l))) + x_l` with Pre-LN on x_0 (§10.5
mandatory). 같은 form 이 user/global side fusion 에서 작동했음 (H008 PASS),
item side 첫 적용.

**왜 이게 진짜 NEW**:
1. 14 H 모두 item-side mechanism 0회. 정확히 §0 의 두 축 중 second axis
   의 *item 측면*.
2. §3.5 4 도메인 disjoint vocab (jaccard 0.7~10%) + target_in_history=0.4%
   → user-side modeling 만으로 cold item 에 대한 representation 부족 가설.
3. H019 retrieval 는 *user 의 history → item* 정보 흐름만 modeling,
   item 자체의 representation 깊이는 안 건드림.

**왜 H042 와 직교 hedge**:
- H042 = output distribution constraint (output level)
- H043 = item representation enrichment (input level)
- 둘 다 fail 시 → next-paradigm (SSL pretraining / GNN / generative)

## 3. Decision tree (post-result)

| Δ vs H019 (Val 0.83720 / OOF 0.8611 / Platform 0.839674) | Action |
|---|---|
| Platform Δ ≥ +0.003pt | item-side axis main lever, sub-H = layers/rank sweep + item_dense_tok 도 cross |
| Platform Δ ∈ [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 |
| Platform Δ ∈ (−0.001, +0.001pt] | item-side modest mechanism 으론 lever 아님 — 더 강한 form (item-side cross-domain attention) 또는 axis pivot |
| Platform Δ < −0.001pt | item-side cross 가 noise, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 3 __init__ params (use_item_side_cross / layers / rank), + 1 module init block (~10 lines), + 1 forward call (~3 lines in _build_token_streams) | Model |
| `train.py` | + 3 argparse + 3 plumbing line | CLI |
| `run.sh` | + `--use_item_side_cross --item_side_cross_layers 2 --item_side_cross_rank 4` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **1,280** (rank=4, 2-layer DCN-V2 on D=64) = 161M total 의 0.0008%, sample budget 안.

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS for model.py + train.py
2. ✅ shellcheck PASS for run.sh
3. ✅ DCNV2CrossBlock forward shape (B=4, T=2, D=64) preserved
4. ✅ NaN-free output
5. ✅ trainable params = 1,280
6. ✅ gradient flow: |grad|.sum=333.08
7. ✅ ablation diff (cross output vs input) = 4.77 (작동 confirm)
8. ✅ md5 verify: 6 unchanged files identical to H019

## 6. Carry-forward
- §17.2 single mutation: item_ns 토큰에 DCN-V2 cross 추가. user/global side / TWIN / NS xattn / 다른 모든 부분 byte-identical to H019.
- §17.4 rotation: NEW first-touch (item_side_modeling axis), AUTO_JUSTIFIED.
- §10.5 LayerNorm on x_0 mandate: DCNV2CrossBlock 의 Pre-LN 자동 통과.
- §10.6 sample budget: trainable params +1.28K (sample-scale 친화).
- §0.5 data-signal-driven: §3.5 4 도메인 disjoint vocab + target_in_history=0.4% 의 item 측 motivation, paper transplant 0.
- §0 P1 룰: same backbone graph 안 user-side + item-side mechanism 둘 다 gradient 공유 (Sequential × Interaction 통합 강화).
- §18.7 + §18.8 H019 carry.

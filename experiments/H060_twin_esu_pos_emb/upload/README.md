# H060_twin_esu_pos_emb — Technical Report

> H057 SUCCESS TEMPLATE-DRIVEN: TWIN-internal mutation, single residual preserved.
> H060 = TWIN ESU 의 top-K MHA 가 rank-aware (1st-top vs 64th-top 차별).
> 현재 ESU 는 "bag of top-K" — 위치 정보 없음. H060 = learnable pos emb 추가.

## 1. Hypothesis & Claim
- Hypothesis: ESU MHA 가 top-K 의 *순위* 인지 시 더 informative attention.
  현재 candidate Q 가 top-K 를 attend 할 때 1st-top 과 64th-top 구별 못함
  → MHA 가 위치 무관하게 처리. Pos emb 가 candidate Q 의 attention bias 에
  rank info 제공.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation
```python
# In TWINBlock.forward, after top-K gather, before ESU MHA:
topk_history = history.gather(1, idx_exp)  # (B, K, D)
if self.use_pos_emb:
    pos_ids = arange(K)
    pos_emb = self.pos_emb(pos_ids)        # (K, D)
    topk_history = topk_history + pos_emb.unsqueeze(0)
# ESU MHA on rank-aware top-K
attended = self.esu(candidate_q, topk_history, topk_history, ...)
```
- pos_emb = nn.Embedding(top_k=64, D=64), 4 도메인 별
- params 추가: 4 × 64 × 64 = ~16K

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | rank-aware ESU main lever, sub-H = RoPE 또는 multi-position |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | top-K rank info 부족, candidate-distance encoding |
| < -0.001pt | retire |

## 4. Files
| File | H019 대비 |
|---|---|
| model.py | + use_pos_emb param in TWINBlock + pos_emb init + forward apply, + use_twin_esu_pos_emb in PCVRHyFormer __init__ |
| train.py | +1 argparse +1 plumbing |
| infer.py | +1 cfg.get (H043 방지) |
| run.sh | + flag |
| 외 .py | byte-identical |

trainable params 추가: ~16K.

## 5. Carry-forward
- §17.2 single mutation: TWIN ESU positional embedding.
- §17.4: rank_aware_esu axis (H057 success template, ESU layer).
- §10.6: trainable params +16K.
- **infer.py flag parity** verified.

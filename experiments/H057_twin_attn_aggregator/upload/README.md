# H057_twin_attn_aggregator — Technical Report

> H019 TWIN aggregator 의 uniform mean → per-user per-domain attention weighted.
> §3.5 4 도메인 imbalance (domain d 8% empty, c outlier max 3894) 직접 attack.
> TWIN-internal mutation, single residual 유지 (H048 fail mode 회피).

## 1. Hypothesis & Claim
- Hypothesis: 4 per-domain TWIN states 의 uniform mean 이 imbalanced domains
  에서 noise 큰 도메인도 동등 가중 → per-user attention weighted aggregation
  이 더 informative twin_state → platform AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation — TWIN aggregator attention

```python
# H019 (uniform mean):
stacked = stack(per_domain_states, dim=1)        # (B, 4, D)
pooled = stacked.mean(dim=1)                      # uniform

# H057 (attention weighted):
stacked = stack(per_domain_states, dim=1)        # (B, 4, D)
attn_logits = self.attn_weights(stacked).squeeze(-1)  # (B, 4)
weights = softmax(attn_logits, dim=-1)           # (B, 4) per-user per-domain
pooled = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # weighted sum
```

**Single residual preserved**: TWIN's gated residual ADD on output unchanged. Only aggregator internal mean→attention.

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | per-domain attention lever, sub-H = multi-head attention or domain-specific Q |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | uniform mean 충분, 4-domain 으로 attention 학습 어려움 |
| < -0.001pt | retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + use_attn flag in TwinRetrievalAggregator class, + 1 PCVRHyFormer __init__ param, + flag pass | Model |
| `train.py` | + 1 argparse + 1 plumbing | CLI |
| `infer.py` | + 1 cfg.get() (H043 방지) | Inference |
| `run.sh` | + `--use_twin_attn_aggregator` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / utils.py / local_validate.py / make_schema.py` | byte-identical | unchanged |

trainable params 추가: **~64** (Linear(D=64, 1)).

## 5. Carry-forward
- §17.2 single mutation: TWIN aggregator mean → attention.
- §17.4: twin_aggregator_attention axis NEW first-touch.
- §10.6: trainable params +64 (sample budget OK).
- §0.5: §3.5 4 도메인 imbalance signal 직접 attack.
- **infer.py flag parity** verified.
- **H048 fail mode 회피**: single residual 유지, output 자체 unchanged.

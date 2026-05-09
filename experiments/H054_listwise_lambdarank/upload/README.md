# H054_listwise_lambdarank — Technical Report

> PARADIGM SHIFT (objective): 14 H 모두 sample-level (BCE/focal). H040 BPR
> (uniform pairwise) REFUTED. H054 = LambdaRank-lite — pairwise weighted by
> misorder severity. Platform metric = AUC = ranking quality, 직접 최적화.

## 1. Hypothesis & Claim
- Hypothesis: lambda-weighted pairwise (sigmoid(-(z_pos - z_neg)) → focus on
  misranked pairs) 가 BPR uniform 보다 효과적 + BCE 와 balance 시 platform
  AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation — LambdaRank-lite

```python
# trainer:
if self.lambdarank_lambda > 0:
    pos_logits = logits[label > 0.5]              # (P,)
    neg_logits = logits[label <= 0.5]             # (N,)
    diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # (P, N)
    # Lambda weight (no grad): high for misranked (pos < neg)
    weight = sigmoid(-diff.detach())              # (P, N)
    pair_loss = -log_sigmoid(diff) * weight
    listwise_loss = pair_loss.mean()
    loss = bce + 0.1 * listwise_loss
```

**왜 paradigm shift**:
- 14 H sample-level loss only (BCE/focal/MSE/KLD).
- H040 BPR REFUTED — uniform pairwise.
- H054 = lambda-weighted (LambdaRank/RankNet form). Hard pair focus = AUC 직접 attack.

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | listwise main lever, sub-H = NDCG-aware weighting |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | weighting form 부족 |
| < -0.001pt | listwise fight BCE, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `trainer.py` | + 1 __init__ param, + LambdaRank loss block | Training |
| `train.py` | + 1 argparse + 1 plumbing | CLI |
| `run.sh` | + `--lambdarank_lambda 0.1` | Entry |
| `README.md` | new | Doc |
| **model.py / dataset.py / infer.py / utils.py / make_schema.py / local_validate.py** | byte-identical | unchanged |

trainable params 추가: **0**. **No model weight change → infer.py flag 변경 불필요** (state_dict 호환).

## 5. Carry-forward
- §17.2 single mutation: LambdaRank loss term.
- §17.4: ranking_objective_lambda axis NEW first-touch.
- §0.5: Platform AUC metric 직접 최적화, paper transplant 0 (RankNet/LambdaRank form 의 단순 적용).

# H040_pairwise_ranking — Technical Report

> CLAUDE.md §0.5 data-signal-driven (R4 reframe): platform metric = AUC =
> ranking quality. 18 H 모두 pointwise BCE 사용 — BCE 는 *sample-level loss*
> 이지 *ranking-level objective* 아님. AUC 직접 최적화 빠진 명백한 blind spot.
> Mutation: BCE → BPR pairwise. 코드 변경 = trainer.py + train.py + run.sh,
> model.py / dataset.py / infer.py byte-identical to H019 (md5 verified).

## 1. Hypothesis & Claim
- Hypothesis: AUC 직접 최적화 (pairwise ranking) 가 pointwise BCE 보다 platform AUC 에서 lift.
- Falsifiable: Δ vs H019 (0.83967) ≥ +0.002pt → ranking objective 가 lever.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — BPR pairwise loss

```python
# Per training batch (B=1024, ~12.4% positive ≈ 127 positives, 897 negatives):
pos_logits = logits[label > 0.5]                  # (P,) — positive samples
neg_logits = logits[label <= 0.5]                 # (N,) — negative samples
neg_idx = randint(0, N, (P, K=4))                 # K=4 random negatives per positive
sampled_neg = neg_logits[neg_idx]                 # (P, K)
diff = pos_logits.unsqueeze(1) - sampled_neg      # (P, K) — (z_pos - z_neg)
loss = -log(sigmoid(diff)).mean()                 # BPR loss

# Edge case fallback: if batch has 0 pos OR 0 neg → fall back to BCE.
```

→ 학습이 (positive, negative) pair 의 *상대 순위* 를 직접 학습. AUC = ranking AUC 와 직접 일치.

## 3. Decision tree (post-result)

| Δ vs H019 (0.83967) | Action |
|---|---|
| ≥ +0.005pt | ranking objective 가 main lever, anchor = H040, sub-H = K↑ (8/16) 또는 hard negative mining |
| [+0.001, +0.005pt] | 약 effect, sub-H 가치 있음 |
| (-0.001, +0.001pt] | NOOP — pointwise BCE 가 이미 ranking 을 충분 |
| < -0.001pt | BPR 가 본 데이터에 안 맞음 (positive sparsity 문제 등), retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `trainer.py` | + BPR dispatch in _train_step + bpr_num_neg in __init__ | Training loop |
| `train.py` | + 'bpr' choice + --bpr_num_neg + plumbing | CLI |
| `run.sh` | + `--loss_type bpr --bpr_num_neg 4` | Entry |
| `README.md` | identity | Doc |
| 다른 모든 .py | byte-identical (md5 verified) | unchanged |

## 5. Carry-forward
- §17.2 single mutation: loss function 만 변경 (BCE → BPR). model graph / data path / inference 전부 byte-identical to H019.
- §17.4 rotation: NEW first-touch (loss objective axis), AUTO_JUSTIFIED.
- §18.7/§18.8 H019 carry.

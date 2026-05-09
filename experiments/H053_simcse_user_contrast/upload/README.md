# H053_simcse_user_contrast — Technical Report

> PARADIGM SHIFT: 14 H 모두 single label supervision. H053 = SimCSE-style
> SSL self-contrastive on backbone output. 14 H 동안 SSL 0회.

## 1. Hypothesis & Claim
- Hypothesis: backbone output (user_repr) 의 두 dropout view 가 같아야 한다는
  SSL constraint 가 user representation 의 robustness 와 discriminative quality
  강화 → platform AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — SimCSE auxiliary

```python
# trainer:
z1 = F.dropout(user_repr, p=0.1, training=True)
z2 = F.dropout(user_repr, p=0.1, training=True)  # different mask
z1n = F.normalize(z1, -1); z2n = F.normalize(z2, -1)
sim = (z1n @ z2n.T) / tau              # (B, B)
targets = arange(B)                     # diag = positive pairs
simcse_loss = CE(sim, targets)
loss = bce + 0.1 * simcse_loss
```

**왜 paradigm shift**:
- 14 H 모두 single label supervision (sparse, 1 signal/user).
- H053 = SSL self-supervision (dense, B-1 negatives via in-batch).
- H052 (user × item) 와 다름 — H050 은 user × user (self-contrastive).
- Gao et al. 2021 SimCSE form 의 recsys 적용 0회.

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | SSL paradigm 작동, sub-H = augmentation 다양화 |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | dropout view 만으론 부족, sequence augmentation 필요 |
| < -0.001pt | SSL fight BCE, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 1 __init__ param, + forward tuple return | Model |
| `trainer.py` | + 3 __init__ params, + SimCSE loss block | Training |
| `train.py` | + 4 argparse + 4 plumbing | CLI |
| `infer.py` | + 1 cfg.get() (H043 방지) | Inference |
| `run.sh` | + `--use_user_simcse --simcse_lambda 0.1 --simcse_temperature 0.1 --simcse_dropout 0.1` | Entry |
| `README.md` | new | Doc |

trainable params 추가: **0** (loss term only).

## 5. Carry-forward
- §17.2 single mutation: SimCSE auxiliary.
- §17.4: ssl_self_supervision axis NEW first-touch (paradigm shift).
- §0.5: 14 H sparse-label 직접 attack via SSL.
- **infer.py flag parity** verified.

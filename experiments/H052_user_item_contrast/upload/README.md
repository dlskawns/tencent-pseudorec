# H052_user_item_contrast — Technical Report

> CLAUDE.md §0.5 data-signal-driven: 14 H 모두 pointwise BCE 단독 supervision.
> H040 BPR (logit pairwise) REFUTED. H052 = REPRESENTATION-level contrastive
> (InfoNCE) — backbone output user_repr 와 item_ns mean item_repr 사이 in-batch
> negatives. SSL-style new supervision form, 14 H 동안 0회.

## 1. Hypothesis & Claim
- Hypothesis: in-batch InfoNCE (positive = (user_i, item_i), negative = (user_i,
  item_j≠i)) 로 user_repr 가 자기 item 을 다른 user item 보다 더 가깝게 학습
  강제 → discriminative user representation → platform AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — User-Item InfoNCE auxiliary

```python
# model.py forward (when use_user_item_contrast=True):
return logits, output, item_repr  # tuple

# trainer._train_step:
if user_repr is not None and self.contrast_lambda > 0:
    u = F.normalize(user_repr, dim=-1)             # (B, D)
    it = F.normalize(item_repr, dim=-1)             # (B, D)
    sim = (u @ it.T) / self.contrast_temperature    # (B, B)
    targets = torch.arange(B)
    contrast_loss = F.cross_entropy(sim, targets)
    loss = bce + self.contrast_lambda * contrast_loss
```

contrast_lambda=0.1 (보수적), contrast_temperature=0.1 (sharpening 강함).

**왜 이게 진짜 NEW**:
1. 14 H 모두 pointwise BCE 단독. H040 BPR (logit pairwise) REFUTED 와 다른
   layer = REPRESENTATION level supervision.
2. SSL-style in-batch contrastive (SimCLR/CLIP form) 본 데이터 첫 적용.
3. trainable params 0 (loss term only) — 순수 supervision form 변경.

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | contrastive supervision main lever, sub-H = lambda + temperature sweep |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | in-batch negatives 부족 (B=1024 weak), 더 강한 form (memory bank) |
| < -0.001pt | contrast 가 BCE 와 fight, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 1 __init__ param, + forward tuple return (~6 lines) | Model |
| `trainer.py` | + 2 __init__ params, + InfoNCE loss block (~12 lines) | Training |
| `train.py` | + 3 argparse + 3 plumbing | CLI |
| `infer.py` | + 1 cfg.get() (H043 방지) | Inference |
| `run.sh` | + `--use_user_item_contrast --contrast_lambda 0.1 --contrast_temperature 0.1` | Entry |
| `README.md` | new | Doc |
| `dataset.py / utils.py / local_validate.py / make_schema.py` | byte-identical | unchanged |

trainable params 추가: **0** (loss term only).

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST PASS (model.py + train.py + trainer.py + infer.py)
2. ✅ shellcheck PASS (run.sh)
3. ✅ InfoNCE math: random reps → loss=2.65 (≈ log(B=8)=2.08, expected ≥ log(B)
   from random)
4. ✅ Identity test: positive identity reps → loss≈0 (math correct)
5. ✅ Gradient flow: |user.grad|=7.95, |item.grad|=8.50

## 6. Carry-forward
- §17.2 single mutation: InfoNCE auxiliary, model + trainer 두 file 변경 (single mechanism).
- §17.4: representation_contrastive axis NEW first-touch.
- §10.6: trainable params +0.
- §0.5: 14 H pointwise BCE 단독 → SSL-style supervision NEW.
- **infer.py flag parity** verified (H043 방지).
- §18.7 + §18.8 H019 carry.

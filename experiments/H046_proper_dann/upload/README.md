# H046_proper_dann — Technical Report

> CLAUDE.md §0.5 data-signal-driven: H044 (GRL on raw_logits[:, 1] aux col)
> **design 결함** — clsfier-aux head + backbone 둘 다 reverse → no actual
> discriminator → loss 발산 (epoch 1=4.93, epoch 5=545). H046 = **proper DANN**:
> separate cohort_head (positive grad, learns timestamp predictor) + GRL
> between backbone and cohort_head (backbone reverses, forgets cohort).

## 1. Hypothesis & Claim
- Hypothesis: backbone 이 timestamp (cohort proxy) 를 못 예측하도록 적절한
  DANN structure (separate discriminator + GRL) 로 강제 → cohort-invariant
  feature → platform transfer 향상.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — proper DANN

```python
# model.py:
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = float(lambd); return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd * grad_output, None

# In PCVRHyFormer.__init__:
if use_dann_cohort:
    self.cohort_head = nn.Linear(d_model, 1)

# In forward:
if self.use_dann_cohort:
    reversed_out = _GradReverse.apply(output, self.dann_cohort_lambda)
    cohort_pred = self.cohort_head(reversed_out).squeeze(-1)
    return logits, cohort_pred  # tuple

# trainer.py:
if isinstance(model_out, tuple):
    raw_logits, cohort_pred = model_out
    ...
    cohort_loss = F.mse_loss(cohort_pred, standardized_log1p_timestamp)
    loss = bce + cohort_loss  # GRL inside model handles backbone reversal
```

**Key fix vs H044**:
- H044: GRL applied on raw_logits[:, 1] (post-clsfier). clsfier-aux head got reverse → 둘 다 forget → no discriminator → loss 발산.
- H046: GRL applied INSIDE model BEFORE cohort_head. cohort_head trains positively, only backbone gets reversed. **Standard DANN structure**.

**왜 dann_cohort_lambda=0.1**: H044 (0.5) 의 1/5, sample-scale 안전. λ↑ sub-H sweep 가능.

## 3. Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | DANN cohort axis main lever, sub-H = lambda sweep + cohort label 다양화 (split_seed bin, user activity) |
| [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | timestamp가 cohort proxy 부족, 다른 cohort signal 필요 |
| < -0.001pt | DANN training 본 setup 에서 destabilize, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + _GradReverse class (~16 lines), + 2 __init__ params, + cohort_head init (~6 lines), + forward tuple return (~5 lines) | Model |
| `trainer.py` | + tuple unpacking + cohort MSE loss in `_train_step` (~10 lines) | Training |
| `train.py` | + 2 argparse + 2 plumbing line | CLI |
| `run.sh` | + `--use_dann_cohort --dann_cohort_lambda 0.1` | Entry |
| `README.md` | new | Doc |
| `dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **65** (cohort_head Linear(64, 1) = 65 params).

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS for model.py + train.py + trainer.py
2. ✅ shellcheck PASS for run.sh
3. ✅ GradReverse forward = identity
4. ✅ GradReverse backward grad ×(−lambd) (lambd=0.1)
5. ✅ Standard DANN test: cohort_head |grad|=4.33 (positive, learns), backbone |grad|=0.28 (small, scaled by 0.1)
6. ✅ md5 verify: 5 unchanged files identical to H019

## 6. Carry-forward
- §17.2 single mutation: cohort_head + GRL 추가 (model + trainer 두 file 변경 — single mechanism class).
- §17.4 rotation: cohort_drift_attack axis (H044 retry), AUTO_JUSTIFIED.
- §10.6 sample budget: trainable params +65.
- §0.5 data-signal-driven: F-A 패턴 직접 attack, paper 정당화 = 본 9 H 의 OOF/Platform divergence.
- §18.7 + §18.8 H019 carry.

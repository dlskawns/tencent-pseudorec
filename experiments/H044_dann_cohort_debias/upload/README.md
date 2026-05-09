# H044_dann_cohort_debias — Technical Report

> CLAUDE.md §0.5 data-signal-driven: F-A 패턴 9 H 누적 (OOF over Platform
> 1.88~2.59pt) + H019/H038/H039 platform 비교가 cohort drift 가 platform
> transfer 실패의 근본 원인 정량 confirm. H038 (aux MSE positive) Platform
> NOOP — H044 = opposite direction (gradient reversal, backbone *forget*
> timestamp = cohort proxy). Ganin & Lempitsky 2015 DANN form, 본 데이터
> F-A 패턴 직접 attack.

## 1. Hypothesis & Claim
- Hypothesis: backbone 이 timestamp (cohort proxy) 를 예측 못 하도록 GRL
  로 adversarial 학습 → cohort-invariant feature 강제 → platform transfer
  향상.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt → cohort attack
  이 H019 mechanism stack 위 추가 lift.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — gradient reversal layer

```python
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd); return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

# H044: aux_timestamp branch:
if self.dann_lambda > 0:
    aux_pred_for_loss = _GradReverse.apply(aux_pred, self.dann_lambda)
else:
    aux_pred_for_loss = aux_pred
mse = F.mse_loss(aux_pred_for_loss, aux_target)
loss = bce + self.aux_lambda * mse
```

**Effect**:
- forward: identity (loss compute 동일)
- backward: aux path gradient 가 backbone 으로 들어갈 때 −dann_lambda 곱
- 결과: backbone 이 timestamp 예측 못 하도록 학습 (DANN adversarial)

**왜 dann_lambda=0.5 (보수적)**: Ganin 권장 1.0 의 절반. sample-scale
(1000 rows) 에서 학습 destabilize 안전, lambda↑ sub-H sweep 가능.

## 3. Decision tree (post-result)

| Δ vs H019 Platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | cohort-attack main lever, sub-H = lambda sweep + cohort label 다양화 |
| [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | timestamp 가 cohort proxy 부족, 다른 cohort signal 시도 |
| < -0.001pt | DANN instability or axis dead, lambda↓ retry 또는 retire |

추가 paired 비교: Δ vs H038 (0.839071) — aux MSE direction (positive H038 vs adversarial H044) specific effect.

## 4. Files
| File | H038 대비 | Purpose |
|---|---|---|
| `trainer.py` | + _GradReverse class (~16 lines), + 1 __init__ param (dann_lambda), + 5 lines branch in 'aux_timestamp' | Training loop |
| `train.py` | + 1 argparse + 1 plumbing line | CLI |
| `run.sh` | + `--dann_lambda 0.5` | Entry |
| `README.md` | new | Doc |
| `model.py / dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **0** (autograd Function 만, no new module).

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS (trainer.py + train.py)
2. ✅ shellcheck PASS (run.sh)
3. ✅ GradReverse forward = identity
4. ✅ GradReverse backward = grad × (−lambda)
5. ✅ lambda=0 boundary (gradient zero, no aux flow)
6. ✅ lambda=1.0 boundary (DANN standard fully reversed)
7. ✅ integration with F.mse_loss (full H044 flow simulation)
8. ✅ md5 verify: 6 unchanged files identical to H038

## 6. Carry-forward
- §17.2 single mutation: GRL + branch 추가, model graph byte-identical to H038, infer.py byte-identical to H038.
- §17.4 rotation: cohort_drift_attack axis NEW first-touch, AUTO_JUSTIFIED.
- §10.6 sample budget: trainable params +0.
- §0.5 data-signal-driven: F-A 패턴 9 H 직접 attack, paper transplant 0.
- §18.7 + §18.8 H019/H038 carry.

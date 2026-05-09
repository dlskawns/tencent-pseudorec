# H047_per_domain_aux — Technical Report

> CLAUDE.md §0.5 data-signal-driven: 14 H 동안 main classifier 1개 만 사용,
> per-domain prediction multi-task 0회. §3.5 4 도메인 disjoint vocab signal —
> 각 도메인이 독립 prediction 신호 가질 가능성. backbone 이 4 도메인 정보
> 를 *대표 1개 logit* 로 압축 → per-domain signal mixing 손실 가능.
> H047 = 4 per-domain auxiliary classifier heads (multi-task BCE on label).

## 1. Hypothesis & Claim
- Hypothesis: 4 per-domain heads 가 각자 같은 main label 예측 → backbone 이
  per-domain prediction balanced 학습 → cold-domain (e.g. domain d frac_empty=8%
  bimodal) 에서도 prediction 유지 → platform AUC 향상.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — per-domain aux heads

```python
# model.py:
if use_per_domain_aux:
    self.per_domain_aux_heads = nn.ModuleList([
        nn.Linear(d_model, 1) for _ in range(4)
    ])

# In forward:
if self.use_per_domain_aux:
    per_domain_aux = []
    for i in range(4):
        valid = (~seq_masks[i]).float().unsqueeze(-1)
        pooled = (seq_tokens[i] * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        aux_logit = self.per_domain_aux_heads[i](pooled).squeeze(-1)
        per_domain_aux.append(aux_logit)
    return logits, per_domain_aux

# trainer.py:
aux_losses = [F.bce_with_logits(a, label) for a in per_domain_aux]
loss = bce + 0.25 * (sum(aux_losses) / 4)
```

**왜 0.25 weight**: 4 heads × 0.25 = 1.0 total → main BCE 와 balanced.

**왜 이게 진짜 NEW**:
1. 14 H 동안 multi-task per-domain prediction 0회. main + per-domain
   classifier multi-task 패턴 첫 시도.
2. §3.5 4 도메인 disjoint vocab + frac_empty 분포 (a/b/c < 2%, d 8%) 의
   imbalanced supervision signal 직접 attack.
3. H012 MoE (uniform routing REFUTED) 와 다름 — explicit per-domain
   prediction supervision, routing 학습 안 함.

## 3. Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | per-domain multi-task axis main lever, sub-H = aux_weight sweep + per-domain head 다양화 (MLP) |
| [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | per-domain supervision noise, axis exhausted |
| < -0.001pt | aux loss 가 main BCE 와 fight, retire 또는 weight↓ |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 1 __init__ param, + per_domain_aux_heads ModuleList init (~6 lines), + forward tuple return + per-domain pooled aux logits (~14 lines) | Model |
| `trainer.py` | + tuple unpacking + per-domain aux BCE loss (~7 lines) | Training |
| `train.py` | + 1 argparse + 1 plumbing line | CLI |
| `run.sh` | + `--use_per_domain_aux` | Entry |
| `README.md` | new | Doc |
| `dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **260** (4 × Linear(64, 1) = 4 × 65 = 260).

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS for model.py + train.py + trainer.py
2. ✅ shellcheck PASS for run.sh
3. ✅ md5 verify: 5 unchanged files identical to H019

## 6. Carry-forward
- §17.2 single mutation: per-domain aux heads 추가, model + trainer 변경 (single mechanism = multi-task per-domain prediction).
- §17.4 rotation: multi_task_per_domain axis NEW first-touch, AUTO_JUSTIFIED.
- §10.6 sample budget: trainable params +260.
- §0.5 data-signal-driven: §3.5 4 도메인 disjoint vocab signal 의 multi-task supervision form, paper transplant 0.
- §18.7 + §18.8 H019 carry.

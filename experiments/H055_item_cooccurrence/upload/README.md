# H055_item_cooccurrence — Technical Report

> PARADIGM SHIFT: 14 H 모두 single label supervision. H055 = Cross-domain
> pool InfoNCE — 같은 user 의 4 도메인 pool 은 같아야 한다는 SSL constraint.
> §3.5 jaccard 0.7~10% (4 도메인 disjoint-but-overlapping) 직접 attack.

## 1. Hypothesis & Claim
- Hypothesis: 같은 user 의 4 도메인 seq pool 이 cross-domain consistency 가져야
  한다는 SSL → user 정체성이 도메인 간 공유 학습 → cold-item generalization
  강화.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation
```python
# trainer:
B, num_d, D = per_domain_pools.shape
domain_idx = torch.randperm(num_d, device=...)[:2]
anchor = per_domain_pools[:, domain_idx[0]]      # (B, D)
positive = per_domain_pools[:, domain_idx[1]]    # (B, D)
sim = (norm(anchor) @ norm(positive).T) / tau    # (B, B)
target = arange(B)
cd_loss = CE(sim, target)
loss = bce + 0.1 * cd_loss
```

**왜 paradigm shift**:
- 14 H 모두 single supervision.
- H045 (cross-domain bridge attention) 와 다름 — H045 = 모듈 add, H055 = supervision form.
- §3.5 4 도메인 jaccard signal 직접 SSL attack.

## 3. Decision tree
| Δ vs H019 platform | Action |
|---|---|
| ≥ +0.003pt | cross-domain SSL 작동, sub-H = lambda sweep + harder negative |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | pool-level 부족, per-token contrastive |
| < -0.001pt | retire |

## 4. Files
| File | H019 대비 |
|---|---|
| model.py | + 1 __init__ param + forward tuple return + per-domain pool compute |
| trainer.py | + 2 __init__ params + InfoNCE block |
| train.py | + 3 argparse + 3 plumbing |
| infer.py | + 1 cfg.get() |
| run.sh | + flag |
| dataset.py / utils.py / make_schema.py / local_validate.py | byte-identical |

trainable params 추가: 0.

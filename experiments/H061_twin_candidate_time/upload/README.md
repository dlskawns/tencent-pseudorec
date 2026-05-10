# H061_twin_candidate_time — Technical Report

> H057 success template TWIN-internal, single residual. H061 = candidate Q
> 의 *upstream enrichment* (GSU scoring + ESU MHA Q 둘 다 영향). H057/H059/H060
> 와 진짜 orthogonal — 다른 H 들 모두 downstream 작용.

## 1. Hypothesis & Claim
- Hypothesis: candidate = item_ns mean 은 시간 무관 → time-irrelevant top-K
  선택 + time-irrelevant ESU attention. candidate 에 max_time_bucket signal
  추가 시 GSU 가 time-relevant top-K 선택 + ESU 가 time-aware Q.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation
```python
candidate = item_ns.mean(dim=1)  # (B, D)
if use_twin_candidate_time:
    max_time = stack(seq_time_buckets[d].max(L) for d in domains).max(domains)  # (B,)
    time_emb = self.candidate_time_proj(log1p(max_time + 1).unsqueeze(-1))  # Linear(1, D)
    candidate = candidate + 0.1 * time_emb
```
- params 추가: ~64 (Linear(1, D))

## 3. Layer orthogonality

| H | TWIN layer |
|---|---|
| H057 | aggregator (downstream of TWIN blocks) |
| H059 | gate (downstream of aggregator) |
| H060 | ESU MHA input (top-K + pos emb) |
| **H061** | **candidate Q (upstream of GSU + ESU)** |

H061 만 upstream → GSU scoring 자체 영향 + 다른 H 들과 진짜 orthogonal.

## 4. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | candidate enrichment main lever |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | time signal 부족 |
| < -0.001pt | retire |

## 5. Files
| File | H019 대비 |
|---|---|
| model.py | +1 __init__ param + candidate_time_proj + candidate enrichment in _compute_twin_residual |
| train.py | +1 argparse +1 plumbing |
| infer.py | +1 cfg.get (H043 방지) |
| run.sh | + flag |
| 외 .py | byte-identical |

trainable params 추가: ~64.

# H062_twin_learnable_gsu — Technical Report

> H057 success template TWIN-internal. H062 = TWIN GSU scoring 의 유일한
> untouched layer 완성. parameter-free 내적 → learnable low-rank W_q/W_k.

## 1. Hypothesis & Claim
- Hypothesis: GSU 의 parameter-free 내적 은 single signal axis 만 score.
  learnable W_q/W_k projection 이 candidate 와 history 의 더 informative
  similarity (low-rank semantic space) 학습 → 더 좋은 top-K 선택.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation
```python
# H019 GSU (parameter-free):
scores = (history * candidate.unsqueeze(1)).sum(-1)

# H062 GSU (learnable):
h_proj = self.gsu_W_k(history)             # Linear(D, 16)
c_proj = self.gsu_W_q(candidate)           # Linear(D, 16)
scores = (h_proj * c_proj.unsqueeze(1)).sum(-1)
```
- params 추가: 4 도메인 × 2 × Linear(64, 16) = ~8K

## 3. TWIN layer coverage matrix

| H | Layer | Status |
|---|---|---|
| **H062** | **GSU scoring (upstream)** | **NEW first-touch** |
| H061 | candidate Q (upstream) | ready |
| H060 | ESU MHA input (top-K + pos) | running |
| H059 | gate (downstream) | running |
| H057 | aggregator (downstream) | **PASS +0.0017** |

H062 cover 시 **TWIN 모든 layer 완성** → 5 H 결과로 어느 layer 가 lever 인지 진단.

## 4. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | learnable GSU 가 main lever, sub-H = full-rank or per-domain rank |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | parameter-free 충분 |
| < -0.001pt | retire |

## 5. Files
| File | H019 대비 |
|---|---|
| model.py | + use_learnable_gsu + gsu_rank in TWINBlock + GSU branch + PCVRHyFormer flag |
| train.py | +2 argparse +2 plumbing |
| infer.py | +2 cfg.get (H043 방지) |
| run.sh | +2 flags |
| 외 .py | byte-identical |

trainable params 추가: ~8K.

# H041_cold_start_branch — Technical Report

> CLAUDE.md §0.5 data-signal-driven (R2 reframe): §3.5 의 강력한 데이터 fact —
> **target_in_history = 0.4%** (any domain). 즉 99.6% 의 prediction 이
> *extrapolation* (user 가 본 적 없는 item 추천). 18 H 모두 single classifier
> 로 두 regime (familiar 0.4% vs novel 99.6%) 동시 처리. cold-start case 에
> capacity 부족 가능성. Mutation: dual classifier (main + cold) + per-sample
> learned gate. trainer.py / dataset.py / infer.py byte-identical to H019.

## 1. Hypothesis & Claim

- Hypothesis: 99.6% extrapolation regime 을 cold_clsfier 가 specialize 학습 → main classifier 가 familiar 0.4% case 에 더 sharp 하게 fit 가능.
- Falsifiable: Δ vs H019 (0.83967) ≥ +0.002pt → cold-start specialization 이 lever.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — Dual Classifier with Learned Gate

```python
# Init: cold_gate.bias = 2.0 → sigmoid(2.0) ≈ 0.88 → main dominates initially.
# 학습 중 backbone-conditioned gate 가 sample 별로 main vs cold 비율 학습.

if use_cold_start_branch:
    gate = sigmoid(cold_gate(output))             # (B, 1) — per-sample, learned
    out_main = clsfier(output)                    # (B, action_num) — existing
    out_cold = cold_clsfier(output)               # (B, action_num) — new, parallel
    return gate * out_main + (1 - gate) * out_cold
else:
    return clsfier(output)                         # H019 path
```

Cold branch 와 gate 모두 backbone output 에서 직접 학습 → 어떤 sample 이 "novel" 인지 모델이 자체 결정.

## 3. Decision tree (post-result)

| Δ vs H019 (0.83967) | gate 학습 결과 | Action |
|---|---|---|
| ≥ +0.005pt | gate 분포 bimodal (high/low split) | cold-start lever 진짜, sub-H = explicit novelty signal (history-candidate sim) |
| [+0.001, +0.005pt] | gate 평균 < 0.7 | 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | gate 평균 ≈ 0.88 (init 그대로) | gate 가 학습 안 됨 — single classifier 로 충분 |
| < -0.001pt | unstable | dual branch 가 학습 destabilize |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + cold_clsfier + cold_gate in `__init__` (~22 lines) + `_compute_classifier` dispatch (~7 lines) | Model |
| `train.py` | + 2 argparse + model_args | CLI |
| `run.sh` | + `--use_cold_start_branch --cold_start_gate_init 2.0` | Entry |
| `README.md` | identity | Doc |
| 다른 모든 .py | byte-identical (md5 verified) | unchanged |

trainable params 추가: cold_clsfier (~4K) + cold_gate (~65) = **~4.4K** (TWIN 71K 의 6%, total 161M 의 0.003%).

## 5. Init equivalence to H019

cold_gate.bias=2.0, weight=0 → 모든 sample 의 init gate ≈ sigmoid(2.0) ≈ 0.88 → main 88% / cold 12%. cold_clsfier 의 random init 출력은 noise 수준 → main 거의 그대로. **학습 시작 시 H019 와 거의 동일 동작**, 학습이 진행되며 gate + cold weights 가 specialize.

## 6. Carry-forward
- §17.2 single mutation: classifier 단일 → dual + gated. 다른 모든 부분 byte-identical to H019.
- §17.4 rotation: NEW first-touch (data-signal-driven specialization), AUTO_JUSTIFIED.
- §10.10 InterFormer bridge gating σ(−2)≈0.12 패턴 변형 — gate init = 0.88 (main 지배), TWIN 의 gate init = 0.12 (TWIN 종속) 와 대칭.
- §18.7/§18.8 H019 carry.

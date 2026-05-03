# H017_recency_exp_decay — Technical Report

> H015 sub-form variant. **Single mutation**: `recency_weight_form 'linear'
> → 'exp'` (auto-normalized to mean=1.0). H015 의 모든 다른 설정 byte-identical.

## Why this variant

H015 = linear recency weighting [0.5, 1.5]. PASS / measurable / noise 결과
관계없이 **shape 변경의 효과 분리** = curve 형태 자체가 mechanism 효과를
좌우하는지 검증.

- linear: 균등 증가 (oldest 0.5 → newest 1.5).
- exp: 기하 증가 (recent emphasis 더 강함). auto-normalize 후 ~[0.55, 1.65].
- 같은 [0.5, 1.5] 범위 — confound 작음.

## Mechanism (H015 vs H017 diff)

```python
# H015 linear:
weights = w_min + (w_max - w_min) × pct                                  # mean = (w_min+w_max)/2

# H017 exp:
weights_raw = w_min × (w_max / w_min)^pct                                # geometric scale
weights = weights_raw / weights_raw.mean()                                # normalize to mean=1.0
```

## Diff vs H015/upload/

| File | Diff | Role |
|---|---|---|
| `run.sh` | +1 flag (`--recency_weight_form exp`) | Entry point |
| `trainer.py` | exp branch + auto-normalize (~10 줄 추가) | Train loop |
| `train.py` | argparse 1 + Trainer 1 key | CLI |
| `README.md` | 변경 | H017 정체성 |
| 다른 8 files | byte-identical with H015 | unchanged |

## Falsification

| Result | Implication |
|---|---|
| Δ vs H015 ≥ +0.001pt + Δ vs H010 ≥ +0.005pt | exp curve 효과, recent emphasis 가 cohort drift 에 더 적합. |
| Δ ∈ (−0.001, +0.001pt vs H015) | Form 변경 효과 거의 없음, linear 와 동일한 결과. |
| Δ < −0.001pt vs H015 | exp 가 학습 disrupt (asymmetry). |

## Triple-H setup (H015 + H016 + H017 동시)

3개 동시 실행 — L2 (cohort drift) multi-form 검증:
- H015 linear weighting [0.5, 1.5] (균등 증가).
- H016 OOF 재정의 (label_time future-only holdout).
- H017 exp weighting [0.5, 1.5] (기하 증가).

셋 다 noise → cohort 가 paradigm 안 ceiling 깨지 못함 → backbone replacement.

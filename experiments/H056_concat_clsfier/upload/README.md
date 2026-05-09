# H056_concat_clsfier — Technical Report

> H048 LESSON-DRIVEN: H048 (user × item bilinear residual ADD) Val +0.0005 / OOF +0.0003
> → **Platform −0.0042** (vanilla baseline 수준). 두 residual ADD on `output` 가
> cohort transfer 깨뜨린 진단. H056 = **same goal (user × item interaction) different
> layer (clsfier input concat)** — NO output residual stacking.

## 1. Hypothesis & Claim
- Hypothesis: user × item interaction 이 platform 에서 lever 일 가능성 (H048
  Val 측에서 시사). 단 mechanism layer 가 *output residual* (H048) 면 transfer
  실패. *clsfier input concat* form 이면 output 변경 없이 user × item 정보 직접
  학습 → platform transfer 살아남.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.

## 2. Mutation — Concat-input clsfier

```python
# H019:
output = backbone(...) + sigmoid(twin_gate) * twin_state  # 1 residual
logits = clsfier(output)  # input dim D

# H056:
output = backbone(...) + sigmoid(twin_gate) * twin_state  # 1 residual (UNCHANGED)
item_repr = item_ns_tokenizer(item_int).mean(dim=1)
logits = clsfier(cat([output, item_repr], -1))  # input dim 2D, NO new residual
```

**Difference vs H048**:
| | H048 | H056 |
|---|---|---|
| user × item layer | output += cross(user, item) | clsfier input = cat(user, item) |
| output residual count | 2 (cross + twin) | 1 (twin only, unchanged) |
| H019 platform path 보존 | 깨짐 (−0.0042) | 보존 (output 동일) |

**왜 이게 다른가:**
- H048: `output` 자체에 cross 추가 → magnitude 증가 → clsfier saturation → cohort transfer 깨짐
- H056: `output` unchanged, clsfier 만 더 풍부한 input. residual stacking 없음.

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | concat layer 진짜 lever, sub-H = item_repr 다양화 |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | concat 만으로 부족 |
| < -0.001pt | concat noise, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 1 __init__ param, + clsfier first Linear input dim 변경 (D→2D), + forward/predict concat | Model |
| `train.py` | + 1 argparse + 1 plumbing | CLI |
| `infer.py` | + 1 cfg.get() (H043 방지) | Inference |
| `run.sh` | + `--use_concat_clsfier` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / utils.py / local_validate.py / make_schema.py` | byte-identical | unchanged |

trainable params 추가: ~4K (Linear(128, 64) vs Linear(64, 64) = 64×64 extra).

## 5. Carry-forward
- §17.2 single mutation: clsfier input concat. output 자체 byte-identical to H019.
- §17.4: clsfier_input_concat axis NEW first-touch.
- §10.6: trainable params +4K.
- §0.5: H048 platform fail 진단 직접 fix — same user×item goal, residual stacking 회피.
- **infer.py flag parity** verified.

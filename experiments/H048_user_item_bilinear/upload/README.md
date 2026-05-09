# H048_user_item_bilinear — Technical Report

> CLAUDE.md §0.5 data-signal-driven: 14 H 분석 결론 — PASS mechanism (H010 NS→S
> xattn, H019 TWIN candidate Q) 모두 *새 정보 흐름 추가*. 14 H 동안 user
> representation × item representation 사이 explicit bilinear interaction 0회.
> H048 = FM-style user × item bilinear cross at top of backbone, residual ADD.

## 1. Hypothesis & Claim
- Hypothesis: backbone output (user_repr) 와 item_repr (item_ns mean) 사이
  explicit bilinear interaction `cross = user_repr ⊙ (W item_repr)` 가 학습
  → user × item 직접 cross 신호가 새 prediction 정보 → platform AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — UserItemBilinearCross

```python
class UserItemBilinearCross(nn.Module):
    def __init__(self, d_model, gate_init=-2.0):
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.cross_gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, user_repr, item_repr):
        item_proj = self.W(item_repr)              # (B, D)
        cross = user_repr * item_proj              # (B, D) element-wise
        cross = self.out_ln(self.proj(cross))
        return torch.sigmoid(self.cross_gate) * cross
```

**Wiring**: TWIN 직전에 적용. item_repr = item_ns_tokenizer(item_int).mean(1).

**왜 이게 진짜 NEW**:
1. 14 H 동안 user × item explicit bilinear 0회.
2. TWIN: item → user_history attention (한 방향). H048: user_repr × item_repr (양 방향 동시 곱).
3. DCN-V2 (H008 PASS) 는 within-user-side cross. H048 = cross-side (user × item).
4. FM (Factorization Machine) 의 top-level 적용 — 14 H 모든 mechanism class 와 직교.

## 3. Decision tree
| Δ vs H019 platform | Action |
|---|---|
| ≥ +0.003pt | user × item axis main lever, sub-H = stacked layers / 다른 cross form |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | bilinear form 부족, 더 강한 form (multi-layer interaction) 또는 retire |
| < -0.001pt | bilinear noise/destabilize, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + UserItemBilinearCross class (~25 lines), + 2 __init__ params, + 1 module init block, + 1 forward call (residual ADD pre-TWIN, both forward+predict) | Model |
| `train.py` | + 2 argparse + 2 plumbing line | CLI |
| `infer.py` | + 2 cfg.get() (H043 사고 방지) | Inference |
| `run.sh` | + `--use_user_item_cross --user_item_cross_gate_init -2.0` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / utils.py / local_validate.py / make_schema.py` | byte-identical | unchanged |

trainable params 추가: ~12K (W + proj + LN + gate, low-rank-ish on D=64).

## 5. Carry-forward
- §17.2 single mutation: UserItemBilinearCross 1개 추가.
- §17.4: user_item_explicit_cross axis NEW first-touch, AUTO_JUSTIFIED.
- §10.10 InterFormer bridge gating σ(-2)≈0.12 mandate 통과.
- §0.5: 14 H "새 정보 흐름 추가" 패턴 직접 추적, paper transplant 0.
- §18.7 + §18.8 H019 carry.
- **infer.py flag parity**: H043 사고 방지, train.py + infer.py + model.py flag set 일치 검증.

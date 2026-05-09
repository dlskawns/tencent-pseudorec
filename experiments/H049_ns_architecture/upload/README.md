# H049_ns_architecture — Technical Report

> CLAUDE.md §0.5 data-signal-driven: 14 H 동안 NS architecture 자체 0회 변경.
> - item_ns_tokens default=2 모든 H 에서 unchanged (item = prediction target,
>   representation slot 부족 가설)
> - NS slot type 차별 embedding 0개 (backbone 이 user/item/dense role 학습으로
>   알아야 함, type-aware processing 부재)
> H049 = (a) item_ns_tokens 2→6 + (b) 4-type slot embedding. 둘 다 NS
> structure 변경 = single concept mutation.

## 1. Hypothesis & Claim
- Hypothesis: NS structure refinement 가 backbone 의 NS-token 처리 specialize
  → item representation 강화 + type-aware attention 패턴 → platform AUC lift.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — NS architecture refinement

```python
# model.py:
if use_ns_slot_type_emb:
    self.ns_type_emb = nn.Embedding(4, d_model)   # 4 types
    type_ids = []
    type_ids.extend([0] * num_user_ns)            # user_ns slots → type 0
    if has_user_dense: type_ids.append(1)          # user_dense → type 1
    type_ids.extend([2] * num_item_ns)             # item_ns slots → type 2
    if has_item_dense: type_ids.append(3)          # item_dense → type 3
    self.register_buffer('ns_type_ids', torch.tensor(type_ids))

# In _build_token_streams (after ns_tokens concat):
if self.use_ns_slot_type_emb:
    type_emb = self.ns_type_emb(self.ns_type_ids)  # (num_ns, D)
    ns_tokens = ns_tokens + type_emb.unsqueeze(0)  # broadcast (B, num_ns, D)
```

**run.sh diff**: `--item_ns_tokens 2 → 6` + `--use_ns_slot_type_emb`.

**왜 이게 진짜 NEW**:
1. 14 H 동안 NS structure (count + type) 변경 0회.
2. item_ns_tokens 2 → 6 = item representation slots 3× expand. backbone 이
   더 풍부한 item summary 받음.
3. NS slot type embedding = backbone 이 어떤 NS 가 어떤 role 인지 explicit
   알 수 있음 (현재는 학습으로만 추정).

## 3. Decision tree
| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | NS architecture axis lever, sub-H = item_ns 더 expand or per-domain NS |
| [+0.001, +0.003pt] | additive |
| (-0.001, +0.001pt] | NS structure 변경 만으로 부족, 다른 axis |
| < -0.001pt | item_ns 6 가 너무 많음, capacity 분배 잘못 |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + 1 __init__ param, + ns_type_emb + buffer + apply (~15 lines) | Model |
| `train.py` | + 1 argparse + 1 plumbing | CLI |
| `infer.py` | + 1 cfg.get() (H043 방지) | Inference |
| `run.sh` | `--item_ns_tokens 2→6` + `--use_ns_slot_type_emb` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / utils.py / local_validate.py / make_schema.py` | byte-identical | unchanged |

trainable params 추가:
- ns_type_emb: 4 × 64 = 256
- 4 extra item NS tokens via RankMixer split: ~16K (4 slots × ~4K params)

## 5. Carry-forward
- §17.2 single mutation: NS architecture refinement (count + type, conceptually 1 mutation = "NS structure").
- §17.4: ns_architecture axis NEW first-touch.
- §10.6: trainable params +~16K.
- §0.5: 14 H 동안 NS structure 변경 0회 = unexplored axis.
- **infer.py flag parity** verified.
- §18.7 + §18.8 H019 carry.

## 6. d_model%T 제약 주의
- H019: T = num_queries × num_sequences + num_ns = 2×4 + 9 = 17 (rankmixer 'half' mode 호환)
- H049: T = 2×4 + 13 = 21 (item_ns 2→6 으로 num_ns 9→13)
- d_model=64 % 21 != 0 — 'full' mode 면 raise. 'half' mode (default in H019) 면 OK.
- run.sh 명시 안 함 → 기본 mode 사용. 만약 OOM/error 시 mode 확인.

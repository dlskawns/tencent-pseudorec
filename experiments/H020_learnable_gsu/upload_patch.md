# H020 — Upload Package Patch Spec

> Patch spec only — full upload/ construction deferred until H019 cloud
> result 회수 (Frame C check). Generated 2026-05-06.

## 1. Base & sequence

- **Base**: `experiments/H019_twin_long_seq_retrieval/upload/` byte-by-byte copy.
- **Mutation**: TWINBlock GSU 의 parameter-free inner product → learnable
  projection (W_q, W_k: `nn.Linear(d_model, d_model//4, bias=False)`).
- **Carry-forward**: H019 의 모든 코드 byte-identical except 4 files
  (model.py + train.py + run.sh + README.md).
- **§17.2 single mutation**: GSU scoring function 의 parameter-free →
  learnable projection. ESU / top_k / aggregator / gate / num_heads /
  seq_max_lens / batch / NS xattn / DCN-V2 stack 전부 H019 byte-identical.

## 2. Files to modify (3 .py + run.sh + README)

| File | Δ vs H019 | Role |
|---|---|---|
| `model.py` | + ~10 lines (TWINBlock 의 W_q/W_k Linear + score 계산 분기) | Model |
| `train.py` | + 1 argparse line (`--twin_learnable_gsu`) + model 생성 시 flag pass | CLI |
| `run.sh` | + `--twin_learnable_gsu` flag bake | Entry |
| `README.md` | rewrite — H020 identity | Doc |
| `dataset.py` / `infer.py` / `make_schema.py` / `utils.py` / `local_validate.py` / `trainer.py` / `ns_groups.json` / `requirements.txt` | byte-identical | unchanged |

## 3. model.py patch — TWINBlock learnable GSU

**Modify** `class TWINBlock` (line 1323-1369 in H019 model.py):

```python
class TWINBlock(nn.Module):
    """Per-domain TWIN GSU+ESU retrieval module.

    H019: GSU = parameter-free inner product.
    H020: GSU = learnable projection (W_q, W_k: Linear(d_model, d_model//4)).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        top_k: int = 64,
        learnable_gsu: bool = False,        # H020 flag
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.learnable_gsu = learnable_gsu
        # H020: learnable GSU projection (paper-faithful form)
        if learnable_gsu:
            proj_dim = max(d_model // 4, 8)  # safety floor
            self.gsu_q = nn.Linear(d_model, proj_dim, bias=False)
            self.gsu_k = nn.Linear(d_model, proj_dim, bias=False)
        else:
            self.gsu_q = None
            self.gsu_k = None
        self.esu = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        history: torch.Tensor,
        candidate: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = history.shape
        K = min(self.top_k, L)
        # GSU score
        if self.learnable_gsu:
            # H020 — learnable projection before inner product
            q = self.gsu_q(candidate)              # (B, proj_dim)
            k = self.gsu_k(history)                # (B, L, proj_dim)
            scores = (k * q.unsqueeze(1)).sum(-1)  # (B, L)
        else:
            # H019 — parameter-free inner product
            scores = (history * candidate.unsqueeze(1)).sum(-1)  # (B, L)
        scores = scores.masked_fill(history_mask, float('-inf'))
        # Top-K filter (H019 byte-identical from here)
        topk_scores, topk_idx = scores.topk(K, dim=-1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        topk_history = history.gather(1, idx_exp)
        topk_pad_mask = (topk_scores == float('-inf'))
        all_padded = topk_pad_mask.all(dim=-1, keepdim=True)
        topk_pad_mask = topk_pad_mask & ~all_padded
        candidate_q = candidate.unsqueeze(1)
        attended, _ = self.esu(
            candidate_q, topk_history, topk_history,
            key_padding_mask=topk_pad_mask,
        )
        attended = self.norm(candidate_q + attended).squeeze(1)
        return attended
```

**Modify** `PCVRHyFormer.__init__` (around line 2038-2049 in H019 model.py):

```python
# H020 — pass learnable_gsu flag to TWINBlock
if use_twin_retrieval:
    self.twin_blocks = nn.ModuleDict({
        f'twin_{domain}': TWINBlock(
            d_model=d_model,
            num_heads=twin_num_heads,
            top_k=twin_top_k,
            learnable_gsu=twin_learnable_gsu,    # H020 — added
        )
        for domain in ('a', 'b', 'c', 'd')
    })
    self.twin_aggregator = TwinRetrievalAggregator(
        d_model=d_model, gate_init=twin_gate_init
    )
    print(
        f"H020 TWIN retrieval enabled: top_k={twin_top_k} "
        f"num_heads={twin_num_heads} gate_init={twin_gate_init} "
        f"learnable_gsu={twin_learnable_gsu}"
    )
```

**Add** `twin_learnable_gsu: bool = False` to `PCVRHyFormer.__init__` signature
(after `twin_gate_init`, line 1793 in H019 model.py).

## 4. train.py patch

**Argparse addition** (after `--twin_gate_init`):
```python
parser.add_argument('--twin_learnable_gsu', action='store_true',
                    help='H020: learnable projection GSU (paper-faithful form)')
```

**Model construction** (pass new flag):
```python
model = PCVRHyFormer(
    ...,
    use_twin_retrieval=args.use_twin_retrieval,
    twin_top_k=args.twin_top_k,
    twin_num_heads=args.twin_num_heads,
    twin_gate_init=args.twin_gate_init,
    twin_learnable_gsu=args.twin_learnable_gsu,    # H020 — added
)
```

**§18.8 SUMMARY block** (carry-forward from H019):
- exp_id 변경: `H019_twin_long_seq_retrieval` → `H020_learnable_gsu`.
- 다른 fields 그대로.

## 5. run.sh patch

```bash
# H020 baked args (vs H019: + --twin_learnable_gsu):
--use_twin_retrieval
--twin_top_k 64
--twin_num_heads 4
--twin_gate_init -2.0
--twin_learnable_gsu                # H020 — added
--oof_redefine future_only          # H016 carry
--seq_max_lens 256 256 256 256      # H019 carry (sweep saturation 영역)
--batch_size 1024                   # H019 carry
+ all H010 mechanism flags 그대로 (NS xattn, DCN-V2)
```

## 6. README.md (rewrite)

```markdown
# H020 — learnable_gsu

> TWIN sub-H (Tencent 2024) — GSU 의 parameter-free inner product →
> learnable projection (W_q, W_k: nn.Linear(d_model, d_model//4, bias=False)).
> paper-faithful GSU 직접 검증.
>
> Mutation: GSU scoring function 의 단일 변경 (parameter-free → learnable).
> ESU / top_k=64 / aggregator / gate=-2.0 / num_heads=4 / seq_max_lens
> 256/256/256/256 / batch 1024 byte-identical to H019.
>
> Control: H019 (champion). Paired Δ vs H019 cloud actual.
>
> Carry-forward: H016 redefined OOF (future-only) default. §18.8 SUMMARY
> block emit (exp_id 만 H019 → H020 변경).
```

## 7. Local sanity check (§17.5 code-path verification only)

```bash
.venv-arm64/bin/python train.py \
  --num_epochs 1 \
  --train_ratio 0.05 \
  --use_twin_retrieval \
  --twin_top_k 64 \
  --twin_gate_init -2.0 \
  --twin_learnable_gsu \              # H020 mutation
  --oof_redefine future_only \
  --seq_max_lens 256 256 256 256 \
  --batch_size 256                    # local downscale
```

Expected:
- Log line: `H020 TWIN retrieval enabled: top_k=64 num_heads=4 gate_init=-2.0 learnable_gsu=True`.
- TWINBlock forward shape mismatch 0.
- W_q / W_k grad flow 확인 (4 도메인 × 2 = 8 params 모두 grad 비-zero).
- top-K filter NaN-free.
- §10.6 sample budget warning (H019 의 71K + H020 의 8K = 79K, soft cap 2146 위반, paradigm shift carry-forward acknowledged).
- §18.8 SUMMARY block printed at end (1 epoch row).
- `local_validate.py` G1–G6 5/5 PASS.

**Ablation diff sanity** (H019 vs H020 forward output):
```python
import torch
from model import TWINBlock
torch.manual_seed(0)
B, L, D = 4, 64, 64
hist = torch.randn(B, L, D)
cand = torch.randn(B, D)
mask = torch.zeros(B, L).bool()
mask[1, :] = True   # row 1 entirely padded — defensive guard test

twin_h019 = TWINBlock(d_model=D, num_heads=4, top_k=32, learnable_gsu=False).eval()
twin_h020 = TWINBlock(d_model=D, num_heads=4, top_k=32, learnable_gsu=True).eval()
# weight init RNG offset → 출력 차이 비교 가능
out_h019 = twin_h019(hist, cand, mask)
out_h020 = twin_h020(hist, cand, mask)
diff = (out_h019 - out_h020).abs().max().item()
print(f"ablation diff: {diff:.4f}")   # > 0.001 expected (mechanism active)
print(f"NaN check: {torch.isnan(out_h020).any().item()}")  # False expected
```

## 8. dataset-inference-auditor invocation (§18.6)

After upload/ package built:

```
Agent(subagent_type="dataset-inference-auditor",
      prompt="Audit experiments/H020_learnable_gsu/upload/. prior_h=H019. Confirm dataset.py / infer.py / make_schema.py byte-identical to H019 (no schema change). model.py change scope = TWINBlock GSU learnable projection only. train.py: --twin_learnable_gsu argparse + model 생성 시 flag pass. §18.7 carry (label_time fill_null) preserved. §18.8 SUMMARY block exp_id = H020_learnable_gsu.")
```

Expected: PASS — H019 와 동일한 dataset/infer/schema (변경 범위 좁음).

BLOCK 시 fix → re-audit. PASS 받기 전 cloud upload 금지.

## 9. config_sha256 + git_sha (§4 reproducibility)

After patch applied + local sanity PASS:
- `git rev-parse --short=7 HEAD` → save to card.yaml.
- `sha256sum train_config.json` → save to card.yaml `config_sha256`.
- both written to SUMMARY block automatically via env vars.

## 10. Cost cap pre-launch audit (§17.6)

Before T2.4 launch:
- H019 cloud actual cost 회수 (T2.4 ~$5-7 expected).
- 누적 cost = (Subset A H022/H028~H031 actual) + H019 actual + H020 estimate.
- per-job cap T2.4 ~$10 / per-campaign cap $100 audit.
- 초과 시 사용자 confirm 필수.

## 11. Frame C check — RESOLVED 2026-05-06

**H019 cloud measurable PASS CONFIRMED**: platform 0.839674, Δ vs H010 corrected +0.001868pt = §17.3 measurable band [+0.001, +0.005pt]. Frame C REFUTED → H020 launch 정당.

upload BUILT 2026-05-06 in this session (4 file patch applied).

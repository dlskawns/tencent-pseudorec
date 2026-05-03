# H019 — Upload Package Patch Spec

> Patch spec only — full upload/ construction deferred. Generated 2026-05-03 (Subset A scaffold).

## 1. Base & sequence

- **Base**: `experiments/H010_ns_to_s_xattn/upload/` byte-by-byte copy
  (corrected anchor mechanism).
- **Mutation**: TWIN GSU+ESU per-domain retrieval module + seq_max_lens
  64-128 → 512 cap (coupled enabling condition).
- **Carry-forward**: H015 §18.7 patch (label_time fill_null) +
  emit_train_summary §18.8 (FIRST-TIME H019 + H018 + H022 simultaneous use).
- **§17.2 single mutation**: TWIN module (GSU+ESU per-domain) is the
  single mechanism class addition. seq_max_lens expansion is enabling
  condition (top-K retrieval meaningless without capacity to retrieve
  from). Per challengers.md ⓒ-§17.2 defensible.

## 2. Files to modify (4 .py + run.sh + README + make_schema)

| File | Δ vs H010 | Role |
|---|---|---|
| `model.py` | + ~80 lines (TWINBlock module + per-domain insertion) | Model |
| `dataset.py` | seq_max_lens 64-128 → 512 cap + §18.7 carry-forward (label_time fill_null) | Data |
| `train.py` | + 3 argparse + emit_train_summary() at end | CLI |
| `make_schema.py` | seq_max_lens 변경 → schema 재생성 | Schema |
| `run.sh` | TWIN flags + cap 512 bake | Entry |
| `README.md` | rewrite — H019 identity | Doc |
| `infer.py` / `utils.py` / `local_validate.py` / `ns_groups.json` / `requirements.txt` | byte-identical | unchanged |

## 3. model.py patch — TWINBlock module

**Insert at top of model.py** (after existing imports):

```python
class TWINBlock(nn.Module):
    """Per-domain TWIN GSU + ESU retrieval module.

    GSU: simple inner product (parameter-free) — score every history token
         against candidate, pick top-K.
    ESU: standard MultiHeadAttention over top-K — full attention with
         candidate as query.

    Args:
        d_model: hidden dim.
        num_heads: ESU MHA heads.
        top_k: K for GSU top-K filter (e.g., 64).
    """
    def __init__(self, d_model: int, num_heads: int = 4, top_k: int = 64):
        super().__init__()
        self.top_k = top_k
        # ESU = standard multi-head attention.
        self.esu = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history: torch.Tensor, candidate: torch.Tensor,
                history_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, L, d_model) — per-domain seq embeddings (L up to cap=512).
            candidate: (B, 1, d_model) — candidate item embedding.
            history_mask: (B, L) — True where padded (ignored).
        Returns:
            retrieved: (B, top_k, d_model) — top-K retrieved + ESU attended.
        """
        B, L, D = history.shape
        # GSU: inner product score — (B, L, D) × (B, 1, D)ᵀ → (B, L).
        scores = (history * candidate).sum(-1)  # (B, L)
        scores = scores.masked_fill(history_mask, float('-inf'))
        # Top-K filter — handle L < top_k via min().
        k = min(self.top_k, L)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # (B, k)
        # Gather top-K embeddings.
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
        topk_history = history.gather(1, idx_expanded)  # (B, k, D)
        # ESU: candidate attends to top-K.
        # candidate as query, topk_history as key/value.
        attended, _ = self.esu(candidate, topk_history, topk_history)  # (B, 1, D)
        # Residual + norm.
        attended = self.norm(candidate + attended)
        return attended.squeeze(1)  # (B, D)
```

**Insert in PCVRHyFormer.__init__** (per-domain encoder section):

```python
# H019 — TWIN per-domain retrieval (one TWINBlock per domain).
if use_twin_retrieval:
    self.twin_blocks = nn.ModuleDict({
        f'twin_{domain}': TWINBlock(d_model, num_heads=4, top_k=twin_top_k)
        for domain in ('a', 'b', 'c', 'd')
    })
```

**Insert in PCVRHyFormer.forward** (after per-domain encoder output,
before NS xattn):

```python
if hasattr(self, 'twin_blocks'):
    # candidate = item embedding — broadcast to per-domain.
    candidate_emb = self.item_id_emb(item_id).unsqueeze(1)  # (B, 1, D)
    for domain in ('a', 'b', 'c', 'd'):
        history = domain_encoder_output[domain]  # (B, L, D)
        history_mask = domain_pad_mask[domain]   # (B, L)
        retrieved = self.twin_blocks[f'twin_{domain}'](
            history, candidate_emb, history_mask
        )  # (B, D)
        # Replace per-domain pooled output with retrieved.
        domain_pooled[domain] = retrieved
```

## 4. dataset.py patch — seq_max_lens cap 512

**Replace** existing seq_max_lens definition (find line with `seq_max_lens
= [...]`):

```python
# H019 — seq_max_lens 64-128 → 512 cap (enabling condition for retrieval).
DEFAULT_SEQ_MAX_LENS = {
    'a': 512, 'b': 512, 'c': 512, 'd': 512,
}
```

**Carry-forward §18.7 patch** (label_time fill_null) — verify already
applied; if not, copy from H015 dataset.py line 553-560.

## 5. train.py patch

**Argparse additions**:
```python
parser.add_argument('--use_twin_retrieval', action='store_true')
parser.add_argument('--twin_top_k', type=int, default=64)
parser.add_argument('--twin_seq_cap', type=int, default=512)
parser.add_argument('--oof_redefine', choices=['random', 'future_only'],
                    default='future_only')   # H016 carry-forward
```

**Model construction** (pass new flags):
```python
model = PCVRHyFormer(
    ...,
    use_twin_retrieval=args.use_twin_retrieval,
    twin_top_k=args.twin_top_k,
)
```

**§18.8 SUMMARY block emit** (mandatory):
```python
# At end of train.py main(), after final epoch loop:
emit_train_summary(
    exp_id="H019_twin_long_seq_retrieval",
    git_sha=os.environ.get('GIT_SHA', 'unknown'),
    cfg_sha=os.environ.get('CONFIG_SHA256', 'unknown'),
    seed=args.seed,
    ckpt_kind='best',
    epoch_history=trainer.epoch_history,
    ...
)
```
(See H018 upload_patch.md §5 for full emit_train_summary() reference impl.)

## 6. make_schema.py patch

- seq_max_lens 변경 → schema 재생성 mandatory.
- §18.5 list type variant check (is_list / is_large_list /
  is_fixed_size_list) 모두 적용 확인.

```bash
.venv-arm64/bin/python make_schema.py \
  --data_path /path/to/data \
  --output schema.json \
  --seq_max_lens 512  # cap
```

## 7. run.sh patch

```bash
# H019 baked args (vs H010):
--use_twin_retrieval
--twin_top_k 64
--twin_seq_cap 512
--oof_redefine future_only       # H016 carry-forward
+ all H010 mechanism flags 그대로 (NS xattn, DCN-V2)
```

## 8. README.md (rewrite)

```markdown
# H019 — twin_long_seq_retrieval

> TWIN (Tencent 2024) GSU+ESU per-domain retrieval. seq_max_lens 64-128
> → 512 cap. Carry-forward H016 redefined OOF (future-only) default.
> §18.8 SUMMARY block emit at end of train.py (concurrent with H018/H022).
>
> Mutation: TWIN module addition (sequence axis paradigm shift, retrieval
> form). Coupled with seq_max_lens cap expansion (enabling condition).
>
> Control: H010 corrected anchor (0.837806). Paired Δ vs H010 corrected.
```

## 9. Local sanity check (§17.5 code-path verification only)

```bash
.venv-arm64/bin/python train.py \
  --num_epochs 1 \
  --train_ratio 0.05 \
  --use_twin_retrieval \
  --twin_top_k 32 \           # smaller for sample-scale
  --twin_seq_cap 256 \        # smaller for sample-scale
  --oof_redefine future_only
```

Expected:
- Log line: `H019 ENABLED: TWIN retrieval (top_k=32, seq_cap=256, num_heads=4)`.
- TWINBlock forward shape mismatch 0.
- top-K filter NaN-free (모든 score 0 case 방지).
- §10.6 sample budget warning (params 추가 ~16K, soft cap 2146 위반,
  acknowledged).
- §18.8 SUMMARY block printed at end (1 epoch row).
- `local_validate.py` G1–G6 5/5 PASS (단 schema regenerated 후).

## 10. dataset-inference-auditor invocation (§18.6)

After upload/ package built:

```
Agent(subagent_type="general-purpose",
      prompt="Audit experiments/H019_twin_long_seq_retrieval/upload/. prior_h=H010. Confirm seq_max_lens 변경 → make_schema.py 재생성 + §18.8 SUMMARY block emit + §18.7 carry-forward.")
```

Expected: PASS — including §18.5 (schema regen), §18.7 (label_time fill_null
carry-forward), §18.8 (SUMMARY block).

BLOCK 시 fix → re-audit. PASS 받기 전 cloud upload 금지.

## 11. config_sha256 + git_sha (§4 reproducibility)

After patch applied + local sanity PASS:
- `git rev-parse --short=7 HEAD` → save to card.yaml.
- `sha256sum train_config.json` → save to card.yaml `config_sha256`.
- both written to SUMMARY block automatically via env vars.

## 12. Cost cap pre-launch audit (§17.6 critical)

Before T3 launch:
- 누적 cost 측정 (Taiji 가격 사용자 확인 + Subset A H018 + H022 estimate).
- Subset A 총: H018 $5 + H019 $15 + H022 $15 = $35.
- + 누적 ~46h Taiji × ? = TBD.
- campaign cap $100 / per-job cap T3 $15 audit.
- 초과 시 사용자 confirm 필수.

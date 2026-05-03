# H018 — Upload Package Patch Spec

> **Out of full-fork scope (Ralph PRD): patch spec only**. Construction
> of `experiments/H018_per_user_recency_weighting/upload/` (10+ file
> fork from H015) is a separate execution step. This document specifies
> file-by-file diffs to apply.

## 1. Base & sequence

- **Base**: `experiments/H015_recency_loss_weighting/upload/` byte-by-byte
  copy.
- **Mutation**: per-user time-decay weighting replaces H015 per-batch
  linear weighting.
- **Carry-forward**: H016 redefined OOF (label_time future-only) on by
  default.
- **§17.2 single mutation**: granularity (per-batch → per-user). exp
  form is sub-spec within same mechanism class (recency loss weighting),
  per challengers.md ④ (multi-mutation 우려 명시 + 정당화).

## 2. Files to modify (4 .py + 1 shell + README)

| File | Δ vs H015 | Role |
|---|---|---|
| `dataset.py` | + ~30 lines (per-domain ts extraction + days_since_last_event derivation) | Data |
| `trainer.py` | replace ~25 lines (H015 linear → H018 exp decay + clip + normalize) | Train loop |
| `train.py` | + 3 argparse args, replace 3 Trainer keys | CLI |
| `run.sh` | replace 3 H015 flags with 3 H018 flags + H016 OOF redefine flag | Entry |
| `README.md` | rewrite — H018 identity | Doc |
| `infer.py` / `model.py` / `utils.py` / `local_validate.py` / `make_schema.py` / `ns_groups.json` / `requirements.txt` | byte-identical | unchanged |

## 3. dataset.py patch

**Location**: `_convert_batch` (around H015 line 553-560).

**H015 baseline (current)**:
```python
timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
# H015: expose label_time for recency-aware loss weighting in trainer.
label_times = (batch.column(self._col_idx['label_time']).fill_null(0)
               .to_numpy(zero_copy_only=False).astype(np.int64))
```

**H018 addition (after H015 lines)**:
```python
# H018 — per-user days_since_last_event derivation.
# Source: max over 4 domain seq ts_fids (§3.1: a→39, b→67, c→27, d→26).
# Each domain_*_seq_<fid> is list<int64> with non-null elements.
TS_FIDS = (
    ('domain_a_seq_39', 'a'),
    ('domain_b_seq_67', 'b'),
    ('domain_c_seq_27', 'c'),
    ('domain_d_seq_26', 'd'),
)
SECONDS_PER_DAY = 86400.0

# Per-row max ts across all 4 domains (using existing _col_idx mapping).
last_event_ts = np.zeros(B, dtype=np.int64)
for col_name, _ in TS_FIDS:
    if col_name not in self._col_idx:
        continue  # schema mismatch tolerated
    col = batch.column(self._col_idx[col_name])
    # list<int64> column — extract per-row max via offsets/values.
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        offsets = col.offsets.to_numpy()
        values = col.values.to_numpy()
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if end > start:
                row_max = int(values[start:end].max())
                if row_max > last_event_ts[i]:
                    last_event_ts[i] = row_max

# days_since_last_event (gap >= 0; clip at 0 if last_event_ts > timestamps).
gap_seconds = np.maximum(timestamps - last_event_ts, 0).astype(np.float32)
days_since_last_event = gap_seconds / SECONDS_PER_DAY  # (B,)
```

**H018 addition to `result` dict** (around H015 line 648):
```python
result = {
    ...
    'label_time': torch.from_numpy(label_times),  # carry-forward H015
    'days_since_last_event': torch.from_numpy(days_since_last_event),  # H018
    ...
}
```

**§18.7 nullability check**: `timestamp` non-null per §3 + domain seq
ts_fid elements non-null per §3.5 → safe `to_numpy()`. New derived
`days_since_last_event` is numpy float32 (no nullability concern).

## 4. trainer.py patch

**Location**: `__init__` H015 args (around H015 line 81-145).

**H015 args (current)**:
```python
use_recency_loss_weighting: bool = False,
recency_weight_min: float = 0.5,
recency_weight_max: float = 1.5,
```

**H018 replacement**:
```python
# H018 — per-user exp time-decay loss weighting (replaces H015 per-batch linear).
use_per_user_recency: bool = False,
recency_tau_days: float = 14.0,
recency_weight_clip_min: float = 0.1,
recency_weight_clip_max: float = 3.0,
```

**`__init__` body (replace H015 state lines 134-145)**:
```python
# H018 — per-user recency weighting state
self.use_per_user_recency: bool = use_per_user_recency
self.recency_tau_days: float = float(recency_tau_days)
self.recency_weight_clip_min: float = float(recency_weight_clip_min)
self.recency_weight_clip_max: float = float(recency_weight_clip_max)

if use_per_user_recency:
    logging.info(
        f"H018 ENABLED: per-user exp recency weighting "
        f"(tau={recency_tau_days:.1f}d, clip=[{recency_weight_clip_min}, "
        f"{recency_weight_clip_max}], normalize mean=1.0)")
```

**`_train_step` weighting branch (replace H015 lines 467-488)**:
```python
# H018 — per-user exp time-decay loss weighting.
# weight = exp(-days_since_last_event / tau), per-batch normalize (mean=1.0),
# then clip to [clip_min, clip_max] for variance safety.
if self.use_per_user_recency:
    days = device_batch['days_since_last_event'].float()  # (B,)
    raw_weights = torch.exp(-days / self.recency_tau_days)  # (B,)
    # Per-batch normalize so mean weight = 1.0 (loss scale preserved).
    norm = raw_weights.mean().clamp(min=1e-6)
    weights = (raw_weights / norm).clamp(
        min=self.recency_weight_clip_min,
        max=self.recency_weight_clip_max,
    )  # (B,)
    # Replace reduction='mean' with weighted mean.
    if self.use_focal_loss:
        loss_per = self._compute_focal_loss(logits, labels, reduction='none')
    else:
        loss_per = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none'
        )
    loss = (weights * loss_per).mean()

    # Diagnostic — last batch every N steps (sample stats for SUMMARY block).
    if self.step % 100 == 0:
        with torch.no_grad():
            w_stats = {
                'mean': weights.mean().item(),
                'p10': weights.quantile(0.1).item(),
                'p50': weights.quantile(0.5).item(),
                'p90': weights.quantile(0.9).item(),
                'clip_lo_pct': (weights <= self.recency_weight_clip_min + 1e-6).float().mean().item(),
                'clip_hi_pct': (weights >= self.recency_weight_clip_max - 1e-6).float().mean().item(),
            }
            logging.info(f"H018 weight_stats step={self.step}: {w_stats}")
else:
    # Default reduction='mean' branch (no recency weighting).
    if self.use_focal_loss:
        loss = self._compute_focal_loss(logits, labels)
    else:
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
```

## 5. train.py patch

**Argparse additions (after H015 args removed)**:
```python
# H018 — per-user recency weighting (replaces H015 per-batch linear).
parser.add_argument('--use_per_user_recency', action='store_true')
parser.add_argument('--recency_tau_days', type=float, default=14.0)
parser.add_argument('--recency_weight_clip_min', type=float, default=0.1)
parser.add_argument('--recency_weight_clip_max', type=float, default=3.0)
# H016 carry-forward — OOF redefine future-only.
parser.add_argument('--oof_redefine', choices=['random', 'future_only'],
                    default='future_only')
```

**Trainer construction (replace H015 keys)**:
```python
trainer = Trainer(
    ...,
    # H018 — per-user recency.
    use_per_user_recency=args.use_per_user_recency,
    recency_tau_days=args.recency_tau_days,
    recency_weight_clip_min=args.recency_weight_clip_min,
    recency_weight_clip_max=args.recency_weight_clip_max,
)
```

**§18.8 SUMMARY block (mandatory addition)**:

After final epoch loop (end of `train.py main`), add:

```python
from collections import defaultdict

def emit_train_summary():
    """§18.8 — single SUMMARY block, copy-paste anchor for verify-claim."""
    git_sha = os.environ.get('GIT_SHA', 'unknown')[:7]
    cfg_sha = os.environ.get('CONFIG_SHA256', 'unknown')[:8]
    seed = args.seed
    ckpt_kind = 'best'  # we always export argmax(val_auc) ckpt

    # epoch_history accumulated by Trainer per epoch (must populate during loop).
    eh = trainer.epoch_history  # list of {epoch, train_loss, val_auc, oof_auc}

    if not eh:
        print("==== TRAIN SUMMARY (H018_per_user_recency_weighting) ====", flush=True)
        print(f"git={git_sha} cfg={cfg_sha} seed={seed} ckpt_exported={ckpt_kind}", flush=True)
        print("epoch | train_loss | val_auc | oof_auc", flush=True)
        print("(no epochs completed)", flush=True)
        print("==== END SUMMARY ====", flush=True)
        return

    best = max(eh, key=lambda h: h.get('val_auc', -1))
    last = eh[-1]

    print(f"==== TRAIN SUMMARY (H018_per_user_recency_weighting) ====", flush=True)
    print(f"git={git_sha} cfg={cfg_sha} seed={seed} ckpt_exported={ckpt_kind}", flush=True)
    print("epoch | train_loss | val_auc | oof_auc", flush=True)
    for h in eh:
        tl = h.get('train_loss', 'N/A')
        tl_s = f"{tl:.4f}" if isinstance(tl, float) else str(tl)
        print(f" {h['epoch']:>3}  |   {tl_s}   | {h['val_auc']:.4f}  | {h.get('oof_auc', 0.0):.4f}", flush=True)
    print(f"best=epoch{best['epoch']}  val={best['val_auc']:.4f}  oof={best.get('oof_auc', 0.0):.4f}", flush=True)
    print(f"last=epoch{last['epoch']}  val={last['val_auc']:.4f}  oof={last.get('oof_auc', 0.0):.4f}", flush=True)
    print(f"overfit={best['val_auc']-last['val_auc']:+.4f} (best_val − last_val)", flush=True)
    # Calibration — needs val predictions. Emit zeros if not computed (placeholder for future).
    pred_mean = trainer.last_val_pred_mean if hasattr(trainer, 'last_val_pred_mean') else 0.0
    label_mean = trainer.last_val_label_mean if hasattr(trainer, 'last_val_label_mean') else 0.0
    ece = trainer.last_val_ece if hasattr(trainer, 'last_val_ece') else 0.0
    print(f"calib pred={pred_mean:.3f} label={label_mean:.3f} ece={ece:.3f}", flush=True)
    print("==== END SUMMARY ====", flush=True)


emit_train_summary()
```

**Trainer.epoch_history population**: must add to `Trainer._train_one_epoch`
or wrapper — append `{'epoch': N, 'train_loss': loss_avg, 'val_auc':
val_auc, 'oof_auc': oof_auc_redefined}` after each epoch eval.
`val_auc` = `best_val_auc` so far / per-epoch val. `oof_auc` = redefined
OOF (H016 default).

## 6. run.sh patch

Replace H015 flags with H018 flags:

```bash
# (H015 baked args — remove)
# --use_recency_loss_weighting --recency_weight_min 0.5 --recency_weight_max 1.5

# (H018 NEW)
--use_per_user_recency
--recency_tau_days 14.0
--recency_weight_clip_min 0.1
--recency_weight_clip_max 3.0
--oof_redefine future_only          # H016 carry-forward
+ all H010/H008 mechanism flags 그대로
```

## 7. README.md (rewrite header)

```markdown
# H018 — per_user_recency_weighting

> Per-user exp time-decay loss weighting. Replaces H015 per-batch linear.
> H016 redefined OOF (future-only) carried forward as default measurement.
> §18.8 SUMMARY block emit at end of train.py.

Mutation: granularity change (per-batch label_time → per-user
days_since_last_event). exp(-days / tau=14), per-batch normalize
(mean=1.0), clip [0.1, 3.0].

Control: H015 (paired Δ → granularity isolation).
```

## 8. Local sanity check (§17.5 code-path verification only)

```bash
.venv-arm64/bin/python train.py \
  --num_epochs 1 \
  --train_ratio 0.05 \
  --use_per_user_recency \
  --recency_tau_days 14.0 \
  --oof_redefine future_only
```

Expected:
- Log line: `H018 ENABLED: per-user exp recency weighting (tau=14.0d, clip=[0.1, 3.0], normalize mean=1.0)`.
- Log line every 100 steps: `H018 weight_stats step=N: {'mean': 1.0..., 'p10': ..., ...}`.
- weight_stats `mean` ≈ 1.0 (normalize 작동).
- `clip_lo_pct` < 0.05, `clip_hi_pct` < 0.05 (clip 거의 안 발동).
- NaN-free.
- §18.8 SUMMARY block printed at end (1 epoch row).
- §10.6 trainable params ≤ 200 (model byte-identical, 추가 0).
- `local_validate.py` G1–G6 5/5 PASS.

## 9. dataset-inference-auditor invocation (§18.6)

After upload/ package built:

```
Agent(subagent_type="general-purpose",
      prompt="Audit experiments/H018_per_user_recency_weighting/upload/. prior_h=H015.")
```

Expected: PASS (all 9 rules) — including §18.7 (label_time fill_null
carry-forward) + §18.8 (SUMMARY block in train.py).

BLOCK 시 fix → re-audit. PASS 받기 전 cloud upload 금지.

## 10. config_sha256 + git_sha (§4 reproducibility)

After patch applied + local sanity PASS:
- `git rev-parse --short=7 HEAD` → save to card.yaml.
- `sha256sum train_config.json` → save to card.yaml `config_sha256`.
- both written to SUMMARY block automatically via env vars.

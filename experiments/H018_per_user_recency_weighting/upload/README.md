# H018 — per_user_recency_weighting

> Single mutation — **per-user exp time-decay loss weighting** (replaces
> H015 per-batch linear). weight = exp(-days_since_last_event / tau=14),
> per-batch normalize mean=1.0, clip [0.1, 3.0]. H010 mechanism (NS xattn)
> + H008 mechanism (DCN-V2 fusion) + envelope byte-identical. Mechanism
> stack 변경 0.
>
> **Mutation per §17.2**: granularity change (per-batch label_time → per-user
> days_since_last_event). exp form은 H017 carry-forward (한 mechanism
> class 안 finer specification). challengers.md ④ 정당화.
>
> **Control**: H015 (paired Δ → granularity isolation). secondary control:
> H010 corrected (anchor 0.837806).

## What changed (vs H015 fork)

| File | Δ | Reason |
|---|---|---|
| `dataset.py` | + ~12 줄 (per-domain max(seq_ts) accumulator + days_since_last_event derivation, exposed in batch dict) | H018 input feature |
| `trainer.py` | replace ~20 줄 (H015 per-batch linear branch → H018 per-user exp decay branch) | H018 mutation core |
| `train.py` | replace 3 args + 3 Trainer keys with 4 args + 4 Trainer keys (H015 legacy kept for sub-H sweep) + §18.8 SUMMARY emit at end (~17 줄) | CLI + verify-claim parser anchor |
| `run.sh` | replace H015 flags with H018 flags + EXP_ID env + identity comments | Entry |
| `README.md` | 본 파일 (rewrite) | Identity |
| 9 다른 .py / .json / .txt | byte-identical | Mechanism stack 변경 0 |

## Mechanism details

### dataset.py — `days_since_last_event` derivation

In `_convert_batch`:
1. Initialize `last_event_ts = np.zeros(B, dtype=np.int64)` before per-domain loop.
2. Inside existing per-domain ts processing block (`if ts_ci is not None:`),
   accumulate: `last_event_ts = np.maximum(last_event_ts, ts_padded.max(axis=1))`.
3. After per-domain loop:
   ```python
   gap_seconds = np.maximum(timestamps - last_event_ts, 0).astype(np.float32)
   days_since_last_event = gap_seconds / 86400.0
   result['days_since_last_event'] = torch.from_numpy(days_since_last_event)
   ```

ts_fids per §3.1: a→39, b→67, c→27, d→26 (already in `self._seq_plan[domain]`).

### trainer.py — per-user exp decay weighting

In `_train_step`:
```python
if self.use_per_user_recency:
    days = device_batch['days_since_last_event'].float()  # (B,)
    raw_weights = torch.exp(-days / self.recency_tau_days)
    norm = raw_weights.mean().clamp(min=1e-6)
    weights = (raw_weights / norm).clamp(
        min=self.recency_weight_clip_min,
        max=self.recency_weight_clip_max,
    )
    loss_per = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    loss = (weights * loss_per).mean()
```

per-batch normalize (mean=1.0) → loss scale 보존 → lr/optim 영향 0 →
paired Δ confound 작음. clip [0.1, 3.0] = variance safety (Frame C 우려).

### §18.8 SUMMARY block

Train.py 끝에서 단일 SUMMARY 블록을 stdout 에 print:
```
==== TRAIN SUMMARY (H018_per_user_recency_weighting, seed=42) ====
git=<sha7> cfg=<sha8> seed=42 ckpt_exported=best
epoch | train_loss | val_auc | oof_auc
(per-epoch trajectory printed above by trainer)
best=epoch?  val=0.XXXX  oof=0.XXXX
last=epoch?  val=0.XXXX  oof=0.XXXX  # (last==best via EarlyStopping load_best)
overfit=+0.0000 (best/last identical via EarlyStopping)
calib pred=N/A label=N/A ece=N/A  (not computed)
==== END SUMMARY ====
```

verify-claim 스킬이 정규식 파싱 (`==== TRAIN SUMMARY (` ... `==== END SUMMARY ====`).

## How to run

```bash
TRAIN_CKPT_PATH=/path/to/h018_seed42  bash run.sh
```

~3-4h wall (T2.4 H010 envelope). Cost ~$5.

## Bring-back artifacts (verify-claim 입력)

1. **§18.8 SUMMARY block** (마지막 ~13 줄 stdout, marker bracketed).
2. **Per-epoch lines** (trainer print: `epoch N: train_loss=... val_auc=...`).
3. **`eval auc: 0.XXXXXX`** (final platform AUC).
4. **inference time** (sec).
5. (선택) **H018 ENABLED 로그**: `H018 ENABLED: per-user exp recency weighting (tau=14.0d, clip=[0.10, 3.00], per-batch normalize mean=1.0)` 가 log 에 있어야 mutation 활성화 confirm.

## Falsification (per predictions.md)

| Δ vs H015 corrected | classification | mechanism implication |
|---|---|---|
| ≥ +0.005pt | strong | per-user granularity 검증 |
| [+0.001, +0.005pt] | measurable | 약 effect, retire 권고 |
| (−0.001, +0.001pt] | noise | granularity REFUTED |
| < −0.001pt | degraded | per-user variance disrupt |

Post H011-H013 trajectory analysis (predictions.md updated): noise outcome
~50% likely. H019 paradigm shift scaffold 사전 준비됨 (deferred per Option B).

## Mechanism reference (H010 carry-forward, byte-identical)

- backbone: PCVRHyFormer (per-domain encoder)
- fusion: DCN-V2 (`--fusion_type dcn_v2`)
- NS xattn: `--use_ns_to_s_xattn --ns_xattn_num_heads 4` (H010 mechanism)
- envelope: 10 epochs × 30% × patience=3 × batch=2048 × lr=1e-4
- seq_max_lens: a=64, b=64, c=128, d=128 (H010 default)

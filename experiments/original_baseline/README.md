# original_baseline — Organizer baseline + leakage-fix smoke envelope

> Working copy of `competition/` (organizer baseline, read-only) + H001 patches
> for honest train/val split + H004 backbone router (dormant). Designed to be
> uploaded to Taiji eval as a single flat folder for both training and inference.

## 1. What this is
- **PCVRHyFormer** (organizer baseline architecture, unchanged).
- **H001 patches active**: `--use_label_time_split` + 10% user OOF holdout —
  exposes whether organizer's row-group split was producing inflated val AUC.
- **H004 backbone router present but dormant**: model.py supports `--backbone {hyformer, onetrans}`,
  default = `hyformer` (organizer baseline path). OneTrans path inactive.
- **Smoke envelope**: `train_ratio=0.05`, `num_epochs=1`, halved `seq_max_lens`,
  `num_workers=2`, `buffer_batches=4` — ~3 min wall.
- **Inference optimizations**: batch_size=1024, num_workers=2, fp16 autocast,
  diagnostic path logging.

## 2. Files
| File | Role |
|---|---|
| `run.sh` | Entry point. Bakes smoke envelope + leak-fix flags. Override with `bash run.sh --num_epochs 3 --train_ratio 0.3`. |
| `train.py` | CLI driver. Routes through label_time split when `--use_label_time_split` is set. |
| `trainer.py` | Per-step training loop, validation, EarlyStopping, sparse re-init. tqdm postfix shows running avg loss. |
| `model.py` | PCVRHyFormer (organizer arch) + OneTrans router (dormant). |
| `dataset.py` | PCVRParquetDataset + `get_pcvr_data_v2` (label_time split). |
| `infer.py` | §13 contract entry. Reads ckpt, batch=1024, fp16 autocast, diagnostic logging. |
| `local_validate.py` | G1–G6 gate runner for predictions.json. |
| `make_schema.py` | Auto-generates schema.json from training parquet metadata. |
| `utils.py` | Logger, EarlyStopping, focal helper. |
| `ns_groups.json` | NS-token feature grouping reference (currently unused — `--ns_groups_json ""`). |
| `requirements.txt` | torch 2.7.1+cu126, etc. |

## 3. Platform contract
The platform must set:
- `TRAIN_DATA_PATH` — directory containing training `*.parquet`.
- `TRAIN_CKPT_PATH` — writable directory for checkpoints + `metrics.json`.
- (optional) `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH` — derive from CKPT if unset.

For inference:
- `EVAL_DATA_PATH` — eval parquet directory.
- `EVAL_RESULT_PATH` — writable dir for `predictions.json`.
- `MODEL_OUTPUT_PATH` — directory containing the trained checkpoint.

## 4. Smoke flow (this run.sh)
1. Upload all 11 files (or `tar` archive) to Taiji as flat directory.
2. `bash run.sh` → trains 1 epoch on 5% data with label_time split + 10% user OOF.
   - ~3 min wall, GPU memory ~8 GB.
   - Produces `metrics.json` with `best_val_AUC`, `best_oof_AUC`, `split_meta`.
3. `python infer.py` (or platform-triggered) → produces `predictions.json`.
   - ~10–20 min wall on full eval set.
   - Logs `[infer] OK: torch path produced N predictions` (or fallback warning).
4. Compare `best_val_AUC` ↔ platform AUC to diagnose split/distribution issues.

## 5. Fuller envelopes
- 30%-data, 3-epoch run: `bash run.sh --train_ratio 0.3 --num_epochs 3` (~30 min).
- 30%-data, 10-epoch run: `bash run.sh --train_ratio 0.3 --num_epochs 10 --patience 3` (~100 min).
- Full data: `bash run.sh --train_ratio 1.0 --num_epochs 5 --patience 3` (multi-hour).

## 6. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256.
- `train_config.json` sidecar at best_model dir; `infer.py` reads it to instantiate matching config.

# H005_focal_loss_calibration — Technical Report

> CLAUDE.md §17.2 one-mutation: PCVRHyFormer-anchor (H001 E_baseline_organizer)
> 와 byte-identical envelope, BCE → focal(α=0.25, γ=2.0) 만 변경.
> §17.4 rotation 첫 충족 (loss_calibration ≠ 직전 H 들의 unified_backbones).
> Required during Final Round per the platform notice.

## 1. Hypothesis & Claim
- Hypothesis: **H005_focal_loss_calibration**
- Claim: BCE 가 12% positive imbalance 환경에서 easy-negative 의 gradient 가
  hard-positive 를 dominate. Focal loss (Lin et al. ICCV 2017) 의 modulating
  factor `(1−p_t)^γ` + class balance `α_t` 가 hard-positive learning 강화 →
  +0.5pt 이상 val_AUC lift 측정.
- Compute tier (suggested): **T2.4 smoke (Taiji)**.
- Expected wall (smoke): ~3 min (E_baseline_organizer 와 동일, focal overhead ≈ 0).

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로.
**Loss 만 BCE → focal(α=0.25, γ=2.0)** (CLI flag).

H001 patches inherited (envelope = organizer-pure, but H005 keeps the
`label_time` patch infrastructure available via `--use_label_time_split` —
not used in this run.sh):

- **`label_time`-aware split** — 코드 보존, run.sh 미사용 (organizer split
  envelope 으로 PCVRHyFormer-anchor 와 paired 비교).
- **10% user_id OOF holdout** — 코드 보존, organizer split 모드라 비활성.

Schema 는 `make_schema.py` 로 자동 생성 (없으면). vocab sizes, list dims,
per-domain `ts_fid` 는 parquet metadata 에서 inferred.

## 3. Files
| File | Purpose |
|---|---|
| `run.sh` | Entry point — platform invokes `bash run.sh`. Bakes `--backbone hyformer --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`. |
| `make_schema.py` | Auto-generates `schema.json` from training parquet metadata. (Identical to H001/H004.) |
| `train.py` | Trainer driver. `--loss_type focal` branch existed in H001; H004 added backbone router which is set to `hyformer` here. |
| `trainer.py` | Pointwise BCE/Focal training loop, AUC + logloss validation. (Identical to H001/H004.) |
| `model.py` | `PCVRHyFormer` wrapper. H004 added OneTrans router; H005 uses `--backbone hyformer` so OneTrans path is dormant. |
| `dataset.py` | `PCVRParquetDataset` + patched `get_pcvr_data_v2` for label_time split. (Identical to H001/H004.) |
| `infer.py` | Reads `backbone`/etc from `train_config.json` so ckpt loads correctly. (Identical to H004.) |
| `local_validate.py` | G1–G6 gate runner. (Identical.) |
| `utils.py` | Logger, EarlyStopping, `sigmoid_focal_loss` helper. (Identical.) |
| `ns_groups.json` | NS-token feature grouping reference (currently unused). |
| `requirements.txt` | torch 2.7.1+cu126, etc. (Identical.) |

## 4. Platform contract
The platform must set:
- `TRAIN_DATA_PATH` — directory containing training `*.parquet`.
- `TRAIN_CKPT_PATH` — writable directory for checkpoints + `metrics.json`.
- (optional) `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH` — derive from CKPT if unset.

Then invoke `bash run.sh`. CLAUDE.md §17.8.7: no local-dev fallback.

## 5. Outputs
- `${TRAIN_CKPT_PATH}/metrics.json` — primary reporting source. Includes:
  - `best_val_AUC` (P2 판정)
  - `best_val_logloss` (P3 판정)
  - `split_meta`, `seed`, `git_sha`, `config_sha256`, `host`
  - `config.loss_type=focal`, `config.focal_alpha=0.25`, `config.focal_gamma=2.0`
- `${TRAIN_CKPT_PATH}/global_step*.best_model/` — model.pt + schema.json +
  train_config.json (used by inference).
- `${TRAIN_LOG_PATH}/train.log` — full training log.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over organizer baseline
- **H001 patches inherited**: label_time split (toggleable, off here), user-OOF
  holdout (toggleable, off here), path defaults, auto-generated schema.json,
  infer.py prior fallback.
- **H004 backbone router inherited**: `--backbone {hyformer, onetrans}` flag
  exists; H005 uses `hyformer` (PCVRHyFormer-anchor 단기 우세).
- **H005 mutation (new)**: `--loss_type bce` → `--loss_type focal --focal_alpha 0.25
  --focal_gamma 2.0`. Lin et al. ICCV 2017 표준. trainer 의 `_train_step` 의 loss
  branch (`loss_type == 'focal'`) 가 자동 라우팅. 코드 변경 0.

## 8. Reproducibility
- All seeds fixed (default 42); cuDNN deterministic mode.
- `metrics.json` records git SHA + config SHA256.
- `train_config.json` stores all CLI args (including `loss_type`, `focal_alpha`,
  `focal_gamma`, `backbone`); infer.py reads them to instantiate the matching
  model for ckpt loading.

# H028 — split_seed variance (cohort saturation diagnostic)

> Measurement H. 9 H 누적 val_auc 0.832~0.836 ceiling — 모든 H 가
> `split_seed=42` 공유 → 같은 val/OOF cohort. H028 = split_seed 42/43/44
> 3 launches → val 변동이 cohort 의존 검증.

## 검증 가설 (#1 hypothesis)

모든 prior H 같은 cohort fit → mechanism 차이 mask. split_seed 변경하면:
- **val 변동 > 0.005pt** → cohort saturation 확정. 향후 H multi-split 의무.
- **val 변동 < 0.001pt** → cohort 가 진짜 ceiling 아님. 다른 root cause.

## How to run (3 seeds)

```bash
TRAIN_CKPT_PATH=/path/h028_s42  bash run.sh --split_seed 42
TRAIN_CKPT_PATH=/path/h028_s43  bash run.sh --split_seed 43
TRAIN_CKPT_PATH=/path/h028_s44  bash run.sh --split_seed 44
```

각 ~3.5h (H023 regime — batch=1024). cost 3 × $5 = $15. parallel 가능.

**중요**: training `--seed 42` 는 모두 동일 (split-only 효과 isolation).

## Bring-back (per launch)

1. §18.8 SUMMARY block (best_val + best_oof)
2. per-epoch lines (val_auc trajectory)
3. `eval auc:` final platform AUC
4. inference time

3 결과 모두 회수 후 mean ± stdev (val 와 platform 따로) 산출.

## Diff vs H022 (mechanism unchanged)

- run.sh: identity comments + EXP_ID + batch=1024 (OOM safety, match H023)
- train.py: exp_id default
- 다른 11 files byte-identical (model.py / dataset.py / trainer.py / 등)

## §17.2 / §17.4 / §17.6

- §17.2 EXEMPT (measurement H, no mutation)
- §17.4 measurement re-entry (methodology framework)
- §17.6 cost $15 within $20 Subset C budget

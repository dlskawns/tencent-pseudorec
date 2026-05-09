# H030 — loss_type focal (loss isolation diagnostic)

> Measurement H. 9 H train_loss scale 다양 (0.12~0.32) — focal vs bce
> ambiguity 가능성 (H011/H012/H013/H018 train_loss ~0.12 = focal-like).
> H023 = bce explicit (best_val 0.8334) baseline. H030 = focal explicit
> paired Δ.

## 검증 가설 (#4 hypothesis)

prior H 의 train_loss 차이 (focal 0.12 vs bce 0.32) 가 진짜 loss config
다름 인지 isolation:
- **focal val_auc 가 bce (H023 0.8334) 와 다르면** → loss type 이 H 마다
  실제 다름 (사용자 override 가능성). cross-H val 비교 invalid carry-forward.
- **focal val_auc ≈ bce** → loss type 효과 없음. train_loss 차이는 다른
  원인 (예: weighting, scaling).

## How to run (1 seed default 42)

```bash
TRAIN_CKPT_PATH=/path/h030  bash run.sh
```

~3.5h (H023 regime — batch=1024). cost ~$5.

## Bring-back

1. §18.8 SUMMARY block
2. per-epoch lines + train_loss scale (focal-like 0.12 expected confirm)
3. `eval auc:` final platform AUC
4. inference time

## Diff vs H022 (mechanism unchanged)

- run.sh: `--batch_size 1024` + `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0` + EXP_ID
- train.py: exp_id default
- 다른 11 files byte-identical

## §17.2 / §17.4 / §17.6

- §17.2 single mutation = loss type (bce → focal).
- §17.4 measurement re-entry (methodology framework). H005 focal pre-correction
  era → invalid. H030 = first valid focal measurement.
- §17.6 cost ~$5.

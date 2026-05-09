# H029 — original default regime (Keskar trap diagnostic)

> Measurement H. 9 H 모두 batch=2048 (사용자 override) 또는 1024 (OOM
> fallback) 사용 → H010 train.py original default (batch=256, lr=1e-4)
> 한 번도 측정 안 됨. F-2 carry-forward: large batch + small lr =
> effective lr underpowered (Keskar generalization gap). H029 = original
> default 처음 측정.

## 검증 가설 (#2 hypothesis)

batch=2048/1024 + lr=1e-4 가 underpowered regime → val_auc 0.832 ceiling.
batch=256 + lr=1e-4 (default) 가:
- **val_auc > 0.840** → underpowered confirmed. 향후 H 모두 batch=256
  default 권장.
- **val_auc ≤ 0.836** → 동일 ceiling. Keskar 가설 REFUTED. 다른 root
  cause.

## How to run (1 seed default 42)

```bash
TRAIN_CKPT_PATH=/path/h029  bash run.sh
```

Wall: ~5-7h expected (8× steps per epoch vs batch=2048). cost $5-7.
Risk: wall 길어 — patience=3 으로 단축 가능. OOM risk: 작은 batch 라
0 위험.

## Bring-back

1. §18.8 SUMMARY block (best_val 가 ceiling 깨는지 핵심)
2. per-epoch lines (convergence rate 비교)
3. `eval auc:` final platform AUC
4. inference time

## Diff vs H022 (mechanism unchanged)

- run.sh: `--batch_size 256` (vs H022 의 2048) + identity + EXP_ID
- train.py: exp_id default
- 다른 11 files byte-identical

## §17.2 / §17.4 / §17.6

- §17.2 single mutation = optimization regime (batch+lr).
- §17.4 measurement re-entry (methodology framework).
- §17.6 cost $5-7 within Subset C budget.

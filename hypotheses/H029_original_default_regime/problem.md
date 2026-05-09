# H029 — Problem (original default regime, Keskar trap diagnostic)

## Background

9 H val_auc ceiling 0.832~0.836. F-2 carry-forward: 사용자가 `--batch_size
2048` override 사용 (H011 이후) → effective lr 1/8 underpowered (Keskar
large-batch generalization gap). H013 가 lr=8e-4 (linear scaling) 시도
했지만 worse — 단 sub-linear (e.g., lr=2e-4) 미시도. **H010 train.py
original default `--batch_size 256 --lr 1e-4` 한 번도 측정 안 됨**.

## Falsifiable claim

batch=256 + lr=1e-4 default regime 가:
- **val_auc > 0.840** → underpowered hypothesis (#2) confirmed. 모든 prior
  H 의 val 측정 invalid (다른 effective lr regime).
- **val_auc ≤ 0.836** → Keskar 가설 REFUTED. ceiling 이 batch/lr 무관.

## Measurement only — NO model mutation

H010 mechanism + envelope byte-identical. **batch_size + lr 만 변경
(single mutation)**.

## Decision tree

| val_auc | Keskar | next |
|---|---|---|
| > 0.840 | confirmed | future H 모두 batch=256 default. anchor 갱신 가능성. |
| 0.836~0.840 | partial | sub-linear scaling sweep (lr=2e-4 / 4e-4) |
| ≤ 0.836 | REFUTED | ceiling 이 batch/lr 무관, cohort/loss/Bayes hypothesis 우선 |

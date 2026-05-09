# H023 — variance_baseline_redo (H022 debug)

> Measurement H redo. H022 produced anomalous result (best_val 0.8322,
> best_epoch=2/5, OOF=N/A) — H023 fixes ambiguity + adds debug.
>
> NO model mutation. H010 mechanism + envelope byte-identical, seed
> varies (42/43/44).

## What changed (vs H022 fork)

| File | Δ | Reason |
|---|---|---|
| `run.sh` | + `--loss_type bce` (explicit bake) | H010 anchor regime 명시 (default 와 동일하지만 reproducibility) |
| `train.py` | + `[H023 DEBUG]` prints around OOF eval (oof_loader/best_model None? OOF eval try/except) | H022 의 OOF=N/A 원인 isolation |
| `train.py` | + `print(f"OOF AUC: ...")` to stdout (mirror logging.info) | platform stdout 으로 user 가 직접 회수 |
| 다른 11 files | byte-identical | mechanism + dataset + model 변경 없음 |

## Why redo

H022 anomaly:
- best_val 0.8322 — H011/H012/H013 (0.8316~0.8336) 영역. H010 corrected
  anchor 0.837806 의 정합 baseline 으로 보기 어려움 (val→platform mean
  −0.003pt 적용 시 platform expected 0.835 영역 — actual 0.838 와 +0.003pt
  off).
- best_epoch=2/5 (early stop ep5) — H010~H018 의 4-6 / 7-9 와 다름.
- OOF=N/A — `oof_loader` 또는 `best_model` 가 None 이거나 evaluate() 가
  raise. H023 debug print 으로 어느 path 인지 확정.
- train_loss 0.21 — H018 0.12 / H015 0.32 과 다른 scale. loss_type
  ambiguity 가능성.

## How to run (3 seeds)

```bash
TRAIN_CKPT_PATH=/path/h023_seed42  bash run.sh --seed 42
TRAIN_CKPT_PATH=/path/h023_seed43  bash run.sh --seed 43
TRAIN_CKPT_PATH=/path/h023_seed44  bash run.sh --seed 44
```

각 ~3-4h. 3 GPU/slot parallel ~3.5h, serial ~10.5h. cost ~$15 total.

## Bring-back artifacts (per seed)

1. **§18.8 SUMMARY block** (`==== TRAIN SUMMARY (...) ====` ~ `==== END SUMMARY ====`).
2. **`[H023 DEBUG]` lines** — oof_loader/best_model None 여부 + (있으면) OOF eval raise message.
3. **`OOF AUC: 0.XXXX`** stdout line — H023 fix 작동 확인.
4. **Per-epoch lines** (trainer print).
5. **`eval auc: 0.XXXXXX`** (final platform AUC).

## Decision tree (post-result)

| OOF eval status | best_val | next action |
|---|---|---|
| OOF computed (e.g., 0.8589) | ≥ 0.8336 | H022 anomaly resolved — σ classification 산출 가능 (3 seeds mean ± stdev). |
| OOF eval RAISED (specific error) | any | error message → root cause fix → H024 또는 H023-sub re-launch. |
| OOF eval SKIPPED (oof_loader None) | any | use_label_time_split path issue → split_meta inspect → fix dataset.py. |
| OOF eval SKIPPED (best_model None) | any | EarlyStopping never saved best — patience or save_best logic bug. |
| best_val 여전히 anomalous (0.8322 영역) | ≤ 0.833 | loss_type bce explicit 도 이런 결과면 H010 anchor regime 자체가 default 가 아닌 user override 였음 → H010 ckpt path 의 train_config 확인 필요. |

## §17.2 EXEMPT (challengers carry-forward from H022)

> "H023 = measurement H, no mechanism mutation. H022 의 debug + bug fix.
> mutation rule 의 'one component 교체' 는 mutation H 적용 — measurement
> H exempt. paired Δ statistical foundation 결정용 infrastructure
> investment."

## §17.4 rotation

`measurement` 카테고리 재진입. §17.4 cite: H022 anomaly 가 framework
broken 신호 → measurement H 재진입 정당화. paradigm shift (H019 deferred)
의사결정의 prerequisite.

## §17.6 cost

per-seed: T2.4 ~$5. total: $15. Subset B 두 번째 사이클 budget 안.

## Mechanism reference (H010 carry-forward, byte-identical)

- backbone: PCVRHyFormer
- fusion: DCN-V2 (`--fusion_type dcn_v2`)
- NS xattn: `--use_ns_to_s_xattn --ns_xattn_num_heads 4` (H010 mechanism)
- envelope: 10 epochs × 30% × patience=3 × batch=2048 × lr=1e-4
- loss: `--loss_type bce` (H023 explicit, default 와 동일)
- seq_max_lens: a=64, b=64, c=128, d=128

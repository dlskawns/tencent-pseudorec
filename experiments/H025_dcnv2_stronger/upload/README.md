# H022 — H010 multi-seed variance baseline (measurement H)

> Measurement H — NO model mutation. H010 mechanism + envelope byte-
> identical, seed only varies (42 / 43 / 44). Output: 3 platform AUC
> values → mean ± stdev → variance baseline for all 9 prior + future
> paired Δ comparisons.

## What this is

H022 = **measurement H**, not mutation. Per §17.2 (one-mutation rule),
mutation H 의 component class 교체 rule 은 적용 안 됨 — measurement H
exempt.

**Goal**: H010 mechanism 의 platform AUC 의 stdev (σ) 측정. σ 가:
- **tight (≤ 0.001pt)** → 기존 9 H paired Δ 분류 trustworthy, 향후 H
  single-seed 측정 valid.
- **moderate ((0.001, 0.005pt])** → marginal Δ (H015 +0.0002 같은) 재분류
  권고. §17.3 threshold raise 검토.
- **large (> 0.005pt)** → 9 H 모두 single-seed 측정 INVALID, future H
  multi-seed (≥ 3) 의무, cost cap audit STRICT.

## Diff vs H010/upload/

| File | Δ | Reason |
|---|---|---|
| `train.py` | + ~17 줄 (§18.8 SUMMARY emit at end) | verify-claim parser anchor |
| `run.sh` | header comment 변경 + EXP_ID env | identity |
| `README.md` | 본 파일 (rewrite) | identity |
| 모든 다른 .py | byte-identical | mechanism unchanged |

## How to run (3 seeds)

```bash
# Per seed (in own ckpt dir):
TRAIN_CKPT_PATH=/path/to/h022_seed42  bash run.sh --seed 42
TRAIN_CKPT_PATH=/path/to/h022_seed43  bash run.sh --seed 43
TRAIN_CKPT_PATH=/path/to/h022_seed44  bash run.sh --seed 44
```

각 launch ~3-4h (H010 envelope 동일). 3 GPU/slot parallel 가능 → wall
~3.5h. serial ~10.5h.

**Note**: seed 42 결과는 H010 corrected re-inference (0.837806) 과 동일
expected (env identical). 검증 차원에서 launch 권장. seed 43/44 NEW.

## Bring-back artifacts (verify-claim 입력)

각 seed 마다:
1. **§18.8 SUMMARY block** (마지막 ~13줄 stdout, `==== TRAIN SUMMARY (H022_h010_multi_seed_variance, seed=N) ====` 부터 `==== END SUMMARY ====` 까지).
2. **Per-epoch lines** (trainer 가 별도 print — `epoch N: train_loss=... val_auc=...`).
3. **`eval auc: 0.XXXXXX`** (final platform AUC).
4. **inference time** (sec).

3 seeds 결과 모두 받으면 verify-claim 가 mean ± stdev → σ classification
산출.

## §17.2 EXEMPT 정당화 (challengers.md ④ cite)

> "H022 = measurement H, no mechanism mutation. H010 byte-identical 3
> 회 학습 (seed 만 변경). §17.2 의 'one component 교체' rule 은 mutation
> H 적용 — measurement H exempt. paired Δ statistical foundation 결정용
> infrastructure investment."

## §17.4 rotation auto-justified

`measurement` 카테고리 first-touch. 직전 H (H015~H018) temporal_cohort,
H019 retrieval_long_seq 와 완전히 다른 axis. rotation strict 충족.

## §17.6 cost

- per-seed: T2.4 ~$5
- total: 3 × $5 = $15 (within Subset A $35 budget, within $100 campaign cap)

## Mechanism reference (H010 carry-forward, byte-identical)

- backbone: PCVRHyFormer (per-domain encoder)
- fusion: DCN-V2 (`--fusion_type dcn_v2`)
- NS xattn: `--use_ns_to_s_xattn --ns_xattn_num_heads 4` (H010 mechanism)
- envelope: 10 epochs × 30% × patience=3
- seq_max_lens: a=64, b=64, c=128, d=128 (H010 default)

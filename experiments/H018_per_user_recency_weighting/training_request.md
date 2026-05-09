# H018 — Cloud Training Request

> Generated per CLAUDE.md §17.8. H018 upload package implemented.

## 1. Hypothesis & claim
- Hypothesis: **H018_per_user_recency_weighting**
- Control: `H015_recency_loss_weighting` (paired Δ primary)
- Mutation: replace per-batch linear weighting with **per-user exp time-decay**
  - `days_since_last_event = max(0, timestamp - max(domain_ts_fids)) / 86400`
  - `weight = exp(-days / tau=14)`
  - per-batch normalize mean=1.0, clip `[0.1, 3.0]`
- Success criterion: corrected platform AUC **Δ vs H015 ≥ +0.005pt**

## 2. Compute tier
- Tier: `T2.4` extended
- Runtime: ~3-4h
- Cost cap: ≤ $5

## 3. Upload manifest
Path: `experiments/H018_per_user_recency_weighting/upload/`

| File | Status vs H015/upload | Role |
|---|---|---|
| `dataset.py` | changed | per-user `days_since_last_event` derivation |
| `trainer.py` | changed | H018 exp weighting logic + weight stats logging |
| `train.py` | changed | H018 CLI + trainer args + §18.8 SUMMARY block |
| `run.sh` | changed | H018 run flags + H016 `--oof_redefine future_only` |
| `README.md` | changed | H018 identity/spec |
| `infer.py` | byte-identical | inference |
| `local_validate.py` | byte-identical | validation harness |
| `make_schema.py` | byte-identical | schema helper |
| `model.py` | byte-identical | model |
| `ns_groups.json` | byte-identical | tokenizer groups |
| `requirements.txt` | byte-identical | deps |
| `utils.py` | byte-identical | utilities |

## 4. Launch command
```bash
bash run.sh
```

H018-specific baked flags:
```text
--use_per_user_recency
--recency_tau_days 14.0
--recency_weight_clip_min 0.1
--recency_weight_clip_max 3.0
--oof_redefine future_only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
batch retry ladder: 1024 -> 512 -> 256 (on non-zero exit)
```

## 5. Required bring-back artifacts
1. `metrics.json` (best_val_AUC, best_oof_AUC, git_sha, config_sha256)
2. train log tail including:
   - `H018 ENABLED: per-user exp recency weighting ...`
   - `H018 weight_stats step=...`
3. §18.8 SUMMARY block:
   - start: `==== TRAIN SUMMARY (H018_per_user_recency_weighting) ====`
   - end: `==== END SUMMARY ====`
4. Platform eval AUC

## 6. Verdict update path
- Update `hypotheses/H018_per_user_recency_weighting/verdict.md`
- Apply decision tree in `card.yaml`:
  - strong: anchor candidate
  - measurable/noise/degraded: retire temporal_cohort and rotate to H019 paradigm shift

## 7. Pre-flight checklist
- [x] upload folder built from H015 baseline
- [x] H018 code patch applied (dataset/trainer/train/run/README)
- [x] card.yaml updated to BUILT status
- [x] local sanity run (`bash run.sh --num_epochs 1 --train_ratio 0.05 --num_workers 0 --batch_size 256 --sparse_lr 0.005`)
- [ ] dataset-inference-auditor pass (§18.6)
- [ ] tar.gz packaging and upload

## 8. Sanity run note (2026-05-03)

- Sample-1000 sanity run executed successfully with H018 package.
- Added stability guard in `trainer.py`:
  - skip batch update when logits/loss/gradients are non-finite
  - prevents parameter corruption and NaN propagation in sample-scale runs
- Validation after guard:
  - train completed and `metrics.json` dumped
  - `local_validate.py` gates: **5/5 PASS**

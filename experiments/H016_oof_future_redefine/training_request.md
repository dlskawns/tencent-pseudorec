# H016 — Cloud Training Request

> Generated 2026-05-03. **Triple-H setup with H015 / H017 동시 launch** (L2
> multi-form attack).

## 1. Hypothesis & Claim
- **H016_oof_future_redefine** = OOF 정의 변경 (random user → label_time future-only).
- Mutation: `--oof_split_type future_label_time`.
- Goal: cohort drift 의 OOF-side attack (Frame C). 9 H 의 OOF stable / Platform
  변동 패턴이 OOF measure 정합성 문제인지 검증.
- **Caveat**: paired Δ baseline 일부 깨짐 (OOF 정의 다름) — Platform 비교만
  valid.

## 2. Compute tier
- T2.4 extended (10ep × 30%, patience=3) ~3-4h.
- 누적 cost: 36h + H016 ~3.5h = ~40h. cap 위협.

## 3. Upload manifest
- 경로: `experiments/H016_oof_future_redefine/upload/`
- tar.gz: 65,220 bytes. 12 files (H010 base + dataset.py + train.py + run.sh + README).

## 4. Run command
```
bash run.sh
```
internal flags: H010 mechanism (NS xattn + DCN-V2) byte-identical 외:
- `--oof_split_type future_label_time` (H016 NEW).
- `--batch_size 2048 --lr 1e-4` (default 명시 bake).

## 5. Bring-back artifacts
- `metrics.json` (best_val_AUC, best_oof_AUC, attn_entropy, oof_split_type
  확인, split_meta 의 oof_cutoff 값).
- log tail (split logging "future_label_time" 모드 확인).
- Platform AUC.
- Wall.

## 6. Verdict path
- **Platform AUC vs H010 corrected anchor (0.837806)** — primary.
- OOF AUC: 단일 측정값 (prior H 비교 invalid, 새 정의).
- card.yaml decision_tree 분기.

## 7. Pre-flight
- [x] H016 코드 빌드 (dataset.py + train.py + run.sh + README).
- [x] ast.parse OK.
- [x] tar.gz.
- [ ] Taiji upload + launch.

## 8. Build status: ✅ BUILT (2026-05-03)

## 9. Mechanism verification (run 후 metrics.json 확인 사항)
- `split_meta.oof_split_type` = 'future_label_time'.
- `split_meta.oof_cutoff` = quantile(label_time, 0.9) value.
- `split_meta.label_time_cutoff` = quantile of label_time < oof_cutoff (val cutoff).
- train cohort ~85% × dataset size.

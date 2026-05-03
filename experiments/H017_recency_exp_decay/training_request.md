# H017 — Cloud Training Request

> Generated 2026-05-03. **Triple-H setup with H015 / H016 동시 launch**.

## 1. Hypothesis & Claim
- **H017_recency_exp_decay** = H015 sub-form variant.
- Mutation: `--recency_weight_form exp` (was `linear` default).
- Goal: linear vs exp curve 효과 분리 (cohort drift 처리 적정 form 결정).

## 2. Compute tier
- T2.4 extended (10ep × 30%, patience=3) ~3-4h.
- 누적 cost: 36h + H017 ~3.5h = ~40h. cap 위협.

## 3. Upload manifest
- 경로: `experiments/H017_recency_exp_decay/upload/`
- tar.gz: 64,789 bytes. 12 files (H015 base + 1 trainer.py + 1 train.py + run.sh + README).

## 4. Run command
```
bash run.sh
```
internal flags: H015 byte-identical 외 `--recency_weight_form exp`.

## 5. Bring-back artifacts
- `metrics.json` (best_val_AUC, best_oof_AUC, attn_entropy, recency_weight_form 확인).
- log tail (H015 ENABLED + recency_weight_form=exp 확인).
- Platform AUC.
- Wall.

## 6. Verdict path
- vs H010 corrected anchor (0.837806).
- vs H015 paired sibling (form 효과 isolated).
- card.yaml decision_tree 분기.

## 7. Pre-flight
- [x] H015 코드 빌드.
- [x] H017 ast.parse OK.
- [x] tar.gz.
- [ ] Taiji upload + launch.

## 8. Build status: ✅ BUILT (2026-05-03)

## 9. Triple-H 동시 launch 권장 순서
1. H015 (먼저, baseline form).
2. H016 (parallel, 다른 form).
3. H017 (parallel, sub-form).
모두 ~3-4h, 동시 실행 시 wall ~4h.

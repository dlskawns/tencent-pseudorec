# Subset A → **Subset B** Launch Guide (H022 + H018 only — H019 deferred)

> Generated 2026-05-03. **Option B 선택 (2026-05-03)**: H019 TWIN 통합
> risk 회피, H018 + H022 만 launch ($20, parallel ~3.5h wall). H019 는
> Subset B 결과 회수 후 dedicated session 에서 build.

## Option B summary

| H | Status | Cost | Wall (parallel) |
|---|---|---|---|
| **H018** | BUILT (사용자) — upload.tar.gz exists | T2.4 ~$5 | ~3-4h |
| **H022** | BUILT (this turn) — upload.tar.gz exists 63KB | T2.4 × 3 ~$15 | ~3.5h (3 GPU/slot) |
| ~~H019~~ | DEFERRED | — | — |

**총 cost**: $20 / $100 per-campaign cap.
**총 wall (parallel)**: ~3.5h (H018 + H022 동시).

---

## Subset A — 3 H 요약

| H | Category | Cost | Wall (parallel) | Status | 핵심 정보 가치 |
|---|---|---|---|---|---|
| **H018** | temporal_cohort (4th) | T2.4 ~$5 | ~3-4h | **READY** (upload BUILT by user, training_request.md exists) | per-user recency final attempt, granularity hypothesis 검증 |
| **H019** | retrieval_long_seq (NEW) | T3 ~$15 | ~6h | upload_patch.md COMPLETE, upload/ NOT BUILT | paradigm shift first entry, sequence axis 강화 |
| **H022** | measurement (NEW) | T2.4 × 3 ~$15 | ~3.5h (3 GPU/slot parallel) | scaffold COMPLETE, H010 + minimal patch only | **모든 paired Δ 신뢰도 검증** (variance baseline) |

**총 cost**: $35 / $100 per-campaign cap (안전 35% 사용).
**총 wall (parallel max)**: ~6h (H019 가 가장 김, H018/H022 그 안에 끝남).

## Pre-launch checklist

### H018 (READY — user-built)

- [x] upload/ package built (사용자 directly)
- [x] upload.tar.gz exists
- [x] training_request.md exists
- [x] §18.7 carry-forward (H015 patch)
- [x] §18.8 emit_train_summary (per upload_patch.md §5)
- [ ] dataset-inference-auditor invoke (§18.6) — **권장 — fix 후 PASS 받고 launch**
- [ ] cloud submit Taiji T2.4

### H019 (upload_patch.md ready, upload/ build needed)

- [x] hypothesis docs (6 files) complete
- [x] card.yaml complete
- [x] upload_patch.md complete
- [ ] **fork H010/upload/** → apply upload_patch.md (4 .py + run.sh + README + make_schema)
- [ ] make_schema.py 재생성 (seq_max_lens 64-128 → 512)
- [ ] local sanity check (§17.5):
      ```
      python train.py --num_epochs 1 --train_ratio 0.05 \
        --use_twin_retrieval --twin_top_k 32 --twin_seq_cap 256 \
        --oof_redefine future_only
      ```
- [ ] dataset-inference-auditor invoke (§18.6)
- [ ] tar.gz build
- [ ] training_request.md write
- [ ] cloud submit T3 (Lambda/RunPod 권장 — Taiji 가격 위협 시)

### H022 (no new upload — H010 + minimal patch)

- [ ] **H010 train.py 검증**: §18.7 (label_time fill_null) + §18.8
      (emit_train_summary) 적용 여부
- [ ] 미적용 시 minimal patch (≤ 30 줄):
  - dataset.py §18.7 fix (H015 carry-forward)
  - train.py emit_train_summary() at end (per H018 upload_patch.md §5)
- [ ] 3 GPU/slot 확보 (parallel) OR sequential schedule
- [ ] 3 launches (seed 42 = H010 result re-use OR re-launch / seed 43, 44 NEW):
      ```
      python train.py --seed 42  # OR re-use H010 0.837806
      python train.py --seed 43
      python train.py --seed 44
      ```
- [ ] cloud submit T2.4 × 3

## Launch sequence options

### Option A: Full parallel (권장, ~6h wall)

3 H 동시 launch → max wall = H019 ~6h.
- t=0: H022 seed 43, H022 seed 44, H018, H019 simultaneous launch
- t=3.5h: H022 (3 seeds), H018 done — partial verification possible
- t=6h: H019 done — full Subset A complete

**조건**: Taiji T2.4 3 slot + Lambda/RunPod T3 1 slot 동시 확보.

### Option B: Sequential T2 first (cheaper, slower, ~12h wall)

H018 + H022 (3 seeds) → H019 last.
- t=0~3.5h: H022 sequential 또는 parallel + H018 parallel
- t=3.5~6h: H019 launch
- t=6~12h: H019 complete

**조건**: T3 slot 부족 시.

### Option C: Stage-gated (most conservative)

H022 first → variance result 확인 → H018/H019 launch decision.
- t=0~3.5h: H022 (parallel 3 seeds)
- t=3.5h: σ classification 결정 → H018/H019 launch decision (cost cap audit)
- t=3.5~9.5h: H018 + H019 parallel

**조건**: σ large 가능성 우려 시 (variance 측정 후 H018/H019 multi-seed
의무 가능 → cost cap 위협 회피).

## Cost cap audit (§17.6 critical)

**Pre-launch**:
- 누적 (H006~H017): ~$? Taiji 가격 사용자 확인 필요 + ~42시간.
- Subset A 추가: $35 ($5 + $15 + $15).
- 누적 + Subset A: 사용자 측정 후 confirm.
- **Per-campaign cap $100** 위반 시 user confirm 필수.

**During-run**:
- 각 H 의 actual cost 가 estimate 초과 (50%+) → abort + investigate.

**Post-Subset-A**:
- 결과 따라 H020+ 결정.
- σ tight + H018 strong + H019 strong → champion candidates 다양 → ensemble 가능.
- σ large → 모든 future H 3-seed 의무 → cost 3× → H020+ 보류.

## Bring-back artifacts (after each H done)

각 H 마다:
1. **§18.8 SUMMARY block** (마지막 ~15 줄 stdout — H018/H019 emit, H022 emit × 3 seeds).
2. **Platform AUC** (Taiji 채점 결과 단일 숫자).
3. **identity** (git_sha + config_sha — SUMMARY block 안에 있음).
4. (선택) **error log** (NaN, OOM, etc. — 있으면).

→ paste 후 verify-claim 스킬 invoke (3 H 모두 처리).

## Decision matrix (post-Subset-A)

| H022 σ | H018 result | H019 result | Action |
|---|---|---|---|
| **tight ≤ 0.001pt** | strong | strong | dual champions → ensemble candidate. H020 = sub-H tau sweep + TWIN top-K sweep. |
| tight | strong | noise/REFUTED | H018 anchor 갱신. retrieval class retire 또는 보류. H020 = H018 sub-H. |
| tight | noise/REFUTED | strong | H019 anchor 갱신. paradigm shift class confirm. H020 = TWIN sub-H. |
| tight | noise | noise | 두 mechanism class 모두 ceiling. H020 = backbone_replacement (HSTU) 또는 debiasing — cost cap audit STRICT. |
| **moderate (0.001, 0.005pt]** | any | any | marginal Δ 재분류 권고. §17.3 threshold +0.01pt 로 raise. H020+ multi-seed RECOMMENDED. |
| **large > 0.005pt** | any | any | 모든 future H 3-seed 의무. cost 3× → H020 paradigm shift 보류 (T3 × 3 = $45+). H020 = ensemble of best Subset A H 또는 measurement-only H. |

## Notes

- **H018 = user-built**: 사용자가 directly upload package 만들었음. card.yaml `build_status.upload_package: BUILT` 반영. dataset-inference-auditor invoke 권장 (independent verification).
- **H019 upload_patch.md complete**: TWINBlock module impl + dataset.py seq_max_lens cap + train.py emit_train_summary spec. fork H010 + apply patch + auditor 후 launch.
- **H022 no new package**: H010 byte-identical + minimal §18.7/§18.8 patch. 가장 cheap to set up.
- **§17.4 rotation 모두 충족**: temporal_cohort (H018) + retrieval_long_seq (H019) + measurement (H022) 3 different categories.
- **§18.8 SUMMARY emit**: H018 (1st), H019 (2nd), H022 (3rd × 3 seeds = 5 total) — verify-claim parser 의 first major test.

---

## Subset C — Methodology validation set (2026-05-04)

> **Trigger**: 9 H 누적 val_auc 0.832~0.836 narrow band → measurement
> framework 자체 의문. paradigm shift (H019 TWIN $15) 보류, methodology
> validation 우선.

### 5 root cause hypotheses + 3 H scaffold

| H | Hypothesis | Mutation | Cost | Wall | tar.gz |
|---|---|---|---|---|---|
| **H028** | #1 cohort saturation (split_seed=42 shared) | --split_seed 42/43/44 sweep | T2.4 × 3 ~$15 | ~3.5h parallel / ~10.5h serial | `H028_split_seed_variance/upload.tar.gz` (63KB) |
| **H029** | #2 Keskar trap (batch=2048 underpowered) | --batch_size 256 + --lr 1e-4 (H010 train.py default) | T2.4 ~$5-7 | ~5-7h | `H029_original_default_regime/upload.tar.gz` (63KB) |
| **H030** | #4 loss_type ambiguity (focal vs bce mixing) | --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0 | T2.4 ~$5 | ~3.5h | `H030_loss_type_focal/upload.tar.gz` (62KB) |

(#3 val set too small + #5 Bayes ceiling — out of scope this iteration,
deferred to future H if Subset C 결과 inconclusive.)

**총 cost**: $25-27 / $100 per-campaign cap (안전 27% 사용).
**총 wall (parallel max)**: ~5-7h (H029 가 가장 김).

### Launch commands (5 cloud jobs total)

```bash
# H028 — split_seed sweep (3 launches)
TRAIN_CKPT_PATH=/path/h028_s42  bash run.sh --split_seed 42
TRAIN_CKPT_PATH=/path/h028_s43  bash run.sh --split_seed 43
TRAIN_CKPT_PATH=/path/h028_s44  bash run.sh --split_seed 44

# H029 — original default regime (1 launch)
TRAIN_CKPT_PATH=/path/h029  bash run.sh

# H030 — focal explicit (1 launch)
TRAIN_CKPT_PATH=/path/h030  bash run.sh
```

### Bring-back (per launch)

1. **§18.8 SUMMARY block** (`==== TRAIN SUMMARY (HXXX, seed=N) ====` ~ `==== END SUMMARY ====`)
2. **Per-epoch lines** (val_auc trajectory + train_loss scale)
3. **`OOF AUC: 0.XXXX` line** (H023 fix carry-forward)
4. **`eval auc: 0.XXXXXX`** (final platform AUC)

5 results 모두 회수 → verify-claim 스킬 invoke.

### Decision matrix (post-Subset C)

| Result pattern | Root cause | Next H |
|---|---|---|
| H028 σ_val > 0.005pt | cohort saturation confirmed | future H multi-split mandatory; H019 TWIN with multi-split |
| H028 σ_val ≤ 0.001pt + H029 val > 0.840 | Keskar trap confirmed | future H batch=256 default; recompute H010 anchor with corrected regime |
| H028 σ_val ≤ 0.001pt + H029 val ≤ 0.836 + H030 |Δ| > 0.005 | loss_type effect, prior cross-H invalid | normalize loss_type explicit per H going forward |
| All ≤ thresholds | Bayes ceiling가 진짜 limit (#5 hypothesis) | task ceiling 인정, ensemble (H024) + multi-modal (P3 phase) 대기 |

### Rotation audit

3 H 모두 `measurement` 카테고리 re-entry. 정당화: H022/H023 sibling +
methodology framework 검증 prerequisite for paradigm shift decisions.
post-Subset C → 결과 따라 다음 H category 결정 (paradigm shift TWIN 또는
다른 mechanism class).

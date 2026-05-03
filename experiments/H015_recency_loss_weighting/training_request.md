# H015 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 2026-05-03.

## 1. Hypothesis & Claim
- Hypothesis: **H015_recency_loss_weighting**.
- **Mechanism**: per-batch linear recency-aware loss weighting (label_time
  → percentile → linear weight [0.5, 1.5], mean = 1.0).
- Goal: 4-layer ceiling diagnosis 의 마지막 가설 **L2 (cohort drift hard
  ceiling)** 직접 검증. paradigm 안 마지막 시도.
- Predicted classifications:
  - **strong** Δ ≥ +0.005pt + P6 gap 줄어듦 → L2 confirmed (Platform ≥ 0.8428).
  - **measurable** Δ ∈ [+0.001, +0.005pt] → L2 partial.
  - **noise** Δ ≤ +0.001pt → **L2 retire, paradigm shift mandatory**.
  - **degraded** Δ < −0.001pt → weighting 학습 disrupt.
- §17.2 정당화: single mutation (loss weighting only, mechanism stack
  byte-identical). mean weight = 1.0 → loss scale 보존.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5.
- **Expected wall: ~3-4h** (H010 envelope 동일).
- 누적 cost: H006~H014 ~32h + H015 ~3.5h = **~36h**. §17.6 cap 위협.

## 3. Upload manifest

경로: `experiments/H015_recency_loss_weighting/upload/`
백업: `experiments/H015_recency_loss_weighting/upload.tar.gz` (65,258 bytes).
총 12 files.

| File | H010/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (5 flags 추가) | Entry point |
| `dataset.py` | 변경 (label_time batch dict 노출, 2줄) | Data |
| `trainer.py` | 변경 (__init__ 3 args + _train_step weighting branch ~20줄) | Train loop |
| `train.py` | 변경 (argparse 3 + Trainer 3 keys) | CLI |
| `model.py` | byte-identical | Model |
| `infer.py` | byte-identical | Inference |
| `utils.py` | byte-identical | helpers |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H015 정체성 |

## 4. Run command
```
bash run.sh
```
internal baked args (5 changes vs H010):
```
--batch_size 2048                                # 사용자 prior regime 명시
--lr 1e-4                                        # H010 default 명시
--use_recency_loss_weighting                     # H015 NEW
--recency_weight_min 0.5                         # H015 NEW
--recency_weight_max 1.5                         # H015 NEW (mean = 1.0)
+ all H010/H008 mechanism flags 그대로
```

## 5. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`,
   `config_sha256`, **use_recency_loss_weighting=true 확인**, weight range
   확인, `attn_entropy_per_layer`.
2. **`train.log` 마지막 ~200 lines** — H015 ENABLED 로그 ("H015 ENABLED:
   recency-aware loss weighting [0.50, 1.50]"), NaN check, peak epoch.
3. **Submission round-trip** — `[infer] OK` + batch heartbeat.
4. **Platform AUC** (eval auc) — 본 H 의 핵심 measurement.
5. **OOF AUC + OOF-Platform gap** — P6 핵심 진단 (cohort drift 검증).

## 6. Verdict update path (post-intake)
- `hypotheses/H015_recency_loss_weighting/verdict.md` 의 P1–P7 채우기.
- 결과에 따라 (`card.yaml.decision_tree_post_result`):
  - **strong** → L2 confirmed, anchor = H015. H016 = recency variants.
  - **measurable** → L2 partial. H016 = recency combo 또는 OOF 재정의.
  - **noise** → **paradigm shift mandatory**. H016 = backbone replacement.
  - **degraded** → H015-sub = weight range 좁힘.
  - **P6 gap 더 벌어짐** → recency direction 잘못, H016 = OOF 재정의.

## 7. Pre-flight checks
- [x] H014 verdict.md REFUTED (L4 retire, 4-layer 종료).
- [x] H015 hypothesis docs (6 files) 완비.
- [x] papers/temporal_cohort/_summary.md (신규 카테고리) 생성.
- [x] H015 코드 패키지 빌드 (4 files changed: dataset/trainer/train + run.sh + README, model.py / infer.py byte-identical).
- [x] ast.parse 5 .py files OK.
- [x] tar.gz 생성 (65,258 bytes).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드.
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity dry-run: `bash run.sh --num_epochs 1 --train_ratio 0.05`
  (H015 ENABLED 로그 + NaN check).
- [ ] launch.

## 8. Build status: ✅ BUILT (2026-05-03)
- Code overlay: H010/upload/ 7 파일 byte-identical 카피, 5 파일 변경
  (run.sh / dataset.py / trainer.py / train.py / README.md).
- ast.parse: 5 .py files OK.
- Mechanism unchanged: model.py / infer.py / utils.py byte-identical (sha256
  검증 가능).
- tar.gz: 65,258 bytes.

## 9. Why recency loss weighting now (rotation 정당화)

- 직전 2 H = H013 (no category, measurement) + H014 (envelope_expansion).
  H015 = `temporal_cohort` 신규 카테고리 first-touch → §10.7 FREE.
- §10.3 challenger rule: H011/H012/H013/H014 모두 H010 anchor 위 mutation
  REFUTED (4 H 누적). H015 = **새 axis (training procedure)** 가 challenger
  사고 적용.
- 4-layer ceiling diagnosis 의 L2 (cohort drift) 가 **마지막 unexplored
  가설**. paradigm 안 마지막 시도.
- §0 north star alignment: training procedure axis = production CTR cohort
  handling 표준 (deployment realism). UNI-REC 안 새 layer.

## 10. Memory + safety

H010 envelope 동일 (batch 2048, seq 64-128, mechanism stack 동일).
- 이전 H010 학습 wall 3:44:54 — memory OK.
- H015 mean weight = 1.0 보존 → memory / loss scale 영향 없음.
- OOM risk 매우 작음.

## 11. Sanity dry-run (사용자 launch 전 권장)

```
bash run.sh --num_epochs 1 --train_ratio 0.05
```

확인 사항:
- `H015 ENABLED: recency-aware loss weighting [0.50, 1.50]` 로그 출력.
- NaN-free.
- `[infer] OK` 로그.

이 모두 통과 시 Taiji full launch 준비 완료.

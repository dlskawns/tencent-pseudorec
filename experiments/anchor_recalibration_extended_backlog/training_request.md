# H010 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 다음 turn 에 코드 패키지
> 빌드 (original_baseline/upload/ 의 12 파일 카피 + run.sh envelope 변경) 후
> 사용자 launch.

## 1. Hypothesis & Claim
- Hypothesis: **H010_anchor_recalibration_extended**.
- Mutation: **envelope only** (mechanism mutation 0).
  - `--num_epochs 1 → 10`
  - `--train_ratio 0.05 → 0.3`
  - `--patience 5 → 3` (H008 F-4 carry-forward)
- Control: original_baseline (smoke envelope, platform AUC ~0.83X).
- Predicted (4 시나리오, predictions.md 참조):
  - **A** anchor extended ∈ [0.825, 0.840] — envelope 효과 작음.
  - **B** anchor extended ∈ [0.840, 0.850] — envelope 효과 large, mechanism reassessment.
  - **C** anchor extended < 0.825 — envelope 룰 재검토.
  - **D** anchor extended ∈ [0.835, 0.840] — most likely. H008 only measurable lift.
- §17.3 binary 임계 적용 안 됨 (measurement objective).

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5 (Taiji 가격 미공개).
- Expected wall: **~2-3.5시간** (patience=3 + plateau early 가정).
- 누적 cost (H006~H009 ~14시간 + H010 ~3시간 = ~17시간) — §17.6 cap 압박 지속.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H010_anchor_recalibration_extended/upload/`
백업: `experiments/H010_anchor_recalibration_extended/upload.tar.gz` (TBD)
총 용량: ~250 KB (original_baseline 동급).

| File | original_baseline 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (envelope flags 3) | Entry point |
| `train.py` | byte-identical | CLI driver |
| `trainer.py` | byte-identical | Train loop |
| `model.py` | byte-identical (mechanism 코드 0) | PCVRHyFormer pure baseline |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `infer.py` | byte-identical | §18 인프라 (새 cfg key 없음) |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H010 정체성 (anchor recalibration) |

총 12 files. **mechanism 코드 변경 0**.

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Run command
```
bash run.sh
```
internal baked args:
```
--num_epochs 10                     # H010 extended (smoke 1 → 10)
--patience 3                         # H010 extended (smoke 5 → 3, H008 F-4)
--train_ratio 0.3                    # H010 extended (smoke 0.05 → 0.3)
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--use_label_time_split --oof_user_ratio 0.1 --split_seed 42
+ (그 외 original_baseline 와 byte-identical args)
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`,
   `config_sha256`, `num_epochs=10`, `train_ratio=0.3`, `patience=3` 모두 기록.
   mechanism flag 없음 (byte-identical baseline).
2. **`train.log` 마지막 ~200 lines** — peak epoch, NaN 검증, plateau timing.
3. **Submission round-trip** — best_model 으로 inference. `[infer] OK: torch
   path produced 609197 predictions` + batch heartbeat 둘 다 보임.
4. **Platform AUC** (eval 환경 score) — 본 H 의 핵심 measurement.
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H010.../verdict.md` 의 P1–P5 채우기 + 시나리오 (A/B/C/D) 분류.
- `hypotheses/INDEX.md` H010 status: `scaffold` → `pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (decision_tree_post_result):
  - **A** → H011 = NS→S xattn 또는 aligned `<id, weight>` pair encoding.
  - **B** → H011 = sub-H 로 H007/H008 단독 paired re-measure.
  - **C** → H011 = envelope 정의 sub-H.
  - **D** → H011 = H008 anchor 위 single mutation.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] H009 verdict.md REFUTED interference 확정 (이미 ✓, 이번 turn).
- [ ] anchor recalibration 우선순위 confirmed (사용자 합의 ✓).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs original_baseline: **mechanism 0**, run.sh envelope flags 3개
  (num_epochs, train_ratio, patience), README 변경.

## 10. Build status: 🚧 PENDING — 다음 turn 패키지 카피 + run.sh 작성

## 11. Why anchor recalibration > mechanism class rotation (this turn)

H009 verdict F-3 정량 노출:
- anchor 정확값 0.83 가정 시 H009 marginal pass.
- anchor 정확값 0.835 가정 시 H009 fail.
- 즉 결론 분류 (additive/sub-additive/interference) 가 anchor 정확값에 의존.
- H011+ 도 같은 모호함 재발 위험 → 한 번 measurement 으로 최대 5+ H 의 paired
  Δ 정확화 (cost-effective).

mechanism class rotation 후보 (H011+ 로 defer):
- NS→S full bidirectional xattn (OneTrans 추출, H007 일반화).
- aligned `<id, weight>` pair encoding (CLAUDE.md §3 mandate).
- multi_domain_fusion (MMoE/PLE).
- DCN-V2 sub-H (layer 2→4, rank 8→16).

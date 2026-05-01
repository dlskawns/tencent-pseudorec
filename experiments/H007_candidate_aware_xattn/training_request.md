# H007 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 다음 turn 에 model.py / train.py / infer.py 통합 후 빌드 완료. 본 문서는 scaffold.

## 1. Hypothesis & Claim
- Hypothesis: **H007_candidate_aware_xattn**
- One-mutation: **`--use_candidate_summary_token`** flag. 신규 클래스 `CandidateSummaryToken` (Pre-LN multi-head cross-attention) per-domain 적용. candidate token = item_ns + item_dense_tok mean pool. seq 시작에 prepend.
- Control: **original_baseline (anchor Platform AUC ~0.83X)**.
- Predicted lift (P2): Δ ≥ **+0.5 pt** vs anchor platform AUC.
- Falsification: Δ < +0.5pt → REFUTED, target_attention 카테고리 일시 archive.

## 2. Compute tier
- `T2.4` smoke 우선 (~5min, ~$1 추정).
- Marginal/REFUTED 시 extended retry (10ep × 30%, ~5h).
- Cost cap: per-job ≤ $5, per-day ≤ $20.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H007_candidate_aware_xattn/upload/`
백업: `experiments/H007_candidate_aware_xattn/upload.tar.gz` (TBD bytes)
총 용량: ~260 KB.

| File | original_baseline 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (1 line: `--use_candidate_summary_token`) | Entry point |
| `train.py` | 변경 (CLI flag + model_args 전달) | CLI driver |
| `trainer.py` | 동일 | Train loop |
| `model.py` | 변경 (CandidateSummaryToken class + PCVRHyFormer 통합) | 신규 클래스 ~80줄 + 통합 ~30줄 |
| `dataset.py` | 동일 | §18.2 universal handler |
| `infer.py` | 변경 (cfg.get 새 key 2개 read-back) | §18 인프라 + new cfg |
| `local_validate.py` | 동일 | G1–G6 |
| `make_schema.py` | 동일 | §18.5 모든 list variant |
| `utils.py` | 동일 | helpers |
| `ns_groups.json` | 동일 | NS ref |
| `requirements.txt` | 동일 | deps |
| `README.md` | 변경 | H007 정체성 |

총 12 files.

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수 (학습).
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Run command
```
bash run.sh
```
internal baked args:
```
(anchor envelope 동일)
+ --use_candidate_summary_token
+ --candidate_summary_num_heads 4
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`, `config_sha256`, `host`, `split_meta`, `use_candidate_summary_token=true`.
2. **`train.log` 마지막 ~200 lines** — `CandidateSummaryToken` init 로그, candidate token shape, NaN 검증.
3. **Submission round-trip** — best_model 으로 inference. `[infer] OK: torch path produced 609197 predictions` + batch heartbeat 둘 다 보임.
4. **Platform AUC** (eval 환경 score).
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H007.../verdict.md` 의 P1–P5 채우기.
- `hypotheses/INDEX.md` H007 status: `scaffold` → `pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 H008 promote (decision tree in card.yaml).

## 8. Pre-flight checks (사용자 launch 전)
- [ ] **anchor (original_baseline) Platform AUC ~0.83X 확정** (이미 ✓).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs anchor: model.py 추가 ~110줄 (CandidateSummaryToken + PCVRHyFormer 통합), train.py 추가 ~10줄 (CLI flag + model_args), infer.py 추가 ~5줄 (cfg.get).

## 10. Build status: 🚧 PENDING — 다음 turn 코드 통합

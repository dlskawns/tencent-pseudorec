# H009 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 다음 turn 에 model.py / train.py / infer.py 통합 후 빌드 완료. 본 문서는 scaffold.

## 1. Hypothesis & Claim
- Hypothesis: **H009_combined_xattn_dcn_v2**
- Stacking: H007 candidate-aware xattn (sequence axis) + H008 DCN-V2 fusion swap (interaction axis) 동시 적용. additivity 가정 검증 + §0 두 축 동시 강화 first direct verification.
- Control: **original_baseline (anchor Platform AUC ~0.83X)**.
- Predicted lift (P2):
  - additive: Δ ∈ [+0.005, +0.010pt] (H007 +0.0035 + H008 +0.0035 ≈ +0.007pt).
  - super-additive: Δ > +0.010pt (paper-grade).
  - sub-additive: Δ ∈ [+0.0035, +0.005pt].
  - interference: Δ < +0.0035pt.
- Falsification: Δ < +0.5pt (binary) → REFUTED.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3 — H008 F-4 carry-forward).
- Cost cap: per-job ≤ $5 (Taiji 가격 미공개 — 사용자 확인).
- Expected wall: **~2-3시간** (patience=3 으로 H006/H007/H008 보다 단축).

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H009_combined_xattn_dcn_v2/upload/`
백업: `experiments/H009_combined_xattn_dcn_v2/upload.tar.gz` (TBD)
총 용량: ~270 KB.

| File | original_baseline 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (`--use_candidate_summary_token --fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8`) | Entry point |
| `train.py` | 변경 (CLI flags 5 + model_args 5: H007 + H008) | CLI driver |
| `trainer.py` | 동일 | Train loop |
| `model.py` | 변경 (CandidateSummaryToken from H007 + DCNV2CrossBlock from H008 + 모든 통합) | 신규 클래스 2개 ~160줄 + 통합 ~50줄 |
| `dataset.py` | 동일 | §18.2 universal handler |
| `infer.py` | 변경 (cfg.get H007 + H008 keys 모두 read-back) | §18 인프라 + new cfg |
| `local_validate.py` | 동일 | G1–G6 |
| `make_schema.py` | 동일 | §18.5 |
| `utils.py` | 동일 | helpers |
| `ns_groups.json` | 동일 | NS ref |
| `requirements.txt` | 동일 | deps |
| `README.md` | 변경 | H009 정체성 (combined stacking) |

총 12 files.

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Run command
```
bash run.sh
```
internal baked args:
```
--num_epochs 10
--patience 3                         # H008 F-4
--train_ratio 0.3
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--use_label_time_split --oof_user_ratio 0.1 --split_seed 42
+ --use_candidate_summary_token       # H007 mechanism
+ --fusion_type dcn_v2                # H008 mechanism
+ --dcn_v2_num_layers 2
+ --dcn_v2_rank 8
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`, `config_sha256`, 두 mutation flags 모두 `true` 기록.
2. **`train.log` 마지막 ~200 lines** — `CandidateSummaryToken` + `DCNV2CrossBlock` init 로그, fusion dispatch 확인, NaN 검증, peak epoch.
3. **Submission round-trip** — best_model 으로 inference. `[infer] OK: torch path produced 609197 predictions` + batch heartbeat 둘 다 보임.
4. **Platform AUC** (eval 환경 score).
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H009.../verdict.md` 의 P1–P5 채우기 + additivity 분류.
- `hypotheses/INDEX.md` H009 status: `scaffold` → `pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (decision_tree_post_result):
  - **additive** → H009 PASS, anchor 갱신, H010 = 다른 axis 탐험.
  - **super-additive** → multi-seed × 3 ablation H 우선.
  - **sub-additive** → ablation H 로 통합 위치 재검토.
  - **interference** → lr/위치/init scale H.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] anchor (original_baseline) Platform AUC ~0.83X 확정 (이미 ✓).
- [ ] H007 PASS marginal 확정 (이미 ✓).
- [ ] H008 PASS 확정 (이미 ✓).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs anchor: model.py 추가 ~210줄 (H007 ~80 + H008 ~80 + integrations ~50), train.py 추가 ~30줄 (CLI flags 5 + model_args 5), infer.py 추가 ~10줄 (cfg.get 5).

## 10. Build status: 🚧 PENDING — 다음 turn 코드 통합 (H007 + H008 merge)

## 11. (선택) 속도 최적화 옵션 A 적용 후보

이전 turn 의 옵션 A — 코드 변경 0, run.sh baked args 만 변경:
- `--batch_size 256 → 512` (lr 1e-4 그대로 유지, 작은 변경)
- `--num_workers 2 → 4`
- `--reinit_sparse_after_epoch 1 → 999`

적용 시 wall ~30~40% 단축 추정 (3h → 2h). 단 H006/H007/H008 와 paired 비교 위해 default envelope 유지 권장. 사용자 결정.

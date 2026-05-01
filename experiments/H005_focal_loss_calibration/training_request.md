# H005 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). Hand this packet to the
> Taiji Training Code page, run, then return with the artifacts listed in §6.

## 1. Hypothesis & Claim
- Hypothesis: **H005_focal_loss_calibration**
- One-mutation: **`--loss_type bce` → `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`**.
  코드 수정 0 (PCVRHyFormer 가 focal 이미 지원). §17.2 깔끔.
- Control: `E_baseline_organizer` (PCVRHyFormer + BCE, val_AUC = **0.8251**).
- Predicted lift (P2): Δ ≥ **+0.5 pt** ⇒ val_AUC ≥ **0.8301**.
- Falsification: Δ < +0.5 pt → REFUTED, focal_loss_calibration 방향 retire (§17.3).

## 2. Compute tier
- `T2.4` smoke (Taiji platform, 사용자 manual launch).
- Expected wall: **~3 min** (E_baseline_organizer 와 동일, focal overhead ≈ 0).
- Cost cap (§17.6): per-job ≤ $5, per-day ≤ $20.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H005_focal_loss_calibration/upload/`
백업: `experiments/H005_focal_loss_calibration/upload.tar.gz` (TBD bytes)
총 용량: ~248 KB — 100 MB cap 여유 충분.

| File | H004 대비 | Role |
|---|---|---|
| `run.sh` | 변경 | Entry point. `--backbone hyformer --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0` baked. OneTrans flag 제거. |
| `train.py` | 동일 | H004 의 backbone router + entropy diagnostic 그대로 (사용 안 됨, dormant). |
| `trainer.py` | 동일 | Train loop. focal branch 가 organizer-supplied. |
| `model.py` | 동일 | PCVRHyFormer + OneTrans router. H005 는 hyformer path 만. |
| `dataset.py` | 동일 | PCVRParquetDataset. |
| `infer.py` | 동일 | backbone routing read-back. |
| `local_validate.py` | 동일 | G1–G6 gate runner. |
| `make_schema.py` | 동일 | Auto schema. |
| `utils.py` | 동일 | Logger, EarlyStopping, sigmoid_focal_loss. |
| `ns_groups.json` | 동일 | NS-token feature group ref. |
| `requirements.txt` | 동일 | torch 2.7.1+cu126, etc. |
| `README.md` | 변경 | H005 anchor 정체성 (§17.8 Final-Round 의무). |

총 12 files.

## 4. Platform env vars (필수)
- `TRAIN_DATA_PATH` — training parquet 디렉토리. 미설정 시 `run.sh` abort.
- `TRAIN_CKPT_PATH` — writable, metrics.json + best_model 경로. 미설정 시 abort.
- `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`, `TRAIN_WORK_PATH` — 미설정 시
  CKPT/{logs,tf_events,work} 자동 derive.

## 5. Run command
플랫폼이 자동으로:
```
bash run.sh
```
호출. `run.sh` 내부 baked args (변경 금지):
```
--num_epochs 1
--seed 42
--ns_tokenizer_type rankmixer
--user_ns_tokens 5         # H001 anchor와 동일 (paired 비교)
--item_ns_tokens 2         # H001 anchor와 동일
--num_queries 2
--ns_groups_json ""
--emb_skip_threshold 1000000
--num_workers 2
--buffer_batches 4
--train_ratio 0.05
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--backbone hyformer        # PCVRHyFormer-anchor (OneTrans archive-pending)
--loss_type focal          # H005 NEW (BCE → focal)
--focal_alpha 0.25         # H005 NEW (Lin et al. 표준)
--focal_gamma 2.0          # H005 NEW (Lin et al. 표준)
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — 핵심 필드:
   - `best_val_AUC` (P2 판정 baseline)
   - `best_val_logloss` (P3 판정 — focal 정상 동작 검증)
   - `seed`, `git_sha`, `config_sha256`, `host`, `python`, `cuda`, `torch`
   - `split_meta`, `total_param_count`
   - `config.loss_type`, `config.focal_alpha`, `config.focal_gamma`
2. **`train.log` 마지막 ~200 lines** — NaN/OOM, loss type 정상 라우팅 확인. 특히
   `Backbone: hyformer` + `loss_type=focal` 줄.
3. **Submission round-trip 증빙** — best_model 으로 `submission/local_validate.py`
   5/5 PASS 로그 (P4).
4. **Wall time** — 실측 분.

## 7. Verdict update path (post-intake)
- `hypotheses/H005.../verdict.md` 의 P1–P5 TBD 채우기.
- `hypotheses/INDEX.md` 의 H005 status: `scaffold` → `pending` (이미) → `done`.
- `experiments/INDEX.md` 새 row: `EXP_ID = E003`, hypothesis_id, val_AUC, logloss,
  config_sha256, git_sha, status.
- `progress.txt` 1 블록 append.
- 결과에 따라:
  - **Δ ≥ +0.5pt**: PCVRHyFormer-anchor 갱신 (val_AUC 새값). H006 = long_seq_retrieval
    (D 도메인 1100 tail) 또는 target_attention. axis-strengthening 카테고리 rotation.
  - **Δ ∈ [+0.0, +0.5pt)**: weak signal. focal 작동, lift 마진 부족. loss_calibration
    카테고리 일시 archive. H006 = axis-strengthening.
  - **Δ < 0**: focal 으로 망가짐. α tuning H (α=0.5) 후보 또는 retire.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) 30-sec sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha (현 워킹트리): TBD — 사용자 launch 직전 `git rev-parse HEAD` 캡쳐.
- config_sha256: card.yaml `repro_meta.config_sha256` 미정 → run 후 metrics.json
  에서 확인.
- Code diff vs H004: `run.sh` baked args 만 변경 (OneTrans 4 flag 제거 + focal 3
  flag 추가). model.py / train.py / trainer.py 등 동일.

## 10. Build status: ✅ COMPLETE — ready to upload

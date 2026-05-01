# H006 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). Hand this packet to the
> Taiji Training Code page, run, then return with the artifacts listed in §6.

## 1. Hypothesis & Claim
- Hypothesis: **H006_longer_encoder_d_domain**
- One-mutation: **`--seq_encoder_type transformer` → `longer`**. organizer
  LongerEncoder (top-K=50 self-attention compression). 코드 수정 0.
- Control: **original_baseline (anchor val_AUC TBD — measurement 대기)**.
- Predicted lift (P2): Δ ≥ **+0.5 pt** vs anchor val_AUC.
- Falsification: Δ < +0.5 pt → REFUTED, long_seq_retrieval 일시 archive.

## 2. Compute tier
- `T2.4` smoke (Taiji platform).
- Expected wall: **~5 min** (anchor ~3분 + LongerEncoder overhead).
- Cost cap: per-job ≤ $5.

## 3. Pre-launch dependency

**⚠️ original_baseline 의 platform AUC measurement 가 1번이라도 통과 (heartbeat + OK 로그) 후 launch.**

원본 anchor 의 platform AUC 가 0.5 (chance) 가 아닌 의미 있는 값 측정된 후에야 H006 의 paired Δ 계산 가능. 그 전엔 anchor 자체 측정이 우선.

## 4. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H006_longer_encoder_d_domain/upload/`
백업: `experiments/H006_longer_encoder_d_domain/upload.tar.gz`
총 용량: ~260 KB.

| File | original_baseline 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (1 line: `--seq_encoder_type longer`) | Entry point |
| `train.py` | 동일 | CLI driver |
| `trainer.py` | 동일 | Train loop |
| `model.py` | 동일 | PCVRHyFormer + OneTrans router (dormant) + LongerEncoder factory |
| `dataset.py` | 동일 | PCVRParquetDataset + §18.2 universal handler |
| `infer.py` | 동일 | §18 인프라 룰 |
| `local_validate.py` | 동일 | G1–G6 |
| `make_schema.py` | 동일 | §18.5 모든 list variant |
| `utils.py` | 동일 | helpers |
| `ns_groups.json` | 동일 | NS ref |
| `requirements.txt` | 동일 | deps |
| `README.md` | 변경 | H006 정체성 |

총 12 files.

## 5. Run command
```
bash run.sh
```
internal baked args (변경 금지):
```
--num_epochs 1
--patience 5
--seed 42
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--ns_groups_json ""
--emb_skip_threshold 1000000
--num_workers 2
--buffer_batches 4
--train_ratio 0.05
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--use_label_time_split        # H001 leak-fix
--oof_user_ratio 0.1
--split_seed 42
--seq_encoder_type longer     # H006 NEW (default 'transformer' → 'longer')
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`, `config_sha256`, `host`, `split_meta`.
2. **`train.log` 마지막 ~200 lines** — encoder type 라우팅 (`Backbone: hyformer` + 어딘가 longer 관련 init 로그) 확인.
3. **Submission round-trip** — best_model 으로 inference 시도. 성공 시:
   - `[infer] OK: torch path produced 609197 predictions` 로그
   - batch heartbeat (`[infer] batch 50/100/...`) 모두 보임
   - heuristic fallback 신호 없음
4. **Platform AUC** (eval 환경 score).
5. **Wall time**.

## 7. Verdict update path (post-intake)

- `hypotheses/H006.../verdict.md` 의 P1–P5 채우기.
- `hypotheses/INDEX.md` 의 H006 status: `scaffold` → `pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라:
  - **Δ ≥ +0.5pt**: H006 PASS. H007 = target_attention (candidate-aware retrieval).
  - **Δ ∈ [+0.0, +0.5pt)**: weak signal. retry on extended envelope (train_ratio=0.3, num_epochs=3).
  - **Δ < 0**: K=50 정보 손실. K tuning H 후보.
  - **P4 fail (heuristic fallback / heartbeat 없음)**: §18 회귀. anchor 자체 검증.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] **anchor (original_baseline) platform AUC 1회 통과 확인**.
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) 30-sec sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json 확인.
- Code diff vs original_baseline: `run.sh` baked args 만 변경 (1 줄: `--seq_encoder_type longer` 추가).

## 10. Build status: ✅ COMPLETE — anchor measurement 후 upload

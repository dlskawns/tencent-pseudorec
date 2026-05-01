# H002 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). Hand this packet to the
> Taiji Training Code page, run, then return with the artifacts listed in §6.

## 1. Hypothesis & Claim
- Hypothesis: **H002_interformer_domain_bridges**
- One-mutation: `MultiSeqHyFormerBlock` 안에 12개 InterFormer-style inter-domain
  bridges 추가 (low-rank r=4, sigmoid-gate init −2.0, pooled-then-broadcast).
- Control: `E_baseline_organizer` (val_AUC = **0.8251**, organizer row-group split,
  train_ratio=0.05, halved seq_max_lens, num_epochs=1).
- Predicted lift (P2): Δ ≥ **+0.5 pt** val_AUC ⇒ val_AUC ≥ **0.8301**.
- Falsification: Δ < +0.5 pt → REFUTED, InterFormer bridge 방향 retire (§17.3).

## 2. Compute tier
- `T2.4` (Taiji platform, 사용자 manual launch).
- Expected wall: **30–45 min** (control 대비 +5% bridge overhead).
- Cost cap (§17.6): per-job ≤ $5, per-day ≤ $20.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)
경로: `experiments/H002_interformer_domain_bridges/upload/`
백업: `experiments/H002_interformer_domain_bridges/upload.tar.gz` (55,181 B)

| File | Bytes | Role |
|---|---|---|
| `run.sh` | 1,961 | Entry point (platform invokes `bash run.sh`) |
| `train.py` | 24,573 | Trainer driver, CLI flags |
| `trainer.py` | 23,039 | Train loop (BCE, AUC eval) |
| `model.py` | 69,421 | PCVRHyFormer + `_DomainBridge` (H002 NEW) |
| `dataset.py` | 49,154 | PCVRParquetDataset, organizer split |
| `infer.py` | 11,961 | §13 contract entry (post-train) |
| `local_validate.py` | 6,949 | G1–G6 gate runner |
| `make_schema.py` | 10,049 | Auto schema from parquet |
| `utils.py` | 11,545 | Logger, EarlyStopping, focal helper |
| `ns_groups.json` | 2,092 | NS-token feature group ref |
| `requirements.txt` | 100 | torch 2.7.1+cu126, etc. |
| `README.md` | 2,973 | Technical report stub (§17.8 Final-Round 의무) |

총 12 files, **≤ 220 KB** — 100 MB cap 여유 충분.

## 4. Platform env vars (필수)
- `TRAIN_DATA_PATH` — training parquet 디렉토리. 미설정 시 `run.sh` abort.
- `TRAIN_CKPT_PATH` — writable, metrics.json + best_model 경로. 미설정 시 abort.
- `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`, `TRAIN_WORK_PATH` — 미설정 시
  CKPT/{logs,tf_events,work} 자동 derive.

> §17.8.7: `run.sh` 는 local-dev fallback 제거됨. 두 필수 env 누락 = abort.

## 5. Run command
플랫폼이 자동으로:
```
bash run.sh
```
호출. `run.sh` 내부 baked args (변경 금지):
```
--num_epochs 1
--seed 42
--train_ratio 0.05
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--enable_inter_domain_bridges
--bridge_rank 4
--bridge_gate_init -2.0
```

## 6. Bring-back artifacts (intake 시 paste 필요)
사용자가 학습 끝나고 가져와야 할 것 — `cloud-intake` skill 이 처리:

1. **`metrics.json`** (TRAIN_CKPT_PATH) — `best_val_AUC`, `best_oof_AUC` (조직자
   split 이라 OOF는 disabled 일 수 있음 → null 허용), `split_meta`, `seed`,
   `git_sha`, `config_sha256`, `host`, `python`, `cuda`, `torch`.
2. **`train.log` 마지막 ~200 lines** — NaN/OOM 여부 확인.
3. **Bridge gate snapshot** — 12개 bridge 의 final `sigmoid(gate)` 값 (P3 mechanism
   check). trainer 가 epoch end 에 dump 하는 위치 확인 후 paste.
4. **Submission round-trip 증빙** — best_model 으로 `submission/local_validate.py`
   5/5 PASS 로그 (P4).
5. **Wall time** — 실측 분.

## 7. Verdict update path (post-intake)
- `hypotheses/H002.../verdict.md` 의 P1–P5 TBD 채우기.
- `hypotheses/INDEX.md` 의 H002 status: `pending` → `done`, verdict 컬럼 채움.
- `experiments/INDEX.md` 새 row: `EXP_ID = E001`, hypothesis_id, val_AUC, OOF_AUC,
  config_sha256, git_sha, status.
- `progress.txt` 1 블록 append (이터레이션 저널).
- 만약 P2 REFUTED — `unified_backbones` 카테고리 retire 신호, H003 (OneTrans
  mixed-causal) 가 같은 카테고리이므로 challengers.md §재진입정당화 필요해짐.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) 30-sec sanity: `bash run.sh --num_epochs 1 --batch_size 32` — 코드패스 검증.
- [ ] launch.

## 9. Repro pin
- git_sha (현 워킹트리): TBD — 사용자 launch 직전 `git rev-parse HEAD` 캡쳐 후 본 파일에 기입.
- config_sha256: card.yaml `repro_meta.config_sha256` 미정 → run 후 metrics.json 에서 확인.
- bridge_param_count: **12,312** (12 bridges × 1,026 params, demo_1000 측정).

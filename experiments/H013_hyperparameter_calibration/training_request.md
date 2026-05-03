# H013 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 2026-05-02.

## 1. Hypothesis & Claim
- Hypothesis: **H013_hyperparameter_calibration**.
- **Measurement diagnostic H** — H010 mechanism + envelope byte-identical.
  run.sh 4 hyperparameter calibration:
  - `--batch_size 2048` (사용자 override 명시 bake).
  - `--lr 8e-4` (linear scaling rule for batch 8×, Goyal et al. 2017).
  - `--num_workers 2 → 4` (IO 완화, Taiji deadlock 위험 8 미만).
  - `--buffer_batches 4 → 8` (큰 배치 IO 부하 완화).
- Goal: 7개 H 누적 ceiling 0.82~0.8408 의 정체 결정.
  - **Frame A** (hyperparameter artifact): Δ ≥ +0.005pt → 모든 prior H 재해석.
  - **Frame B** (mechanism limit): Δ ≤ +0.001pt → Track B (long-seq P2).
  - **Frame C** (cohort drift): P6 gap > 2.5pt → cohort H 우선.
- §17.2 정당화: H012 F-2 (hyperparameter bias 노출) 의 명시 trigger + Linear
  scaling rule = standard practice, arbitrary tuning 아님.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5.
- Expected wall: **~2.5-3.5h** (lr 큰 효과로 patience=3 trigger 빠를 수
  있음, IO 4/8 로 단축).
- 누적 cost: H006~H012 ~24h + H013 ~3h = **~27h**. §17.6 cap 임박.

## 3. Upload manifest

경로: `experiments/H013_hyperparameter_calibration/upload/`
백업: `experiments/H013_hyperparameter_calibration/upload.tar.gz` (64,569 bytes).
총 12 files.

| File | H010/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (4 hyperparameter flags) | Entry point |
| `train.py` | byte-identical | CLI driver |
| `model.py` | byte-identical | Model architecture |
| `infer.py` | byte-identical | §18 인프라 |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | Data |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H013 정체성 |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Run command
```
bash run.sh
```
**중요**: 사용자가 추가 batch override 안 해도 됨 (run.sh 가 baked).
internal baked args:
```
--batch_size 2048                         # H013 NEW (사용자 override 명시 bake)
--lr 8e-4                                 # H013 NEW (linear scaling)
--num_workers 4                           # H013 NEW
--buffer_batches 8                        # H013 NEW
--num_epochs 10 --patience 3 --seed 42    # 그대로
--train_ratio 0.3                          # 그대로
--seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128"
--use_label_time_split --oof_user_ratio 0.1 --split_seed 42
--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8     # H008 anchor
--use_ns_to_s_xattn --ns_xattn_num_heads 4 --log_attn_entropy   # H010 anchor
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`,
   `config_sha256`, **batch_size=2048, lr=8e-4 명시 확인**, mutation flags,
   `attn_entropy_per_layer`.
2. **`train.log` 마지막 ~200 lines** — NaN check (lr 8e-4 발산 위험), peak
   epoch (lr 큰 효과로 빠른 수렴 가능), GPU utilization 가능 시.
3. **Submission round-trip** — `[infer] OK: torch path produced N predictions`
   + batch heartbeat.
4. **Platform AUC** (eval auc) — 본 H 의 핵심 measurement.
5. **Wall time** (학습 + inference) — IO efficiency P7 검증.

## 7. Verdict update path (post-intake)
- `hypotheses/H013_hyperparameter_calibration/verdict.md` 의 P1–P7 채우기 +
  decision tree 분기.
- `hypotheses/INDEX.md` H013 status: `code_build_pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (`card.yaml.decision_tree_post_result`):
  - **strong (Δ ≥ +0.005pt)** → Frame A. 모든 prior H 재해석. anchor = H013.
    H014 = mechanism H (new ranking).
  - **measurable** → 부분적. anchor 갱신 검토. H014 = long-seq P2 또는 cohort H.
  - **noise** → Frame B. anchor 유지. H014 = long-seq P2 entry.
  - **degraded** → H013-sub = lr 4e-4.
  - **NaN abort** → H013-sub = lr 4e-4 또는 warmup 추가.
  - **P6 gap > 2.5pt** → Frame C confirm. cohort H 우선.

## 8. Pre-flight checks (사용자 launch 전)
- [x] H012 verdict.md REFUTED Frame B 확정.
- [x] H013 hypothesis docs (6 files) 완비.
- [x] H013 코드 패키지 빌드 (model.py byte-identical with H010 + run.sh
  4 changes).
- [x] ast.parse 5 .py files OK.
- [x] tar.gz 생성 (`upload.tar.gz`, 64,569 bytes).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드.
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity dry-run: `bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05`
  (single batch NaN check).
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json (batch=2048, lr=8e-4 명시 확인).
- Code diff vs H010/upload/:
  - `model.py / train.py / infer.py / trainer.py / dataset.py`: **byte-identical**.
  - `run.sh`: 4 flags 변경 (batch, lr, num_workers, buffer_batches).
  - `README.md`: H013 정체성.

## 10. Build status: ✅ BUILT (2026-05-02)

- Code overlay: H010/upload/ 10 파일 byte-identical 카피, 2 파일 변경
  (run.sh / README.md).
- ast.parse: 5 .py files OK.
- Mechanism unchanged: model.py byte-identical (sha256 검증 가능).
- tar.gz: 64,569 bytes.

## 11. Sanity dry-run (사용자 launch 전 권장)

특히 lr 8e-4 가 batch 2048 와 stable 한지 1-epoch smoke 권장:
```
bash run.sh --num_epochs 1 --train_ratio 0.05
```
NaN-free + `lr=0.0008, batch_size=2048` 로그 + `[infer] OK` 모두 보이면
Taiji full launch 준비 완료. NaN/divergence 발생 시 lr 4e-4 sub-H.

## 12. Why hyperparameter calibration now (rotation 정당화)

- 직전 2 H primary_category: H011 feature_engineering, H012 multi_domain_fusion.
- H013 = **measurement H, primary_category 없음** → §10.7 / §17.4 rotation
  룰 미발동.
- 다음 mechanism H (H014) 시 직전 2 mechanism H = H011/H012 차단 (rotation
  유지).
- §17.2 정당화: H012 F-2 명시 trigger + Linear scaling rule (Goyal et al.
  2017) standard practice + 4 changes single concern.
- 결과가 모든 prior H 재해석에 영향 → measurement integrity check 의
  prerequisite.

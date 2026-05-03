# H011 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 2026-05-01.

## 1. Hypothesis & Claim
- Hypothesis: **H011_aligned_pair_encoding**.
- Mechanism: **Aligned `<id, weight>` pair encoding at input embedding lookup** (Option α — per-row L1-normalized weighted mean, parameter-free).
  - For verified shared fids `{62, 63, 64, 65, 66, 89, 90, 91}` (8 fids, 출처 `competition/ns_groups.json _note_shared_fids` + `eda/out/aligned_audit.json`): replace baseline mean-pool with weighted mean using `user_dense_feats` slice as weights.
  - non-aligned fids + dense-only `{61, 87}` 변경 없음. `user_dense_proj` 그대로.
  - 통합 위치: `RankMixerNSTokenizer.forward` 내부 mean-pool 분기. NS xattn 출력 + DCN-V2 fusion 입력 byte-identical → H009 위치 충돌 회피.
- CLAUDE.md §4.8 mandate (aligned 한쌍 이동) 직접 구현. baseline 이 룰 미통과 상태 (현재 user_int / user_dense 분리 처리) 패치.
- Predicted (paired classifications):
  - **strong PASS** Δ vs H010 ≥ +0.005pt → input-stage explicit binding 효과 큼 (Platform ≥ 0.8458).
  - **measurable** Δ ∈ [+0.001, +0.005pt] → mechanism 작동 confirmed.
  - **noise** Δ ∈ (−0.001, +0.001pt] → baseline implicit binding 충분 (Frame B).
  - **degraded** Δ < −0.001pt → REFUTED.
- Falsification (binary): Δ < +0.001pt → §17.3 strict 미달.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5 (Taiji 가격 미공개).
- Expected wall: **~2.5-3.5시간** (params 추가 0 → H010 와 동급 envelope).
- 누적 cost: H006~H010 ~18h + H011 ~3h = ~21h. §17.6 cap 압박 지속.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H011_aligned_pair_encoding/upload/`
백업: `experiments/H011_aligned_pair_encoding/upload.tar.gz` (~64 KB)
총 12 files.

| File | H010/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (`--use_aligned_pair_encoding --aligned_pair_fids 62 63 64 65 66 89 90 91` 추가) | Entry point |
| `train.py` | 변경 (argparse 2 + aligned_user_dense_specs 산출 + model_args 2 keys) | CLI driver |
| `model.py` | 변경 (RankMixerNSTokenizer __init__ + forward weighted-mean 분기, PCVRHyFormer __init__ 2 args + _build_token_streams 분기) | PCVRHyFormer + DCN-V2 + NS xattn + aligned encoding |
| `infer.py` | 변경 (aligned_user_dense_specs 재구성 + cfg.get 2 keys) | §18 인프라 + new cfg |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H011 정체성 |

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
--patience 3
--train_ratio 0.3
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--use_label_time_split --oof_user_ratio 0.1 --split_seed 42
--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8     # H008 anchor
--use_ns_to_s_xattn --ns_xattn_num_heads 4 --log_attn_entropy   # H010 anchor
+ --use_aligned_pair_encoding                                   # H011 NEW
+ --aligned_pair_fids 62 63 64 65 66 89 90 91                   # H011 NEW
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`, `config_sha256`, 모든 mutation flags 기록 (`use_aligned_pair_encoding=true`, `aligned_pair_fids=[62,63,64,65,66,89,90,91]`).
2. **`train.log` 마지막 ~200 lines** — H011 aligned dispatch 로그 (`H011 aligned_pair_encoding ENABLED: 8 fids ...`), attn entropy per layer, NaN 검증, peak epoch.
3. **Submission round-trip** — best_model 으로 inference. `[infer] OK: torch path produced N predictions` + batch heartbeat 둘 다 보임.
4. **Platform AUC** (eval 환경 score) — 본 H 의 핵심 measurement.
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H011_aligned_pair_encoding/verdict.md` 의 P1–P6 채우기 + paired classification.
- `hypotheses/INDEX.md` H011 status: `scaffold` → `code_build_pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (`card.yaml.decision_tree_post_result`):
  - **strong_pass / measurable** → anchor 갱신 (H011 = 새 baseline). H012 = orthogonal axis.
  - **noise** → Frame B 채택. H012 = MMoE/PLE 또는 sub-form.
  - **degraded** → REFUTED. 매핑/scale 재확인.

## 8. Pre-flight checks (사용자 launch 전)
- [x] H010 verdict.md PASS additive 확정.
- [x] H011 hypothesis docs (6 files) 완비 (CLAUDE.md §3 verified facts 반영).
- [x] P0 audit PASS (`eda/out/aligned_audit.json` — 8 fids 모두 1000/1000).
- [x] gap #3 dense value stats PASS — Option α 선택 근거 확보.
- [x] H011 코드 패키지 빌드 + ast.parse + dummy forward sanity.
- [x] tar.gz 생성 (`experiments/H011_aligned_pair_encoding/upload.tar.gz`).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity: `bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs H010/upload/:
  - `model.py` +20줄 (RankMixerNSTokenizer aligned_dense_specs param + weighted-mean branch + PCVRHyFormer 2 args).
  - `train.py` +30줄 (argparse 2 + aligned_user_dense_specs build + model_args 2).
  - `infer.py` +20줄 (aligned_user_dense_specs rebuild + cfg.get 2).
  - `run.sh` +2 flags.
  - `README.md` H011 정체성.

## 10. Build status: ✅ BUILT (2026-05-01)

- Code overlay: H010/upload/ 9 파일 byte-identical 카피, 5 파일 변경 (run.sh / train.py / model.py / infer.py / README.md).
- ast.parse: 5 .py files OK.
- Dummy forward sanity: Tests 1-5 + param-free 모두 PASS (Pattern X 18M finite, Pattern Y signed finite, baseline path, uniform weights ≡ baseline @ 4e-7, zero-row clamp safe).
- tar.gz: 65,507 bytes.

## 11. Sanity dry-run (사용자 launch 전 권장)
```
bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05
```
NaN-free + `H011 aligned_pair_encoding ENABLED: 8 fids ...` 로그 + DCNV2CrossBlock + NSToSCrossAttention init 로그 모두 보이면 Taiji full launch 준비 완료.

## 12. Why aligned_pair_encoding now (rotation 정당화)

- 직전 2 H primary_category: H009 hybrid (target_attention + sparse_feature_cross), H010 target_attention.
- H011 = `feature_engineering` 신규 카테고리 first-touch → §10.7 rotation FREE.
- §0 UNI-REC north star: sequence × interaction 통합. sequence axis 3번 (H006/H007/H010), interaction axis 1번 (H008). **input stage** 0번 → 새 axis 측정 의무.
- §4.8 mandate (aligned 한쌍 이동) 직접 구현 + baseline 룰 위반 patch.
- H010 F-1 carry-forward: NS-only enrichment safe pattern → input-stage 도 같은 원리 (downstream byte-identical, interference 위험 0).

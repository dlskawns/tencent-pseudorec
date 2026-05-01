# H010 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 다음 turn 에 model.py /
> train.py / infer.py 통합 후 빌드 완료. 본 문서는 scaffold.

## 1. Hypothesis & Claim
- Hypothesis: **H010_ns_to_s_xattn**.
- Mechanism: **OneTrans NS→S bidirectional cross-attention** 직접 구현 (paper-grade).
  - H007 (1-token candidate xattn) 의 N_NS-token 일반화.
  - H008 anchor (DCN-V2 fusion) 위 single mutation stacking.
  - 통합 위치: per-domain seq encoder 출력 → S concat (L_total=384) → NSToSCrossAttention → enriched NS tokens (B, 7, D) → 기존 query decoder + DCN-V2 fusion 그대로. **NS dimension 변경 없음 → H009 위치 충돌 회피 by 설계**.
- Predicted (paired classifications):
  - **super-additive** Δ vs H008 ≥ +0.005pt → paper-grade lift, NS×S 통합 가치 confirmed (Platform ≥ 0.8437).
  - **additive** Δ vs H008 ∈ [+0.001, +0.005pt] (Platform ∈ [0.8397, 0.8437]).
  - **noise** Δ vs H008 ∈ [−0.001, +0.001pt] (mechanism 일반화 가치 marginal).
  - **interference** Δ vs H008 < −0.001pt → REFUTED, 위치 충돌 회피 설계 가설 무효.
- Falsification (binary): Δ vs anchor < +0.5pt → REFUTED.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5 (Taiji 가격 미공개).
- Expected wall: **~2.5-3.5시간**.
- 누적 cost: H006~H009 ~14h + H010 ~3h = ~17h. §17.6 cap 압박 지속.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H010_ns_to_s_xattn/upload/`
백업: `experiments/H010_ns_to_s_xattn/upload.tar.gz` (TBD)
총 용량: ~280 KB.

| File | H008/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (`--use_ns_to_s_xattn --ns_xattn_num_heads 4 --log_attn_entropy`) | Entry point |
| `train.py` | 변경 (CLI flags 3 + model_args 3) | CLI driver |
| `trainer.py` | 변경 (attn_entropy 측정 hook, §10.9 룰) | Train loop |
| `model.py` | 변경 (`NSToSCrossAttention` 클래스 추가 ~80줄 + MultiSeqHyFormerBlock 통합 ~30줄) | PCVRHyFormer + DCN-V2 + NS xattn |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `infer.py` | 변경 (3 cfg.get keys 추가) | §18 인프라 + new cfg |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H010 정체성 (NS→S xattn paper-grade) |

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
--patience 3
--train_ratio 0.3
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--use_label_time_split --oof_user_ratio 0.1 --split_seed 42
--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8     # H008 anchor 그대로
+ --use_ns_to_s_xattn                    # H010 NEW
+ --ns_xattn_num_heads 4                 # H010 NEW (default = num_heads)
+ --log_attn_entropy                     # H010 NEW (§10.9 룰)
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`, `config_sha256`, 모든 mutation flags 기록 (`use_ns_to_s_xattn=true`, `fusion_type=dcn_v2`).
2. **`train.log` 마지막 ~200 lines** — `NSToSCrossAttention` init 로그, attn entropy per layer 측정값, fusion dispatch 확인, NaN 검증, peak epoch.
3. **Submission round-trip** — best_model 으로 inference. `[infer] OK: torch path produced 609197 predictions` + batch heartbeat 둘 다 보임.
4. **Platform AUC** (eval 환경 score) — 본 H 의 핵심 measurement.
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H010.../verdict.md` 의 P1–P6 채우기 + paired classification (super-additive / additive / noise / interference).
- `hypotheses/INDEX.md` H010 status: `scaffold` → `pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (decision_tree_post_result):
  - **super-additive** → anchor 갱신 (H010 = 새 baseline). H011 = orthogonal axis.
  - **additive** → anchor 갱신. H011 = 다른 axis.
  - **noise** → anchor 갱신 보류. H011 = orthogonal axis 또는 NS xattn sub-H.
  - **interference** → REFUTED. H011 = NS xattn 통합 위치 변경 sub-H.
  - **REFUTED (Δ vs anchor < +0.5pt)** → H011 = aligned pair encoding 또는 multi_domain_fusion.
  - **attn entropy violation** → §10.9 abort. NS-token granularity 또는 hard routing sub-H.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] H009 verdict.md REFUTED interference 확정 (이미 ✓).
- [ ] H010 hypothesis docs (6 files) 완비 (이미 ✓ this turn).
- [ ] H010 코드 패키지 빌드 (다음 turn — H008/upload/ over-overlay + NSToSCrossAttention 추가).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs H008/upload/: model.py +110줄 (NSToSCrossAttention 클래스 + 통합), train.py +15줄 (CLI flags 3 + model_args 3), trainer.py +20줄 (attn_entropy hook), infer.py +5줄 (cfg.get 3).

## 10. Build status: ✅ BUILT (this turn)

- Code overlay: H008/upload/ 11 파일 (run.sh 제외) byte-identical 카피.
- 변경 파일 (5):
  - `model.py` — `NSToSCrossAttention` 클래스 추가 (~80 lines, L1462+) + MultiSeqHyFormerBlock __init__ 인자 3개 + self.use_ns_to_s_xattn + ns_xattn instance + forward 추가 step + PCVRHyFormer __init__ 인자 2개 + self.use_ns_to_s_xattn + blocks 호출 args 3개 + collect_attn_entropies 확장.
  - `train.py` — argparse 2 flags (--use_ns_to_s_xattn, --ns_xattn_num_heads) + model_args 2 keys + entropy 분기 (`backbone == 'onetrans' or use_ns_to_s_xattn`) + n_tokens 계산 분기.
  - `trainer.py` — byte-identical (entropy 수집은 model.collect_attn_entropies() 호출만).
  - `infer.py` — cfg.get 2 keys (use_ns_to_s_xattn, ns_xattn_num_heads).
  - `README.md` — H010 정체성 + 통합 위치 명시.
- run.sh — 이미 작성됨 (3 flags baked: --use_ns_to_s_xattn, --ns_xattn_num_heads 4, --log_attn_entropy).
- tar.gz: experiments/H010_ns_to_s_xattn/upload.tar.gz (65 KB).
- Sanity:
  - Syntax: `ast.parse` OK for model.py / train.py / infer.py.
  - Dummy forward: `NSToSCrossAttention(d=64, h=4)` → (2, 7, 64), entropy=4.8947 (random input below threshold 5.65), params=16896 (≈ 16K transfer.md §⑥ 예상 일치).

## 11. Sanity dry-run (사용자 launch 전 권장)
```
bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05
```
NaN-free + NSToSCrossAttention init 로그 + DCNV2CrossBlock init 로그 둘 다 보이면 Taiji full launch 준비 완료.

## 11. Why NS→S xattn now (vs anchor recalibration backlog)

사용자 가치 align:
- anchor recalibration = measurement H, mechanism lift 0. cost-effective signal 작음 (anchor 정확값 의존성은 H011 부터 H008 paired 비교 가 주가 되면 영향 작음).
- NS→S xattn = paper-grade mechanism (OneTrans), H007 (PASS marginal) 자연 일반화, H008 anchor on champion stacking sub-H, H009 위치 충돌 회피 by 설계.
- §0 north star (sequence × interaction 통합) 의 sequence axis 강화 paper-grade lift 시도.
- PASS 시 anchor 갱신 + H011+ 의 새 ground truth.

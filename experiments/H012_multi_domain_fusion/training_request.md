# H012 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 2026-05-02.

## 1. Hypothesis & Claim
- Hypothesis: **H012_multi_domain_fusion**.
- Mechanism: **Multi-domain MoE on NS-tokens (MMoE single-layer)**.
  - 4 expert FFNs (= 도메인 a/b/c/d) + softmax gate per NS-token.
  - 통합 위치: `MultiSeqHyFormerBlock.forward`, NS xattn 직후, query
    decoder + DCN-V2 fusion 전.
  - NS dimension preserved → DCN-V2 fusion input token stack unchanged →
    H009 위치 충돌 회피 + H011 input-stage cohort drift 회피.
  - parameter-add ~66.7K (sanity 측정), anchor envelope 면제 후 budget 안.
- Source: MMoE (Ma et al. KDD 2018, arXiv:1810.10739) — task→domain mapping.
- 데이터 motivation (`eda/out/domain_facts.json`):
  - 4 도메인 item Jaccard ≤ 0.10 (max a_vs_d=10%, min a_vs_c=0.7%) — 거의
    disjoint vocab.
  - length p50: a=577 / b=405 / c=322 / d=1035. 3× 차이.
  - frac_empty: a=0.5% / d=8%. d 양극 분포.
- Predicted (paired classifications):
  - **strong PASS** Δ vs H010 ≥ +0.005pt + P3 specialized (Platform ≥ 0.8458).
  - **measurable** Δ ∈ [+0.001, +0.005pt] + specialized.
  - **noise** Δ ∈ (−0.001, +0.001pt] + uniform (Frame B).
  - **degraded** Δ < −0.001pt + specialized → interference.
  - **collapse** P3 entropy < 0.69 → §10.9 abort.
- Falsification (binary): Δ < +0.001pt → §17.3 sample-scale relaxed 미달.

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5.
- Expected wall: **~3-3.5시간** (H010 envelope, +66.7K params 미미).
- 누적 cost: H006~H011 ~21h + H012 ~3h = **~24h**. §17.6 cap 압박 지속.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H012_multi_domain_fusion/upload/`
백업: `experiments/H012_multi_domain_fusion/upload.tar.gz` (66,469 bytes)
총 12 files.

| File | H010/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (`--use_multi_domain_moe --num_experts 4 --moe_ffn_hidden 128` 추가) | Entry point |
| `train.py` | 변경 (argparse 3 + model_args 3 + MoE entropy diagnostic block) | CLI driver |
| `model.py` | 변경 (MultiDomainMoEBlock 클래스 ~60줄 + MultiSeqHyFormerBlock __init__/forward + PCVRHyFormer __init__ + collect_moe_gate_entropies()) | PCVRHyFormer + DCN-V2 + NS xattn + MMoE |
| `infer.py` | 변경 (cfg.get 3 keys) | §18 인프라 + new cfg |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H012 정체성 |

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
+ --use_multi_domain_moe                                        # H012 NEW
+ --num_experts 4                                               # H012 NEW
+ --moe_ffn_hidden 128                                          # H012 NEW
```

## 6. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`,
   `config_sha256`, mutation flags, **`moe_gate_entropy_per_block`** (신규
   H012 진단), `moe_collapse_threshold`, `moe_uniform_threshold`,
   `moe_collapse_violation`, plus 기존 `attn_entropy_per_layer`.
2. **`train.log` 마지막 ~200 lines** — H012 dispatch 로그 (`MultiDomainMoEBlock` init
   confirmation), MoE gate entropy per block, NS xattn entropy 같이.
3. **Submission round-trip** — `[infer] OK: torch path produced N predictions`
   + batch heartbeat.
4. **Platform AUC** (eval auc) — 본 H 의 핵심 measurement.
5. **Wall time** (학습 + inference).

## 7. Verdict update path (post-intake)
- `hypotheses/H012_multi_domain_fusion/verdict.md` 의 P1–P6 채우기 + paired classification.
- `hypotheses/INDEX.md` H012 status: `scaffold` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (`card.yaml.decision_tree_post_result`):
  - **strong_pass / measurable** + P3 specialized → anchor 갱신, H013 = orthogonal.
  - **noise** + P3 uniform → Frame B 채택, H013 = NS xattn sub-H 또는 cohort 처리.
  - **degraded** + P3 specialized → interference, H013 = 위치 변경 sub-H.
  - **collapse** P3 entropy < 0.69 → §10.9 abort, H013 = num_experts ≤ 2.

## 8. Pre-flight checks (사용자 launch 전)
- [x] H011 verdict.md REFUTED degraded 확정.
- [x] H012 hypothesis docs (6 files) 완비.
- [x] domain_facts.json 산출 (motivation 정량 근거).
- [x] H012 코드 패키지 빌드 + ast.parse + dummy forward sanity (5 tests PASS).
- [x] tar.gz 생성 (`upload.tar.gz`, 66,469 bytes).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity dry-run: `bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05`.
- [ ] launch.

## 9. Repro pin
- git_sha: TBD — launch 직전 캡쳐.
- config_sha256: TBD — run 후 metrics.json.
- Code diff vs H010/upload/:
  - `model.py` +~80줄 (MultiDomainMoEBlock 클래스 ~60 + MultiSeqHyFormerBlock 수정 ~10 + PCVRHyFormer 수정 ~10 + collect_moe_gate_entropies()).
  - `train.py` +~40줄 (argparse 3 + model_args 3 + MoE entropy diagnostic ~30).
  - `infer.py` +~5줄 (cfg.get 3).
  - `run.sh` +3 flags.
  - `README.md` H012 정체성.

## 10. Build status: ✅ BUILT (2026-05-02)

- Code overlay: H010/upload/ 8 파일 byte-identical 카피, 5 파일 변경
  (run.sh / train.py / model.py / infer.py / README.md).
- ast.parse: 5 .py files OK.
- Dummy forward sanity (5 tests):
  - Test 1 shape + finite: (4, 7, 64) PASS, |out|.max=3.62.
  - Test 2 param count: 66,692 (예상값 일치).
  - Test 3 gate entropy at random init: 1.378 (uniform log(4)≈1.386 근접 — 학습 전 expected).
  - Test 4 residual delta: 0.083 (> 0, bounded).
  - Test 5 gradient flow: 20/20 params with grad.
- tar.gz: 66,469 bytes.

## 11. Sanity dry-run (사용자 launch 전 권장)
```
bash run.sh --num_epochs 1 --batch_size 32 --train_ratio 0.05
```
NaN-free + `MultiDomainMoEBlock` init confirm + `moe_gate_entropy_per_block`
log + `[infer] OK` 모두 보이면 Taiji full launch 준비 완료.

## 12. Why multi_domain_fusion now (rotation 정당화)

- 직전 2 H primary_category: H010 target_attention, H011 feature_engineering.
- H012 = `multi_domain_fusion` 신규 카테고리 first-touch → §10.7 rotation FREE.
- §0 UNI-REC north star: sequence × interaction 통합. multi_domain_fusion =
  sequence-side fusion 강화 (orthogonal axis to interaction).
- 데이터 motivation 강함: 4 도메인 Jaccard ≤ 0.10 (거의 disjoint), length 3× 차이,
  frac_empty 16× 차이 → explicit specialization 필요.
- H010 F-1 안전 stacking 패턴 (NS-only enrichment, downstream byte-identical) 직접 적용.
- H011 F-1 carry-forward (input-stage 위험) 회피 — NS-token level (post-encoder).

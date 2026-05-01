# H007_candidate_aware_xattn — Technical Report

> CLAUDE.md §17.2 one-mutation: original_baseline anchor 와 byte-identical
> envelope, `--use_candidate_summary_token` flag 만 추가. 신규 클래스
> `CandidateSummaryToken` per-domain 적용.
> §17.4 rotation 추가 충족 (target_attention, 첫 적용).
> §18 inference 인프라 룰 inherit from original_baseline.

## 1. Hypothesis & Claim
- Hypothesis: **H007_candidate_aware_xattn**
- Claim: H006 (LongerEncoder top-K=50) 의 random/probability-based candidate
  selection 한계 (verdict F-1) 직접 후속. **candidate item embedding 을 attention
  query 로** 만들어 per-domain history sequence 에 cross-attention →
  candidate-relevant events 가 weighted pool. 1 candidate-attended summary
  token 을 seq 시작에 prepend → 다운스트림 query decoder + RankMixer fusion
  자동 candidate-aware 경유.
- 차용: mechanism class (DIN/CAN/SIM/TWIN/HSTU/OneTrans family idea), modern
  multi-head cross-attention 구현 (2018 archaic activation MLP 아님).
- Compute tier: **T2.4 smoke (Taiji)**.
- Expected wall: ~5 분 (anchor 3분 + candidate xattn ~2분 추정).

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로.
**`CandidateSummaryToken` per-domain 적용 + seq prepend** — `--use_candidate_summary_token`
flag.

§18 inference 인프라 룰 모두 inherit:
- §18.1: `infer.py` 의 dataset batch_size 생성자 인자 (256 default, override 금지)
- §18.2: dataset.py 의 dim==1 universal handler
- §18.3: 진단 로그 (MODEL_OUTPUT_PATH, ckpt_dir, WARNING/FALLBACK/OK)
- §18.4: INFER_BATCH_SIZE=256, INFER_NUM_WORKERS=2, no autocast, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- §18.5: make_schema.py 모든 list-type variant 검출

## 3. Files
| File | Purpose |
|---|---|
| `run.sh` | Entry point. Bakes `--use_candidate_summary_token` (encoder/loss 등 그대로). |
| `train.py` | CLI driver. NEW H007 flags: `--use_candidate_summary_token`, `--candidate_summary_num_heads`. |
| `trainer.py` | tqdm 진행률 (epoch / step% / avg loss). |
| `model.py` | PCVRHyFormer + OneTrans router (dormant) + **NEW `CandidateSummaryToken` class** (~80줄, line ~1192) + `_build_token_streams` 통합 (candidate token mean pool + per-domain summary prepend). |
| `dataset.py` | PCVRParquetDataset + universal dim==1 handler. |
| `infer.py` | §18 인프라 룰 + use_candidate_summary_token cfg.get read-back. |
| `local_validate.py` | G1–G6 gate runner. |
| `make_schema.py` | Auto schema (모든 list variant). |
| `utils.py` | Logger, EarlyStopping. |
| `ns_groups.json` | NS-token feature ref. |
| `requirements.txt` | torch 2.7.1+cu126. |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Outputs
- `metrics.json` — `best_val_AUC`, `best_oof_AUC`, `split_meta`, repro meta.
  `config.use_candidate_summary_token=true` 기록.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — tqdm postfix + heartbeat 로그. `CandidateSummaryToken` 모듈
  init 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over original_baseline
- **CandidateSummaryToken class** (model.py line ~1192): Pre-LN multi-head
  cross-attention. Q=candidate (1 token), K=V=seq_tokens (L tokens). padding
  mask aware + all-pad-row guard.
- **Per-domain modules**: 4 domains 각각 독립 `CandidateSummaryToken` 인스턴스.
  domain-specific candidate-history pattern 학습.
- **Candidate token 구성**: `(item_ns + item_dense_tok)` 의 mean pool → (B, 1, D).
  paper 의 raw item ID embedding 과 다름 (organizer representation 활용).
- **Prepend (start of seq)**: candidate summary 가 position 0 → RoPE 0 = "context
  anchor" 역할. append 도 가능 (별도 ablation).
- **다운스트림 자동 candidate-aware**: query decoder + RankMixer 단계가 L+1 위치
  seq 를 처리 → position 0 의 candidate-attended summary 가 cross-attention
  의 K 로 들어가 informative.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + use_candidate_summary_token flag.
- `train_config.json` sidecar; `infer.py` reads `use_candidate_summary_token` +
  `candidate_summary_num_heads` to instantiate matching modules for ckpt loading.

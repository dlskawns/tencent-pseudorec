# H008_dcn_v2_block_fusion — Technical Report

> CLAUDE.md §17.2 one-mutation: original_baseline anchor 와 byte-identical
> envelope, `--fusion_type dcn_v2` flag 만 추가.
> §0 P1 직접 충족 (block-level gradient sharing, concat-late anti-pattern 회피).
> §17.4 rotation 추가 충족 (sparse_feature_cross, 첫 적용).
> §18 inference 인프라 룰 inherit from original_baseline.

## 1. Hypothesis & Claim
- Hypothesis: **H008_dcn_v2_block_fusion**
- Claim: H007 verdict F-1 (target_attention mechanism PASS marginal) 의 직접
  후속. **interaction axis 강화** — `MultiSeqHyFormerBlock` step 3 의 token
  fusion 을 `RankMixerBlock` (token-mixing MLP) 에서 `DCNV2CrossBlock`
  (explicit polynomial cross with x_0 residual) 로 swap. 같은 위치, 같은
  역할 (decoded_q + NS tokens fusion), 다른 mechanism (token-mixing →
  explicit polynomial cross). **block-level gradient sharing 보존** — seq
  결과와 NS interaction tokens 가 같은 block 안 cross 에서 polynomial
  interaction. **§0 P1 조건 직접 충족** (concat-late 안티패턴 회피).
- 차용: DCN-V2 (Wang et al. WWW 2021), production CTR 표준 lever. low-rank
  cross (rank=8) 로 params 절약.
- Compute tier: **T2.4 smoke (Taiji)**.
- Expected wall: ~5 분.

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로.
**`MultiSeqHyFormerBlock` step 3 fusion 만 swap** (`--fusion_type dcn_v2`).

§18 inference 인프라 룰 모두 inherit:
- §18.1: `infer.py` 의 dataset batch_size 생성자 인자 (256 default, override 금지)
- §18.2: dataset.py 의 dim==1 universal handler
- §18.3: 진단 로그 (MODEL_OUTPUT_PATH, ckpt_dir, WARNING/FALLBACK/OK)
- §18.4: INFER_BATCH_SIZE=256, INFER_NUM_WORKERS=2, no autocast, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- §18.5: make_schema.py 모든 list-type variant 검출

## 3. Files
| File | Purpose |
|---|---|
| `run.sh` | Entry point. Bakes `--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8`. |
| `train.py` | CLI driver. NEW H008 flags: `--fusion_type`, `--dcn_v2_num_layers`, `--dcn_v2_rank`. |
| `trainer.py` | tqdm 진행률 (epoch / step% / avg loss). |
| `model.py` | PCVRHyFormer + OneTrans router (dormant) + **NEW `DCNV2CrossBlock` class** (~80줄, line ~415) + `MultiSeqHyFormerBlock` fusion dispatch (rankmixer | dcn_v2). |
| `dataset.py` | PCVRParquetDataset + universal dim==1 handler. |
| `infer.py` | §18 인프라 룰 + fusion_type cfg.get read-back. |
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
  `config.fusion_type=dcn_v2`, `config.dcn_v2_num_layers=2`, `config.dcn_v2_rank=8` 기록.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — tqdm postfix + heartbeat 로그. `DCNV2CrossBlock` init 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over original_baseline
- **DCNV2CrossBlock class** (model.py line ~415): per-token polynomial cross.
  - Pre-LN on x_0 (CLAUDE.md §10.5 MANDATORY).
  - Stack 2 cross layers → polynomial degree 3.
  - Low-rank approx W = U V^T, rank=8 (D/8 saving).
  - `x_{l+1} = x_0 ⊙ (V_l(U_l(x_l))) + x_l`.
  - Same I/O signature as RankMixerBlock: (B, T, D) → (B, T, D).
- **MultiSeqHyFormerBlock fusion dispatch**: `fusion_type` arg routes step 3
  to RankMixerBlock (default, anchor) or DCNV2CrossBlock (H008).
- **PCVRHyFormer constructor**: `fusion_type`, `dcn_v2_num_layers`,
  `dcn_v2_rank` args propagated to MultiSeqHyFormerBlock stack.
- **Block-level integration preserved**: decoded_q (per-domain seq results)
  + NS tokens (interaction features) 가 같은 cross block 통과 → seq +
  interaction 한 block 안 gradient 공유. **§0 P1 조건 직접 충족**.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + fusion_type + dcn_v2_num_layers + dcn_v2_rank.
- `train_config.json` sidecar; `infer.py` reads cfg keys to instantiate matching
  fusion module for ckpt loading.

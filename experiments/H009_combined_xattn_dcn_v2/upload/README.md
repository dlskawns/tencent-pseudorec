# H009_combined_xattn_dcn_v2 — Technical Report

> CLAUDE.md §17.2 stacking sub-H: H007 (candidate-aware xattn, sequence axis)
> + H008 (DCN-V2 fusion swap, interaction axis) 동시 적용.
> additivity 가정 검증 + §0 두 축 동시 강화 first direct verification.
> 두 mechanism 모두 `MultiSeqHyFormerBlock` 안 작동 → **block-level gradient
> sharing** (§0 P1 직접 충족, concat-late anti-pattern 회피).
> §17.4 hybrid (target_attention + sparse_feature_cross), 정당화 = stacking.
> §18 inference 인프라 룰 inherit.

## 1. Hypothesis & Claim
- Hypothesis: **H009_combined_xattn_dcn_v2**
- Stacking:
  - H007 mechanism (sequence axis): `CandidateSummaryToken` per-domain.
    candidate token = (item_ns + item_dense_tok) mean pool. seq 시작에 prepend.
  - H008 mechanism (interaction axis): `MultiSeqHyFormerBlock` step 3 fusion
    `RankMixerBlock` → `DCNV2CrossBlock` swap. low-rank polynomial cross.
- Predicted lift (P2):
  - additive: Δ ∈ [+0.005, +0.010pt] (H007 +0.0035 + H008 +0.0035 ≈ +0.007pt).
  - super-additive: Δ > +0.010pt (paper-grade).
  - sub-additive: Δ ∈ [+0.0035, +0.005pt] (interference).
- Compute tier: **T2.4 extended (10 epoch × 30%, patience=3)**, ~2-3시간 wall.

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로.
**두 mutation 동시 활성** — `--use_candidate_summary_token --fusion_type dcn_v2`.

§18 inference 인프라 룰 모두 inherit (batch=256 default + PYTORCH_CUDA_ALLOC_CONF +
universal handler + 진단 로그).

## 3. Files
| File | Purpose |
|---|---|
| `run.sh` | Entry point. Bakes H007 + H008 flags 동시. patience=3 (H008 F-4). |
| `train.py` | CLI driver. H007 + H008 flags 모두 (5개) + model_args 전달. |
| `trainer.py` | tqdm 진행률 (epoch / step% / avg loss). |
| `model.py` | PCVRHyFormer + OneTrans router (dormant) + **`CandidateSummaryToken` (H007)** + **`DCNV2CrossBlock` (H008)** + 모든 통합. |
| `dataset.py` | PCVRParquetDataset + universal dim==1 handler. |
| `infer.py` | §18 인프라 룰 + H007 + H008 cfg.get read-back. |
| `local_validate.py` | G1–G6. |
| `make_schema.py` | Auto schema (모든 list variant). |
| `utils.py` | helpers. |
| `ns_groups.json` | NS-token feature ref. |
| `requirements.txt` | torch 2.7.1+cu126. |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Outputs
- `metrics.json` — `best_val_AUC`, `best_oof_AUC`, repro meta. 두 mechanism flag
  (`use_candidate_summary_token=true`, `fusion_type=dcn_v2`) 모두 기록.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — `CandidateSummaryToken` + `DCNV2CrossBlock` init 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over original_baseline (stacking)
- **H007 mechanism (sequence axis)**:
  - `CandidateSummaryToken` per-domain (Pre-LN multi-head cross-attention).
  - Q=candidate (1 token), K=V=seq_tokens. padding mask + all-pad guard.
  - 4 도메인 각각 독립 module.
  - 통합: per-domain seq encoder 출력 → CandidateSummaryToken → seq 시작 prepend.
- **H008 mechanism (interaction axis)**:
  - `DCNV2CrossBlock` token-wise polynomial cross.
  - Pre-LN on x_0 (CLAUDE.md §10.5 MANDATORY).
  - 2 stacked layers → polynomial degree 3.
  - Low-rank W = U V^T, rank=8 (D/8 saving).
  - `MultiSeqHyFormerBlock` step 3 fusion dispatch (rankmixer | dcn_v2).
- **Block-level integration preserved**: 두 mechanism 모두
  `MultiSeqHyFormerBlock` 안 작동 → seq encoder + candidate summary 가 query
  decoder 통과 → DCN-V2 cross 에서 NS tokens 와 polynomial interaction. **§0 P1
  조건 직접 충족**.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + 두 mutation flags.
- `train_config.json` sidecar; `infer.py` reads cfg keys to instantiate
  matching modules for ckpt loading.

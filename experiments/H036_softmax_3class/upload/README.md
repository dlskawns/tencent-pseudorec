# H010_ns_to_s_xattn — Technical Report

> CLAUDE.md §17.2 stacking sub-H on H008 champion: H008 envelope + H008 mechanism
> (DCN-V2 fusion) byte-identical PLUS `--use_ns_to_s_xattn` 추가.
> Paper-grade source: OneTrans (Tencent WWW 2026, arXiv:2510.26104) — NS→S
> bidirectional half. H007 (1-token candidate xattn) 의 N_NS-token 일반화.
> 통합 위치 = per-domain seq encoder 출력 직후 (fusion 이전), NS dimension 변경
> 없음 → H009 위치 충돌 회피 by 설계.
> §17.4 target_attention 재진입정당화 (paper-grade, H007 일반화).
> §10.9 attn entropy active 적용 (threshold 0.95 × log(384) ≈ 5.65).
> §18 inference 인프라 룰 inherit.

## 1. Hypothesis & Claim
- Hypothesis: **H010_ns_to_s_xattn**.
- Stacking on H008 champion (Platform 0.8387):
  - H008 mechanism (interaction axis): `MultiSeqHyFormerBlock` step 3 fusion =
    `DCNV2CrossBlock` (low-rank polynomial cross with x₀ residual). 그대로 유지.
  - H010 mechanism (sequence axis): `NSToSCrossAttention` per-block, applied
    after seq encoders, before query decoder. NS tokens (Q) attend
    bidirectionally to per-domain S tokens concatenated (K=V, L_total=384).
    Output: enriched NS tokens (B, 7, D) — **NS dimension preserved**.
- Predicted (paired classifications vs H008):
  - **super-additive** Δ vs H008 ≥ +0.005pt → paper-grade lift (Platform ≥ 0.8437).
  - **additive** Δ vs H008 ∈ [+0.001, +0.005pt] (Platform ∈ [0.8397, 0.8437]).
  - **noise** Δ vs H008 ∈ [−0.001, +0.001pt] (mechanism 일반화 가치 marginal).
  - **interference** Δ vs H008 < −0.001pt → REFUTED.
- Compute tier: **T2.4 extended (10 epoch × 30%, patience=3)**, ~2.5-3.5h wall.

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로 +
H008 DCN-V2 fusion 그대로 + **H010 NSToSCrossAttention 매 block 마다 추가**:

`MultiSeqHyFormerBlock.forward` step sequence:
1. Per-domain seq encoders → 4 도메인 S tokens (B, L_i, D).
2. **(H010 NEW) NS xattn**: S concat (B, L_total=384, D) + NS tokens (B, 7, D)
   → `NSToSCrossAttention` → enriched NS tokens (B, 7, D). residual update.
3. Per-domain query decoder (Nq=2 queries → seq cross-attention).
4. Fusion = `DCNV2CrossBlock` (H008 anchor) — decoded_q + enriched NS tokens
   의 polynomial cross.
5. Split back per-domain.

§18 inference 인프라 룰 모두 inherit (batch=256 default + PYTORCH_CUDA_ALLOC_CONF +
universal handler + 진단 로그).

## 3. Files
| File | H008 대비 | Purpose |
|---|---|---|
| `run.sh` | 변경 (3 flags 추가) | Entry point |
| `train.py` | 변경 (CLI flags 2 + model_args 2 + entropy 분기 확장) | CLI driver |
| `trainer.py` | byte-identical | Train loop |
| `model.py` | 변경 (`NSToSCrossAttention` 클래스 추가 + MultiSeqHyFormerBlock + PCVRHyFormer + collect_attn_entropies 통합) | PCVRHyFormer + DCN-V2 + NS xattn |
| `dataset.py` | byte-identical | PCVRParquetDataset |
| `infer.py` | 변경 (cfg.get 2개 추가) | §18 인프라 + new cfg |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | Auto schema |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS-token feature ref |
| `requirements.txt` | byte-identical | torch 2.7.1+cu126 |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Outputs
- `metrics.json` — `best_val_AUC`, `best_oof_AUC`, repro meta. mutation flags
  (`use_ns_to_s_xattn=true`, `fusion_type=dcn_v2`) 모두 기록. `attn_entropy_per_layer`
  + `attn_entropy_threshold` (0.95×log(384)≈5.65) + `attn_entropy_violation`.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — `NSToSCrossAttention` init 메시지 + per-block ns_xattn._last_entropy.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over H008 anchor
- **H010 mechanism (sequence axis, paper-grade NS→S bidirectional)**:
  - `NSToSCrossAttention` per-block module:
    - LN-Pre 분리 (LN(Q), LN(K)/LN(V)).
    - Multi-head cross-attention: Q=NS tokens (N_NS=7), K=V=S tokens concat (L_total=384).
    - Padding mask aware + all-pad guard.
    - Softmax-attention entropy logging (§10.9 룰, threshold 5.65).
    - Output projection + dropout.
    - Residual: `enriched_ns = ns_tokens + ns_xattn(ns_tokens, s_concat, mask_concat)`.
  - 통합 위치: `MultiSeqHyFormerBlock` step 1 직후 (per-domain seq encoders 출력)
    + step 2 직전 (query decoder 통과 전). **NS dimension 변경 없음**.
- **H008 mechanism (interaction axis) 유지**:
  - `DCNV2CrossBlock` token-wise polynomial cross (Pre-LN x₀ + 2 cross layers + low-rank rank=8).
  - `MultiSeqHyFormerBlock` step 3 fusion `RankMixerBlock` → `DCNV2CrossBlock` swap.
- **Layer-level integration**: NS tokens 와 S tokens 이 같은 cross-attention layer
  안에서 학습 → **layer-level gradient sharing** (block-level H008 보다 강한 통합).
  enriched NS tokens 가 DCN-V2 cross 에서 decoded queries 와 polynomial interaction
  → 두 axis (sequence enrichment + interaction cross) 가 sequential 단계에서 연결됨.
  §0 P1 ("seq + interaction 한 블록 gradient 공유") 강한 형태.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + 모든 mutation flags +
  attn_entropy diagnostic.
- `train_config.json` sidecar; `infer.py` reads cfg keys to instantiate
  matching modules for ckpt loading.

## 9. Why H010 = NS→S xattn (vs anchor recalibration backlog)

사용자 가치 align:
- anchor recalibration = measurement H, mechanism lift 0. cost-effective signal
  작음 (anchor 정확값 의존성은 H011 부터 H008 paired 비교 가 주가 되면 영향 작음).
- NS→S xattn = paper-grade mechanism (OneTrans), H007 (PASS marginal) 자연
  일반화, H008 anchor on champion stacking sub-H, H009 위치 충돌 회피 by 설계.
- §0 north star (sequence × interaction 통합) 의 sequence axis 강화 paper-grade
  lift 시도.

## 10. Carry-forward (§10.5 / §10.9)
- §10.5 LayerNorm on x₀ MANDATORY: `NSToSCrossAttention` LN-Pre 분리 (Q + KV).
  `DCNV2CrossBlock` Pre-LN x₀ 그대로.
- §10.9 OneTrans softmax-attention entropy abort: 본 H 가 **두 번째 active 적용**
  (H004 첫). threshold 0.95 × log(384) ≈ 5.65. 모든 layer < threshold 의무.
  초과 시 verdict.md `attn_entropy_violation: true` + abort 후속 H.

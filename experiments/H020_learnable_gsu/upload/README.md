# H020_learnable_gsu — Technical Report

> CLAUDE.md §17.2 sub-H of H019 champion (TWIN paradigm shift first entry,
> cloud measurable PASS 0.839674, Δ vs H010 corrected +0.001868pt).
> Single mutation: TWINBlock GSU 의 parameter-free inner product →
> learnable projection (W_q, W_k: nn.Linear(d_model, d_model//4, bias=False))
> 추가. paper-faithful TWIN GSU 형태 (Tencent 2024 RecSys, arXiv:2302.02352).
> H019 의 모든 다른 부분 (top_k=64, ESU, aggregator, gate=-2.0, num_heads=4,
> seq_max_lens 256/256/256/256, batch=1024, NS xattn, DCN-V2) byte-identical.
> §17.4 retrieval_long_seq re-entry justified (RE_ENTRY_JUSTIFIED — 4 사유,
> challengers.md 참조).

## 1. Hypothesis & Claim
- Hypothesis: **H020_learnable_gsu**.
- Sub-H of H019 (champion, retrieval mechanism class lever confirmed):
  - H019 mechanism (retrieval form): TWIN GSU+ESU per-domain. 그대로 유지.
  - H020 mechanism (scoring axis): GSU 의 parameter-free inner product →
    learnable projection. backbone embedding space 와 retrieval scoring
    space 분리.
- Predicted (paired classifications vs H019 cloud actual 0.839674):
  - **strong** Δ vs H019 ≥ +0.003pt → learnable scoring lift, anchor = H020.
  - **measurable** Δ ∈ [+0.001, +0.003pt] → 약 effect, sub-H 후보.
  - **noise** Δ ∈ (−0.001, +0.001pt] → scoring NOOP, retrieval selection axis retire.
  - **degraded** Δ < −0.001pt → projection rank reduction, sub-H = dim 확장 또는 retire.
- Compute tier: **T2.4 (~3.5h, ~$5-7)**, H019 동급.

## 2. What this code does

H019 anchor 인프라 (label_time-aware split, OOF holdout, NS xattn, DCN-V2,
TWINBlock per-domain) 그대로 + **TWINBlock GSU 에 W_q/W_k Linear projection 추가**:

`TWINBlock.forward` GSU 분기:
- H019: `scores = (history * candidate.unsqueeze(1)).sum(-1)` — backbone embedding 직접 inner product.
- H020 (this H): `scores = (W_k(history) * W_q(candidate).unsqueeze(1)).sum(-1)` — projection 후 inner product.

W_q, W_k = `nn.Linear(d_model=64, d_model//4=16, bias=False)` per domain.
4 도메인 × (W_q + W_k) × (64×16) = **8,192 trainable params 추가**.

§18 inference 인프라 룰 모두 inherit (H019 carry-forward).

## 3. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | 변경 (+ `--twin_learnable_gsu` flag, EXP_ID, 주석) | Entry point |
| `train.py` | 변경 (+ argparse `--twin_learnable_gsu` + model_args) | CLI driver |
| `model.py` | 변경 (TWINBlock에 W_q/W_k projection + PCVRHyFormer 에 flag pass) | PCVRHyFormer + TWIN |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | PCVRParquetDataset (§18.7 carry) |
| `infer.py` | byte-identical | §18 인프라 |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | Auto schema |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS-token feature ref |
| `requirements.txt` | byte-identical | torch 2.7.1+cu126 |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Outputs
- `metrics.json` — `best_val_AUC`, `best_oof_AUC`, repro meta. mutation flag
  `twin_learnable_gsu=true` 기록. attn entropy diagnostic carry-forward.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — `H020 TWIN retrieval enabled: top_k=64 num_heads=4 gate_init=-2.0 learnable_gsu=True` 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over H019 anchor
- **H020 mechanism (scoring axis, paper-faithful TWIN GSU)**:
  - `TWINBlock.__init__` 에 `learnable_gsu: bool = False` param.
  - `learnable_gsu=True` 시 `gsu_q`, `gsu_k` Linear projection (d_model → d_model//4, bias=False) 생성.
  - `forward` 에서 분기:
    - learnable_gsu=False (H019 default): backbone embedding 직접 inner product.
    - learnable_gsu=True (H020): projection 후 inner product.
- **H019 mechanism (retrieval form) 유지**:
  - per-domain TWINBlock (4 도메인) → top_k=64 filter → ESU MultiheadAttention(candidate Q) → (B, D).
  - TwinRetrievalAggregator: 4 (B, D) → mean → Linear → LayerNorm → sigmoid(gate=-2.0)≈0.12 residual ADD post-backbone.
- **§17.2 single mutation**: GSU scoring function 의 parameter-free → learnable projection 1단계. ESU / top_k / aggregator / gate / num_heads 전부 H019 byte-identical.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + `twin_learnable_gsu=true` flag.
- `train_config.json` sidecar; `infer.py` reads cfg keys to instantiate matching modules for ckpt loading (H019 carry, no infer.py change needed).

## 9. Why H020 = Learnable GSU (vs H021 per-domain top_k)

H020 과 H021 = H019 mechanism class 안 직교 sub-H:
- H020 = scoring quality axis (parameter-free → learnable projection).
- H021 = scoring quantity axis (uniform K → per-domain K).
- 두 axis 직교 → 동시 cloud submit, paired 비교 framework (predictions.md 참조).

H020 우선 이유:
- TWIN paper 의 GSU = learnable scorer (별도 projection). H019 의 simplified parameter-free 와 직접 차이.
- backbone embedding space 가 retrieval scoring 에 잘 맞는다는 가정 검증.
- §17.2 single mutation 깔끔.

## 10. Carry-forward (§10.5 / §10.9 / §10.10)
- §10.5 LayerNorm on x₀ MANDATORY: H019 carry (DCN-V2 Pre-LN, NS xattn LN-Pre 분리).
- §10.9 OneTrans softmax-attention entropy abort: ESU attention entropy 측정 carry-forward (H019 동일 기준).
- §10.10 InterFormer bridge gating σ(−2)≈0.12: H019 의 twin_gate 그대로 유지.
- §17.2 one-mutation: GSU scoring function 만 변경.
- §17.4 rotation: retrieval_long_seq re-entry RE_ENTRY_JUSTIFIED.
- §18.7 nullable to_numpy: H015 carry (dataset.py 변경 없음).
- §18.8 emit_train_summary: H019 carry, exp_id 만 H020 로 변경.

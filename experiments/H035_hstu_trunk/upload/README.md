# H035_hstu_trunk — Technical Report

> CLAUDE.md §17.2 paradigm shift sub-H — backbone_replacement class first-touch.
> Trigger: data_ratio=1 → eval 0.837785 (< ratio=0.3 의 0.839674) ceiling signal +
> retrieval_long_seq 4회 연속 (H019/H020/H021/H034) 후 강제 rotation.
> Single mutation: per-domain seq encoder 의 type 을 transformer (softmax MHA +
> GELU FFN) → HSTU (pointwise silu-attention + gated linear unit) 로 swap.
> Paper-faithful core (Meta 2024 ICML, arXiv:2402.17152, eq. 1-3).
> §17.4 backbone_replacement first-touch AUTO_JUSTIFIED.

## 1. Hypothesis & Claim
- Hypothesis: **H035_hstu_trunk**.
- Paradigm shift sub-H of H019:
  - H019 mechanism (sequence axis): per-domain TransformerEncoder (softmax MHA + GELU FFN).
  - H035 mechanism (sequence axis): per-domain HSTUEncoder (pointwise silu-attention + U gate).
  - 다른 모든 부분 byte-identical to H019 (TWIN GSU+ESU, NS xattn, DCN-V2, gate=-2.0, top_k=64, seq 256, batch 1024).
- Predicted (paired classifications vs H019 cloud actual 0.839674):
  - **strong** Δ ≥ +0.005pt → mechanism class lever 진짜, anchor = H035, 0.85 도달 가능성.
  - **measurable** Δ ∈ [+0.001, +0.005pt] → 약 effect, sub-H 후보.
  - **noise** Δ ∈ (-0.001, +0.001pt] → mechanism class 도 dead → cohort drift 강제 attack (H036).
  - **degraded** Δ < -0.001pt → HSTU 본 데이터 부적합 (Frame B), retire.
- Compute tier: **T2.4 (~3.5h, ~$5-7)**, H019/H020/H021 동급.

## 2. What this code does

H019 anchor 인프라 그대로 + **per-domain seq encoder 를 HSTU 로 swap**:

`HSTUEncoder` (paper eq. 1-3):
```
Step 1: h = SiLU(LN(x) · W_uvqk).chunk(4) → U, V, Q, K
Step 2: A = silu(QK^T / sqrt(head_dim)) / L          # length-normalized pointwise
        Y = A @ V
Step 3: out = LN(W_o · (Y ⊙ U)) + x                  # gated residual
```

Mask handling: padded keys → score 0 (silu(0) = 0, no contribution).

Multi-head reshape per Vaswani style. RoPE arg signature 호환 (HSTU 는 사용 안 함).

per-domain × per-block params:
- HSTU encoder per layer = 20,736 (uvqk Linear 16,384 + proj_out 4,096 + 2 LN 256)
- TransformerEncoder per layer = 54,144
- HSTU 가 transformer 의 0.38× → −264K 절약 (4 도메인 × 2 hyformer block 합산)

§18 inference 인프라 룰 모두 inherit (H019 carry).

## 3. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | + `--seq_encoder_type hstu` flag, EXP_ID, 주석 | Entry point |
| `train.py` | + `--seq_encoder_type` choices 에 'hstu' 추가 + help text | CLI driver |
| `model.py` | + HSTUEncoder class (~50 lines) + create_sequence_encoder dispatch 'hstu' 분기 | PCVRHyFormer + HSTU |
| 다른 모든 파일 | byte-identical | unchanged |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `MODEL_OUTPUT_PATH` (inference).

## 5. Outputs
- `metrics.json` — `seq_encoder_type=hstu` 기록. attn entropy diagnostic carry (silu-attention 이라 정보용).
- `train.log` — `H035 HSTU per-domain seq encoder` 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over H019 anchor
- **H035 mechanism (paradigm shift, sequence axis)**:
  - HSTUEncoder class — paper-faithful core (silu-attention + gated linear unit).
  - create_sequence_encoder dispatch 에 'hstu' 분기 추가.
  - TransformerEncoder 와 동일 forward signature → drop-in swap.
- **H019 mechanism (다른 부분) 유지**:
  - TWIN GSU+ESU per-domain retrieval (top_k=64, gate=-2.0, num_heads=4).
  - NS xattn + DCN-V2 fusion (interaction axis).
- **§17.2 single mutation**: per-domain seq encoder type 의 단일 변경.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + `seq_encoder_type=hstu` flag.
- `train_config.json` sidecar; `infer.py` reads cfg 기반 모델 재구성 (H019 carry — `seq_encoder_type` cfg 기록 시 ckpt loading 정상).

## 9. Why H035 = HSTU trunk replacement

H035 = 첫 backbone_replacement 카테고리 진입. 다른 후보들과 비교:

| 후보 | source | 비용 | 정당화 |
|---|---|---|---|
| **HSTU (H035)** | Meta 2024 paper-faithful | T2.4 ~$5-7 | 본 H |
| OneTrans full | Tencent 대회 organizer | T3 ~$15+ | H035 noise 시 후보 |
| InterFormer | Meta CIKM 2025 | T3 ~$15+ | 후순위 |
| HSTU full form (RAB 추가) | paper full | T2.4 ~$8 | H035 PASS measurable 시 sub-H |

H035 우선 이유:
- T2.4 cost-effective (낮은 risk-적정 비용).
- paper-faithful core (relative attention bias 등 paper-specific 부가 요소 제외, 핵심 silu+gate 만).
- params 작음 (transformer 의 0.38×) → §10.6 sample budget 친화.

## 10. Carry-forward (§10.5 / §10.9 / §10.10)
- §10.5 LayerNorm on x₀ MANDATORY: HSTU 가 norm_in (Pre-LN) + norm_out 둘 다 적용. mandatory 충족.
- §10.9 OneTrans softmax-attention entropy abort: HSTU silu-attention 은 softmax 가 아님. **threshold 적용 명시 예외** — 측정만 carry, abort 적용 안 함 (H035 에서 §10.9 룰 갱신 가능성).
- §10.10 InterFormer bridge gating σ(−2): TWIN twin_gate 그대로.
- §17.2 one-mutation: per-domain seq encoder type 만 변경.
- §17.4 rotation: backbone_replacement first-touch AUTO_JUSTIFIED.
- §18.7 nullable to_numpy: H015 carry.
- §18.8 emit_train_summary: H019 carry, exp_id 만 H035 로 변경.

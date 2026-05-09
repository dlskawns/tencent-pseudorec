# H021_per_domain_top_k — Technical Report

> CLAUDE.md §17.2 sub-H of H019 champion (TWIN paradigm shift first entry,
> cloud measurable PASS 0.839674, Δ vs H010 corrected +0.001868pt).
> Single mutation: TWINBlock 의 top_k policy uniform=64 → per-domain
> `{seq_a:64, seq_b:64, seq_c:64, seq_d:96}`. domain d 만 50% 확장
> (§3.5 p90=2215, K=64 의 top 2.9% — 4 도메인 중 가장 under-served).
> uniform K=128 sweep flat 영역 (75% conservative end).
> **TWINBlock class byte-identical to H019** — PCVRHyFormer wiring + train.py
> argparse + run.sh 만 변경. **trainable params 추가 0**.
> H019 의 모든 다른 부분 (GSU, ESU, aggregator, gate=-2.0, num_heads=4,
> seq_max_lens 256/256/256/256, batch=1024, NS xattn, DCN-V2) byte-identical.
> §17.4 retrieval_long_seq re-entry (3회 연속) RE_ENTRY_JUSTIFIED — 5 사유
> (challengers.md). H020 (scoring axis) 와 직교 sub-H, paired 비교 framework.

## 1. Hypothesis & Claim
- Hypothesis: **H021_per_domain_top_k**.
- Sub-H of H019 (champion, retrieval mechanism class lever confirmed):
  - H019 mechanism (retrieval form): TWIN GSU+ESU per-domain. 그대로 유지.
  - H021 mechanism (quantity axis): top_k policy uniform → domain-aware.
    domain d (p90=2215, longest) 만 K=64→96 으로 확장. 다른 도메인은 K=64 floor 유지.
- Predicted (paired classifications vs H019 cloud actual 0.839674):
  - **strong** Δ vs H019 ≥ +0.003pt → per-domain K lift, anchor = H021.
  - **measurable** Δ ∈ [+0.001, +0.003pt] → 약 effect, H020 결과와 paired stack 가능성.
  - **noise** Δ ∈ (−0.001, +0.001pt] → uniform K=64 globally saturated, retrieval quantity axis dead.
  - **degraded** Δ < −0.001pt → K=96 noise injection, sub-H K=80 또는 retire.
- Compute tier: **T2.4 (~3.5h, ~$5-7)**, H019 동급.

## 2. What this code does

H019 anchor 인프라 (label_time-aware split, OOF holdout, NS xattn, DCN-V2,
TWINBlock per-domain) 그대로 + **PCVRHyFormer 의 TWIN block instantiation 시
도메인별 top_k 적용**:

`PCVRHyFormer.__init__` TWIN block construction:
- H019: `top_k=twin_top_k` (single int, 모든 도메인 동일).
- H021 (this H): `--twin_top_k_per_domain "64,64,64,96"` 로 per-domain dict 받아서
  도메인별 다른 K 로 4 TWINBlock instantiate.

**TWINBlock class 자체는 byte-identical to H019** (이미 top_k 를 init param 으로
받기 때문). 변경 범위 = PCVRHyFormer wiring + train.py argparse + run.sh.

§3.5 quantitative motivation:

| 도메인 | p90 seq len | K=64 의 vs p90 비율 | H021 K | H021 K 의 비율 |
|---|---|---|---|---|
| seq_a | 1562 | 4.1% | 64 | 4.1% (변경 없음) |
| seq_b | 1393 | 4.6% | 64 | 4.6% (변경 없음) |
| seq_c | 887 | 7.2% | 64 | 7.2% (변경 없음) |
| seq_d | 2215 | **2.9%** | **96** | **4.3%** (50% 확장) |

trainable params 추가 **0** (top_k 는 hyperparam).

§18 inference 인프라 룰 모두 inherit (H019 carry-forward).

## 3. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | 변경 (+ `--twin_top_k_per_domain "64,64,64,96"` flag, EXP_ID, 주석) | Entry point |
| `train.py` | 변경 (+ argparse `--twin_top_k_per_domain` + parsing logic + model_args) | CLI driver |
| `model.py` | 변경 (PCVRHyFormer __init__ wiring: per-domain K dict 처리, TWINBlock class byte-identical) | PCVRHyFormer + TWIN |
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
  `twin_top_k_per_domain="64,64,64,96"` 기록. attn entropy diagnostic carry-forward.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — `H021 enabled: per-domain K = {'seq_a': 64, 'seq_b': 64, 'seq_c': 64, 'seq_d': 96}` 메시지 + `H021 TWIN per-domain K: seq_a=64 seq_b=64 seq_c=64 seq_d=96` 메시지.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over H019 anchor
- **H021 mechanism (quantity axis, per-domain K)**:
  - `train.py` argparse `--twin_top_k_per_domain` (comma-separated 4-int string).
  - parsing logic: string → dict `{seq_a, seq_b, seq_c, seq_d → int}`.
  - `PCVRHyFormer.__init__` wiring: TWIN block instantiation 시 도메인별 K 적용.
  - **TWINBlock class byte-identical**: 이미 top_k 를 init param 으로 받음.
- **H019 mechanism (retrieval form) 유지**:
  - per-domain TWINBlock (4 도메인) → top_k filter → ESU MultiheadAttention(candidate Q) → (B, D).
  - TwinRetrievalAggregator: 4 (B, D) → mean → Linear → LayerNorm → sigmoid(gate=-2.0)≈0.12 residual ADD post-backbone.
- **§17.2 single mutation**: top_k policy 의 uniform → domain-aware. TWINBlock 내부 변경 0. ESU / GSU / aggregator / gate / num_heads 전부 H019 byte-identical.

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256 + `twin_top_k_per_domain` flag.
- `train_config.json` sidecar; `infer.py` reads cfg keys to instantiate matching modules for ckpt loading. **infer.py 변경 없음** — `twin_top_k` 는 (int 또는 dict) 둘 다 PCVRHyFormer 가 지원.

## 9. Why H021 = Per-domain top_k (vs H020 learnable GSU)

H020 과 H021 = H019 mechanism class 안 직교 sub-H:
- H020 = scoring quality axis (parameter-free → learnable projection).
- H021 = scoring quantity axis (uniform K → per-domain K).
- 두 axis 직교 → 동시 cloud submit, paired 비교 framework.

H021 우선 이유:
- §3.5 quantitative motivation 가장 강함: domain d K=64 = top 2.9%, paper TWIN K/L=1.3% 보다 높지만 4 도메인 중 가장 under-served.
- params 추가 0 → cost-effective.
- single-domain change → paired Δ 해석 가장 깔끔.

## 10. Carry-forward (§10.5 / §10.9 / §10.10)
- §10.5 LayerNorm on x₀ MANDATORY: H019 carry (DCN-V2 Pre-LN, NS xattn LN-Pre 분리).
- §10.9 OneTrans softmax-attention entropy abort: ESU attention entropy 측정 carry-forward. domain d K=96 → threshold 0.95 × log(96) ≈ 4.34.
- §10.10 InterFormer bridge gating σ(−2)≈0.12: H019 의 twin_gate 그대로 유지.
- §17.2 one-mutation: top_k policy 만 변경.
- §17.4 rotation: retrieval_long_seq re-entry (3회 연속) RE_ENTRY_JUSTIFIED.
- §18.7 nullable to_numpy: H015 carry (dataset.py 변경 없음).
- §18.8 emit_train_summary: H019 carry, exp_id 만 H021 로 변경.

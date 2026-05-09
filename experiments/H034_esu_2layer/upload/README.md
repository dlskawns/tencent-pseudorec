# H034_esu_2layer — Technical Report

> CLAUDE.md §17.2 sub-H of H019 champion (TWIN paradigm shift, cloud
> measurable PASS 0.839674). Single mutation: TWINBlock 의 ESU 를
> 1-layer MultiheadAttention → 2-layer stack (intermediate residual + LayerNorm).
> H020 (scoring quality axis) / H021 (scoring quantity axis) 와 직교 axis A3
> = retrieved token 처리 capacity. H020/H021 모두 NOOP 일 때 가장 informative.
> H019 의 모든 다른 부분 (GSU, top_k=64, aggregator, gate=-2.0, num_heads=4,
> seq_max_lens, batch) byte-identical.
> §17.4 retrieval_long_seq re-entry (4회 연속) RE_ENTRY_JUSTIFIED.

## 1. Hypothesis & Claim
- Hypothesis: **H034_esu_2layer**.
- Sub-H of H019 (champion):
  - H019 mechanism: TWIN GSU + 1-layer ESU. 그대로.
  - H034 mechanism (capacity axis): ESU 를 2-layer MHA stack 으로 확장.
    각 layer 후 residual + LayerNorm (Pre-LN style).
- Predicted (paired classifications vs H019 cloud actual 0.839674):
  - **strong** Δ ≥ +0.003pt → ESU capacity bottleneck confirmed.
  - **measurable** Δ ∈ [+0.001, +0.003pt] → 약 effect.
  - **noise** Δ ∈ (−0.001, +0.001pt] → 1-layer ESU sufficient → retrieval class 의 capacity axis dead → cohort/HSTU pivot.
  - **degraded** Δ < −0.001pt → 2-layer over-capacity (학습 instability).
- Compute tier: **T2.4 (~3.5h, ~$5-7)**, H019 동급.

## 2. What this code does

H019 anchor 인프라 그대로 + **TWINBlock ESU 를 multi-layer 로 확장**:

`TWINBlock.__init__`:
- `esu_num_layers=1` (H019 default): `self.esu` = single MHA, `self.norm` = single LN. byte-identical.
- `esu_num_layers≥2` (H034): `self.esu_layers` = ModuleList[MHA × N], `self.esu_norms` = ModuleList[LN × N].

`TWINBlock.forward` ESU 분기:
- num_layers=1: `out = norm(candidate_q + esu(...))` — H019.
- num_layers=2: 각 layer 마다 `x = ln(x + layer(x, topk_history, topk_history))`. residual + LayerNorm 누적.

per-domain trainable params 추가: ESU MHA 1 layer ≈ 16K params. 2-layer = +16K per domain × 4 domains = **+64K total**.

## 3. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | + `--twin_esu_num_layers 2` flag, EXP_ID, 주석 | Entry point |
| `train.py` | + argparse `--twin_esu_num_layers` + model_args | CLI driver |
| `model.py` | TWINBlock multi-layer ESU + PCVRHyFormer 에 flag pass | PCVRHyFormer + TWIN |
| 다른 모든 파일 | byte-identical | unchanged |

## 4. Outputs
- `metrics.json` — `twin_esu_num_layers=2` 기록.
- `train.log` — `H034 TWIN retrieval enabled: ... esu_num_layers=2` 메시지.

## 5. Why H034 = ESU 2-layer (vs H020/H021)

H020/H021/H034 = H019 mechanism class 안 3개 직교 axis sub-H:
- H020 = **scoring quality** axis (parameter-free → learnable projection).
- H021 = **scoring quantity** axis (uniform K → per-domain K).
- H034 = **capacity** axis (ESU 1-layer → 2-layer).

H034 가 가장 informative 한 시나리오:
- H020 noise + H021 noise → scoring axis 전체 saturated → capacity 가 진짜 lever 인지 검증.
- H020/H021 PASS 시: H034 결과는 별도 axis 신호 (orthogonal lever).

## 6. Carry-forward
- §17.2 one-mutation: ESU layer 수만 변경.
- §17.4 rotation: retrieval_long_seq 4회 연속 RE_ENTRY_JUSTIFIED.
- §10.5/§10.9/§10.10 H019 carry. ESU multi-layer Pre-LN style.
- §18.7/§18.8 H019 carry.

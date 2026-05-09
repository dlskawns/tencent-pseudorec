# H035 — Method Transfer

## ① Source

- **Zhai, J. et al. 2024 (Meta)** — "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU). ICML 2024. arXiv:2402.17152. **본 H 의 1차 source**.
  - 핵심: pointwise silu-attention + gated linear unit (U projection).
  - 트릴리언-파라미터 scale, autoregressive recommendation 환경 검증.
- **Vaswani et al. 2017** — Transformer baseline (softmax MHA + FFN). 본 H 의 *비교 control* (per-domain seq encoder 의 default).
- 카테고리 (`backbone_replacement/`): NEW first-touch. 미경험 paradigm shift class.

## ② Original mechanism (HSTU paper, eq. 1-3)

```
Input: x ∈ (B, L, D)
Step 1: Project to U/V/Q/K
    h = SiLU(LN(x) · W_uvqk)              # W_uvqk: (D → 4D)
    U, V, Q, K = h.chunk(4, dim=-1)         # each (B, L, D)

Step 2: Pointwise attention
    A = silu(QK^T / sqrt(d)) / L            # length-normalized, NO softmax
    Y = A @ V                               # (B, L, D)

Step 3: Gated output
    out = LN(W_o · (Y ⊙ U)) + x             # gated by U, residual
```

핵심 차이 (vs Transformer):
- **Attention func**: softmax → silu (pointwise, no probability normalization).
- **Nonlinearity**: FFN (linear → GELU → linear) → multiplicative gate (Y ⊙ U).
- **Length norm**: 없음 → /L (paper: prevent attention collapse for long L).
- **Positional**: RoPE → implicit (silu attention 의 자체).

## ③ What we adopt (H035 mutation)

- **Mechanism**: paper eq. 1-3 의 minimum viable form 으로 `HSTUEncoder` 클래스 추가. `TransformerEncoder` 와 동일 forward signature → drop-in swap.
- **변경 내용 (3 files)**:
  - `model.py`: `HSTUEncoder` class 신규 (~50 lines), `create_sequence_encoder` dispatch 에 `'hstu'` 분기 (1 line).
  - `train.py`: `--seq_encoder_type` choices 에 `'hstu'` 추가 + help text.
  - `run.sh`: `--seq_encoder_type hstu` flag bake.
- **CLI**: `--seq_encoder_type hstu` (default `transformer`).
- **다른 모든 부분 byte-identical to H019**: TWIN GSU+ESU, NS xattn, DCN-V2, gate=-2.0, top_k=64, seq 256, batch 1024.

## ④ What we modify (NOT a clone)

- **Per-domain encoder swap (not full backbone)**: paper trunk = HSTU layer stack 전체. 본 H = per-domain seq encoder 만 HSTU, NS xattn / DCN-V2 / candidate xattn 등 transformer-style 그대로. **Frame C 위험 인지** — challengers.md.
- **Minimum viable form**: paper full 형태에는 relative attention bias (rab) + 추가 normalization tricks. 본 H = 핵심 silu + gate 만. PASS strong → sub-H = full form.
- **No causal masking**: paper 는 autoregressive generation. 본 데이터 binary classification → bidirectional self-attention.
- **Hidden mult = 1**: paper formulation. transformer 의 4× FFN 과 다름 — HSTU 의 본질이 FFN 없는 architecture.
- **§17.2 single mutation**: per-domain seq encoder type 의 단일 변경. NS xattn / DCN-V2 / TWIN / aggregator / gate 전부 H019 byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: NEW backbone class — softmax-attention → silu-attention. UNI-REC sequence axis 의 mechanism class 자체 교체.
- **Interaction reference**: 변경 없음 (DCN-V2 fusion).
- **Bridging mechanism**: 변경 없음 (NS xattn).
- **Training procedure**: 변경 없음.
- **primary_category**: `backbone_replacement` (NEW first-touch — §17.4 rotation auto-justified).
- **Innovation axis**: §0 backbone 표 의 paradigm shift first-class entry. paper-grade form, 대회 organizer (Tencent) 와 다른 회사 (Meta) 의 다른 paradigm 도입.
- **OneTrans / InterFormer / PCVRHyFormer 와의 관계**:
  - OneTrans (single-stream + mixed-causal): 별도 backbone class (`--backbone onetrans`). H035 와 직교.
  - InterFormer: bridge gating 변경 없음.
  - PCVRHyFormer: per-domain encoder backbone 유지, encoder layer type 만 transformer → HSTU.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: HSTU encoder per layer = ~21K (vs TransformerEncoder ~54K). **HSTU 가 transformer 보다 params 0.38×**.
- 4 도메인 × 2 hyformer block = 8 encoder. 총 HSTU = 168K vs transformer 432K. **−264K 절약**.
- §10.6 sample budget 친화 — H019 (paradigm shift carry) 보다도 더 작음.
- Sample-scale viability hard test: **local sanity 1 epoch + 1000-row → loss finite + HSTU forward NaN-free + create_sequence_encoder('hstu') dispatch 정상**. 검증 완료 (T0 sanity PASS).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: HSTU 가 norm_in (Pre-LN) + norm_out (Post-projection LN) 둘 다 적용. mandatory 충족.
- **§10.6 sample budget cap**: 위반 안 함 (HSTU params < transformer).
- **§10.7 카테고리 rotation**: `backbone_replacement` first-touch auto-justified.
- **§10.9 OneTrans softmax-attention entropy abort**: HSTU 의 silu-attention 은 softmax 가 아니라 entropy metric 의미 다름. **threshold 적용 명시 예외** — 측정만 carry-forward (정보용), abort 적용 안 함.
- **§10.10 InterFormer bridge gating σ(−2)**: TWIN twin_gate 그대로 유지 (변경 없음).
- **§17.2 one-mutation**: per-domain seq encoder type 만 변경.
- **§17.3 binary success**: Δ vs H019 ≥ +0.005pt → PASS strong (paradigm shift first-class entry 임계). [+0.001, +0.005pt] measurable. < +0.001pt REFUTED.
- **§17.4 rotation**: backbone_replacement first-touch auto-justified.
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. campaign cap $100 친화.
- **§18.6 dataset-inference-auditor**: H035 upload/ ready 직전 PASS 의무. dataset.py / infer.py / make_schema.py byte-identical → audit 범위 좁음 (model.py + train.py + run.sh).
- **§18.7 nullable to_numpy**: H015 carry (영향 없음).
- **§18.8 emit_train_summary**: H019 carry, exp_id 만 H035 로 변경.

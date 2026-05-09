# H034 — Method Transfer

## ① Source

- **Chang, J. et al. 2024 (Tencent RecSys)** — TWIN. ESU = MultiheadAttention (paper 명시 layer 수 없음, transformer block-style 가능).
- **Vaswani et al. 2017** — Transformer multi-layer encoder. Pre-LN style residual + LayerNorm.
- **H019** — base. 1-layer ESU 의 capacity 검증.
- 카테고리 (`retrieval_long_seq/`): re-entry 4회. capacity axis sub-H.

## ② Original mechanism

H019 ESU:
```python
attended, _ = self.esu(candidate_q, topk_history, topk_history, key_padding_mask=topk_pad_mask)
attended = self.norm(candidate_q + attended).squeeze(1)
```
1-layer MHA + 1 LayerNorm.

## ③ What we adopt (H034 mutation)

`TWINBlock.esu` 를 multi-layer stack 으로 확장:
- num_layers=1 (default, H019 byte-identical): self.esu + self.norm 단일.
- num_layers≥2 (H034): self.esu_layers ModuleList + self.esu_norms ModuleList.

forward 분기:
```python
if self.esu_num_layers == 1:
    # H019 path
    attended, _ = self.esu(candidate_q, topk_history, topk_history, ...)
    attended = self.norm(candidate_q + attended).squeeze(1)
else:
    # H034 path
    x = candidate_q
    for layer, ln in zip(self.esu_layers, self.esu_norms):
        attn_out, _ = layer(x, topk_history, topk_history, ...)
        x = ln(x + attn_out)
    attended = x.squeeze(1)
```

per-domain trainable params: ESU MHA 1-layer ≈ 16K. 2-layer = +16K per domain × 4 = **+64K total**.

CLI: `--twin_esu_num_layers 2`.

## ④ What we modify (NOT a clone)

- **Layer 수 = 2 (paper unspecified)**: paper TWIN 은 ESU layer 수 명시 안 함. 본 H = transformer block convention (2-layer minimum 으로 multi-layer effect 검증).
- **Pre-LN style**: residual + LN 누적 (paper Pre-LN 권장 패턴).
- **§17.2 single mutation**: ESU layer 수만 변경. GSU / top_k / aggregator / gate 전부 H019 byte-identical.

## ⑤ UNI-REC alignment

- Sequential reference: H019 carry.
- Interaction reference: 변경 없음.
- primary_category: `retrieval_long_seq` (re-entry, 4회 연속).
- Innovation axis: capacity axis = retrieval mechanism 의 token 처리 깊이.

## ⑥ Sample-scale viability

- 추가 params: +64K (4 도메인 × 16K). TWIN module = H019 71K + 64K = 135K. total 161M 의 0.084%.
- §10.6 sample budget 위반 인지 (paradigm shift class exempt, H019 carry).
- T0 sanity: TWINBlock(esu_num_layers=2) forward NaN-free 검증 완료.
- 1-layer vs 2-layer ablation diff 0.375 (mechanism active).

## ⑦ Carry-forward rules

- §17.2 one-mutation: ESU layer 수만 변경.
- §17.3 binary success: Δ vs H019 ≥ +0.003pt strong / [+0.001, +0.003pt] measurable / (−0.001, +0.001pt] noise / < −0.001pt degraded.
- §17.4 rotation: 4회 연속 RE_ENTRY_JUSTIFIED.
- §10.5/§10.9: H019 carry. 2 ESU layer 모두 entropy 측정.
- §10.10: H019 의 twin_gate σ(-2)=0.12 그대로.
- §18.6/§18.7/§18.8: H019 carry.

# H045_cross_domain_bridge — Technical Report

> CLAUDE.md §0.5 data-signal-driven: §3.5 4 도메인 disjoint vocab (jaccard
> 0.7~10%) — disjoint 같지만 0.7~10% overlap 실재. 14 H 동안 cross-domain
> feature transfer 0회. TWIN aggregator 는 per-domain 독립 처리 후 mean →
> 도메인 간 정보 흐름 부재. H045 = per-domain encoder output 사이 explicit
> MultiheadAttention bridge → cross-domain semantic 흐름 추가.

## 1. Hypothesis & Claim
- Hypothesis: 4 도메인 sequence encoder output 이 서로 attend → cross-
  domain feature transfer 학습 → user 의 다른 도메인 활동 정보가 candidate
  prediction 에 활용됨. 14 H 모두 도메인 독립 처리 한 한계 attack.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt → cross-domain
  axis 가 lever, sub-H 가치.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — CrossDomainBridge

```python
# Per-domain seq tokens (4 × (B, T, D)) → masked mean-pool → 4 × (B, D)
# → stack (B, 4, D) → MultiheadAttention(num_heads=4) across 4 tokens
# → mean over domain dim → (B, D) → Linear+LN+gated residual

per_domain = [masked_mean(seq, mask) for seq, mask in zip(seq_tokens, seq_masks)]
stacked = torch.stack(per_domain, dim=1)               # (B, 4, D)
attended, _ = self.attn(stacked, stacked, stacked)
attended = self.attn_ln(attended + stacked)
pooled = attended.mean(dim=1)                          # (B, D)
bridged = self.out_ln(self.proj(pooled))
output = output + sigmoid(self.bridge_gate) * bridged  # gate init=-2 ≈ 0.12
```

**Why bridge BEFORE TWIN**: cross-domain context informs backbone output
*before* TWIN's per-domain retrieval. TWIN candidate (item NS) attention
operates on bridge-aware base.

**왜 이게 진짜 NEW**:
1. 14 H 동안 cross-domain feature transfer 0회. TWIN aggregator (mean) /
   per-domain encoder 모두 도메인 독립 처리.
2. §3.5 jaccard 0.7~10% overlap signal 직접 attack — 도메인 간 vocab 가
   완전 disjoint 가 아닌 부분 overlap 이므로 cross-domain semantic 학습 가능.
3. §10.10 InterFormer bridge gating σ(-2)≈0.12 패턴 적용 — H019 TWIN gate
   와 같은 form, 진입 conservative.

## 3. Decision tree (post-result)

| Δ vs H019 platform (0.839674) | Action |
|---|---|
| ≥ +0.003pt | cross-domain axis main lever, sub-H = num_heads sweep + stacked layers |
| [+0.001, +0.003pt] | additive 약 effect, sub-H 가치 |
| (-0.001, +0.001pt] | cross-domain mechanism 부족, 더 강한 form (per-token cross-attn vs current pooled token) 또는 axis pivot |
| < -0.001pt | bridge 가 noise / gate 학습 안 됨, retire |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `model.py` | + CrossDomainBridge class (~50 lines), + 3 __init__ params, + 1 module init block, + 1 forward call (residual ADD before TWIN) | Model |
| `train.py` | + 3 argparse + 3 plumbing line | CLI |
| `run.sh` | + `--use_cross_domain_bridge --cross_domain_num_heads 4 --cross_domain_gate_init -2.0` | Entry |
| `README.md` | new | Doc |
| `trainer.py / dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **21,057** (4-head MHA on D=64 + LN + Linear + gate). 161M total 의 0.013%.

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS for model.py + train.py
2. ✅ shellcheck PASS for run.sh
3. ✅ Forward shape (B=4, D=64) preserved
4. ✅ NaN-free output
5. ✅ trainable params = 21,057
6. ✅ gradient flow through all 4 domain tokens
7. ✅ defensive: all-padded masks NaN-free
8. ✅ ablation diff (bridge output vs zero) = 0.349 (gate=0.12, 작동 confirm)
9. ✅ md5 verify: 6 unchanged files identical to H019

## 6. Carry-forward
- §17.2 single mutation: CrossDomainBridge 1개 추가. user-side / TWIN / NS xattn / 다른 모든 부분 byte-identical to H019.
- §17.4 rotation: cross_domain_modeling axis NEW first-touch, AUTO_JUSTIFIED.
- §10.10 InterFormer bridge gating σ(-2)≈0.12 mandate 통과.
- §10.6 sample budget: trainable params +21K (sample-scale 안).
- §0.5 data-signal-driven: §3.5 4 도메인 jaccard overlap signal 직접 attack, paper transplant 0.
- §18.7 + §18.8 H019 carry.

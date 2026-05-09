# H035 — Literature References

## Primary
- **Zhai, J. et al. 2024** — "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (HSTU). ICML 2024. arXiv:2402.17152. (Meta)
  - 본 H 의 1차 source. paper-faithful core (silu-attention + gated linear unit).
  - GitHub: github.com/facebookresearch/generative-recommenders.
- **Vaswani, A. et al. 2017** — "Attention is All You Need". Transformer baseline (softmax MHA + FFN). 본 H 의 비교 control.
- **H019** — TWIN paradigm shift first entry, cloud measurable PASS 0.839674. base + retrieval mechanism.

## Secondary (alternative paradigm shifts)
- **OneTrans (Tencent, WWW 2026, arXiv:2510.26104)** — single-stream + mixed-causal mask. 별도 backbone class. H035 noise 시 후보.
- **InterFormer (Meta, CIKM 2025, arXiv:2411.09852)** — 3-arch + bidirectional bridges. 별도 backbone class.
- **PCVRHyFormer (organizer baseline)** — per-domain encoder + RankMixer NS tokenizer. 본 H 의 backbone (HSTU 가 그 안 per-domain encoder swap).

## HSTU 의 paper claims (참고용)
- 트릴리언-파라미터 scale, recommendation task SOTA.
- standard transformer 보다 1.5~3× 빠른 inference.
- pointwise attention 의 dense token interaction 이 long-history retrieval 에 잘 작동.

## Carry-forward refs (from H019)
- **TWIN (Chang et al. 2024)** — H019 carry, per-domain GSU+ESU 그대로.
- **DCN-V2 (Wang et al. 2021)** — H008 carry, fusion mechanism.
- **NS→S xattn (OneTrans)** — H010 carry.

## What's NOT a clone

- 본 H 는 **HSTU paper 의 1:1 재현 아님**:
  - paper full form 에는 relative attention bias (rab) — 본 H = 미적용.
  - paper recommendation = autoregressive generation — 본 데이터 = binary classification (bidirectional attention 사용).
  - paper trunk = 모든 layer HSTU stack — 본 H = per-domain seq encoder 만 HSTU, NS xattn / DCN-V2 등 transformer-style 그대로.
  - paper scale = 트릴리언 params + 1B+ events — 본 H = 161M params + sample-scale.
  - paper hidden_mult = N/A (no FFN expansion). 본 H = 동일 (HSTU 의 본질).

## H010~H034 carry-forward refs

- **H010 (NS→S xattn) PASS additive** — anchor 0.837806 corrected.
- **H011~H013 REFUTED** — input/MoE/hyperparameter mutations.
- **H014 REFUTED (L4 dense form retire)**.
- **H015~H018 marginal/REFUTED** — temporal_cohort sibling 4 H.
- **H019 cloud measurable PASS** — TWIN paradigm shift first entry. **data_ratio=1 → 0.837785 ceiling 강한 신호**.
- **H020 (in flight)** — TWIN sub-H scoring axis.
- **H021 (in flight)** — TWIN sub-H quantity axis.
- **H033 (built)** — TWIN combined H020 ∘ H021.
- **H034 (built)** — TWIN sub-H capacity axis (ESU 2-layer).

H035 는 retrieval class 4 H 누적 후 **첫 backbone_replacement** — paradigm shift family 의 다음 level.

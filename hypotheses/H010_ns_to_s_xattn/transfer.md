# H010 — Method Transfer

## ① Source

- **OneTrans** (Tencent UNI-REC team, "OneTrans: Single-stream Transformer for Unified Recommendation with Mixed-Token Causal Attention", arXiv:2510.26104, WWW 2026).
  - 본 H 의 paper-grade source. NS→S bidirectional cross-attention 의 정수.
- **H007 (candidate_aware_xattn)** precedent — 1-token candidate query 의 single mutation form. PASS marginal at Platform 0.8352. 본 H 가 query 수 일반화 (1 → N_NS=7).
- 카테고리 family (target_attention): DIN, DIEN, CAN, SIM, TWIN, HSTU, OneTrans.

## ② Original mechanism (OneTrans NS→S half)

OneTrans 의 mixed causal attention 의 4 sub-mask 중 본 H 가 채택하는 것은 **NS→S bidirectional half**:

```
M[i, j] = 1 if can_attend(token_i, token_j) else -inf
NS→S: bidirectional but j.timestamp < candidate.timestamp  (본 H 가 채택)
S→S: causal upper-triangular                              (별도 H)
NS→NS: full self-attention                                (별도 H)
Candidate→{S, NS}: full up to candidate.timestamp         (H007 가 단일 query subset)
```

**Modern transformer-style implementation** (paper convention):
- Pre-LN.
- Multi-head cross-attention (Q=NS-tokens, K=V=S-tokens).
- Padding mask aware.
- Residual + output projection.

## ③ What we adopt

- **Mechanism class**: NS-tokens 가 S-tokens 에 multi-head bidirectional cross-attention. minimum viable form.
- **신규 module `NSToSCrossAttention`** (~80줄):
  - Input: `ns_tokens (B, N_NS=7, D)`, `s_tokens_concat (B, L_total=384, D)` (4 도메인 concat: a 64 + b 64 + c 128 + d 128), `padding_mask (B, L_total)`.
  - Pre-LN 분리 적용 (LN(Q), LN(K)/LN(V)).
  - Multi-head cross-attention: Q=NS tokens (N_NS=7), K=V=S tokens (L_total=384).
  - Softmax-attention entropy 측정 → §10.9 룰 적용 (threshold 0.95 × log(384) ≈ 5.65).
  - Output: enriched NS tokens `(B, N_NS=7, D)` — **NS dimension 변경 없음** (H009 위치 충돌 회피 by 설계).
  - Residual: `enriched_ns = ns_tokens + xattn_output` (Post-LN 또는 Pre-LN 표준 residual).
  - num_heads = 4 (default = anchor's `num_heads`).
  - dropout = anchor's `dropout_rate` (0.01).
- **`MultiSeqHyFormerBlock` 통합 위치**:
  1. Per-domain seq encoder 출력 → 4 도메인 S tokens.
  2. **신규 step**: 4 도메인 S tokens concat (L_total=384) + NS tokens (B, 7, D) → `NSToSCrossAttention` → enriched NS tokens (B, 7, D). **S tokens, query decoder, DCN-V2 입력 토큰 stack 변경 없음**.
  3. 기존 query decoder (per-domain Nq=2 queries → seq cross-attention) 통과.
  4. 기존 DCN-V2 fusion (H008 anchor 그대로) — decoded_q + enriched NS tokens 의 polynomial cross.
- **PCVRHyFormer constructor 인자**: `use_ns_to_s_xattn: bool = False`, `ns_xattn_num_heads: int = 4`.
- **CLI flags**: `--use_ns_to_s_xattn`, `--ns_xattn_num_heads 4`.
- **H008 mechanism 유지**: `--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8`.
- **인프라**: §18 룰 inherit (batch=256 default + PYTORCH_CUDA_ALLOC_CONF + universal handler + 진단 로그).

## ④ What we modify (NOT a clone)

- **Mixed causal mask 의 NS→S half 만 채택**: paper 의 4 sub-mask 중 1개만. S→S causal half (paper 의 sequence axis) 는 PCVRHyFormer 의 transformer encoder 가 이미 처리. NS→NS self-attention 은 별도 H. 단일 mutation 정신 유지.
- **Backbone replacement 아님**: paper 는 single-stream transformer. 우리는 PCVRHyFormer baseline + DCN-V2 anchor 그대로 + 1 cross-attention layer 추가. **mechanism injection** 형태. paper 의 100M user 규모 dependency 회피.
- **Per-domain seq encoder 보존**: paper 는 S-token 4 도메인 merged stream + 도메인 ID embedding. 우리는 PCVRHyFormer 의 per-domain seq encoder 통과 후 concat → NS xattn. domain dimension 보존.
- **timestamp 기반 causal masking 미적용**: paper 의 NS→S 가 candidate timestamp 까지 attend. 우리는 padding mask 만 (label_time split 으로 leakage 이미 차단). timestamp masking 은 별도 H (mixed causal mask injection backlog).
- **Single layer**: paper 는 multiple transformer layers. 우리는 1 cross-attention layer 만 추가. layer 수는 sub-H.
- **NS tokens 가 query, S tokens 가 K/V**: paper 는 single-stream attention block 안에서 모든 token 이 Q/K/V 모두. 우리는 cross-attention (NS→S only). 단순화.
- **§17.2 one-mutation**: NSToSCrossAttention 한 클래스 추가 + flag 분기. anchor (H008) 위 single mutation. NS-token 수, num_queries, fusion_type, dcn_v2 hyperparam 등 변경 없음.

## ⑤ UNI-REC alignment

- **Sequential reference**: OneTrans / DIN / CAN family — NS-tokens 가 S-tokens 에 cross-attend → user feature tokens 이 user history 에 의해 enriched. H007 의 1-token candidate (= item embedding) 일반화.
- **Interaction reference**: DCN-V2 (H008 anchor) 그대로 — enriched NS tokens 와 decoded queries 의 explicit polynomial cross.
- **Bridging mechanism**: NS tokens 와 S tokens 이 같은 cross-attention layer 안에서 학습 → **layer-level gradient 공유** (block-level H008 보다 강한 통합). enriched NS tokens 가 DCN-V2 cross 에서 decoded queries 와 polynomial interaction → 두 axis (sequence enrichment + interaction cross) 가 sequential 단계에서 연결됨. §0 P1 ("seq + interaction 한 블록 gradient 공유") 강한 형태.
- **primary_category**: `target_attention` (§17.4 rotation 재진입). 정당화 = H007 일반화 + paper-grade mechanism (OneTrans).
- **Innovation axis**: H007 의 1-token candidate query → N_NS-token query. **paper-grade** (OneTrans NS→S bidirectional 직접 구현). UNI-REC north star (sequence × interaction 통합) 의 sequence axis 강화 paper-grade lift 시도.

### §재진입정당화 (target_attention)

H007 (target_attention 첫) → H008 (sparse_feature_cross) → H009 (hybrid) → H010 (target_attention 재진입).

§10.7 룰 적용 분석:
- 2회 연속 아님 (H007/H008/H009/H010 — H008, H009 가 사이).
- 재진입 정당화 (필수):
  1. **Single-token query → N-token query 일반화**: H007 candidate (1 token) 가 PASS marginal — generalization 단계 자연.
  2. **Paper-grade mechanism**: OneTrans NS→S bidirectional 의 직접 구현. 우리 자체 변형이 아닌 paper-grade source.
  3. **H009 위치 충돌 회피**: H010 통합 위치 (fusion 이전, NS dimension 변경 없음) 가 H009 위치 충돌 패턴 (candidate prepend → seq encoder 출력 변경 → DCN-V2 입력 변경) 명시적 회피.
  4. **H008 anchor on champion**: H010 = H008 + NS xattn stacking sub-H 합법 (H007 단독 + H008 단독 검증 후).

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params (single shared NSToSCrossAttention module):
  - LN(Q) + LN(K)/LN(V): 2 × (2 × D) = 256 params.
  - Q proj + K proj + V proj + out proj: 4 × D² = 4 × 4096 = 16,384 params.
  - **Total**: ~16,640 params (단일 layer, single shared module across 4 domains via concat).
- Total params: ~198M (H008 동일) + 16K = ~198M.
- §10.6 cap 면제 (anchor envelope 동일).
- Sample-scale (extended 30% × 10 epoch ≈ 51M sample steps): 16K params 학습 충분.
- §10.9 attn entropy: N_tokens=384 (S concat). threshold = 0.95 × log(384) ≈ 5.65. 측정 mandatory.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀ MANDATORY**: cross-attention block 의 Pre-LN 분리 적용 (LN(Q), LN(K)/LN(V)). DCN-V2 cross stack (H008) 도 그대로 (첫 step Pre-LN x₀).
- **§10.6 sample budget cap**: anchor envelope 동일 (H008 envelope).
- **§10.7 카테고리 rotation**: target_attention 재진입 정당화 — 본 transfer.md §⑤ 명시.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 가 룰의 두 번째 active 적용. threshold 0.95 × log(384) ≈ 5.65. 모든 layer < threshold 의무. 초과 시 verdict.md `attn_entropy_violation: true` 표시 + abort.
- **§10.10 InterFormer bridge gating σ(−2)**: 본 H 는 새 bridge/gate 추가 없음 — 미적용.
- **§17.2 one-mutation**: NSToSCrossAttention 한 클래스 추가 + flag 분기. ✓ stacking on H008 champion (sub-H 합법).
- **§17.3 binary success**: Δ ≥ +0.5pt platform AUC vs anchor (original_baseline).
- **§17.4 카테고리 rotation 재진입 정당화 명시**.
- **§17.5 sample-scale = code-path verification only**: extended 결과는 mechanism 효과 measurement.
- **§17.6 cost cap**: extended ~3시간, T2 cap 안. 누적 cost ~17시간 (H006~H010).
- **§17.7 falsification-first**: predictions.md 에 paired vs H008 sub-criterion (super-additive / additive / noise / interference 분류).
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§18 inference 인프라 룰**: original_baseline 패키지에서 inherit + H010 의 두 cfg key (use_ns_to_s_xattn, ns_xattn_num_heads) read-back 추가.

# H010 — NS-tokens → S-tokens full bidirectional cross-attention

## What we're trying to explain

H007 verdict F-1: candidate-as-attention-query mechanism class (target_attention 카테고리) 가 우리 데이터에서 +0.5pt 임계 도달 — sequence axis 강화 first confirmed PASS. **단 H007 은 candidate 1 token 만 query 로 사용**.

H008 verdict F-1: DCN-V2 explicit polynomial cross (sparse_feature_cross 카테고리) PASS, +0.0035pt vs H007. **현재 최고 (Platform 0.8387)**.

H009 verdict F-1, F-2: H007 + H008 stacking 이 strongest single (H008) 보다 못함 — combined 0.8364, classic overfit signature (OOF +0.001 / platform −0.0023). **interference 위치 가설**: candidate token prepend → seq encoder 출력 candidate-mixed → DCN-V2 cross 입력 변경. 두 mechanism 이 서로의 input 을 변경.

본 H = **H007 mechanism 의 paper-grade 일반화 + H009 위치 충돌 회피**:
- **NS-tokens 7개 모두** (user_ns 5 + item_ns 2) 가 **S-tokens (per-domain seq encoder 출력)** 에 cross-attention. H007 의 1-token candidate query → 7-token NS query.
- **OneTrans NS→S bidirectional** 직접 구현. paper-grade mechanism.
- **통합 위치**: per-domain seq encoder 출력 → NS-tokens 가 cross-attend → enriched NS-tokens → 기존 query decoder + RankMixer/DCN-V2 fusion 통과. **NS dimension 변경 없음** (just enriched). DCN-V2 cross 입력 토큰 stack 자체는 변경 없이 NS tokens 만 enrichment → **H009 위치 충돌 회피**.
- **H008 anchor 위 single mutation** (DCN-V2 fusion 유지 + NS xattn 추가).

§0 P1 룰 직접 충족 — NS×S layer-level gradient 공유 (block-level 보다 강한 통합). OneTrans NS→S 정신 직접 구현.

## Why now

- **H007 F-1 직접 일반화**: target_attention mechanism class 가 single-token query 로 작동 confirmed → 모든 NS tokens 로 확장 시 lift 강화 가설.
- **H009 F-2 carry-forward**: interference 위치 가설 → **fusion 이전, NS dimension 변경 없는** 통합 위치 채택. 두 mechanism (H008 DCN-V2 + H010 NS xattn) 이 서로의 input 변경 안 함.
- **OneTrans paper-grade source**: arXiv:2510.26104 (Tencent WWW 2026). NS→S bidirectional 의 정수. H004 OneTrans backbone 자체 (smoke-only, invalid) 보다 **mechanism injection** 으로 PCVRHyFormer baseline + DCN-V2 anchor 위 single mutation. paper 의 backbone replacement risk 회피.
- **§17.4 카테고리 rotation 재진입 정당화**: H007 (target_attention) → H008 (sparse_feature_cross) → H009 (hybrid stacking) → H010 (target_attention 재진입). H007 의 일반화 + paper-grade mechanism 추출 = §10.7 재진입 정당화.
- **§17.2 single mutation 깔끔**: 한 클래스 추가 (`NSToSCrossAttention`) + flag 분기. H008 anchor 위 stacking on champion.
- **§0 P1 layer-level gradient sharing**: NS tokens 와 S tokens 이 같은 attention layer 안에서 학습됨. block-level (H008) 보다 강한 통합 — concat-late anti-pattern 의 정반대.
- **앞으로의 anchor candidate**: PASS 시 H010 = 새 anchor (Platform > 0.8387). H011+ 부터 H010 anchor 위 single mutation.

## Scope
- In:
  - 신규 클래스 `NSToSCrossAttention` (model.py 확장, ~80줄):
    - Input: `ns_tokens (B, N_NS, D)`, `s_tokens_concat (B, L_total, D)` (4 도메인 concat), `padding_mask (B, L_total)`.
    - Pre-LN 분리 적용 (LN(Q) + LN(K)/LN(V)).
    - Multi-head cross-attention: Q=NS tokens (N_NS=7), K=V=S tokens (L_total=384, 4 도메인 concat).
    - Output: enriched NS tokens `(B, N_NS, D)` — NS dimension 변경 없음.
    - num_heads = 4 (default = anchor's `num_heads`).
    - softmax-attention entropy 측정 → §10.9 룰 적용 (threshold 0.95 × log(L_total) ≈ 5.65).
  - `MultiSeqHyFormerBlock` 통합:
    - Constructor 인자: `use_ns_to_s_xattn: bool = False`.
    - 통합 위치: per-domain seq encoder 출력 (concat to L_total=384) → `NSToSCrossAttention` → enriched NS → query decoder + DCN-V2 fusion (그대로).
  - PCVRHyFormer constructor 인자 추가: `use_ns_to_s_xattn`, `ns_xattn_num_heads`.
  - CLI flags: `--use_ns_to_s_xattn`, `--ns_xattn_num_heads 4`.
  - infer.py: 두 cfg keys read-back 추가.
  - **H008 mechanism 유지**: `--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8` (anchor 그대로).
  - 그 외 모든 config: H008 와 byte-identical (extended envelope, patience=3).
  - §18 인프라 룰 모두 inherit.
- Out:
  - NS xattn 의 layer 수 (1→2), num_heads tuning — sub-H.
  - Cascading vs concat 도메인 통합 변형 — 별도 H.
  - NS xattn + candidate summary token (H007) 동시 적용 — H009 와 같은 위치 충돌 risk, 별도 stacking H.
  - Backbone replacement (OneTrans full) — H004 sub-H, full-data 도착 후.

## UNI-REC axes
- **Sequential axis**: 변경 없음 — TransformerEncoder per-domain 그대로. **단 NS tokens 가 sequence axis 정보 enrichment**.
- **Interaction axis**: 변경 없음 — DCN-V2 fusion (H008 anchor) 그대로.
- **Bridging mechanism**: NS tokens 와 S tokens 이 같은 cross-attention layer 안에서 학습 → **layer-level gradient 공유** (block-level H008 보다 강한 통합). §0 P1 이상.
- **primary_category**: `target_attention` (§17.4 rotation 재진입, transfer.md §재진입정당화 + paper-grade mechanism).
- **Innovation axis**: candidate single-token query (H007) → all NS-tokens query 일반화. **paper-grade** (OneTrans NS→S bidirectional 직접 구현).

## Success / Failure conditions
**§17.3 binary lift 임계 적용**:

- **Success**: Δ vs anchor (original_baseline) **platform AUC** ≥ **+0.5 pt**. **+ 4 부수 게이트**:
  1. Train NaN-free 완주 (또는 patience=3 early stop 정상).
  2. Inference: §18 인프라 통과 (batch heartbeat + `[infer] OK` 로그).
  3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, best_val_AUC, best_oof_AUC, use_ns_to_s_xattn=true, fusion_type=dcn_v2}` 기록.
  4. infer.py 가 새 cfg key read-back → strict load 통과.
  5. §10.9 attn entropy < 5.65 (= 0.95 × log(384)).
- **Failure**: Δ < +0.5pt → REFUTED. target_attention 일반화 가설 약화.

**부수 sub-criterion (paired vs H008 strongest single)**:
- **super-additive**: Δ vs H008 ≥ +0.005pt (paper-grade, NS×S 통합 가치 confirmed).
- **additive**: Δ vs H008 ∈ [+0.001, +0.005pt] (mechanism class 강화 confirm).
- **noise**: Δ vs H008 ∈ [−0.001, +0.001pt] (NS xattn 효과 marginal).
- **interference**: Δ vs H008 < −0.001pt (H009 와 같은 위치 충돌 — REFUTED, mechanism class 한계).

## Frozen facts referenced
- Anchor (original_baseline) Platform AUC: ~0.83X (smoke).
- H007 PASS marginal: Platform 0.8352 (target_attention 1-token query).
- H008 PASS: Platform 0.8387 (sparse_feature_cross, **현재 최고**).
- H009 REFUTED interference: Platform 0.8364 (combined H007+H008, 위치 충돌).
- H006/H007/H008/H009 모두 extended envelope (10ep × 30%, patience=3-5).
- §10.5 LayerNorm on x₀ MANDATORY.
- §10.9 OneTrans softmax-attention entropy abort threshold = 0.95 × log(N_tokens).
- §18 인프라 룰.
- OneTrans paper: arXiv:2510.26104 (Tencent WWW 2026).
- `papers/unified_backbones/onetrans_tencent.md` — H010 의 paper-grade source.

## Inheritance from prior H

- **H007 F-1** → mechanism class 작동 confirmed → 일반화 단계.
- **H007 F-2** → val ↔ platform 정합 expected.
- **H008 F-1** → DCN-V2 fusion (interaction axis) 유지 (anchor on champion).
- **H008 F-4** → patience=3 + early stop aggressive (H009 에서 trigger 안 됐지만 envelope discipline 유지).
- **H009 F-2** → interference 위치 충돌 회피 → fusion 이전 + NS dimension 변경 없는 통합 위치 채택.
- **H009 F-4** → H008 anchor 위 stacking on champion.
- **H004 F-1** → §10.9 attn entropy 룰 active 적용 (threshold 0.95 × log(384) ≈ 5.65).
- **§18 인프라 룰**: original_baseline 패키지 inherit.

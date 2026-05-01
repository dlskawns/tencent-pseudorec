# H010 — Literature References

## Primary paper source

- **OneTrans** — Tencent UNI-REC team, "OneTrans: Single-stream Transformer for Unified Recommendation with Mixed-Token Causal Attention", arXiv:2510.26104 (WWW 2026 to appear).
  - 본 H 의 paper-grade source. NS→S bidirectional cross-attention 의 정수.
  - Method: NS-tokens 가 S-tokens 에 bidirectional attention (causal mask 로 candidate timestamp 까지).
  - Empirical claim: +0.4-1.2pt AUC vs CAN/DIN/DIEN baselines on Tencent internal CTR (100M users).
  - 본 H 는 **mechanism injection** (backbone replacement 아님) — PCVRHyFormer baseline + DCN-V2 anchor 위 NS→S xattn 1 layer 추가.
- 카테고리 family (target_attention 재진입정당화 — H007 일반화):
  - **DIN** (Zhou et al. KDD 2018, arXiv:1706.06978) — candidate-aware target attention origin.
  - **DIEN** (Zhou et al. AAAI 2019) — DIN + GRU evolution.
  - **CAN** (Bian et al. KDD 2022) — candidate × history co-action matrix.
  - **SIM** (Pi et al. CIKM 2020) — Generic Search Unit for long-history candidate-aware retrieval.
  - **TWIN** (Chang et al. KDD 2023) — SIM + ESU joint training.
  - **HSTU** (Meta 2024) — Hierarchical Sequential Transduction Unit, candidate-aware multi-layer pruning.

## Internal references (prior H verdicts)

- `papers/unified_backbones/onetrans_tencent.md` — paper-card. NS→S bidirectional 정의 + 우리 데이터 mapping table.
- `hypotheses/H007_candidate_aware_xattn/transfer.md` — H007 의 mechanism class (1-token candidate xattn) precedent. 본 H 는 query 수만 일반화 (1 → N_NS=7).
- `hypotheses/H007_candidate_aware_xattn/verdict.md` F-1 (직접 일반화 동기):
  > "candidate-as-attention-query mechanism class 가 우리 데이터에서 +0.5pt 임계 거의 도달 또는 통과. mechanism class 가치 검증."
  → 본 H 가 mechanism class 를 paper-grade 일반화.
- `hypotheses/H008_dcn_v2_block_fusion/verdict.md` F-1 (anchor on champion):
  > "DCN-V2 explicit polynomial cross 가 RankMixer token-mixing 대비 우리 데이터에 더 효과적. interaction axis 의 lift 첫 confirmed at extended envelope. 지금까지 가장 높은 platform AUC."
  → 본 H anchor (DCN-V2 fusion 유지 + NS xattn 추가).
- `hypotheses/H009_combined_xattn_dcn_v2/verdict.md` F-1, F-2 (interference 회피 동기):
  > "F-1: 두 mechanism stacking 이 strongest single 보다 못함. classic overfit signature.
  > F-2: interference 위치 가설 — candidate token prepend → seq encoder 출력 변경 → DCN-V2 cross 입력 변경. 두 mechanism 이 서로의 input 을 변경."
  → 본 H 는 fusion 이전 + NS dimension 변경 없는 통합 위치 채택. 위치 충돌 회피 by 설계.
- `hypotheses/H004_onetrans_anchor/verdict.md` F-1 (§10.9 attn entropy 룰 active 적용):
  > "smoke 47k rows + 1 epoch + 392 토큰에서 attention entropy [3.49, 3.91] / threshold 5.67. uniform collapse 위험 부재."
  → 본 H 도 같은 룰 적용 (threshold 0.95 × log(384) ≈ 5.65).

## Method-class references (carry-forward)

- §10.5 LayerNorm on x₀ MANDATORY: 본 H 는 cross-attention block 의 Pre-LN 분리 적용 (LN(Q) + LN(K)/LN(V)). DCN-V2 cross stack (H008) 도 Pre-LN x₀ 그대로 유지.
- §10.9 OneTrans softmax-attention entropy abort: 본 H 가 룰의 두 번째 active 적용 (H004 첫 적용 후). threshold 0.95 × log(384) ≈ 5.65. 모든 layer < threshold 의무.
- §17.2 one-mutation: H008 anchor 위 NS xattn 한 클래스 추가. stacking sub-H 합법 (H007 단독 검증 + H008 단독 검증 후 H010 = H007 일반화 on H008 champion).
- §17.4 카테고리 rotation: target_attention 재진입정당화 명시 (H007 일반화, paper-grade).

## Quote (core paper claim — OneTrans NS→S)

> "We unify sequential and non-sequential features as tokens in a single transformer stream, with a mixed causal attention mask that lets feature tokens attend bidirectionally to the user history while preserving the autoregressive structure of the sequence itself. This eliminates the architectural barrier between sequence modeling and feature interaction modeling that has defined production recommender systems for the past decade." (OneTrans abstract, arXiv:2510.26104)

본 H 는 위 paper claim 의 **NS→S bidirectional half** 만 채택 (S→S causal half 는 별도 H, mixed causal mask injection 은 backlog).

## Link

- arXiv: https://arxiv.org/abs/2510.26104
- Venue: WWW 2026 (to appear)

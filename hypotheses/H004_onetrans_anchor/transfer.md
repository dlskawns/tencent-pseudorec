# H004 — Method Transfer

## ① Source
[`papers/unified_backbones/onetrans_tencent.md`](../../papers/unified_backbones/onetrans_tencent.md) — OneTrans (Tencent UNI-REC team, "OneTrans: Single-stream Transformer for Unified Recommendation with Mixed-Token Causal Attention", arXiv:2510.26104, WWW 2026 to appear).

## ② Original mechanism

OneTrans 는 production CTR 시스템에서 분리되어 발전한 두 axis (sequence modeling: SASRec/DIN/HSTU vs feature interaction: DCN-V2/CAN/PLE) 를 **하나의 transformer stream 안 token 으로 통합**. 핵심 메커니즘 4 가지:

1. **Token taxonomy**:
   - **S-tokens**: per-event sequence embeddings (per-domain or merged), positional encoding 추가.
   - **NS-tokens**: scalar/dense/multi-val features 를 N 개 group 으로 parameter-free chunking (RankMixer 와 유사). 각 group → 1 token of d_model.
   - **Candidate token**: 추론 대상 item 의 embedding + bucket id (timestamp 기반).

2. **Mixed causal attention mask**:
   - S→S: causal upper-triangular (시퀀스 future 차단).
   - NS→S: full bidirectional, candidate timestamp 미만 S-token 만 (label leakage 방지).
   - NS→NS: full self-attention.
   - Candidate→{S, NS}: candidate timestamp 까지 attend.

3. **Pyramid pruning**: layer ℓ 에서 attention probability mass top-K_ℓ S-token 만 보존, K_ℓ monotone decreasing → compute O(r^ℓ · L · d).

4. **Output head**: candidate token 의 final-layer hidden state → linear → logit.

Paper claims: (a) **Universality** — NS=0 + pruning off ⇒ SASRec, S=0 ⇒ FT-Transformer; (b) **Compute** — pyramid pruning 으로 FLOPs 3–5× 감소; (c) **Empirical** — Tencent internal 100M user CTR 에서 +0.4–1.2 pt AUC vs CAN/DIN/DIEN.

## ③ What we adopt

- **Single-stream transformer block**: PCVRHyFormer 의 `MultiSeqHyFormerBlock` (per-domain encoder → query decoder → RankMixerBlock fusion) 을 폐기, OneTrans single-stream block 으로 교체.
- **Token taxonomy** (paper 그대로):
  - S-tokens: 4 도메인 시퀀스 (a/b/c/d) 의 이벤트 embedding + 도메인 식별 embedding 추가. d_model=64.
  - NS-tokens: PCVRHyFormer baseline 의 `RankMixerNSTokenizer` 가 만드는 user_ns_tokens=5 + item_ns_tokens=2 = **7 NS tokens** 그대로 재사용 (parameter-free chunking).
  - Candidate token: 기존 `cand_*` features → 1 token.
- **Mixed causal attention mask**: paper 의 sub-mask 정의 그대로:
  ```
  M[i, j] = 1 if can_attend(token_i, token_j) else -inf
  S→S: causal (i ≥ j in sequence order)
  NS→S: bidirectional but j.timestamp < candidate.timestamp
  NS→NS: full
  Candidate→all: full up to candidate.timestamp
  ```
- **Output head**: candidate token 의 final hidden state → `nn.Linear(d_model, 1)` → sigmoid (BCE/focal 호환).
- **§10.9 룰 적용 (첫 적용)**: 매 layer 의 attention prob 의 평균 entropy 기록 → `metrics.json` `attn_entropy_per_layer` 필드 + abort threshold check.

## ④ What we modify (NOT a clone)

- **Pyramid pruning 미적용**: paper 의 compute optimization 이라 anchor 단계 (smoke 47k rows + full TBD) 에 불필요. K_ℓ = L (모든 token 보존), full attention 으로 단순화. **이유**: pruning 은 별도 H (anchor 결정 후) 의 mutation 자료. 본 H anchor 가 pruning 없이도 학습 가능함을 검증해야 미래 H 가 paired Δ 측정 가능.
- **Per-domain S-token (NOT merged stream)**: paper 는 S-token sequence 를 per-domain 또는 merged 둘 다 가능. 우리는 **per-domain + 도메인 ID embedding** 으로 고정 — 도메인 dimension 보존 + interpretability. merged stream 은 별도 H.
- **NS-token 수 고정**: paper 는 N 자유. 우리는 RankMixerNSTokenizer 의 **user=5 + item=2 = 7 NS tokens** PCVRHyFormer 와 동일하게 고정. 이유: PCVRHyFormer-anchor 와 paired 비교 가능. NS-token granularity 변경은 별도 H.
- **mixed causal mask 의 ambiguous 점 보수적 해석**: paper 본문 "NS→S bidirectional up to the candidate position" 에서 "candidate position" 의 정의가 시퀀스 인덱스 vs timestamp 모호. 우리는 **timestamp 기반** (candidate.timestamp 미만 S-token 만) — 더 보수적, label_time leakage 방지 (CLAUDE.md §4.3 룰 강건 충족).
- **Sparse param 분류**: PCVRHyFormer 의 dual optimizer 는 high-cardinality embedding 을 Adagrad sparse 로, dense 를 AdamW 로. OneTrans single-stream block 의 **Q/K/V projection, FFN, layernorm 은 모두 dense AdamW**. domain embedding 은 low-cardinality (4 개) → AdamW. token embedding (S-token feature lookup) 은 H001 과 동일 분류 유지.
- **Output head 위치**: paper 는 candidate token 의 final hidden 만. 우리는 candidate token + 첫번째 NS-token (= mean-pooled user representation) 의 concat → linear. 이유: H001 baseline 의 classifier 가 query token concat 형태였음, 본 H anchor 가 minimal change 로 paired 비교 가능하도록 head dim 만 정렬.

## ⑤ UNI-REC alignment

- **Sequential reference**: SASRec / HSTU — S→S causal sub-mask 이 SASRec 의 self-attention 과 동등. 도메인 ID embedding 으로 4 도메인 differentiation.
- **Interaction reference**: DCN-V2 / CAN / FwFM — NS→NS full self-attention 이 DCN-V2 의 explicit cross 를 attention 으로 재구현. NS→S 는 user feature × user history 의 token-level explicit cross.
- **Bridging mechanism**: 한 transformer block 안 attention layer 의 Q/K/V 를 S/NS 가 공유 → **layer 마다 두 축 gradient 공유**. PCVRHyFormer 의 block fusion (RankMixerBlock 한 번) 대비 layer-level fusion (블록당 attention 한 번 + FFN 한 번 = layer 수 × 통합 횟수). §0 P1 정의 ("seq + interaction 한 블록 gradient 공유") 보다 강한 조건 — layer 단위.
- **primary_category**: `unified_backbones` (재진입; challengers.md §재진입정당화 + H002 verdict.md F-3 인용).
- **Innovation axis**: 통합 깊이 = block fusion → layer/token fusion. PCVRHyFormer-anchor 와 OneTrans-anchor 가 공존하면 미래 H 들이 두 통합 깊이 중 어느 쪽이 우리 데이터에 적합한지 paired 비교 가능.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 예상 trainable params: PCVRHyFormer 198M 와 비슷 — embedding tables 가 dominant, single-stream block 자체는 <1M.
- §10.6 soft cap (≤ 2146) 100x+ 초과. 그러나 본 H 는 **anchor**이므로 cap 면제 (H001 과 동일).
- demo_1000 (47k rows train) 에서의 학습 결과의 generalization 은 주장 안 함. `claim_scope: "demo-only"`.
- Full-data 도착 시 동일 config 로 재실행 → E_H004.full.
- §10.9 룰: smoke 결과 attn entropy ≥ 0.95·log(N) 한 layer 라도 발견되면 verdict.md 에 plus-flag, full-data 시도 보류.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x0 MANDATORY**: OneTrans single-stream block 은 Pre-LN convention 사용 (paper 본문 "PreLayerNorm + residual"). 첫 layer 입력에 LayerNorm 통과 → 자동 충족.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 가 룰의 **첫 active 적용**. trainer 에 epoch end 마다 random batch 의 attention prob 에서 layer 별 평균 entropy 측정해 metrics.json 에 dump. threshold = 0.95·log(N_tokens). 초과 시 verdict.md `attn_entropy_violation: true` 표시 + full-data 학습 보류.
- **§10.10 InterFormer bridge gating σ(−2) init**: 본 H 는 새 bridge 추가 없음 — 미적용.
- **§17.2 one-mutation-per-experiment**: 본 H 는 **anchor 면제**. backbone 전체 replacement 라 single mutation 정의 미적합. challengers.md §재진입정당화 + 본 transfer.md §⑤ 에 명시 정당화.
- **§17.5 sample-scale = code-path verification only**: 본 H smoke 결과는 anchor 자격 (val_AUC ≥ 0.7, NaN-free, submission round-trip) 검증 용도. paper 의 lift claim 검증은 별도 H + full-data.
- **§17.7 falsification-first**: predictions.md 에 4 게이트 모두 명시. 어느 하나라도 fail → REFUTED + carry-forward.
- **§17.8 cloud handoff discipline**: training_request.md 생성 + flat upload + run.sh local-fallback 제거 + git_sha pin.
- **§4.3 label_time-aware split MANDATORY**: H001 패치 그대로 재사용. organizer row-group split 모드도 control 비교용으로 옵션 유지.
- **§4.4 OOF holdout 10%**: H001 과 동일하게 seed=42, oof_user_ratio=0.10. metrics.json 에 best_val_AUC + best_oof_AUC 양쪽 기록.

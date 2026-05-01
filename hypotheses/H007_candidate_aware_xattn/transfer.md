# H007 — Method Transfer

## ① Source

차용 = **mechanism class** ("candidate as attention query"), 특정 paper 1:1 reproduce 아님. paper family:

- **DIN** (Deep Interest Network) — Zhou et al. KDD 2018. arXiv:1706.06978. Candidate-aware target attention 의 origin paper.
- **DIEN** (Zhou et al. AAAI 2019) — DIN + GRU evolution.
- **DSIN** (Feng et al. IJCAI 2019) — session-aware DIN.
- **CAN** (Co-action Network) — Bian et al. KDD 2022. Candidate × history co-action matrix.
- **SIM** (Pi et al. CIKM 2020) — Generic Search Unit (GSU) for long-history candidate-aware retrieval.
- **TWIN** (Chang et al. KDD 2023) — SIM + ESU joint training.
- **HSTU** (Meta 2024) — Hierarchical Sequential Transduction Unit, candidate-aware multi-layer pruning.
- **OneTrans** (Tencent WWW 2026, arXiv:2510.26104) — single-stream candidate token + mixed-causal mask.

## ② Original mechanism (mechanism class 전체)

PCVRHyFormer baseline 의 query decoder (`MultiSeqQueryGenerator`) 가 per-domain **learnable Nq queries** 로 history 압축 — candidate-irrelevant pool. 같은 user 라도 추천 candidate 에 따라 봐야 할 history events 가 다를 텐데 그걸 무시.

Candidate-aware mechanism class:
1. Candidate item embedding 만들기 (paper 마다 다름: ID embedding / feature pool / co-action matrix).
2. Candidate 를 attention query 로 → history seq 에 cross-attention.
3. Candidate-relevant history events 가 weighted 되어 pooled representation 생성.
4. Pooled representation → downstream classifier.

**Modern transformer-style implementation**:
- Pre-LN convention.
- Multi-head cross-attention (Q=candidate, K=V=seq).
- Padding mask aware.
- Residual + output projection.

## ③ What we adopt

- Mechanism class: **candidate-aware multi-head cross-attention pooling**. minimum viable form.
- 신규 module `CandidateSummaryToken`:
  - Input: `seq_tokens (B, L, D)`, `padding_mask (B, L)`, `candidate_token (B, 1, D)`.
  - Pre-LN 분리 적용 (LN(Q), LN(K)/LN(V)).
  - Multi-head cross-attention: Q=candidate (1 token), K=V=seq_tokens (L tokens).
  - Output: 1 candidate-attended pooled summary `(B, 1, D)`.
  - num_heads = 4 (default = anchor's `num_heads`).
- Per-domain instantiation: 4 도메인 (a/b/c/d) 각각 독립 `CandidateSummaryToken` module.
- 통합 위치: per-domain seq encoder 출력 직후, query decoder 직전. Candidate summary 를 seq 시작에 **prepend** (L → L+1).
- Candidate token 구성: `item_ns + item_dense_tok` 가 만든 (B, 7, D) 토큰들의 **mean pool** → (B, 1, D).
- CLI flag: `--use_candidate_summary_token`.

## ④ What we modify (NOT a clone)

- **paper 의 dedicated DIN architecture 미적용**: DIN paper 는 candidate-aware attention 이 main path (다른 mechanism 거의 없음). 우리는 **augment** (PCVRHyFormer 의 query decoder + RankMixer fusion 다 보존, candidate summary token 만 추가). 효과 약화 가능성 (challengers Frame 1) 있지만 §17.2 one-mutation 깔끔 + 다운스트림 호환.
- **candidate token = mean pool of item_ns + item_dense_tok**: paper 들의 raw item ID embedding 사용과 다름. 우리 organizer baseline 의 representation 활용. learnable weighted pool 또는 first token 사용은 별도 H.
- **prepend (start of seq)**: append (end) 도 가능. prepend 채택 이유: seq 시작 position RoPE 0 이 candidate "context anchor" 역할로 자연스러움. append 시 candidate 가 마지막 event 처럼 보이는 risk.
- **Per-domain 독립 module**: 4 modules. shared module 도 가능. 독립 채택 이유: 도메인별 다른 candidate-history pattern 학습. 단 trainable params 4x.
- **2018 DIN 의 element-wise concat + MLP 미적용**: 우리는 standard multi-head cross-attention block. paper 의 archaic 한 implementation 이 아니라 modern transformer convention.
- **§17.2 one-mutation**: CandidateSummaryToken 클래스 추가 + flag 분기 + forward 통합 = 한 mechanism 추가. NS-token 수, num_queries, num_blocks 등 다른 hyperparameter 변경 없음.

## ⑤ UNI-REC alignment

- **Sequential reference**: DIN/CAN/SIM/TWIN/HSTU/OneTrans family — per-event sequential representation 에 candidate-aware read-out 추가.
- **Interaction reference**: item embedding (item_int + item_dense) 가 cross-attention query 로 작용 → item × user-history interaction layer-level 추가. RankMixer fusion downstream 보존 → 두 axis 모두 강화.
- **Bridging mechanism**: candidate token 이 4 도메인 seq 모두에 동일 query 로 들어감 → candidate 가 cross-domain integration 단위. CLAUDE.md §0 P1 정의 ("seq + interaction 한 블록 gradient 공유") 충족.
- **primary_category**: `target_attention` (§17.4 rotation 첫 충족 추가).
- **Innovation axis**: H006 의 random/probability-based selection → candidate-aware selection. mechanism class 이동.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: per-domain (Pre-LN ×2 + Q proj + KV proj + out proj) × 4 domains. 대략 4 × (d_model² × 3 + 2 × d_model × 2) ≈ 4 × (12,288 + 256) ≈ **50K params**.
- Total params: PCVRHyFormer ~198M + 50K ≈ +0.025% (negligible).
- Sample-scale (5%-data 47k rows): 50K params 학습 가능 — paper-grade attention layer 라 데이터 크기 의존성 작음. anchor envelope 동일.
- §10.6 cap 면제 (anchor 와 동일).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x0**: 본 H 는 Pre-LN (Q + K/V) 분리 적용. ✓
- **§10.6 sample budget cap**: anchor envelope 동일.
- **§10.7 카테고리 rotation**: H007 = 첫 target_attention. 추가 충족.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 의 cross-attention 도 softmax. 만약 attention prob uniform collapse 시 lift 약화 신호 — instrumentation 별도 sub-H.
- **§10.10 InterFormer bridge gating σ(−2)**: 본 H 는 새 bridge/gate 추가 없음 — 미적용.
- **§17.2 one-mutation**: CandidateSummaryToken 한 클래스 추가 + flag 분기. ✓
- **§17.3 binary success**: Δ ≥ +0.5pt platform AUC vs anchor.
- **§17.4 카테고리 rotation 추가 충족**.
- **§17.5 sample-scale = code-path verification only**: smoke 결과는 mechanism 효과 measurement.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§18 inference 인프라 룰**: original_baseline 패키지에서 그대로 inherit (batch=256 default, PYTORCH_CUDA_ALLOC_CONF, universal handler, 진단 로그). infer.py 에 `use_candidate_summary_token` cfg.get read-back 추가만.

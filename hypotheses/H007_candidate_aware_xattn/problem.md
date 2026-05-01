# H007 — Candidate-aware cross-attention pooling per domain

## What we're trying to explain

H006 (LongerEncoder, top-K=50 self-attention compression) REFUTED platform 0.82 vs anchor 0.83. F-1: top-K selection 이 **candidate 와 무관하게** self-attention probability 로 선택 → random 에 가까움 + 우리 envelope 의 seq_max_lens=128 < K=50 일 때 단순 truncation. 정보 손실만 발생.

본 H = **그 fail 의 직접 후속**. PCVRHyFormer baseline 의 `MultiSeqQueryGenerator` 가 per-domain 으로 **learnable Nq queries 만** 사용해 history 압축 — candidate item 이 무엇이든 같은 query → candidate-irrelevant pooling. 같은 user 라도 어떤 item 을 추천하느냐에 따라 history 에서 보아야 할 events 가 다른데 그걸 무시.

Candidate-aware cross-attention pooling = **candidate item embedding 을 query 로 만들어 history seq 에 cross-attention** → "이 후보에 관련 있는 history events 만 weighted" representation. CTR 도메인 표준 lever (DIN family 의 mechanism class).

본 H 의 차용 = **mechanism class** ("candidate as attention query"), 특정 paper 의 archaic 한 implementation 아님. 우리는 **modern multi-head cross-attention (Pre-LN)** 으로 구현.

## Why now

- **H006 refutation 직접 후속**: F-1 carry-forward — random/probability-based selection 한계 → candidate-aware 로 재구성.
- **§17.4 카테고리 rotation**: H001/H002/H004 unified_backbones, H005 loss_calibration, H006 long_seq_retrieval. **H007 = target_attention 첫 적용** → rotation 추가 충족.
- **paper grade reference**: DIN (Zhou et al. KDD 2018), CAN (KDD 2022), SIM (CIKM 2020), TWIN (KDD 2023), HSTU (Meta 2024), OneTrans (Tencent WWW 2026 candidate token). 영감 풍부, 검증된 lever.
- **§18 infra ready**: original_baseline 패키지가 §18 룰 모두 적용 (batch=256 default, PYTORCH_CUDA_ALLOC_CONF, universal handler, 진단 로그). H007 inherit.

## Scope
- In:
  - 신규 클래스 `CandidateSummaryToken` (model.py 확장, ~80줄):
    - Input: per-domain `seq_tokens (B, L, D)`, `padding_mask (B, L)`, `candidate_token (B, 1, D)`.
    - Output: `(B, 1, D)` candidate-attended pooled summary.
    - Architecture: Pre-LN (Q + K/V) → multi-head cross-attention → residual → output projection.
  - PCVRHyFormer 통합:
    - Constructor 인자: `use_candidate_summary_token: bool = False`, `candidate_summary_num_heads: int = 4`.
    - Forward path: `_build_token_streams` 안에서 candidate token (item_ns + item_dense_tok mean pool) 만들어 per-domain 으로 `CandidateSummaryToken` 호출 → seq 시작에 prepend → seq_tokens (B, L+1, D), padding_mask 도 갱신.
    - 다운스트림 (`MultiSeqQueryGenerator`, `MultiSeqHyFormerBlock` stack) 자동 candidate-aware 경유.
  - CLI flag: `--use_candidate_summary_token`.
  - infer.py: `cfg.get` 으로 train_config 에서 새 인자 read-back.
  - 그 외 모든 config: anchor 와 byte-identical envelope (smoke: train_ratio=0.05, num_epochs=1, halved seq_max_lens, NS=5+2=7, num_queries=2, BCE, label_time split).
  - §18 인프라 룰 모두 inherit from original_baseline.
- Out:
  - candidate token 구성 변경 (예: item_id 직접 embedding) — 별도 H.
  - num_queries 변경 — T constraint 위배.
  - CAN co-action / HSTU hierarchical 같은 candidate-aware 의 modern 변형 — H008+ carry-forward.
  - target attention 의 multi-layer stacking — 별도 H.

## UNI-REC axes
- **Sequential axis**: per-domain seq encoder 출력에 candidate-aware pooling 추가. seq representation 을 candidate 관점에서 read-out.
- **Interaction axis**: candidate item embedding (item_int + item_dense pool) 가 cross-attention query 로 작용 → item × user-history interaction layer-level 추가.
- **Bridging mechanism**: candidate token 이 4 도메인 seq 모두에 동일 query 로 들어감 → candidate 가 cross-domain 통합 단위. RankMixerBlock fusion downstream 그대로 → block-level fusion 보존.
- **primary_category**: `target_attention` (§17.4 rotation 첫 충족 추가).
- **Innovation axis**: H006 의 random/probability-based selection → candidate-aware selection. mechanism class 이동.

## Success / Failure conditions
**§17.3 binary lift 임계 적용**:

- **Success**: Δ vs anchor (original_baseline) **platform AUC** ≥ **+0.5 pt**. **+ 4 부수 게이트**:
  1. Train 1 epoch NaN-free 완주.
  2. Inference: §18 인프라 통과 (batch heartbeat + `[infer] OK` 로그, no fallback, platform AUC ≠ 0.5).
  3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, best_val_AUC, best_oof_AUC, split_meta, use_candidate_summary_token}` 모두 채워짐.
  4. infer.py 가 새 cfg key 읽어 모델 재구성 시 strict load 통과.
- **Failure**:
  - Δ < +0.5pt platform AUC → REFUTED. target_attention 카테고리 일시 archive.
  - 게이트 1–4 중 1개라도 fail → 코드/계약 회귀.

## Frozen facts referenced
- Anchor (original_baseline) Platform AUC: ~0.83X (확정).
- H006 verdict F-1, F-3, F-5 carry-forward.
- §18 인프라 룰 (CLAUDE.md 신설 2026-04-28).
- DIN/CAN/SIM/TWIN/HSTU/OneTrans paper family — mechanism class reference.
- §10.6 sample budget cap: anchor 면제.

## Inheritance from prior H (carry-forward)

- **H006 F-1**: random/probability-based candidate selection 한계 → 본 H 가 candidate-aware 로 mechanism class 이동.
- **H006 F-3**: paired Δ 는 platform AUC 으로만. 본 H 도 platform AUC 기준.
- **H006 F-5**: §18 인프라 룰 H007 패키지 inherit.
- **H004 verdict F-1 (P3 PASS)**: OneTrans candidate token 도 같은 mechanism class. 본 H 가 더 minimal viable form (1 candidate token + 1 cross-attention layer per domain).
- **H005 verdict F-1**: BCE 12% imbalance 영역 충분 → 본 H BCE 유지.
